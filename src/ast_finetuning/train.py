"""Command-line training entry point."""

import argparse
import json
import random
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, default_collate
from transformers import ASTFeatureExtractor

from .data import ASTMelSpectrogram, FMADataset
from .model import ASTFineTuner


def _prepare_batch(batch: object, device: torch.device) -> dict[str, torch.Tensor]:
    """Convert the dataset's possible list/tuple output to model inputs."""
    if isinstance(batch, list):
        if batch and all(isinstance(item, dict) for item in batch):
            batch = default_collate(batch)
        elif len(batch) == 2:
            batch = {"input_values": batch[0], "labels": batch[1]}
    elif isinstance(batch, tuple) and len(batch) == 2:
        batch = {"input_values": batch[0], "labels": batch[1]}

    if not isinstance(batch, dict):
        raise TypeError("Expected a mapping or (input_values, labels) batch")
    return {key: value.to(device) for key, value in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ajuste fino do AST em arquivos de áudio locais.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("datasets/fma_tracks.csv"),
        help="CSV de metadados do FMA.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/ast-finetuned"))
    parser.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--subset", choices=["small", "medium", "large"], default="small", help="Subconjunto do FMA a ser usado.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate recomendado para fine-tuning do backbone AST completo.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ASTFeatureExtractor.from_pretrained(args.model_name)
    train_mel_transform = ASTMelSpectrogram(
        mean=extractor.mean,
        std=extractor.std,
        is_training=True,
    )
    validation_mel_transform = ASTMelSpectrogram(
        mean=extractor.mean,
        std=extractor.std,
        is_training=False,
    )
    train_set = FMADataset(
        args.data_dir,
        args.metadata,
        subset=args.subset,
        train=True,
        mel_transform=train_mel_transform,
    )
    val_set = FMADataset(
        args.data_dir,
        args.metadata,
        subset=args.subset,
        train=False,
        class_names=train_set.classes_,
        mel_transform=validation_mel_transform,
    )
    labels = train_set.classes_
    if len(val_set) == 0:
        raise ValueError("O conjunto de validação do FMA está vazio.")
    print(
        f"train_samples={len(train_set)} "
        f"validation_samples={len(val_set)} "
        f"num_classes={len(labels)} "
        f"classes={labels}"
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)

    model = ASTFineTuner.from_backbone(args.model_name, num_labels=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.csv"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            batch = _prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            batch_size = batch["labels"].size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (output["logits"].argmax(dim=-1) == batch["labels"]).sum().item()
            train_total += batch_size

        train_loss = train_loss_sum / max(1, train_total)
        train_accuracy = train_correct / max(1, train_total)
        model.eval()
        val_loss_sum = 0.0
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _prepare_batch(batch, device)
                output = model(**batch)
                batch_size = batch["labels"].size(0)
                val_loss_sum += output["loss"].item() * batch_size
                correct += (output["logits"].argmax(dim=-1) == batch["labels"]).sum().item()
                total += batch_size
        val_loss = val_loss_sum / max(1, total)
        val_accuracy = correct / max(1, total)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
        }
        history.append(metrics)
        pl.DataFrame(history).write_csv(history_path)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"validation_loss={val_loss:.4f} "
            f"validation_accuracy={val_accuracy:.4f}"
        )
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            model.save_pretrained(args.output_dir)
            extractor.save_pretrained(args.output_dir)
            (args.output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
