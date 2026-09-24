"""Command-line training entry point."""

import argparse
import json
import random
from pathlib import Path

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
        default=Path("datasets/fma_metadata/tracks.csv"),
        help="CSV de metadados do FMA.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/ast-finetuned"))
    parser.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--subset", choices=["small", "medium", "large"], default="small", help="Subconjunto do FMA a ser usado.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ASTFeatureExtractor.from_pretrained(args.model_name)
    mel_transform = ASTMelSpectrogram(mean=extractor.mean, std=extractor.std)
    train_set = FMADataset(
        args.data_dir,
        args.metadata,
        subset=args.subset,
        train=True,
        mel_transform=mel_transform,
    )
    val_set = FMADataset(
        args.data_dir,
        args.metadata,
        subset=args.subset,
        train=False,
        class_names=train_set.classes_,
        mel_transform=mel_transform,
    )
    labels = train_set.classes_
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)

    model = ASTFineTuner.from_backbone(args.model_name, num_labels=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = _prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _prepare_batch(batch, device)
                output = model(**batch)
                correct += (output["logits"].argmax(dim=-1) == batch["labels"]).sum().item()
                total += batch["labels"].numel()
        accuracy = correct / total if total else 0.0
        print(f"epoch={epoch:03d} train_loss={train_loss / max(1, len(train_loader)):.4f} val_accuracy={accuracy:.4f}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            model.save_pretrained(args.output_dir)
            extractor.save_pretrained(args.output_dir)
            (args.output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
