"""Command-line training entry point."""

import argparse
import json
import random
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader, default_collate
from transformers import ASTFeatureExtractor

from .data import ASTMelSpectrogram, FMADataset, GTZANDataset
from .metrics import macro_metrics, save_best_epoch_metrics
from .model import DEFAULT_LORA_TARGET_MODULES, FINETUNING_METHODS, ASTFineTuner


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
    parser.add_argument(
        "--dataset",
        choices=["fma_small", "fma_medium", "gtzan"],
        default="fma_small",
        help="Dataset de treino. Cada dataset gera um modelo separado, pois os gêneros são diferentes.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("datasets/fma_tracks.csv"),
        help="CSV de metadados do FMA (ignorado no GTZAN).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Pasta do checkpoint. Padrão: checkpoints/ast-{dataset}-{finetuning}.",
    )
    parser.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument(
        "--pitch-shift-semitones",
        type=int,
        nargs="*",
        default=[-1, 1],
        help=(
            "Deslocamentos de pitch (em semitons) usados como data augmentation no treino do GTZAN; "
            "passe a flag sem valores para desativar. Ignorado no FMA."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--finetuning",
        choices=FINETUNING_METHODS,
        default="head",
        help=(
            "Metodologia de ajuste fino: 'head' treina apenas a cabeça Softmax com o backbone AST congelado; "
            "'lora' treina adaptadores LoRA no backbone junto com a cabeça."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate dos parâmetros treináveis (cabeça classificadora e, com LoRA, os adaptadores).",
    )
    parser.add_argument("--lora-rank", type=int, default=1, help="Rank r das matrizes LoRA.")
    parser.add_argument("--lora-alpha", type=int, default=2, help="Fator de escala alpha do LoRA.")
    parser.add_argument("--lora-dropout", type=float, default=0.2, help="Dropout aplicado na entrada dos adaptadores LoRA.")
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_MODULES),
        help="Camadas lineares do AST que recebem adaptadores LoRA (ex.: q_proj k_proj v_proj o_proj fc1 fc2).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path("checkpoints") / f"ast-{args.dataset}-{args.finetuning}"
    return args


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
    if args.dataset.startswith("fma_"):
        fma_subset = args.dataset.removeprefix("fma_")
        train_set = FMADataset(
            args.data_dir,
            args.metadata,
            subset=fma_subset,
            split="training",
            mel_transform=train_mel_transform,
        )
        val_set = FMADataset(
            args.data_dir,
            args.metadata,
            subset=fma_subset,
            split="validation",
            class_names=train_set.classes_,
            mel_transform=validation_mel_transform,
        )
    else:
        train_set = GTZANDataset(
            args.data_dir,
            split="train",
            mel_transform=train_mel_transform,
            pitch_shift_semitones=tuple(args.pitch_shift_semitones),
            device=device,
        )
        val_set = GTZANDataset(
            args.data_dir,
            split="validation",
            class_names=train_set.classes_,
            mel_transform=validation_mel_transform,
        )
    labels = train_set.classes_
    if len(val_set) == 0:
        raise ValueError(f"O conjunto de validação do {args.dataset.upper()} está vazio.")
    print(
        f"dataset={args.dataset} "
        f"train_samples={len(train_set)} "
        f"validation_samples={len(val_set)} "
        f"num_classes={len(labels)} "
        f"classes={labels}"
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)

    model = ASTFineTuner.from_backbone(
        args.model_name,
        num_labels=len(labels),
        finetuning_method=args.finetuning,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(args.lora_target_modules),
    ).to(device)
    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameter_count = sum(parameter.numel() for _, parameter in trainable_parameters)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"finetuning={args.finetuning} "
        f"trainable_parameters={trainable_parameter_count} "
        f"total_parameters={total_parameter_count} "
        f"trainable_ratio={trainable_parameter_count / total_parameter_count:.4%}"
    )
    if args.finetuning == "head":
        print(f"trainable_modules={[name for name, _ in trainable_parameters]}")
    optimizer = torch.optim.AdamW(
        (parameter for _, parameter in trainable_parameters),
        lr=args.learning_rate,
    )
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
        val_targets: list[torch.Tensor] = []
        val_predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _prepare_batch(batch, device)
                output = model(**batch)
                batch_size = batch["labels"].size(0)
                val_loss_sum += output["loss"].item() * batch_size
                val_targets.append(batch["labels"].cpu())
                val_predictions.append(output["logits"].argmax(dim=-1).cpu())
        y_true = torch.cat(val_targets).numpy()
        y_pred = torch.cat(val_predictions).numpy()
        val_loss = val_loss_sum / max(1, len(y_true))
        val_accuracy = float((y_true == y_pred).mean())
        val_macro = macro_metrics(y_true, y_pred, num_classes=len(labels))
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            **{f"validation_{name}": value for name, value in val_macro.items()},
        }
        history.append(metrics)
        pl.DataFrame(history).write_csv(history_path)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"validation_loss={val_loss:.4f} "
            f"validation_accuracy={val_accuracy:.4f} "
            f"validation_precision_macro={val_macro['precision_macro']:.4f} "
            f"validation_recall_macro={val_macro['recall_macro']:.4f} "
            f"validation_f1_macro={val_macro['f1_macro']:.4f}"
        )
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            save_best_epoch_metrics(args.output_dir, epoch, y_true, y_pred, labels)
            model.save_pretrained(args.output_dir)
            extractor.save_pretrained(args.output_dir)
            (args.output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
