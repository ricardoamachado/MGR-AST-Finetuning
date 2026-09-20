"""Command-line training entry point."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import ASTFeatureExtractor

from .data import AudioDataset, build_manifest
from .model import ASTFineTuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune o AST em arquivos de áudio locais.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--metadata", type=Path, default=None, help="CSV/Parquet com colunas path,label.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/ast-finetuned"))
    parser.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest, labels = build_manifest(args.data_dir, args.metadata)
    extractor = ASTFeatureExtractor.from_pretrained(args.model_name)
    dataset = AudioDataset(manifest, extractor)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = max(1, int(len(indices) * args.val_ratio))
    val_set = Subset(dataset, indices[:split])
    train_set = Subset(dataset, indices[split:])
    if not train_set:
        raise ValueError("Poucos arquivos para separar treino e validação.")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ASTFineTuner.from_backbone(args.model_name, num_labels=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
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
