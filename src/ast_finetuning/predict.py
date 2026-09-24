"""Predict the class of one audio file using a fine-tuned checkpoint."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from transformers import ASTFeatureExtractor

from .data import ASTMelSpectrogram
from .model import ASTFineTuner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ast-finetuned"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.loads((args.checkpoint / "labels.json").read_text(encoding="utf-8"))
    extractor = ASTFeatureExtractor.from_pretrained(args.checkpoint)
    mel_transform = ASTMelSpectrogram(mean=extractor.mean, std=extractor.std)
    model = ASTFineTuner.from_pretrained(args.checkpoint, num_labels=len(labels)).to(device).eval()
    waveform, sample_rate = torchaudio.load(args.audio)
    mel_spectrogram = mel_transform(waveform, sample_rate).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(mel_spectrogram.to(device))["probabilities"][0]
    index = int(probabilities.argmax())
    print(json.dumps({"label": labels[index], "probability": float(probabilities[index])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
