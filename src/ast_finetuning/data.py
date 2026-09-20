"""Audio dataset discovery and preprocessing."""

from pathlib import Path

import polars as pl
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def build_manifest(dataset_dir: Path, metadata_path: Path | None = None) -> tuple[pl.DataFrame, list[str]]:
    """Build a path/label manifest from folders or a CSV/Parquet metadata file."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if metadata_path is not None:
        metadata_path = metadata_path.expanduser().resolve()
        frame = pl.read_csv(metadata_path) if metadata_path.suffix.lower() == ".csv" else pl.read_parquet(metadata_path)
        required = {"path", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"O metadata precisa conter as colunas {sorted(required)}.")
        frame = frame.select("path", "label").with_columns(
            pl.col("path").map_elements(
                lambda value: str((metadata_path.parent / value).resolve())
                if not Path(value).is_absolute()
                else str(Path(value).resolve()),
                return_dtype=pl.String,
            ),
            pl.col("label").cast(pl.String),
        )
    else:
        paths = sorted(path for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)
        if not paths:
            raise FileNotFoundError(f"Nenhum áudio encontrado em {dataset_dir}.")
        rows = [{"path": str(path), "label": path.parent.name} for path in paths]
        frame = pl.DataFrame(rows)

    frame = frame.filter(pl.col("path").map_elements(lambda value: Path(value).is_file(), return_dtype=pl.Boolean))
    if frame.is_empty():
        raise FileNotFoundError("O manifesto não contém arquivos de áudio existentes.")
    labels = sorted(frame.get_column("label").unique().to_list())
    if len(labels) < 2:
        raise ValueError("São necessárias pelo menos duas classes para classificação.")
    return frame.with_columns(pl.col("label").replace({label: index for index, label in enumerate(labels)}).cast(pl.Int64)), labels


class AudioDataset(Dataset):
    """Load audio, resample it, and convert it to the representation expected by AST."""

    def __init__(self, manifest: pl.DataFrame, feature_extractor: ASTFeatureExtractor, sample_rate: int = 16_000):
        self.manifest = manifest
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return self.manifest.height

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.manifest.row(index, named=True)
        waveform, source_rate = torchaudio.load(row["path"])
        waveform = waveform.mean(dim=0)
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)
        features = self.feature_extractor(
            waveform.numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        )
        return {"input_values": features.input_values.squeeze(0), "labels": torch.tensor(row["label"], dtype=torch.long)}
