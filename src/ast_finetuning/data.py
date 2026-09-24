"""Audio dataset discovery and preprocessing."""

from pathlib import Path

import polars as pl
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor
from sklearn.preprocessing import LabelEncoder

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def build_manifest(
    dataset_dir: Path, metadata_path: Path | None = None) -> tuple[pl.DataFrame, list[str]]:
    """Build a path/label manifest from folders or a CSV/Parquet metadata file."""
    dataset_dir = dataset_dir.expanduser().resolve()
    if metadata_path is not None:
        metadata_path = metadata_path.expanduser().resolve()
        frame = (
            pl.read_csv(metadata_path)
            if metadata_path.suffix.lower() == ".csv"
            else pl.read_parquet(metadata_path)
        )
        required = {"path", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"O metadata precisa conter as colunas {sorted(required)}."
            )
        frame = frame.select("path", "label").with_columns(
            pl.col("path").map_elements(
                lambda value: (
                    str((metadata_path.parent / value).resolve())
                    if not Path(value).is_absolute()
                    else str(Path(value).resolve())
                ),
                return_dtype=pl.String,
            ),
            pl.col("label").cast(pl.String),
        )
    else:
        paths = sorted(
            path
            for path in dataset_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not paths:
            raise FileNotFoundError(f"Nenhum áudio encontrado em {dataset_dir}.")
        rows = [{"path": str(path), "label": path.parent.name} for path in paths]
        frame = pl.DataFrame(rows)

    frame = frame.filter(
        pl.col("path").map_elements(
            lambda value: Path(value).is_file(), return_dtype=pl.Boolean
        )
    )
    if frame.is_empty():
        raise FileNotFoundError("O manifesto não contém arquivos de áudio existentes.")
    labels = sorted(frame.get_column("label").unique().to_list())
    if len(labels) < 2:
        raise ValueError("São necessárias pelo menos duas classes para classificação.")
    return frame.with_columns(
        pl.col("label")
        .replace({label: index for index, label in enumerate(labels)})
        .cast(pl.Int64)
    ), labels


class AudioDataset(Dataset):
    """Load audio, resample it, and convert it to the representation expected by AST."""

    def __init__(
        self,
        manifest: pl.DataFrame,
        feature_extractor: ASTFeatureExtractor,
        sample_rate: int = 16_000,
    ):
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
            waveform = torchaudio.functional.resample(
                waveform, source_rate, self.sample_rate
            )
        features = self.feature_extractor(
            waveform.numpy(), sampling_rate=self.sample_rate, return_tensors="pt"
        )
        return {
            "input_values": features.input_values.squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }


class FMADataset(Dataset):
    """Load FMA dataset."""

    def __init__(
        self, dataset_dir: Path, metadata_path: Path, subset="small", train=True):
        assert subset in {"small", "medium", "large"}, "Subset must be one of 'small', 'medium', or 'large'."
        self.data_dir = Path(dataset_dir)
        self.metadata_path = Path(metadata_path)
        self.subset = subset
        df = pl.read_csv(metadata_path, skip_rows=1, null_values=[""])
        df = df.rename({df.columns[0]: "track_id"})
        df = df.filter(pl.col("track_id") != "track_id").with_columns(
            pl.col("track_id").cast(pl.Int64)
        )
        df_subset = df.filter(pl.col("subset") == subset).drop_nulls(
            subset=["genre_top"]
        )
        # Train and validation splits are defined in the metadata file, so we filter accordingly.
        if train:
            df_subset = df_subset.filter(pl.col("split") == "training")
        else:
            df_subset = df_subset.filter(pl.col("split") != "training")
        # Get the track_ids and genres as lists.
        self.track_ids = df_subset["track_id"].cast(pl.Int64).to_list()
        self.genres = df_subset["genre_top"].to_list()
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.genres)
        self.length = len(self.track_ids)

    def _get_audio_path(self, track_id):
        # The FMA dataset organizes audio files in folders based on the first three digits of the track ID.
        track_id_str = f"{track_id:06d}"
        folder = track_id_str[:3]
        return self.data_dir / f"fma_{self.subset}" / folder / f"{track_id_str}.mp3"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        audio_path = self._get_audio_path(track_id)

        # Load the audio file using torchaudio. If it fails, create a silent waveform.
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            # Create a silent 30s waveform.
            sample_rate = 22050  # Default sample rate for FMA dataset
            waveform = torch.zeros(2, 30 * sample_rate)

        # Get the label for the original index and convert it to a tensor.
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return waveform, label_tensor
