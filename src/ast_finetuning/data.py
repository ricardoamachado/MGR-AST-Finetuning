"""Audio dataset discovery and preprocessing."""

from pathlib import Path

import polars as pl
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


class ASTMelSpectrogram(torch.nn.Module):
    """Convert waveforms using the same Kaldi fbank preprocessing as AST."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mels: int = 128,
        hop_length: int = 160,
        win_length: int = 400,
        target_frames: int = 1024,
        mean: float = -4.2677393,
        std: float = 4.5689974,
        is_training: bool = True,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_frames = target_frames
        self.mean = mean
        self.std = std
        self.is_training = is_training
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length

        expected_sample_rate = 16_000
        expected_hop_length = 160
        expected_win_length = 400
        if (
            sample_rate != expected_sample_rate
            or hop_length != expected_hop_length
            or win_length != expected_win_length
        ):
            raise ValueError(
                "A configuração do AST exige sample_rate=16000, "
                "hop_length=160 e win_length=400."
            )

    def forward(
        self, waveform: torch.Tensor, source_sample_rate: int
    ) -> torch.Tensor:
        # Garante entrada mono 1D [amostras]
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        # Reamostragem se necessário
        if source_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, source_sample_rate, self.sample_rate
            )

        # This is the same preprocessing path used by ASTFeatureExtractor when
        # torchaudio is available: Hanning window, 25 ms frames, 10 ms shift,
        # pre-emphasis, DC-offset removal, and a 512-point FFT selected by Kaldi.
        log_mel = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0),
            sample_frequency=self.sample_rate,
            num_mel_bins=self.n_mels,
            window_type="hanning",
            frame_length=self.win_length * 1000.0 / self.sample_rate,
            frame_shift=self.hop_length * 1000.0 / self.sample_rate,
            dither=0.0,
        )

        total_frames = log_mel.shape[0]

        # Random crop during training; AST-compatible initial crop otherwise.
        if total_frames > self.target_frames:
            if self.is_training:
                start = torch.randint(
                    0, total_frames - self.target_frames + 1, (1,)
                ).item()
            else:
                start = 0
            log_mel = log_mel[start : start + self.target_frames, :]
        elif total_frames < self.target_frames:
            log_mel = torch.nn.functional.pad(
                log_mel, (0, 0, 0, self.target_frames - total_frames)
            )

        # ASTFeatureExtractor normalizes after padding/truncation as
        # (log_mel - mean) / (std * 2).
        log_mel = (log_mel - self.mean) / (self.std * 2.0)

        # Return [target_frames, n_mels] -> [1024, 128].
        return log_mel.contiguous()


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
    """Load FMA files and expose Mel spectrograms instead of raw waveforms."""

    def __init__(
        self,
        dataset_dir: Path,
        metadata_path: Path,
        subset="small",
        train=True,
        class_names: list[str] | None = None,
        mel_transform: ASTMelSpectrogram | None = None,
    ):
        if subset not in {"small", "medium", "large"}:
            raise ValueError("Subset must be one of 'small', 'medium', or 'large'.")
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
        self.classes_ = sorted(class_names or set(self.genres))
        self.class_to_index = {name: index for index, name in enumerate(self.classes_)}
        unknown_classes = sorted(set(self.genres) - set(self.class_to_index))
        if unknown_classes:
            raise ValueError(f"Classes ausentes no conjunto de treino: {unknown_classes}")
        self.labels = [self.class_to_index[genre] for genre in self.genres]
        self.mel_transform = mel_transform or ASTMelSpectrogram()
        self.length = len(self.track_ids)
        audio_paths = [self._get_audio_path(track_id) for track_id in self.track_ids]
        missing_paths = [path for path in audio_paths if not path.is_file()]
        if missing_paths:
            examples = ", ".join(str(path) for path in missing_paths[:3])
            raise FileNotFoundError(
                f"{len(missing_paths)} arquivos de áudio do FMA não foram encontrados. "
                f"Exemplos: {examples}"
            )

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

        # Do not replace failed reads with silence: that would make every failed
        # sample identical and can produce chance-level accuracy without errors.
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as error:
            raise RuntimeError(
                f"Não foi possível carregar o áudio FMA: {audio_path}"
            ) from error
        mel_spectrogram = self.mel_transform(waveform, sample_rate)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_values": mel_spectrogram, "labels": label}
