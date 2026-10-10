"""Audio I/O helpers and the data-module base class."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def read_audio(path: str | Path, start: int = 0, frames: int = -1, sample_rate: int | None = None) -> Tensor:
    """Read ``[channels, samples]`` float32 audio; ``sample_rate`` is checked when given (no resampling)."""
    audio, rate = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    if sample_rate is not None and rate != sample_rate:
        raise ValueError(f"{path}: expected {sample_rate} Hz, found {rate} Hz")
    return torch.from_numpy(np.ascontiguousarray(audio.T))


def read_mono(path: str | Path, start: int = 0, frames: int = -1, sample_rate: int | None = None) -> Tensor:
    """Read audio averaged over channels, ``[samples]``."""
    return read_audio(path, start, frames, sample_rate).mean(dim=0)


def num_frames(path: str | Path) -> int:
    return int(sf.info(str(path)).frames)


def load_manifest(path: str | Path) -> list[tuple[Path, int | None]]:
    """A JSON list of ``[path, num_samples]`` pairs (or bare paths); relative paths are resolved against the file."""
    path = Path(path)
    entries = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        file, length = (row[0], row[1] if len(row) > 1 else None) if isinstance(row, (list, tuple)) else (row, None)
        file = Path(str(file)).expanduser()
        entries.append((file if file.is_absolute() else (path.parent / file).resolve(), length))
    return entries


def peak_normalize(mixture: Tensor, sources: Tensor) -> tuple[Tensor, Tensor]:
    """Scale mixture and sources together so that the largest absolute sample is one."""
    peak = torch.maximum(mixture.abs().amax(), sources.abs().amax())
    if float(peak) > 0:
        mixture, sources = mixture / peak, sources / peak
    return mixture, sources


class AudioDataModule(pl.LightningDataModule):
    """Batches ``(mixture [C, T], sources [N, (C,) T], key)``; subclasses provide the datasets.

    Training batches are shuffled with ``drop_last=True``; evaluation keeps the dataset order.
    """

    def __init__(self, *, batch_size: int = 1, eval_batch_size: int = 1, num_workers: int = 0,
                 pin_memory: bool = False, persistent_workers: bool = False):
        super().__init__()
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self._datasets: dict[str, Dataset] = {}

    # Subclasses implement these.
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    def val_dataset(self) -> Dataset:
        raise NotImplementedError

    def test_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def world_size(self) -> int:
        trainer = getattr(self, "trainer", None)
        return max(1, int(getattr(trainer, "world_size", 1) or 1))

    def setup(self, stage: str | None = None) -> None:
        wanted = {"fit": ("train", "val"), "validate": ("val",), "test": ("test",), None: ("train", "val", "test")}
        for split in wanted.get(stage, ()):
            if split not in self._datasets:
                self._datasets[split] = getattr(self, f"{split}_dataset")()

    def _loader(self, split: str, **kwargs: Any) -> DataLoader:
        self.setup({"train": "fit", "val": "validate", "test": "test"}[split])
        return DataLoader(self._datasets[split], num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, **kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", batch_size=self.eval_batch_size, shuffle=False)
