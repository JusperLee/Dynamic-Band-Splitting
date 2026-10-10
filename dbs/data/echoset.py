"""EchoSet: two-speaker mixtures with noise and reverberation (16 kHz, mono).

Each split directory holds ``mix.json``, ``s1.json`` and ``s2.json``, lists of
``[path, num_samples]``. Training takes a random crop of ``segment`` seconds
(clips shorter than that are skipped); validation and test use whole clips.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import AudioDataModule, load_manifest, read_mono


class EchoSetDataset(Dataset):
    def __init__(self, directory: str | Path, *, sample_rate: int = 16000, segment: float | None = None,
                 num_speakers: int = 2):
        directory = Path(directory)
        self.mixtures = load_manifest(directory / "mix.json")
        self.sources = [load_manifest(directory / f"s{i + 1}.json") for i in range(num_speakers)]
        if any(len(entries) != len(self.mixtures) for entries in self.sources):
            raise ValueError(f"manifest lengths differ in {directory}")
        self.sample_rate = int(sample_rate)
        self.samples = None if segment is None else int(segment * sample_rate)
        self.items = list(range(len(self.mixtures)))
        if self.samples is not None:
            self.items = [i for i in self.items if (self.mixtures[i][1] or 0) >= self.samples]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        path, length = self.mixtures[item]
        start, frames = 0, -1
        if self.samples is not None:
            if length != self.samples:
                start = np.random.randint(0, length - self.samples)
            frames = self.samples
        mixture = read_mono(path, start, frames, self.sample_rate)
        sources = torch.stack([read_mono(entries[item][0], start, frames, self.sample_rate)
                               for entries in self.sources])
        return mixture.unsqueeze(0), sources, path.name


class EchoSetDataModule(AudioDataModule):
    sources = ("speaker1", "speaker2")

    def __init__(self, *, train_dir: str | None = None, valid_dir: str | None = None, test_dir: str | None = None,
                 sample_rate: int = 16000, segment: float = 3.0, **loader):
        super().__init__(**loader)
        self.dirs = dict(train=train_dir, val=valid_dir, test=test_dir)
        self.sample_rate, self.segment = int(sample_rate), float(segment)

    def _dataset(self, split: str, segment: float | None) -> Dataset:
        if not self.dirs[split]:
            raise ValueError(f"set the EchoSet {split} directory")
        return EchoSetDataset(self.dirs[split], sample_rate=self.sample_rate, segment=segment)

    def train_dataset(self) -> Dataset:
        return self._dataset("train", self.segment)

    def val_dataset(self) -> Dataset:
        return self._dataset("val", None)

    def test_dataset(self) -> Dataset:
        return self._dataset("test", None)
