"""DnR v2: on-the-fly training mixtures and full-length evaluation tracks (44.1 kHz, mono).

Evaluation directories hold ``mix.json``, ``s1.json``, ``s2.json`` and
``s3.json`` (lists of ``[path, num_samples]``); the music/sfx/speech slot of
each stem list is inferred from its file names. Training mixtures are built
on the fly from one random clip per source:

* ``train_list``: a ``.npy`` list of HDF5 files with ``music_VAD``, ``sfx_VAD``
  and ``speech_VAD`` datasets of voice-activity segments (the paper recipe);
* ``train_dir``: alternatively, a directory with the same JSON manifests as
  the evaluation sets, cropped at random positions.

Each source is cropped to ``segment`` seconds, scaled by a random gain in
``gain_db`` and dropped with probability ``drop_prob``; the mixture is the sum
of the sources, and mixture and sources are peak-normalized together.
"""
from __future__ import annotations

import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from .common import AudioDataModule, load_manifest, num_frames, peak_normalize, read_mono

SOURCES = ("music", "sfx", "speech")
SLOTS = {"s1": "music", "s2": "sfx", "s3": "speech"}
KEYWORDS = (("speech", ("speech", "dialog", "voice")), ("sfx", ("sfx", "effect", "fx")), ("music", ("music",)))


def _pad_or_crop(waveform: Tensor, samples: int, rng) -> Tensor:
    if waveform.shape[-1] < samples:
        return F.pad(waveform, (0, samples - waveform.shape[-1]))
    if waveform.shape[-1] == samples:
        return waveform
    start = rng.randint(0, waveform.shape[-1] - samples)
    return waveform[start:start + samples]


def _gain_and_drop(waveform: Tensor, gain_db: Sequence[float], drop_prob: float, rng) -> Tensor:
    waveform = waveform * 10.0 ** (rng.uniform(*gain_db) / 20.0)
    if rng.random() < drop_prob:
        waveform = torch.zeros_like(waveform)
    return waveform


def _guess_source(path: Path) -> str | None:
    tokens = f"{path.name.lower()} {path.parent.name.lower()}"
    for source, words in KEYWORDS:
        if any(word in tokens for word in words):
            return source
    return None


def read_dnr_manifests(directory: str | Path) -> tuple[list[Path], dict[str, list[Path]]]:
    """Mixture paths and ``{source: stem paths}`` from the JSON manifests of a DnR split."""
    directory = Path(directory)
    mixtures = [path for path, _ in load_manifest(directory / "mix.json")]
    slots = {slot: [path for path, _ in load_manifest(directory / f"{slot}.json")] for slot in SLOTS}
    if len({len(mixtures), *(len(paths) for paths in slots.values())}) != 1:
        raise ValueError(f"manifest lengths differ in {directory}")
    assigned: dict[str, str] = {}
    for slot, paths in slots.items():
        votes = Counter(filter(None, (_guess_source(path) for path in paths[:16])))
        if votes and votes.most_common(1)[0][0] not in assigned.values():
            assigned[slot] = votes.most_common(1)[0][0]
    remaining = [source for source in SOURCES if source not in assigned.values()]
    for slot in SLOTS:
        if slot not in assigned:
            assigned[slot] = SLOTS[slot] if SLOTS[slot] in remaining else remaining[0]
            remaining.remove(assigned[slot])
    return mixtures, {assigned[slot]: paths for slot, paths in slots.items()}


class DnRVADMixDataset(Dataset):
    """Training mixtures from voice-activity segments stored in HDF5 files (see the module docstring)."""

    def __init__(self, train_list: str | Path, *, sample_rate: int = 44100, segment: float = 3.0,
                 num_items: int = 1000, gain_db: Sequence[float] = (-10.0, 10.0), drop_prob: float = 0.1,
                 normalize: bool = True, max_open_files: int = 8):
        train_list = Path(train_list)
        self.paths = [Path(str(p.decode() if isinstance(p, bytes) else p)) for p in np.load(train_list, allow_pickle=True)]
        self.paths = [p if p.is_absolute() else (train_list.parent / p).resolve() for p in self.paths]
        if not self.paths:
            raise ValueError(f"empty training list {train_list}")
        self.samples = int(round(segment * sample_rate))
        self.num_items = int(num_items)
        self.gain_db, self.drop_prob, self.normalize = tuple(gain_db), float(drop_prob), bool(normalize)
        self.max_open_files = int(max_open_files)
        self._handles: OrderedDict = OrderedDict()  # per-worker LRU of open HDF5 files

    def __len__(self) -> int:
        return self.num_items

    def _file(self, path: Path):
        import h5py

        handle = self._handles.pop(path, None) or h5py.File(path, "r")
        self._handles[path] = handle
        while len(self._handles) > self.max_open_files:
            self._handles.popitem(last=False)[1].close()
        return handle

    def _source(self, name: str, rng) -> Tensor:
        segments = self._file(self.paths[rng.randrange(len(self.paths))])[f"{name}_VAD"]
        waveform = torch.from_numpy(np.asarray(segments[rng.randrange(segments.shape[0])])).float()
        if waveform.ndim > 1:
            waveform = waveform.reshape(-1, waveform.shape[-1]).mean(0)
        return _gain_and_drop(_pad_or_crop(waveform, self.samples, rng), self.gain_db, self.drop_prob, rng)

    def __getitem__(self, index: int):
        sources = torch.stack([self._source(name, random) for name in SOURCES])
        mixture = sources.sum(0, keepdim=True)
        if self.normalize:
            mixture, sources = peak_normalize(mixture, sources)
        return mixture, sources, f"train_{index}"


class DnRCropMixDataset(Dataset):
    """Training mixtures from random crops of the stems listed in a DnR manifest directory."""

    def __init__(self, directory: str | Path, *, sample_rate: int = 44100, segment: float = 3.0,
                 num_items: int = 1000, gain_db: Sequence[float] = (-10.0, 10.0), drop_prob: float = 0.1,
                 normalize: bool = True):
        _, self.stems = read_dnr_manifests(directory)
        self.sample_rate = int(sample_rate)
        self.samples = int(round(segment * sample_rate))
        self.num_items = int(num_items)
        self.gain_db, self.drop_prob, self.normalize = tuple(gain_db), float(drop_prob), bool(normalize)
        self._frames: dict[Path, int] = {}

    def __len__(self) -> int:
        return self.num_items

    def _source(self, name: str, rng) -> Tensor:
        path = self.stems[name][rng.randrange(len(self.stems[name]))]
        if path not in self._frames:
            self._frames[path] = num_frames(path)
        start = rng.randint(0, max(0, self._frames[path] - self.samples))
        waveform = read_mono(path, start, self.samples, self.sample_rate)
        return F.pad(waveform, (0, self.samples - waveform.shape[-1]))

    def __getitem__(self, index: int):
        sources = torch.stack([self._source(name, random) for name in SOURCES])
        sources = torch.stack([_gain_and_drop(s, self.gain_db, self.drop_prob, random) for s in sources])
        mixture = sources.sum(0, keepdim=True)
        if self.normalize:
            mixture, sources = peak_normalize(mixture, sources)
        return mixture, sources, f"train_{index}"


class DnRTrackDataset(Dataset):
    """Full-length tracks of a DnR split; the mixture is the sum of the stems (or ``mix.json`` with ``use_mix_file``)."""

    def __init__(self, directory: str | Path, *, sample_rate: int = 44100, normalize: bool = True,
                 use_mix_file: bool = False):
        self.mixtures, self.stems = read_dnr_manifests(directory)
        self.sample_rate, self.normalize, self.use_mix_file = int(sample_rate), bool(normalize), bool(use_mix_file)

    def __len__(self) -> int:
        return len(self.mixtures)

    def __getitem__(self, index: int):
        sources = [read_mono(self.stems[name][index], sample_rate=self.sample_rate) for name in SOURCES]
        length = min(source.shape[-1] for source in sources)
        sources = torch.stack([source[:length] for source in sources])
        if self.use_mix_file:
            mixture = read_mono(self.mixtures[index], sample_rate=self.sample_rate)[:length].unsqueeze(0)
        else:
            mixture = sources.sum(0, keepdim=True)
        if self.normalize:
            mixture, sources = peak_normalize(mixture, sources)
        return mixture, sources, self.mixtures[index].name


class DnRDataModule(AudioDataModule):
    """DnR v2 data module; ``num_iter_per_epoch`` is the number of training batches per epoch on every device."""

    sources = SOURCES

    def __init__(self, *, train_list: str | None = None, train_dir: str | None = None, valid_dir: str | None = None,
                 test_dir: str | None = None, sample_rate: int = 44100, segment: float = 3.0,
                 num_iter_per_epoch: int = 1000, gain_db: Sequence[float] = (-10.0, 10.0), drop_prob: float = 0.1,
                 normalize: bool = True, max_open_files: int = 8, **loader):
        super().__init__(**loader)
        self.train_list, self.train_dir, self.valid_dir, self.test_dir = train_list, train_dir, valid_dir, test_dir
        self.sample_rate, self.segment = int(sample_rate), float(segment)
        self.num_iter_per_epoch = int(num_iter_per_epoch)
        self.mix_options = dict(sample_rate=self.sample_rate, segment=self.segment, gain_db=tuple(gain_db),
                                drop_prob=float(drop_prob), normalize=bool(normalize))
        self.normalize, self.max_open_files = bool(normalize), int(max_open_files)

    def train_dataset(self) -> Dataset:
        num_items = self.num_iter_per_epoch * self.batch_size * self.world_size
        if self.train_list:
            return DnRVADMixDataset(self.train_list, num_items=num_items, max_open_files=self.max_open_files,
                                    **self.mix_options)
        if self.train_dir:
            return DnRCropMixDataset(self.train_dir, num_items=num_items, **self.mix_options)
        raise ValueError("set data.train_list (HDF5 VAD segments) or data.train_dir (JSON manifests)")

    def _tracks(self, directory: str | None, split: str) -> Dataset:
        if not directory:
            raise ValueError(f"set the DnR {split} directory")
        return DnRTrackDataset(directory, sample_rate=self.sample_rate, normalize=self.normalize)

    def val_dataset(self) -> Dataset:
        return self._tracks(self.valid_dir, "validation")

    def test_dataset(self) -> Dataset:
        return self._tracks(self.test_dir, "test")
