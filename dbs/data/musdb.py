"""MUSDB18-HQ (44.1 kHz stereo; vocals, drums, bass, other).

A split directory holds one folder per track with ``mixture.wav`` and one WAV
file per stem. Two training recipes are provided:

* :class:`MusdbDataModule` (BSRNN and BS-RoFormer, following
  Music-Source-Separation-Training): every stem is a random chunk of a random
  track, rejected while its mean absolute amplitude is below ``min_mean_abs``,
  and scaled by a random loudness factor; validation and test use full tracks.
* :class:`MusdbChunkDataModule` (Mel-RoFormer): aligned chunks with
  cross-track stem remixing, gain, polarity and channel-swap augmentation,
  drawn as a pure function of ``(seed, epoch, index)``; the last
  ``holdout_tracks`` training tracks are held out for validation.

In both recipes the training mixture is the sum of the stems.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .common import AudioDataModule, num_frames, read_audio

STEMS = ("vocals", "drums", "bass", "other")


def list_tracks(directory: str | Path) -> list[Path]:
    """Track folders of a split that contain ``mixture.wav``, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"MUSDB18-HQ split not found: {directory}")
    return sorted(path for path in directory.iterdir() if (path / "mixture.wav").is_file())


def _stereo(audio: Tensor) -> Tensor:
    return audio.expand(2, -1) if audio.shape[0] == 1 else audio[:2]


class MusdbTrackDataset(Dataset):
    """Full tracks: ``mixture.wav`` and the stems, trimmed to a common length."""

    def __init__(self, directory: str | Path, *, stems: Sequence[str] = STEMS, sample_rate: int = 44100):
        self.tracks = list_tracks(directory)
        self.stems, self.sample_rate = tuple(stems), int(sample_rate)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, index: int):
        track = self.tracks[index]
        mixture = _stereo(read_audio(track / "mixture.wav", sample_rate=self.sample_rate))
        stems = [_stereo(read_audio(track / f"{stem}.wav", sample_rate=self.sample_rate)) for stem in self.stems]
        length = min(mixture.shape[-1], *(stem.shape[-1] for stem in stems))
        return mixture[:, :length], torch.stack([stem[:, :length] for stem in stems]), track.name


class MusdbRandomMixDataset(Dataset):
    """Random-mix training chunks of Music-Source-Separation-Training (``dataset_type=1``)."""

    def __init__(self, directory: str | Path, *, num_items: int, chunk_samples: int = 131584,
                 min_mean_abs: float = 1e-3, loudness: Sequence[float] | None = (0.5, 1.5),
                 stems: Sequence[str] = STEMS, max_tries: int = 1000):
        self.tracks = [(track, min(num_frames(track / f"{stem}.wav") for stem in stems))
                       for track in list_tracks(directory)]
        if not self.tracks:
            raise ValueError(f"no MUSDB18-HQ tracks in {directory}")
        self.num_items, self.chunk = int(num_items), int(chunk_samples)
        self.min_mean_abs, self.max_tries = float(min_mean_abs), int(max_tries)
        self.loudness = None if loudness is None else tuple(loudness)
        self.stems = tuple(stems)

    def __len__(self) -> int:
        return self.num_items

    def _chunk(self, path: Path, length: int) -> Tensor:
        start = random.randint(0, length - self.chunk) if self.chunk <= length else 0
        audio = _stereo(read_audio(path, start, min(self.chunk, length)))
        return F.pad(audio, (0, self.chunk - audio.shape[-1]))

    def _source(self, stem: str) -> Tensor:
        for _ in range(self.max_tries):
            track, length = random.choice(self.tracks)
            audio = self._chunk(track / f"{stem}.wav", length)
            if audio.abs().mean() >= self.min_mean_abs:  # skip near-silent chunks
                break
        return audio

    def __getitem__(self, index: int):
        stems = torch.stack([self._source(stem) for stem in self.stems])
        if self.loudness is not None:
            gains = torch.tensor([random.uniform(*self.loudness) for _ in self.stems], dtype=torch.float32)
            stems = stems * gains[:, None, None]
        return stems.sum(0), stems, f"train_{index}"


def split_holdout(directory: str | Path, holdout_tracks: int) -> tuple[list[Path], list[Path]]:
    """Training tracks (with ``vocals.wav``) split into ``(train, holdout)``; the holdout is the last tracks by name."""
    tracks = [track for track in list_tracks(directory) if (track / "vocals.wav").is_file()]
    if len(tracks) <= holdout_tracks:
        raise ValueError(f"need more than {holdout_tracks} training tracks, found {len(tracks)}")
    return tracks[:-holdout_tracks], tracks[-holdout_tracks:]


class MusdbChunkDataset(Dataset):
    """Aligned four-stem chunks with paired augmentation (Mel-RoFormer recipe).

    Every draw (track, position, remix donor, gains, polarity, channel swap) is
    a pure function of ``(seed, epoch, index)``; call :meth:`set_epoch` before
    each epoch. Each stem may be replaced by the same-position chunk of a donor
    track, then scaled by a random gain of up to ``gain_db`` dB and a random
    sign; the channels of all stems are swapped together at random, and the
    stems are rescaled if the mixture peak exceeds one.
    """

    def __init__(self, tracks: Sequence[Path], *, samples: int, size: int, seed: int, augment: bool = True,
                 remix_stem_probability: float = 0.33, gain_db: float = 6.0, min_rms: float = 1e-4,
                 activity_tries: int = 12, stems: Sequence[str] = STEMS):
        self.tracks = list(tracks)
        self.samples, self.size, self.seed = int(samples), int(size), int(seed)
        self.augment, self.remix_stem_probability, self.gain_db = bool(augment), float(remix_stem_probability), float(gain_db)
        self.min_rms, self.activity_tries, self.stems = float(min_rms), max(1, int(activity_tries)), tuple(stems)
        self._frames: dict[int, int] = {}
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()  # visible to persistent workers

    def set_epoch(self, epoch: int) -> None:
        self.epoch.fill_(int(epoch))

    def __len__(self) -> int:
        return self.size

    def _track_frames(self, index: int) -> int:
        if index not in self._frames:
            track = self.tracks[index]
            self._frames[index] = min(num_frames(track / f"{stem}.wav") for stem in ("mixture", *self.stems))
        return self._frames[index]

    def _read(self, track: Path, stem: str, start: int) -> np.ndarray:
        import soundfile as sf

        values = sf.read(str(track / f"{stem}.wav"), start=start, frames=self.samples, dtype="float32",
                         always_2d=True)[0]
        return np.pad(values, ((0, self.samples - values.shape[0]), (0, 0)))

    def _start(self, rng: random.Random, frames: int) -> int:
        return rng.randrange(max(1, frames - self.samples + 1)) if frames > self.samples else 0

    def __getitem__(self, index: int):
        rng = random.Random((self.seed + int(self.epoch.item()) * 7_907) * 1_000_003 + index)
        for _ in range(self.activity_tries):
            track_index = rng.randrange(len(self.tracks))
            start = self._start(rng, self._track_frames(track_index))
            stems = [self._read(self.tracks[track_index], stem, start) for stem in self.stems]
            mixture = sum(stems)
            if float(np.sqrt(np.mean(np.square(mixture)))) >= self.min_rms:
                break
        if self.augment:
            donor_index = rng.randrange(len(self.tracks))
            if donor_index != track_index:
                donor_start = self._start(rng, self._track_frames(donor_index))
                for position, stem in enumerate(self.stems):
                    if rng.random() < self.remix_stem_probability:
                        stems[position] = self._read(self.tracks[donor_index], stem, donor_start)
            for position in range(len(stems)):
                stems[position] = stems[position] * 10.0 ** (rng.uniform(-self.gain_db, self.gain_db) / 20.0)
                if rng.random() < 0.5:
                    stems[position] = -stems[position]
            if rng.random() < 0.5:
                stems = [stem[:, ::-1].copy() for stem in stems]
            mixture = sum(stems)
            peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
            if peak > 1.0:
                stems = [stem / peak for stem in stems]
                mixture = mixture / peak
        sources = torch.from_numpy(np.ascontiguousarray(np.stack(stems).transpose(0, 2, 1)))
        return torch.from_numpy(np.ascontiguousarray(mixture.T)), sources, f"train_{index}"


class MusdbWindowDataset(Dataset):
    """Windows of ``samples`` from the holdout tracks for fast validation.

    Up to about ``max_windows`` evenly strided windows per track; windows whose
    stems are nearly silent are skipped and the mixture is the stem sum.
    """

    def __init__(self, tracks: Sequence[Path], *, samples: int, max_windows: int = 12, min_rms: float = 1e-3,
                 stems: Sequence[str] = STEMS):
        self.samples, self.stems = int(samples), tuple(stems)
        self.windows: list[tuple[Path, int]] = []
        for track in tracks:
            frames = min(num_frames(track / f"{stem}.wav") for stem in self.stems)
            if frames < self.samples:
                continue
            available = max(1, frames // self.samples)
            for window in range(0, available, max(1, available // max_windows)):
                start = window * self.samples
                if float(self._stems(track, start).square().mean().sqrt()) >= min_rms:
                    self.windows.append((track, start))

    def _stems(self, track: Path, start: int) -> Tensor:
        return torch.stack([read_audio(track / f"{stem}.wav", start, self.samples) for stem in self.stems])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        track, start = self.windows[index]
        stems = self._stems(track, start)
        return stems.sum(0), stems, f"{track.name}@{start}"


class MusdbDataModule(AudioDataModule):
    """Music-Source-Separation-Training recipe; ``steps_per_epoch`` training batches per epoch on every device."""

    sources = STEMS

    def __init__(self, *, train_dir: str | None = None, valid_dir: str | None = None, test_dir: str | None = None,
                 sample_rate: int = 44100, chunk_samples: int = 131584, min_mean_abs: float = 1e-3,
                 steps_per_epoch: int = 1000, loudness: Sequence[float] | None = (0.5, 1.5), **loader):
        super().__init__(**loader)
        self.dirs = dict(train=train_dir, val=valid_dir, test=test_dir)
        self.sample_rate, self.chunk_samples, self.min_mean_abs = int(sample_rate), int(chunk_samples), float(min_mean_abs)
        self.steps_per_epoch, self.loudness = int(steps_per_epoch), loudness

    def _dir(self, split: str) -> str:
        if not self.dirs[split]:
            raise ValueError(f"set the MUSDB18-HQ {split} directory")
        return self.dirs[split]

    def train_dataset(self) -> Dataset:
        return MusdbRandomMixDataset(self._dir("train"), num_items=self.steps_per_epoch * self.batch_size * self.world_size,
                                     chunk_samples=self.chunk_samples, min_mean_abs=self.min_mean_abs,
                                     loudness=self.loudness)

    def val_dataset(self) -> Dataset:
        return MusdbTrackDataset(self._dir("val"), sample_rate=self.sample_rate)

    def test_dataset(self) -> Dataset:
        return MusdbTrackDataset(self._dir("test"), sample_rate=self.sample_rate)


class MusdbChunkDataModule(AudioDataModule):
    """Mel-RoFormer recipe: deterministic training chunks, holdout-window validation, full-track test."""

    sources = STEMS

    def __init__(self, *, train_dir: str | None = None, test_dir: str | None = None, sample_rate: int = 44100,
                 chunk_samples: int = 88200, steps_per_epoch: int = 1000, seed: int = 20260919,
                 holdout_tracks: int = 10, valid_windows: int = 12, augment: bool = True,
                 remix_stem_probability: float = 0.33, gain_db: float = 6.0, **loader):
        super().__init__(**loader)
        self.train_dir, self.test_dir, self.sample_rate = train_dir, test_dir, int(sample_rate)
        self.chunk_samples, self.steps_per_epoch, self.seed = int(chunk_samples), int(steps_per_epoch), int(seed)
        self.holdout_tracks, self.valid_windows = int(holdout_tracks), int(valid_windows)
        self.augmentation = dict(augment=bool(augment), remix_stem_probability=float(remix_stem_probability),
                                 gain_db=float(gain_db))

    def _split(self) -> tuple[list[Path], list[Path]]:
        if not self.train_dir:
            raise ValueError("set the MUSDB18-HQ train directory")
        return split_holdout(self.train_dir, self.holdout_tracks)

    def train_dataset(self) -> Dataset:
        return MusdbChunkDataset(self._split()[0], samples=self.chunk_samples, seed=self.seed,
                                 size=self.steps_per_epoch * self.world_size * self.batch_size, **self.augmentation)

    def val_dataset(self) -> Dataset:
        return MusdbWindowDataset(self._split()[1], samples=self.chunk_samples, max_windows=self.valid_windows)

    def test_dataset(self) -> Dataset:
        if not self.test_dir:
            raise ValueError("set the MUSDB18-HQ test directory")
        return MusdbTrackDataset(self.test_dir, sample_rate=self.sample_rate)

    def train_dataloader(self) -> DataLoader:
        # An explicit seeded sampler keeps the batch composition identical across runs with the same seed.
        self.setup("fit")
        dataset = self._datasets["train"]
        rank = int(getattr(getattr(self, "trainer", None), "global_rank", 0) or 0)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=True, seed=self.seed)
        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, drop_last=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)
