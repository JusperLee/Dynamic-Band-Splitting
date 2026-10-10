"""Bounded-memory, shared-scale STFT previews for the four audio players."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import rfft


FFT_SIZE = 2048
HOP_SIZE = 512
BATCH_FRAMES = 128
MAX_WIDTH = 1600
MAX_HEIGHT = 400
DB_MIN = -80.0
DB_MAX = 0.0
_TRACKS = frozenset(("mix", "speech", "music", "sfx"))


def _palette():
    """Quiet ink green through jade to warm gold; identical for every track."""
    positions = np.array([0, 0.2, 0.45, 0.7, 0.88, 1])
    colors = np.array([
        [12, 31, 29], [18, 57, 49], [30, 101, 76],
        [88, 148, 97], [184, 180, 104], [247, 211, 130],
    ])
    scale = np.linspace(0, 1, 256)
    return np.column_stack([
        np.interp(scale, positions, colors[:, channel]) for channel in range(3)
    ]).round().astype(np.uint8)


PALETTE = _palette()


def _validate(waveforms, sample_rate):
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("The spectrogram sample rate must be a positive integer.")
    if not waveforms or not set(waveforms).issubset(_TRACKS):
        raise ValueError("Spectrogram tracks must be mix, speech, music, or sfx.")
    length = None
    for key, audio in waveforms.items():
        if not isinstance(audio, np.ndarray) or audio.ndim != 1 or audio.size == 0:
            raise ValueError(f"The {key} spectrogram needs a nonempty mono waveform.")
        if audio.dtype.kind not in "fi":
            raise ValueError(f"The {key} waveform must contain real numeric samples.")
        if length is not None and audio.size != length:
            raise ValueError("All spectrogram tracks must have the same duration.")
        length = audio.size
        # Do not allocate a full-length boolean mask for long recordings.
        for start in range(0, length, 1_048_576):
            if not np.isfinite(audio[start:start + 1_048_576]).all():
                raise ValueError(f"The {key} waveform contains non-finite samples.")
    return length


def _amplitude_raster(audio):
    """Return time × ascending-frequency peaks without retaining a full STFT.

    Frames are centered at t = 0, hop / sr, ... with zero padding at the edges.
    Peak pooling preserves short events and narrow tones when multiple frames
    or bins share a pixel. This is a visualization, not a power-density plot.
    """
    length = audio.size
    frame_count = (length + HOP_SIZE - 1) // HOP_SIZE
    width = min(MAX_WIDTH, frame_count)
    height = min(MAX_HEIGHT, FFT_SIZE // 2 + 1)
    raster = np.zeros((width, height), dtype=np.float32)
    frequency_starts = np.linspace(0, FFT_SIZE // 2 + 1, height + 1, dtype=int)[:-1]
    window = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)
    window_sum = window.sum()
    offsets = np.arange(FFT_SIZE, dtype=np.int64) - FFT_SIZE // 2
    for first in range(0, frame_count, BATCH_FRAMES):
        centers = np.arange(first, min(first + BATCH_FRAMES, frame_count), dtype=np.int64) * HOP_SIZE
        indices = centers[:, None] + offsets[None, :]
        valid = (indices >= 0) & (indices < length)
        # Only this bounded batch is copied; the caller's waveforms stay intact.
        frames = audio[np.clip(indices, 0, length - 1)].astype(np.float32)
        frames *= valid
        frames *= window
        magnitude = np.abs(rfft(frames, axis=1, workers=1)) / window_sum
        pooled = np.maximum.reduceat(magnitude, frequency_starts, axis=1)
        columns = np.minimum((centers * width) // length, width - 1)
        np.maximum.at(raster, columns, pooled)
    return raster


def write_spectrograms(job, waveforms: dict[str, np.ndarray], sample_rate: int) -> dict:
    """Write PNG heatmaps and metadata using one reference and dB range.

    All supplied waveforms must be mono, equally long and finite. The shared
    reference is the maximum STFT magnitude across every supplied track, so
    identical amplitudes get identical colors even when switching tracks.
    Silent recordings use reference 1.0 and render uniformly at the dB floor.
    No sample normalization or track-specific contrast enhancement is applied.
    """
    length = _validate(waveforms, sample_rate)
    rasters = {key: _amplitude_raster(audio) for key, audio in waveforms.items()}
    reference = max(float(raster.max()) for raster in rasters.values()) or 1.0
    destination = Path(job)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for key, raster in rasters.items():
        # Clamp before log to keep silence finite and avoid runtime warnings.
        decibels = 20 * np.log10(np.maximum(raster / reference, 10 ** (DB_MIN / 20)))
        levels = np.rint((np.clip(decibels, DB_MIN, DB_MAX) - DB_MIN) * (255 / (DB_MAX - DB_MIN)))
        # Image row zero is the highest frequency; t=0 is the leftmost column.
        pixels = PALETTE[levels.astype(np.uint8).T[::-1]]
        filename = f"{key}-spectrogram.png"
        Image.fromarray(pixels).save(destination / filename, optimize=True)
        metadata[key] = {
            "filename": filename,
            "width": raster.shape[0],
            "height": raster.shape[1],
            "duration": length / sample_rate,
            "min_hz": 0.0,
            "max_hz": sample_rate / 2,
            "frequency_scale": "linear",
            "time_scale": "linear",
            "db_min": DB_MIN,
            "db_max": DB_MAX,
            "reference_amplitude": reference,
            "fft_size": FFT_SIZE,
            "hop_samples": HOP_SIZE,
            "pooling": "maximum_magnitude",
            "orientation": "low_frequency_bottom",
            "colormap": "ink_jade_gold",
        }
    return metadata
