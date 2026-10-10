"""DnR v2 separation and the frequency partition of every inference window.

Both released models (DBS-BSRNN and DBS-TIGER) are loaded with
``from_pretrained`` from the Hugging Face Hub or a local ``save_pretrained``
directory. Each window runs the full model with ``return_aux=True``; the
boundaries shown in the studio are the hard Top-K boundaries of that call.
"""
from __future__ import annotations

import hashlib
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from dbs.models import DBSBSRNN, DBSTIGER

SAMPLE_RATE = 44100
SOURCE_NAMES = ("music", "sfx", "speech")
MODELS = {
    "bsrnn": {"label": "DBS-BSRNN", "cls": DBSBSRNN, "repo": "JusperLee/DBS-BSRNN-DnR"},
    "tiger": {"label": "DBS-TIGER", "cls": DBSTIGER, "repo": "JusperLee/DBS-TIGER-DnR"},
}


def weights_source(model_key: str) -> str:
    """Hub repo id or local directory; override with ``DBS_BSRNN_WEIGHTS`` / ``DBS_TIGER_WEIGHTS``."""
    return os.environ.get(f"DBS_{model_key.upper()}_WEIGHTS", MODELS[model_key]["repo"])


def weights_file(model_key: str) -> Path:
    """The ``model.safetensors`` of a model (downloaded and cached on first use)."""
    source = weights_source(model_key)
    if Path(source).is_dir():
        return Path(source) / "model.safetensors"
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source, "model.safetensors"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Backend:
    """Load a model once, then separate mono 44.1 kHz float arrays on CPU or CUDA."""

    def __init__(self, model_key: str, device: str = "cpu"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("Use CPU or CUDA; the complex-STFT models are not run on MPS.")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this computer; select CPU.")
        began = time.monotonic()
        self.device = torch.device(device)
        spec = MODELS[model_key]
        source = weights_source(model_key)
        self.model = spec["cls"].from_pretrained(source).to(self.device).eval()
        self.sample_rate = int(self.model.sample_rate)
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError("The studio expects the 44.1 kHz DnR v2 models.")
        self.source_names = SOURCE_NAMES
        n_fft = int(getattr(self.model, "win", None) or self.model.stft.n_fft)
        self.n_fft = n_fft
        self.coarse_boundaries_hz = [self._bin_hz(a) for a, _ in self.model.band_slices] + [self.sample_rate / 2]
        # TIGER's grouped 1x1 convolutions are slow on the default CPU backends:
        # one thread and no NNPACK during separate() (identical outputs).
        self.tiger_cpu = model_key == "tiger" and self.device.type == "cpu"
        self.info = {
            "architecture": spec["label"],
            "weights": source,
            "weights_sha256": sha256(weights_file(model_key)),
            "sample_rate": self.sample_rate,
            "n_fft": n_fft,
            "source_names": list(self.source_names),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "load_seconds": round(time.monotonic() - began, 3),
            "route_source": "hard_boundaries from each full model forward(return_aux=True)",
        }

    def _bin_hz(self, index: int) -> float:
        return min(float(index) * self.sample_rate / self.n_fft, self.sample_rate / 2)

    def _partition(self, aux: dict, start: int, valid: int, window: int, index: int, elapsed: float) -> dict:
        boundaries, probabilities, selected, counts = [], [], [], []
        for (lo, _), hard, probs in zip(aux["band_slices"], aux["hard_boundaries"], aux["boundary_probs"]):
            positions = torch.nonzero(hard[0].detach().cpu() > 0.5, as_tuple=False).flatten().tolist()
            selected.append([int(lo + p) for p in positions])
            boundaries.append([self._bin_hz(lo + p) for p in positions])
            probabilities.append(probs[0].detach().float().cpu().tolist())
            counts.append(len(positions))
        return {"index": index, "start": start / self.sample_rate, "end": (start + valid) / self.sample_rate,
                "nominal_end": (start + window) / self.sample_rate, "valid_samples": valid,
                "padding_samples": window - valid, "boundaries_hz": boundaries, "boundary_bins": selected,
                "boundary_probabilities": probabilities, "token_counts": counts,
                "region_bins": [[int(a), int(b)] for a, b in aux["band_slices"]],
                "inference_seconds": round(elapsed, 4)}

    def separate(self, audio: np.ndarray, window_seconds: float = 3.0, hop_seconds: float = 3.0,
                 progress: Callable[[float, str], None] | None = None) -> dict:
        """Uniform overlap-add over zero-padded windows, as in the DnR v2 evaluation; output keeps the input length."""
        values = np.asarray(audio)
        if values.ndim != 1 or len(values) == 0 or not np.issubdtype(values.dtype, np.floating):
            raise ValueError("Expected a nonempty floating-point mono waveform [samples].")
        if not np.isfinite(values).all():
            raise ValueError("Audio contains NaN or infinite samples.")
        if not (math.isfinite(window_seconds) and math.isfinite(hop_seconds)):
            raise ValueError("Window duration and hop must be finite.")
        if not 0.25 <= window_seconds <= 30 or not 0.05 <= hop_seconds <= window_seconds:
            raise ValueError("Use a 0.25-30 s window and a hop between 0.05 s and the window.")
        length = len(values)
        window, hop = int(round(window_seconds * self.sample_rate)), int(round(hop_seconds * self.sample_rate))
        offsets, offset = [], 0
        while True:
            offsets.append(offset)
            if offset + window >= length:
                break
            offset += hop
        outputs = np.zeros((len(self.source_names), length), dtype=np.float32)
        weights = np.zeros(length, dtype=np.float32)
        partitions, began = [], time.monotonic()
        threads = torch.get_num_threads()
        context = torch.backends.nnpack.flags(enabled=False) if self.tiger_cpu else nullcontext()
        try:
            if self.tiger_cpu:
                torch.set_num_threads(1)
            with torch.inference_mode(), context:
                for index, offset in enumerate(offsets):
                    if progress:
                        progress(index / len(offsets), f"Separating window {index + 1}/{len(offsets)}")
                    valid = min(window, length - offset)
                    segment = np.zeros(window, dtype=np.float32)
                    segment[:valid] = values[offset:offset + valid]
                    tick = time.monotonic()
                    estimate, aux = self.model(torch.from_numpy(segment).view(1, 1, -1).to(self.device),
                                               return_aux=True)
                    estimate = estimate.float().cpu().numpy()[0, :, :valid]
                    if not np.isfinite(estimate).all():
                        raise RuntimeError("Model returned invalid audio; no output was published.")
                    outputs[:, offset:offset + valid] += estimate
                    weights[offset:offset + valid] += 1
                    partitions.append(self._partition(aux, offset, valid, window, index, time.monotonic() - tick))
        finally:
            torch.set_num_threads(threads)
        outputs /= weights[None, :]
        elapsed = time.monotonic() - began
        metadata = dict(self.info, window_seconds=window / self.sample_rate, hop_seconds=hop / self.sample_rate,
                        window_count=len(partitions), audio_seconds=length / self.sample_rate,
                        inference_seconds=round(elapsed, 3),
                        real_time_factor=round(elapsed / (length / self.sample_rate), 4),
                        overlap_add="uniform mean", short_window="right zero padding, cropped to the input length",
                        input_channels=1, output_channels=1, gain_adjustment=1.0)
        if progress:
            progress(1.0, "Separation complete")
        return {"stems": {name: outputs[i] for i, name in enumerate(self.source_names)},
                "sample_rate": self.sample_rate, "windows": partitions,
                "coarse_boundaries_hz": self.coarse_boundaries_hz, "metadata": metadata}
