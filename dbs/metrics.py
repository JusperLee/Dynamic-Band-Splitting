"""Separation metrics (higher is better) and the per-dataset evaluation reports.

Estimates and references are ``[B, N, T]`` (mono) or ``[B, N, C, T]``; every
metric returns ``[B, N]`` and is computed in float64.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .losses import best_permutation, pairwise_neg_sdr, reorder

EPS = 1e-8


def _channels(x: Tensor) -> Tensor:
    return x.unsqueeze(2) if x.ndim == 3 else x


def snr(estimate: Tensor, target: Tensor, eps: float = EPS) -> Tensor:
    """Signal-to-noise ratio in dB, averaged over channels."""
    estimate, target = _channels(estimate).double(), _channels(target).double()
    noise = estimate - target
    value = 10 * torch.log10(target.square().sum(-1) / (noise.square().sum(-1) + eps) + eps)
    return value.mean(-1)


def si_sdr(estimate: Tensor, target: Tensor, zero_mean: bool = True, eps: float = EPS) -> Tensor:
    """Scale-invariant SDR in dB, averaged over channels."""
    estimate, target = _channels(estimate).double(), _channels(target).double()
    if zero_mean:
        estimate = estimate - estimate.mean(-1, keepdim=True)
        target = target - target.mean(-1, keepdim=True)
    dot = (estimate * target).sum(-1, keepdim=True)
    projection = dot * target / (target.square().sum(-1, keepdim=True) + eps)
    noise = estimate - projection
    value = 10 * torch.log10(projection.square().sum(-1) / (noise.square().sum(-1) + eps) + eps)
    return value.mean(-1)


def global_sdr(estimate: Tensor, target: Tensor, eps: float = EPS) -> Tensor:
    """MUSDB18 "global" SDR of Music-Source-Separation-Training: one energy ratio over channels and time."""
    estimate, target = _channels(estimate).double(), _channels(target).double()
    numerator = target.square().sum((-2, -1)) + eps
    denominator = (target - estimate).square().sum((-2, -1)) + eps
    return 10 * torch.log10(numerator / denominator)


def mixture_as_estimates(mixture: Tensor, target: Tensor) -> Tensor:
    """The unprocessed mixture repeated once per source (the "no separation" baseline), shaped like ``target``."""
    if mixture.ndim == 2:
        mixture = mixture.unsqueeze(1)
    if target.ndim == 3:
        if mixture.shape[1] != 1:
            raise ValueError("a mono target needs a mono mixture")
        return mixture.expand(-1, target.shape[1], -1)
    return mixture.unsqueeze(1).expand(-1, target.shape[1], -1, -1)


METRICS = {
    "snr": lambda est, tgt, mix: snr(est, tgt),
    "sisdr": lambda est, tgt, mix: si_sdr(est, tgt),
    "sisdri": lambda est, tgt, mix: si_sdr(est, tgt) - si_sdr(mixture_as_estimates(mix, tgt), tgt),
    "sdr": lambda est, tgt, mix: global_sdr(est, tgt),
}


def source_scores(estimates: Tensor, targets: Tensor, mixture: Tensor, names: Sequence[str]) -> dict[str, Tensor]:
    """``{metric: [B, N]}`` for the requested metric names (see :data:`METRICS`)."""
    unknown = set(names) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics {sorted(unknown)}; choose from {sorted(METRICS)}")
    return {name: METRICS[name](estimates, targets, mixture) for name in names}


# --------------------------------------------------------------------------- evaluation reports
# A report maps one test item (a batch of one) to a flat dict of floats.


def dnr_report(estimates: Tensor, targets: Tensor, mixture: Tensor, sources: Sequence[str], **_) -> dict[str, float]:
    """SI-SDR and SNR per source, after SI-SDR permutation search (``pit_*``) and in output order (``fixed_*``)."""
    _, perm = best_permutation(pairwise_neg_sdr(estimates.double(), targets.double(), kind="sisdr"))
    row = {}
    for order, est in (("pit", reorder(estimates, perm)), ("fixed", estimates)):
        for metric, function in (("sisdr", si_sdr), ("snr", snr)):
            values = function(est, targets).mean(0)
            for name, value in zip(sources, values.tolist()):
                row[f"{order}_{name}_{metric}"] = value
            row[f"{order}_avg_{metric}"] = float(values.mean())
    return row


def echoset_report(estimates: Tensor, targets: Tensor, mixture: Tensor, sources: Sequence[str], **_) -> dict[str, float]:
    """SDRi and SI-SDRi over the unprocessed mixture (TIGER EchoSet protocol).

    SDR is BSS-Eval SDR with a 512-tap distortion filter (``fast_bss_eval``);
    SI-SDR is permutation-invariant and not zero-meaned.
    """
    import fast_bss_eval

    row = {}
    for suffix, est in (("", estimates), ("_baseline", mixture_as_estimates(mixture, targets))):
        est, ref = est[0].float(), targets[0].float()
        row[f"sdr{suffix}"] = float(-fast_bss_eval.sdr_pit_loss(est, ref).mean())
        pairwise = pairwise_neg_sdr(est[None].double(), ref[None].double(), kind="sisdr", zero_mean=False)
        row[f"si_sdr{suffix}"] = float(-best_permutation(pairwise)[0].mean())
    row["sdri"] = row["sdr"] - row["sdr_baseline"]
    row["si_sdri"] = row["si_sdr"] - row["si_sdr_baseline"]
    return row


def musdb_report(estimates: Tensor, targets: Tensor, mixture: Tensor, sources: Sequence[str],
                 museval: bool = False, sample_rate: int = 44100, **_) -> dict[str, float]:
    """Global SDR per stem; with ``museval`` also the BSS-Eval v4 median SDR over one-second frames."""
    values = global_sdr(estimates, targets)[0]
    row = {f"{name}_sdr": value for name, value in zip(sources, values.tolist())}
    row["avg_sdr"] = float(values.mean())
    if museval:
        import museval as bss

        references = _channels(targets)[0].transpose(-1, -2).cpu().double().numpy()  # [N, T, C]
        separated = _channels(estimates)[0].transpose(-1, -2).cpu().double().numpy()
        sdr, _, _, _ = bss.evaluate(references, separated, win=sample_rate, hop=sample_rate)
        for name, value in zip(sources, np.nanmedian(sdr, axis=1).tolist()):
            row[f"{name}_museval_sdr"] = value
    return row


REPORTS = {"dnr": dnr_report, "echoset": echoset_report, "musdb": musdb_report}


def summarize(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean over items; museval scores use the median over tracks (SiSEC 2018), averaged over stems."""
    summary = {}
    for key in rows[0]:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[key] = float(np.nanmedian(values) if "museval" in key else np.mean(values))
    museval_keys = [key for key in summary if key.endswith("_museval_sdr")]
    if museval_keys:
        summary["avg_museval_sdr"] = float(np.mean([summary[key] for key in museval_keys]))
    return summary
