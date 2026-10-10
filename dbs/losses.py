"""Training objectives on separated waveforms.

* ``pit_neg_sisdr`` / ``pit_neg_snr``: permutation-invariant negative SI-SDR or
  SNR (exhaustive search over source orders), used for DnR DBS-BSRNN and EchoSet.
* ``waveform_stft_mae``: fixed-order waveform MAE plus real/imaginary STFT MAE,
  the TIGER DnR objective (TIGER paper, Appendix D).

Each objective maps ``(estimates, targets)`` to ``(loss, aligned_estimates)``,
where ``aligned_estimates`` are reordered to the targets for the PIT objectives.
"""
from __future__ import annotations

from itertools import permutations
from typing import Callable

import torch
from torch import Tensor


def pairwise_neg_sdr(estimates: Tensor, targets: Tensor, kind: str = "sisdr", zero_mean: bool = True,
                     eps: float = 1e-8) -> Tensor:
    """Negative SDR of every (estimate, target) pair: ``[B, N, T]`` -> ``[B, N_estimate, N_target]``.

    ``kind="sisdr"`` projects each estimate onto the target (scale-invariant);
    ``kind="snr"`` uses the target itself as the signal.
    """
    if estimates.shape != targets.shape or targets.ndim != 3:
        raise ValueError(f"expected [batch, sources, time] inputs, got {tuple(estimates.shape)} "
                         f"and {tuple(targets.shape)}")
    if kind not in ("sisdr", "snr"):
        raise ValueError(f"kind must be 'sisdr' or 'snr', got {kind!r}")
    if zero_mean:
        targets = targets - targets.mean(dim=2, keepdim=True)
        estimates = estimates - estimates.mean(dim=2, keepdim=True)
    target = targets.unsqueeze(1)
    estimate = estimates.unsqueeze(2)
    if kind == "sisdr":
        dot = (estimate * target).sum(dim=3, keepdim=True)
        projection = dot * target / ((target ** 2).sum(dim=3, keepdim=True) + eps)
        noise = estimate - projection
    else:
        projection = target
        noise = estimate - target
    ratio = (projection ** 2).sum(dim=3) / ((noise ** 2).sum(dim=3) + eps)
    return -10 * torch.log10(ratio + eps)


def best_permutation(pairwise: Tensor) -> tuple[Tensor, Tensor]:
    """Exhaustive PIT on a ``[B, N_estimate, N_target]`` loss matrix.

    Returns the per-item loss averaged over sources ``[B]`` and the
    permutation ``[B, N]`` such that estimate ``perm[b, i]`` is paired with target ``i``.
    """
    num_sources = pairwise.shape[-1]
    if num_sources > 6:
        raise ValueError("exhaustive PIT is limited to six sources")
    perms = torch.tensor(list(permutations(range(num_sources))), dtype=torch.long, device=pairwise.device)
    target_index = torch.arange(num_sources, device=pairwise.device)
    loss_per_perm = pairwise[:, perms, target_index].mean(dim=-1)
    min_loss, index = loss_per_perm.min(dim=1)
    return min_loss, perms[index]


def reorder(estimates: Tensor, perm: Tensor) -> Tensor:
    """Apply a ``[B, N]`` permutation from :func:`best_permutation` to ``[B, N, ...]`` estimates."""
    return estimates[torch.arange(estimates.shape[0], device=estimates.device)[:, None], perm]


def pit_neg_sdr(estimates: Tensor, targets: Tensor, kind: str = "sisdr", zero_mean: bool = True) -> tuple[Tensor, Tensor]:
    min_loss, perm = best_permutation(pairwise_neg_sdr(estimates, targets, kind=kind, zero_mean=zero_mean))
    return min_loss.mean(), reorder(estimates, perm)


def waveform_stft_mae(estimates: Tensor, targets: Tensor, n_fft: int = 2048,
                      hop_length: int = 512) -> tuple[Tensor, Tensor]:
    """Waveform MAE + STFT real MAE + STFT imaginary MAE, in fixed source order and fp32."""
    if estimates.shape != targets.shape:
        raise ValueError(f"shape mismatch: {tuple(estimates.shape)} vs {tuple(targets.shape)}")
    estimates_f, targets_f = estimates.float(), targets.float()
    window = torch.hann_window(n_fft, dtype=torch.float32).to(estimates.device)
    kwargs = dict(n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    estimate_spec = torch.stft(estimates_f.reshape(-1, estimates.shape[-1]), **kwargs)
    target_spec = torch.stft(targets_f.reshape(-1, targets.shape[-1]), **kwargs)
    wave = (estimates_f - targets_f).abs().mean()
    real = (estimate_spec.real - target_spec.real).abs().mean()
    imag = (estimate_spec.imag - target_spec.imag).abs().mean()
    return wave + (real + imag), estimates


OBJECTIVES: dict[str, Callable[..., tuple[Tensor, Tensor]]] = {
    "pit_neg_sisdr": lambda est, tgt: pit_neg_sdr(est, tgt, kind="sisdr"),
    "pit_neg_snr": lambda est, tgt: pit_neg_sdr(est, tgt, kind="snr"),
    "waveform_stft_mae": waveform_stft_mae,
}


def build_objective(name: str, **options) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
    """Look up an objective by name; ``options`` are forwarded (e.g. ``n_fft`` for the STFT MAE)."""
    if name not in OBJECTIVES:
        raise ValueError(f"unknown objective {name!r}; choose from {sorted(OBJECTIVES)}")
    function = OBJECTIVES[name]
    return (lambda est, tgt: function(est, tgt, **options)) if options else function
