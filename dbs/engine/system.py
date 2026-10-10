"""The Lightning module shared by every training recipe."""
from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Mapping, Sequence

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from ..losses import best_permutation, build_objective, pairwise_neg_sdr, reorder
from ..metrics import source_scores
from .config import instantiate, to_plain
from .inference import separate

LIGHTNING_SCHEDULER_KEYS = ("monitor", "interval", "frequency", "strict")


def ratio_weight(spec: float | Mapping[str, Any] | None, epoch: int) -> float:
    """Weight of the boundary ratio loss: a constant, or ``{max, start, end}`` for a linear ramp over epochs."""
    if spec is None:
        return 0.0
    if not isinstance(spec, Mapping):
        return float(spec)
    peak, start, end = float(spec["max"]), int(spec.get("start", 0)), int(spec.get("end", 0))
    if peak <= 0 or epoch < start:
        return 0.0
    if epoch >= end or end <= start:
        return peak
    return peak * (epoch - start) / (end - start)


class SeparationSystem(pl.LightningModule):
    """Training and validation around a separation model.

    Batches are ``(mixture [B, C, T], sources [B, N, (C,) T], keys)``.

    * ``objective``: ``"model"`` trains on the model's own loss
      (``model(mixture, target)``); any name in :data:`dbs.losses.OBJECTIVES`
      is applied to the separated sources, plus ``ratio_loss_weight`` times the
      DBS boundary ratio loss (a constant or a ``{max, start, end}`` epoch ramp).
    * ``validation``: ``inference`` protocol (see :func:`dbs.engine.inference.separate`),
      optional ``loss`` objective logged as ``val_loss``, and ``metrics`` logged as
      ``val_{source}_{metric}`` and ``val_avg_{metric}``. With ``order: pit`` the
      estimates are first matched to the references by SI-SDR.
    * ``router_weight_decay``: weight decay of a separate parameter group for
      ``model.router`` (DBS-MelRoFormer).
    """

    def __init__(self, model: nn.Module, *, sources: Sequence[str], sample_rate: int,
                 optimizer: Mapping[str, Any], scheduler: Mapping[str, Any] | None = None,
                 objective: str = "model", objective_options: Mapping[str, Any] | None = None,
                 ratio_loss_weight: float | Mapping[str, Any] | None = None,
                 router_weight_decay: float | None = None, fail_on_nonfinite: bool = False,
                 validation: Mapping[str, Any] | None = None, config: Mapping[str, Any] | None = None):
        super().__init__()
        self.model = model
        self.sources = tuple(sources)
        self.sample_rate = int(sample_rate)
        self.optimizer_spec = dict(to_plain(optimizer))
        self.scheduler_spec = None if scheduler is None else dict(to_plain(scheduler))
        self.objective = None if objective == "model" else build_objective(objective, **dict(objective_options or {}))
        self.ratio_loss_weight = to_plain(ratio_loss_weight)
        self.router_weight_decay = router_weight_decay
        self.fail_on_nonfinite = bool(fail_on_nonfinite)
        validation = dict(to_plain(validation or {}))
        self.inference = dict(validation.get("inference") or {"mode": "full"})
        self.val_objective = build_objective(validation["loss"]) if validation.get("loss") else None
        self.val_metrics = tuple(validation.get("metrics", ()))
        self.val_order = validation.get("order", "fixed")
        if self.val_order not in ("fixed", "pit"):
            raise ValueError(f"validation.order must be 'fixed' or 'pit', got {self.val_order!r}")
        self.config = None if config is None else to_plain(config)
        parameters = inspect.signature(model.forward).parameters
        self._takes_epoch = "current_epoch" in parameters
        self._takes_aux = "return_aux" in parameters

    def _separate_with_aux(self, mixture: Tensor) -> tuple[Tensor, Tensor | None]:
        kwargs = {}
        if self._takes_epoch:
            kwargs["current_epoch"] = self.current_epoch
        if self._takes_aux:
            kwargs["return_aux"] = True
        output = self.model(mixture, **kwargs)
        if isinstance(output, tuple):
            return output[0], output[1].get("ratio_loss")
        return output, None

    def training_step(self, batch, batch_idx: int) -> Tensor:
        mixture, targets = batch[0], batch[1]
        log = partial(self.log, on_step=True, on_epoch=True, sync_dist=True, batch_size=mixture.shape[0])
        if self.objective is None:
            loss = self.model(mixture, targets)
        else:
            estimates, ratio = self._separate_with_aux(mixture)
            main, _ = self.objective(estimates.float(), targets.float())
            weight = ratio_weight(self.ratio_loss_weight, self.current_epoch)
            loss = main
            log("train/main_loss", main)
            if ratio is not None and weight > 0:
                loss = loss + weight * ratio
                log("train/ratio_loss", ratio)
                log("train/ratio_weight", weight)
        if self.fail_on_nonfinite and not bool(torch.isfinite(loss).all()):
            raise FloatingPointError(f"non-finite training loss at step {self.global_step}")
        log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        mixture, targets = batch[0], batch[1]
        estimates = separate(self.model, mixture, self.inference, self.sample_rate).float()
        targets = targets.float()
        log = partial(self.log, on_step=False, on_epoch=True, sync_dist=True, batch_size=mixture.shape[0])
        if self.val_objective is not None:
            loss, _ = self.val_objective(estimates, targets)
            log("val_loss", loss, prog_bar=True)
        if self.val_order == "pit":
            _, perm = best_permutation(pairwise_neg_sdr(estimates.double(), targets.double(), kind="sisdr"))
            estimates = reorder(estimates, perm)
        for metric, values in source_scores(estimates, targets, mixture, self.val_metrics).items():
            for index, name in enumerate(self.sources):
                log(f"val_{name}_{metric}", values[:, index].mean())
            log(f"val_avg_{metric}", values.mean(), prog_bar=True)

    def on_train_epoch_start(self) -> None:
        # Datasets whose draws depend on the epoch (MUSDB18-HQ chunk stream).
        dataset = getattr(self.trainer.train_dataloader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.current_epoch)

    def configure_optimizers(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        router = getattr(self.model, "router", None)
        if self.router_weight_decay is not None and isinstance(router, nn.Module):
            router_ids = {id(p) for p in router.parameters()}
            groups: Any = [
                {"params": [p for p in parameters if id(p) not in router_ids]},
                {"params": [p for p in parameters if id(p) in router_ids],
                 "weight_decay": float(self.router_weight_decay)},
            ]
        else:
            groups = parameters
        optimizer = instantiate(self.optimizer_spec, groups)
        if self.scheduler_spec is None:
            return optimizer
        spec = dict(self.scheduler_spec)
        lightning = {key: spec.pop(key) for key in LIGHTNING_SCHEDULER_KEYS if key in spec}
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": instantiate(spec, optimizer), **lightning}}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.config is not None:
            checkpoint["dbs_config"] = self.config
