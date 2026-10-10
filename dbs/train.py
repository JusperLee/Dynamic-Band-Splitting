"""Train a separation model from an experiment config.

    python -m dbs.train configs/dnr/dbs_bsrnn.yaml data.train_list=/data/dnr/train.npy \
        data.valid_dir=/data/dnr/val data.test_dir=/data/dnr/test

Positional ``key.subkey=value`` arguments override the config. ``--resume``
continues from a Lightning checkpoint (optimizer, scheduler and epoch included).
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from .engine import SeparationSystem, instantiate, load_config, to_plain


def _device_count(devices: Any) -> int:
    if isinstance(devices, (list, tuple)):
        return len(devices)
    if devices in (-1, "-1", "auto"):
        return max(1, torch.cuda.device_count())
    try:
        return int(devices)
    except (TypeError, ValueError):
        return 1


def build_trainer(cfg: DictConfig, callbacks: list, logger: Any, run_dir: Path) -> pl.Trainer:
    options = dict(to_plain(cfg.trainer))
    options.setdefault("default_root_dir", str(run_dir))
    ddp = dict(to_plain(cfg.get("ddp") or {}))
    if _device_count(options.get("devices", 1)) > 1 and "strategy" not in options:
        from pytorch_lightning.strategies import DDPStrategy

        options["strategy"] = DDPStrategy(
            find_unused_parameters=bool(ddp.get("find_unused_parameters", True)),
            process_group_backend=ddp.get("backend"),
            timeout=timedelta(seconds=int(ddp.get("timeout_seconds", 7200))),
        )
    return pl.Trainer(callbacks=callbacks, logger=logger, **options)


def build_system(cfg: DictConfig, datamodule) -> SeparationSystem:
    model = instantiate(cfg.model)
    return SeparationSystem(model, sources=datamodule.sources, sample_rate=datamodule.sample_rate,
                            config=to_plain(cfg), **to_plain(cfg.system))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="config overrides, key.subkey=value")
    parser.add_argument("--resume", type=Path, default=None, help="Lightning checkpoint to resume from")
    args = parser.parse_intermixed_args(argv)

    cfg = load_config(args.config, args.overrides)
    torch.set_float32_matmul_precision("highest")
    if cfg.get("seed") is not None:
        pl.seed_everything(int(cfg.seed), workers=True)
    run_dir = Path(cfg.experiment.dir) / str(cfg.experiment.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    datamodule = instantiate(cfg.data)
    system = build_system(cfg, datamodule)
    print(f"{type(system.model).__name__}: {sum(p.numel() for p in system.model.parameters()):,} parameters",
          flush=True)

    callbacks = [instantiate(spec) for spec in cfg.get("callbacks") or []]
    logger = (instantiate(cfg.logger) if cfg.get("logger")
              else pl.loggers.CSVLogger(str(run_dir), name="logs"))
    if logger:
        callbacks.append(pl.callbacks.LearningRateMonitor())
    trainer = build_trainer(cfg, callbacks, logger, run_dir)
    trainer.fit(system, datamodule=datamodule, ckpt_path=None if args.resume is None else str(args.resume))


if __name__ == "__main__":
    main()
