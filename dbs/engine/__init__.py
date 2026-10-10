"""Training engine: experiment configs, the Lightning system, inference protocols and schedules.

Names are imported on first use, so ``from dbs.engine import separate`` works
without the training dependencies (PyTorch Lightning, OmegaConf).
"""
from importlib import import_module

_EXPORTS = {
    "separate": ".inference",
    "WarmupStepDecay": ".schedulers",
    "SeparationSystem": ".system",
    "ratio_weight": ".system",
    "instantiate": ".config",
    "load_config": ".config",
    "locate": ".config",
    "to_plain": ".config",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
