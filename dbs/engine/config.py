"""Experiment configs: YAML with ``_base_`` inheritance, dotted overrides, and ``_target_`` instantiation."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from omegaconf import DictConfig, OmegaConf


def _drop_replaced(base: DictConfig, child: DictConfig) -> None:
    """Remove base mappings that the child redefines with a different ``_target_`` (replace, don't merge)."""
    for key, value in child.items():
        if not isinstance(value, DictConfig) or not isinstance(base.get(key), DictConfig):
            continue
        if "_target_" in value and value.get("_target_") != base[key].get("_target_"):
            base.pop(key)
        else:
            _drop_replaced(base[key], value)


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> DictConfig:
    """Load ``path``; a top-level ``_base_: other.yaml`` is merged underneath (relative to ``path``).

    A mapping whose ``_target_`` differs from the base replaces the base
    mapping instead of being merged into it. ``overrides`` are
    ``key.subkey=value`` strings applied last.
    """
    path = Path(path)
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is not None:
        parent = load_config(path.parent / base)
        _drop_replaced(parent, cfg)
        cfg = OmegaConf.merge(parent, cfg)
    overrides = list(overrides)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def to_plain(node: Any) -> Any:
    """Resolve interpolations and convert a config node to plain Python containers."""
    if isinstance(node, DictConfig) or OmegaConf.is_list(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def locate(name: str) -> Any:
    """Import ``package.module.Attribute``."""
    module, _, attribute = name.rpartition(".")
    if not module:
        raise ValueError(f"expected a dotted path, got {name!r}")
    return getattr(importlib.import_module(module), attribute)


def instantiate(spec: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``spec["_target_"]`` with the remaining keys (nested ``_target_`` mappings are built first)."""
    spec = dict(to_plain(spec))
    target = locate(spec.pop("_target_"))
    built = {key: _build(value) for key, value in spec.items()}
    built.update(kwargs)
    return target(*args, **built)


def _build(value: Any) -> Any:
    if isinstance(value, Mapping) and "_target_" in value:
        return instantiate(value)
    if isinstance(value, list):
        return [_build(item) for item in value]
    return value
