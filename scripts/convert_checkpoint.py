"""Convert a training checkpoint into a ``from_pretrained`` directory.

    python scripts/convert_checkpoint.py configs/dnr/dbs_bsrnn.yaml epoch=180.ckpt exports/DBS-BSRNN-DnR

Accepts checkpoints written by ``dbs.train`` and the checkpoints of the original
training code (``audio_model.*`` keys). Tensors of the original code that the
released models do not use are dropped: the untrained mixture-context
projections of DBS-BSRNN when ``use_mixture_context`` is off, and the unused
correction heads and per-region aliases of the shared similarity projection of
DBS-TIGER. The export is reloaded and checked against the checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dbs.engine import instantiate, load_config, locate  # noqa: E402

UNUSED = (".neural_correction.",)


def model_state(checkpoint: dict, expected: set[str]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    for prefix in ("model.", "audio_model."):
        if any(key.startswith(prefix) for key in state):
            state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
            break
    dropped = [key for key in state if key not in expected]
    unknown = [key for key in dropped if not (
        any(tag in key for tag in UNUSED) or ".spec_context_proj." in key
        or (key.startswith("chunkers.") and ".shared_similarity_projection." in key))]
    if unknown:
        raise KeyError(f"unexpected checkpoint tensors: {unknown[:5]}")
    return {key: value for key, value in state.items() if key in expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("overrides", nargs="*", help="config overrides, key.subkey=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    model = instantiate(cfg.model).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(model_state(checkpoint, set(model.state_dict())), strict=True)
    model.save_pretrained(args.output)

    restored = locate(cfg.model._target_).from_pretrained(args.output).eval()
    for name, tensor in model.state_dict().items():
        if not torch.equal(tensor, restored.state_dict()[name]):
            raise RuntimeError(f"{name} differs after export")
    channels = 2 if getattr(model, "stereo", False) else 1
    sample_rate = int(getattr(model, "sample_rate", None) or cfg.data.sample_rate)  # DBS-MelRoformer has none
    mixture = torch.randn(1, channels, sample_rate)
    with torch.inference_mode():
        same = torch.equal(model(mixture), restored(mixture))
    print(f"exported {sum(p.numel() for p in model.parameters()):,} parameters to {args.output} "
          f"(reload bitwise identical: {same})")


if __name__ == "__main__":
    main()
