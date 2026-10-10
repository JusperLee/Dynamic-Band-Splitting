"""Evaluate a separation model on the full test recordings of a benchmark.

    # a released model
    python -m dbs.evaluate configs/dnr/dbs_bsrnn.yaml --pretrained JusperLee/DBS-BSRNN-DnR \
        data.test_dir=/data/dnr/test
    # a checkpoint written by dbs.train
    python -m dbs.evaluate configs/dnr/dbs_bsrnn.yaml --checkpoint runs/dnr-dbs-bsrnn/checkpoints/last.ckpt \
        data.test_dir=/data/dnr/test --rows runs/dnr-dbs-bsrnn/test_rows.json

The protocol (inference windows, metrics) comes from the ``evaluation`` block
of the config and falls back to its validation settings. Positional
``key=value`` arguments override the config.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .engine import instantiate, load_config, locate, separate, to_plain
from .metrics import REPORTS, summarize

REPORT_BY_DATA = {"DnRDataModule": "dnr", "EchoSetDataModule": "echoset",
                  "MusdbDataModule": "musdb", "MusdbChunkDataModule": "musdb"}


def load_model(model_cfg: dict[str, Any], checkpoint: str | None = None, pretrained: str | None = None) -> nn.Module:
    """Build ``model_cfg`` and load weights from a ``dbs.train`` checkpoint, a state dict, or a Hub repo/directory."""
    if pretrained is not None:
        options = {key: value for key, value in model_cfg.items() if key != "_target_"}
        return locate(model_cfg["_target_"]).from_pretrained(pretrained, **options)
    model = instantiate(model_cfg)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        if all(key.startswith("model.") for key in state):
            state = {key.removeprefix("model."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    return model


def main(argv: Sequence[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("overrides", nargs="*", help="config overrides, key.subkey=value")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", help="checkpoint written by dbs.train, or a plain state dict")
    weights.add_argument("--pretrained", help="Hugging Face repo id or a save_pretrained directory")
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--output", type=Path, default=None, help="summary JSON (default: print only)")
    parser.add_argument("--rows", type=Path, default=None, help="per-recording scores JSON")
    parser.add_argument("--museval", action="store_true", help="MUSDB18-HQ: also compute BSS-Eval v4 SDR")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N recordings")
    args = parser.parse_intermixed_args(argv)

    cfg = load_config(args.config, args.overrides)
    evaluation = dict(to_plain(cfg.get("evaluation") or {}))
    validation = dict(to_plain(cfg.system.get("validation") or {}))
    inference = evaluation.get("inference") or validation.get("inference") or {"mode": "full"}
    report_name = evaluation.get("report") or REPORT_BY_DATA[str(cfg.data._target_).rsplit(".", 1)[-1]]
    report = REPORTS[report_name]

    datamodule = instantiate(cfg.data)
    dataset = datamodule.test_dataset() if args.split == "test" else datamodule.val_dataset()
    model = load_model(to_plain(cfg.model), args.checkpoint, args.pretrained).to(args.device).eval()
    device_type = torch.device(args.device).type
    autocast = (torch.autocast(device_type, dtype=torch.float16)
                if evaluation.get("autocast") and device_type == "cuda" else contextlib.nullcontext())

    rows = []
    count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    with torch.inference_mode():
        for index in range(count):
            mixture, sources, key = dataset[index]
            with autocast:
                estimates = separate(model, mixture.unsqueeze(0).to(args.device), inference, datamodule.sample_rate)
            row = report(estimates.float().cpu(), sources.unsqueeze(0), mixture.unsqueeze(0),
                         sources=datamodule.sources, museval=args.museval, sample_rate=datamodule.sample_rate)
            rows.append({"key": str(key), **row})
            print(json.dumps(rows[-1]), flush=True)

    summary = summarize([{k: v for k, v in row.items() if k != "key"} for row in rows])
    result = {"config": str(args.config), "weights": args.checkpoint or args.pretrained, "split": args.split,
              "report": report_name, "inference": inference, "recordings": len(rows), "summary": summary}
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    if args.rows is not None:
        args.rows.parent.mkdir(parents=True, exist_ok=True)
        args.rows.write_text(json.dumps(rows, indent=2))
    return summary


def cli() -> None:
    """Console entry point (``dbs-evaluate``); exits with status 0 on success."""
    main()


if __name__ == "__main__":
    cli()
