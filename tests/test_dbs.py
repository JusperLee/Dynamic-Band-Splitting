"""Tests of the dbs package: models, losses, metrics, configs and the train/evaluate CLIs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from dbs.engine import load_config
from dbs.losses import pit_neg_sdr
from dbs.metrics import global_sdr, si_sdr, snr
from dbs.models import DBSBSRNN, DBSBSRoformer, DBSMelRoformer, DBSTIGER

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Parameter counts of the paper's models (the original training code builds the same architectures).
PARAMETERS = {
    "dnr/dbs_bsrnn": 4_713_557, "dnr/dbs_tiger": 1_013_535,
    "echoset/dbs_bsrnn": 7_099_781, "echoset/dbs_tiger": 982_683,
    "musdb/dbs_bsrnn": 21_729_977, "musdb/dbs_bs_roformer": 33_858_965, "musdb/dbs_mel_roformer": 19_095_843,
}

SMALL = {
    DBSBSRNN: dict(dim=16, depth=1, rnn_hidden_size=8),
    DBSTIGER: dict(out_channels=12, in_channels=24, num_blocks=1, upsampling_depth=2),
    DBSBSRoformer: dict(dim=16, depth=1),
    DBSMelRoformer: dict(dim=16, depth=1),
}


@pytest.mark.parametrize("name", sorted(PARAMETERS))
def test_config_builds_the_paper_model(name):
    from dbs.engine import instantiate

    model = instantiate(load_config(CONFIGS / f"{name}.yaml").model)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS[name]


@pytest.mark.parametrize("cls", list(SMALL), ids=lambda c: c.__name__)
def test_forward_shapes_and_from_pretrained(cls, tmp_path):
    torch.manual_seed(0)
    model = cls(**SMALL[cls]).eval()
    stereo = cls in (DBSBSRoformer, DBSMelRoformer)
    mixture = torch.randn(1, 2 if stereo else 1, 22050)
    with torch.inference_mode():
        estimate = model(mixture)
    assert estimate.shape[-1] == mixture.shape[-1]
    assert estimate.shape[1] == (4 if stereo else 3)
    model.save_pretrained(tmp_path)
    restored = cls.from_pretrained(tmp_path).eval()
    with torch.inference_mode():
        assert torch.equal(restored(mixture), estimate)


def test_dbs_return_aux():
    model = DBSBSRNN(**SMALL[DBSBSRNN]).eval()
    with torch.inference_mode():
        estimate, aux = model(torch.randn(1, 1, 22050), return_aux=True)
    assert estimate.shape == (1, 3, 22050)
    assert [int(h[0].sum()) for h in aux["hard_boundaries"]] == [32, 32, 32, 32]


def test_losses_and_metrics():
    torch.manual_seed(0)
    targets = torch.randn(2, 3, 4000)
    estimates = targets + 0.1 * torch.randn_like(targets)
    loss, aligned = pit_neg_sdr(estimates[:, [2, 0, 1]], targets)
    assert torch.equal(aligned, estimates)
    assert torch.allclose(-loss, si_sdr(estimates, targets).mean().float(), atol=1e-4)
    assert torch.allclose(si_sdr(3 * estimates, targets), si_sdr(estimates, targets))
    assert (snr(estimates, targets) > 15).all() and (global_sdr(estimates, targets) > 15).all()


def _write(path: Path, audio: np.ndarray, rate: int) -> list:
    sf.write(path, audio, rate)
    return [str(path), len(audio)]


@pytest.fixture
def dnr_data(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        folder = tmp_path / split
        folder.mkdir()
        rows = {key: [] for key in ("mix", "s1", "s2", "s3")}
        for i in range(2):
            stems = [(0.1 * rng.standard_normal(2 * 44100)).astype(np.float32) for _ in range(3)]
            for key, name, audio in zip(("s1", "s2", "s3", "mix"), ("music", "sfx", "speech", "mix"),
                                        [*stems, sum(stems)]):
                rows[key].append(_write(folder / f"{i}_{name}.wav", audio, 44100))
        for key, value in rows.items():
            (folder / f"{key}.json").write_text(json.dumps(value))
    return tmp_path


def test_train_then_evaluate(dnr_data, tmp_path):
    from dbs.evaluate import main as evaluate
    from dbs.train import main as train

    overrides = [f"data.train_dir={dnr_data}/train", f"data.valid_dir={dnr_data}/val",
                 f"data.test_dir={dnr_data}/test", "data.segment=1.0", "data.num_iter_per_epoch=2",
                 "data.num_workers=0", "data.pin_memory=false", "model.dim=16", "model.depth=1",
                 "model.rnn_hidden_size=8"]
    train([str(CONFIGS / "dnr/dbs_bsrnn.yaml"), *overrides, f"experiment.dir={tmp_path}/runs",
           "trainer.accelerator=cpu", "trainer.devices=1", "trainer.precision=32-true", "trainer.max_epochs=1",
           "trainer.num_sanity_val_steps=0", "trainer.enable_progress_bar=false"])
    run = tmp_path / "runs" / "dnr-dbs-bsrnn"
    rows = [r for r in csv.DictReader((run / "logs/version_0/metrics.csv").open()) if r.get("val_avg_snr")]
    assert rows and np.isfinite(float(rows[-1]["val_loss"]))
    summary = evaluate([str(CONFIGS / "dnr/dbs_bsrnn.yaml"), "--checkpoint", str(run / "checkpoints/last.ckpt"),
                        "--device", "cpu", *overrides])
    assert np.isfinite(summary["pit_avg_sisdr"]) and set(summary) >= {"fixed_music_sisdr", "pit_avg_snr"}


@pytest.mark.parametrize("samples", [600, 1024, 30000])
def test_chunked_inference_handles_short_audio(samples):
    from dbs.engine import separate

    model = DBSBSRNN(**SMALL[DBSBSRNN]).eval()
    mixture = 0.1 * torch.randn(1, 1, samples)
    with torch.inference_mode():
        estimate = separate(model, mixture, {"mode": "chunked", "chunk_seconds": 1.0, "hop_seconds": 0.5}, 44100)
    assert estimate.shape == (1, 3, samples) and torch.isfinite(estimate).all()
