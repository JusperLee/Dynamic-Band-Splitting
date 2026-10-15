<p align="center">
  <img src="project_page/figures/logo.svg" width="88" height="88" alt="DBS logo: dynamic frequency bands">
</p>

<h1 align="center">Learning Where to Split:<br>Dynamic Band Splitting for Audio Source Separation</h1>

<p align="center">
  <strong>Bingzhe Zhang<sup>*</sup>, <a href="https://cslikai.cn/">Kai Li</a><sup>*†</sup>, <a href="https://xlhu.cn/index.html">Xiaolin Hu</a><sup>†</sup></strong><br>
  Tsinghua University<br>
  <sup>*</sup> Equal contribution &nbsp;·&nbsp; <sup>†</sup> Corresponding authors
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.14993">📄 Paper</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/pdf/2610.14993">PDF</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/papers/2610.14993">🤗 HF Paper</a> &nbsp;|&nbsp;
  <a href="https://cslikai.cn/Dynamic-Band-Splitting/">🌐 Project Page</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/spaces/JusperLee/DBS-Demo">🎧 Demo</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/JusperLee/DBS-BSRNN-DnR">🤗 DBS-BSRNN</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/JusperLee/DBS-TIGER-DnR">🤗 DBS-TIGER</a>
</p>

<p align="center">
  <a href="https://visitor-badge.laobi.icu/"><img src="https://visitor-badge.laobi.icu/badge?page_id=JusperLee.Dynamic-Band-Splitting&amp;left_text=Visitors&amp;left_color=%23555555&amp;right_color=%23007ec6&amp;radius=0" alt="Visitors (README views)" title="Counts README badge views, not unique visitors"></a>
  <a href="https://github.com/JusperLee/Dynamic-Band-Splitting/stargazers"><img src="https://img.shields.io/github/stars/JusperLee/Dynamic-Band-Splitting?style=flat-square&amp;logo=github" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square" alt="License: Apache 2.0"></a>
  <a href="#-installation"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11 or newer"></a>
  <a href="https://github.com/JusperLee/Dynamic-Band-Splitting/actions/workflows/tests.yml"><img src="https://github.com/JusperLee/Dynamic-Band-Splitting/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests"></a>
</p>

> **Dynamic Band Splitting (DBS)** learns where to split the frequency axis for each mixture, while keeping a fixed number of band tokens.

## 💥 News

- **[2026-10-15]** 📄 Our paper, [Learning Where to Split: Dynamic Band Splitting for Audio Source Separation](https://arxiv.org/abs/2610.14993), is now available on arXiv! Read the [PDF](https://arxiv.org/pdf/2610.14993) and join the [discussion on Hugging Face](https://huggingface.co/papers/2610.14993).
- **[2026-10-10]** 🚀 We release the code, pretrained [DBS-BSRNN](https://huggingface.co/JusperLee/DBS-BSRNN-DnR) and [DBS-TIGER](https://huggingface.co/JusperLee/DBS-TIGER-DnR) models, and an interactive [listening studio](https://huggingface.co/spaces/JusperLee/DBS-Demo)! Upload audio or video, listen to speech, music and sound effects, and explore the model's dynamic frequency bands.

## 🧩 Overview

Band-split separators group STFT bins into a fixed set of bands, and the same boundaries are used
for every mixture. **Dynamic Band Splitting (DBS)** predicts the boundaries from the mixture
instead, while keeping the number of band tokens fixed. Within each coarse frequency region, a
content-conditioned scorer rates every bin and hard Top-*K* picks the boundaries; content-weighted
pooling turns each band into one token; the unchanged separator backbone (BSRNN, TIGER,
BS-RoFormer or Mel-RoFormer) processes the tokens; and the separated tokens are expanded back to
the full STFT grid and decoded into complex masks. DBS-MelRoFormer uses a continuous variant: a
router shifts the interior region boundaries by a bounded offset, and bins are pooled into and
scattered from its 128 bands with soft memberships.

<p align="center"><img src="project_page/assets/overall.png" width="100%" alt="Overview of Dynamic Band Splitting"></p>

## ⚡ Installation

Python 3.11 or newer. Install a PyTorch build for your machine first, then the package:

```bash
git clone https://github.com/JusperLee/Dynamic-Band-Splitting.git
cd Dynamic-Band-Splitting
pip install -r requirements.txt      # exact versions of the tested environment
pip install --no-deps -e .
```

`pip install -e .` alone installs the models only; add `.[train]` for training and evaluation.
The clone does not download the project-page videos (Git LFS `fetchexclude`).

## 🎧 Separating audio with the released models

```python
import soundfile as sf
import torch
from dbs.models import DBSBSRNN
from dbs.engine import separate

model = DBSBSRNN.from_pretrained("JusperLee/DBS-BSRNN-DnR").eval()
audio, rate = sf.read("soundtrack.wav", dtype="float32", always_2d=True)  # 44.1 kHz
mixture = torch.from_numpy(audio.mean(1)).view(1, 1, -1)
with torch.inference_mode():
    music, sfx, speech = separate(model, mixture, {"mode": "chunked", "chunk_seconds": 3.0, "hop_seconds": 1.5},
                                  sample_rate=44100)[0]
```

For DBS-TIGER, load `DBSTIGER.from_pretrained("JusperLee/DBS-TIGER-DnR")` and use 6 s windows with
a 3 s hop (`"chunk_seconds": 6.0, "hop_seconds": 3.0`), as in its training recipe.

`model(mixture, return_aux=True)` also returns the predicted boundaries of every region. The
interactive studio in [`demo/`](demo/) shows them in sync with playback.

## 🏋️ Training and evaluation

Every experiment is one YAML file in [`configs/`](configs/); `key=value` arguments override it.

```bash
# DnR v2: DBS-BSRNN
python -m dbs.train configs/dnr/dbs_bsrnn.yaml \
    data.train_list=/data/dnr/train_vad.npy data.valid_dir=/data/dnr/val data.test_dir=/data/dnr/test
python -m dbs.evaluate configs/dnr/dbs_bsrnn.yaml --checkpoint runs/dnr-dbs-bsrnn/checkpoints/last.ckpt \
    data.test_dir=/data/dnr/test --output runs/dnr-dbs-bsrnn/test.json --rows runs/dnr-dbs-bsrnn/rows.json
# the released weights
python -m dbs.evaluate configs/dnr/dbs_tiger.yaml --pretrained JusperLee/DBS-TIGER-DnR data.test_dir=/data/dnr/test
```

| Config | Model |
| --- | --- |
| `dnr/dbs_bsrnn.yaml`, `dnr/dbs_tiger.yaml` | DBS-BSRNN (128), DBS-TIGER (128) on DnR v2 |
| `echoset/dbs_bsrnn.yaml`, `echoset/dbs_tiger.yaml` | DBS-BSRNN, DBS-TIGER on EchoSet |
| `musdb/dbs_bsrnn.yaml`, `musdb/dbs_bs_roformer.yaml`, `musdb/dbs_mel_roformer.yaml` | DBS-BSRNN, DBS-BS-RoFormer, DBS-MelRoFormer on MUSDB18-HQ |

The configs reproduce the original training runs (GPU count, batch size, precision, schedule);
change `trainer.devices` and `data.batch_size` for other hardware. Details per dataset:

- **DnR v2** ([download](https://zenodo.org/records/6949108)). Validation and test directories hold
  `mix.json`, `s1.json`, `s2.json`, `s3.json` (lists of `[wav path, samples]`). The paper's
  training data is an `.npy` list of HDF5 files with `music_VAD`, `sfx_VAD` and `speech_VAD`
  segment datasets (`data.train_list`); a training directory with the same JSON manifests can be
  used instead (`data.train_dir`). Scores are SI-SDR on the 973 full-length test tracks, with
  overlap-add windows of 3 s / 1.5 s hop (DBS-BSRNN) and 6 s / 3 s hop (DBS-TIGER).
- **EchoSet** ([download](https://huggingface.co/datasets/JusperLee/EchoSet)). Each split holds
  `mix.json`, `s1.json`, `s2.json`. SDRi (BSS-Eval, `fast_bss_eval`) and SI-SDRi over the mixture,
  one forward pass per 6 s mixture.
- **MUSDB18-HQ** ([download](https://zenodo.org/records/3338373)). Standard track folders. Scores
  are the global SDR of Music-Source-Separation-Training; `--museval` adds BSS-Eval v4 median SDR.

Run `python -m pytest tests` for the package tests.

## 📊 Results

**DnR v2** (SI-SDR, dB; 973 full-length test tracks)

| Model | Music | Speech | SFX | Avg. | Params (M) |
| --- | --- | --- | --- | --- | --- |
| DBS-BSRNN (128) | 6.48 | 15.01 | 5.73 | 9.07 | 4.7 |
| DBS-TIGER (128) | 7.78 | 15.90 | 6.70 | 10.13 | 1.0 |

**MUSDB18-HQ** (four-stem SDR, dB; single models trained only on MUSDB18-HQ)

| Model | Vocals | Drums | Bass | Other | Avg. | Params (M) |
| --- | --- | --- | --- | --- | --- | --- |
| DBS-BSRNN | 6.70 | 6.44 | 4.58 | 4.01 | 5.43 | 21.7 |
| DBS-BS-RoFormer | 7.64 | 7.72 | 5.29 | 4.72 | 6.34 | 33.9 |
| DBS-MelRoFormer | 10.02 | 9.30 | 6.96 | 5.98 | 8.07 | 19.1 |

**EchoSet** (dB; 2,650 test mixtures)

| Model | SDRi | SI-SDRi | Params (M) |
| --- | --- | --- | --- |
| DBS-BSRNN | 13.21 | 12.66 | 7.1 |
| DBS-TIGER | 14.55 | 14.04 | 1.0 |

Comparisons with the corresponding fixed-band models are in the paper and on the
[project page](https://cslikai.cn/Dynamic-Band-Splitting/).

## 🗂️ Repository layout

```
dbs/            package: models (dbs.models), data, losses, metrics, engine, train/evaluate CLIs
configs/        one YAML per experiment
scripts/        convert_checkpoint.py (training checkpoint -> from_pretrained directory)
demo/           Gradio studio (Hugging Face Space)
project_page/   https://cslikai.cn/Dynamic-Band-Splitting/
tests/
```

## 📝 Citation

```bibtex
@misc{zhang2026dbs,
  title         = {Learning Where to Split: Dynamic Band Splitting for Audio Source Separation},
  author        = {Zhang, Bingzhe and Li, Kai and Hu, Xiaolin},
  year          = {2026},
  eprint        = {2610.14993},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  doi           = {10.48550/arXiv.2610.14993},
  url           = {https://arxiv.org/abs/2610.14993}
}
```

## 📄 License

Apache License 2.0. Parts of the code are adapted from BS-RoFormer, Music-Source-Separation-Training
and TIGER under the MIT License; see [THIRD_PARTY.md](THIRD_PARTY.md).

