---
title: DBS Listening Studio
emoji: 🎧
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12.12"
app_file: app.py
license: apache-2.0
---

# DBS · Listening Studio

Separate speech, music and sound effects from a video or audio file with the released DnR v2
models (**DBS-BSRNN**, **DBS-TIGER**) and watch the predicted frequency boundaries of every
inference window move with the content. Video keeps its picture; audio shows a clickable STFT
spectrogram. The interface is available in English and Chinese.

**Learning Where to Split: Dynamic Band Splitting for Audio Source Separation**

Bingzhe Zhang\*, Kai Li\*†, Xiaolin Hu† · Tsinghua University

[Paper](https://arxiv.org/abs/2610.14993) ·
[PDF](https://arxiv.org/pdf/2610.14993) ·
[HF Paper](https://huggingface.co/papers/2610.14993) ·
[Code](https://github.com/JusperLee/Dynamic-Band-Splitting) ·
[Project page](https://cslikai.cn/Dynamic-Band-Splitting/)

The studio displays the paper title, authors and affiliations. Set
`DBS_PAPER_URL=https://arxiv.org/abs/2610.14993` in the Space's Variables settings or in your
local environment to enable the Paper button.

## Run locally

From the repository root:

```bash
pip install -r requirements.txt && pip install --no-deps -e .
pip install -r demo/requirements.txt          # FFmpeg must be on PATH
python demo/app.py                            # http://127.0.0.1:7860
```

The weights are downloaded from the Hugging Face Hub on first use
(`JusperLee/DBS-BSRNN-DnR`, `JusperLee/DBS-TIGER-DnR`). `--bsrnn` / `--tiger` accept a Hub repo
id or a local `save_pretrained` directory, `--device cuda` runs on a GPU, and `--sample-video`
sets the clip shown on start-up. Tests: `cd demo && python -m pytest`.

## Hugging Face ZeroGPU

To deploy this studio on ZeroGPU, copy `requirements-space.txt` to the Space's
`requirements.txt`, keep Python 3.12.12, and select ZeroGPU in the Space hardware settings.
The pinned Spaces SDK initializes both models on CUDA before the mounted Gradio app starts.
Only audio separation requests a GPU; decoding, spectrograms, video exports and downloads run
on CPU. Cached results do not request a GPU. Local CPU or dedicated CUDA runs do not require
the Spaces SDK. GPU access on the Hub is subject to the visitor's queue and daily quota.
GPU allocation estimates depend on clip length and overlap and remain within the current
anonymous visitor's two-minute allowance. Use shorter clips or a larger hop if a job exceeds
its allocation.

## What is shown

- Every window runs the full model with `return_aux=True`; the boundaries drawn are its hard
  Top-K boundaries (four regions, 128 bands). Bold colored lines are a sample of those
  boundaries, not importance scores. Switching stems keeps the boundaries, since they are
  predicted from the mixture.
- Windows are 3 s with a 3 s hop by default (0.5-15 s windows, hop at least 0.1 s and at most the
  window). Overlapping windows are averaged with uniform weights, as in the DnR v2 evaluation;
  the last window is zero-padded and the output is cropped to the input length.
- Input is decoded to 44.1 kHz mono. Stems share one gain, applied only to prevent clipping and
  recorded in `results.json`. Files are limited to 10 minutes and 500 MB.
- Downloads: the separated WAV files, the boundary and metadata JSON files, and either the dubbed
  MP4 files (video) or the spectrogram PNG files (audio).
- On CPU, DBS-TIGER runs single-threaded with NNPACK disabled during separation, which is faster
  for its grouped convolutions and gives the same output.

## Citation

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
