"""Gradio studio for the released DnR v2 models (DBS-BSRNN and DBS-TIGER)."""
from __future__ import annotations

import argparse
import functools
import hashlib
import html
import json
import logging
import math
import os
from pathlib import Path
import re
import time
import zipfile

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
from gpu_runtime import ZERO_GPU, gpu, start_zero_gpu

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from media_io import prepare_media, prepare_video, write_outputs, mux_stem
from spectrogram import write_spectrograms
from backend import MODELS, Backend, weights_file, weights_source
from ui_text import COPY, t, bi, friendly_error
from research_info import research_panel

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODEL_LABELS = {key: spec["label"] for key, spec in MODELS.items()}
WEIGHTS: dict[str, Path] = {}  # model key -> local weights file (tests); default: the Hub
EXPORT_NAMES = ("speech.wav", "music.wav", "sfx.wav", "speech.mp4", "music.mp4",
                "sfx.mp4", "boundaries.json", "results.json")
DOWNLOAD_NAMES = ("DBS-results.zip", *EXPORT_NAMES)
AUDIO_EXPORT_NAMES = ("speech.wav", "music.wav", "sfx.wav", "mix-spectrogram.png",
                      "speech-spectrogram.png", "music-spectrogram.png", "sfx-spectrogram.png",
                      "boundaries.json", "results.json")
ALL_DOWNLOAD_NAMES = set(DOWNLOAD_NAMES) | set(AUDIO_EXPORT_NAMES)
MEDIA_EXTENSIONS = [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
                    ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".aiff", ".aif", ".opus"]
SAMPLE = Path(os.environ.get("DBS_SAMPLE_MEDIA", ROOT / "sample.mp4"))
DEVICE = os.environ.get("DBS_DEVICE", "cuda" if ZERO_GPU else "cpu")
LOG = logging.getLogger("dbs")


def validate_settings(window, hop, language="en"):
    window, hop = float(window), float(hop)
    if not all(math.isfinite(v) for v in (window, hop)):
        raise ValueError(t("finite", language))
    if not 0.5 <= window <= 15:
        raise ValueError(t("window_range", language))
    if not 0.1 <= hop <= window:
        raise ValueError(t("hop_range", language))
    return window, hop


def model_checkpoint(model_key):
    """Local weights file of a model (downloaded from the Hub on first use; ``WEIGHTS`` overrides)."""
    if model_key not in MODEL_LABELS:
        raise ValueError(t("choose_model"))
    return Path(WEIGHTS[model_key]) if model_key in WEIGHTS else weights_file(model_key)


@functools.lru_cache(maxsize=2)
def get_backend(checkpoint: str, checkpoint_sha256: str, model_key="bsrnn"):
    return Backend(model_key, device=DEVICE)


def initialize_gpu_runtime():
    """Load both models before ZeroGPU packs tensors for its GPU workers."""
    if not ZERO_GPU:
        return
    if DEVICE != "cuda":
        raise ValueError("ZeroGPU requires DBS_DEVICE=cuda.")
    for model_key in MODELS:
        checkpoint = model_checkpoint(model_key)
        get_backend(str(checkpoint), digest(checkpoint), model_key)
    start_zero_gpu()


def separation_budget(audio, window, hop, checkpoint, checksum, model_key, language, progress):
    """Reserve more time for longer clips and overlapping inference windows."""
    window_samples, hop_samples = round(window * 44100), round(hop * 44100)
    windows = 1 + max(0, (len(audio) - window_samples + hop_samples - 1) // hop_samples)
    # The pinned SDK scales Blackwell allocations by 1.5; keep the effective
    # request within the anonymous visitor's two-minute allowance.
    return min(80, max(20, math.ceil(10 + 0.2 * windows * window_samples / 44100)))


@gpu(duration=separation_budget)
def separate_audio(audio, window, hop, checkpoint, checksum, model_key, language, progress):
    """Only model computation leases a GPU; inputs and results remain on CPU."""
    backend = get_backend(checkpoint, checksum, model_key)

    def report(fraction, message=""):
        match = re.search(r"Separating window (\d+)/(\d+)", message)
        description = t("separating_window", language, current=match[1], total=match[2]) if match else t("separating", language)
        progress(0.15 + 0.70 * float(fraction), desc=description)

    result = backend.separate(audio, window_seconds=window, hop_seconds=hop, progress=report)
    if ZERO_GPU:
        import torch

        if not torch.cuda.is_initialized() or next(backend.model.parameters()).device.type != "cuda":
            raise RuntimeError("ZeroGPU did not initialize a CUDA model.")
        result["metadata"]["hardware"] = "zerogpu"
        result["metadata"]["cuda_initialized"] = True
        result["metadata"]["gpu_name"] = torch.cuda.get_device_name()
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def job_key(path, window, hop, model_key="bsrnn"):
    record = {"video": digest(path), "checkpoint": digest(model_checkpoint(model_key)),
              "window": window, "hop": hop, "pipeline": 1}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:24], record


def frame(job_id):
    return (f'<iframe class="studio-frame" src="/studio/{job_id}" '
            'title="Media, separated stems and frequency bands" data-aria-en="Media, separated stems and frequency bands" data-aria-zh="媒体、分离音轨与动态频带" allow="autoplay; fullscreen" '
            'style="width:100%;height:1100px;border:0;border-radius:18px;background:#fcfbf8" '
            'loading="eager"></iframe>')


def export_names(manifest):
    return AUDIO_EXPORT_NAMES if manifest.get("media_type") == "audio" else EXPORT_NAMES


def downloads(job):
    manifest = cached_manifest(job)
    return [str(job / name) for name in ("DBS-results.zip", *export_names(manifest))] if manifest else []


def download_panel(job):
    """Show every export in a responsive panel, with no clipped file table."""
    manifest = cached_manifest(job)
    if not manifest:
        return ""
    model = MODEL_LABELS.get(manifest.get("meta", {}).get("model_key", "bsrnn"), "DBS")
    summary = bi(f"{model} · {manifest['duration']:g} s · {manifest['window_seconds']:g} s window / {manifest['hop_seconds']:g} s hop",
                 f"{model} · {manifest['duration']:g} 秒 · {manifest['window_seconds']:g} 秒窗口 / {manifest['hop_seconds']:g} 秒间隔")

    def link(name, label, bundle=False):
        size = (job / name).stat().st_size
        size_label = f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.1f} KB"
        return (f'<a class="{"download-bundle" if bundle else "download-item"}" '
                f'href="/download/{html.escape(job.name, quote=True)}/{name}" download="{name}">'
                f'<span class="download-copy"><b>{label}</b>'
                f'<span class="download-filename">{name}</span></span>'
                f'<span class="download-size">{size_label} <span aria-hidden="true">↓</span></span></a>')

    visual_group = (
        bi("Spectrograms", "频谱图"), bi("PNG · Shared intensity scale", "PNG · 统一强度色阶"),
        tuple((f"{key}-spectrogram.png", bi(en, zh)) for key, en, zh in
              (("mix", "Original", "原声"), ("speech", "Speech", "人声"),
               ("music", "Music", "音乐"), ("sfx", "Sound effects", "音效")))
    ) if manifest.get("media_type") == "audio" else (
        bi("Separated videos", "分离视频"), bi("MP4 · Video + selected stem", "MP4 · 画面 + 对应音轨"),
        (("speech.mp4", bi("Speech video", "人声视频")), ("music.mp4", bi("Music video", "音乐视频")), ("sfx.mp4", bi("Sound effects video", "音效视频")))
    )
    groups = (
        (bi("Audio stems", "音轨"), bi("WAV · 44.1 kHz", "WAV · 44.1 kHz"), (("speech.wav", bi("Speech", "人声")), ("music.wav", bi("Music", "音乐")), ("sfx.wav", bi("Sound effects", "音效")))),
        visual_group,
        (bi("Frequency bands & settings", "频带与参数"), bi("JSON · Reproducible data", "JSON · 可复现数据"), (("boundaries.json", bi("Dynamic band boundaries", "动态频带边界")), ("results.json", bi("Inference & export details", "推理参数与导出信息")))),
    )
    cards = "".join(
        f'<section class="download-group"><h3>{title}<small>{subtitle}</small></h3>'
        + "".join(link(name, label) for name, label in entries) + '</section>'
        for title, subtitle, entries in groups
    )
    return ('<section class="download-panel" aria-label="Downloads" data-aria-en="Downloads" data-aria-zh="下载结果">'
            f'<div class="download-heading"><div><div class="section-title"><span>03</span> {bi("Downloads", "下载结果")}</div>'
            f'<p>{summary}</p></div>{link("DBS-results.zip", bi(f"Download all · {len(export_names(manifest))} files", f"下载全部 · {len(export_names(manifest))} 个文件"), bundle=True)}</div>'
            f'<div class="download-grid">{cards}</div></section>')



def cached_manifest(job):
    try:
        data = json.loads((job / "manifest.json").read_text())
        media_name = "mixture.wav" if data.get("media_type") == "audio" else "preview.mp4"
        required = (media_name, "DBS-results.zip", *export_names(data))
        if not all((job / name).is_file() and (job / name).stat().st_size for name in required):
            return None
        if data.get("media_type") == "audio" and not all(
                data.get("spectrograms", {}).get(key, {}).get("url") for key in ("mix", "speech", "music", "sfx")):
            return None
        return data if data.get("ready") and data.get("partitions") else None
    except (ValueError, OSError):
        return None


def no_progress(*args, **kwargs):
    """A picklable callback for direct, non-Gradio use of the media pipeline."""
    pass


def process_video(video, window=3.0, hop=3.0, progress=no_progress, model_key="bsrnn", language="en"):
    window, hop = validate_settings(window, hop, language)
    path = Path(video) if video else SAMPLE
    if not path.is_file():
        raise ValueError(t("missing_video", language))
    try:
        checkpoint = model_checkpoint(model_key)
    except Exception as exc:
        raise ValueError(t("missing_weight", language, model=MODEL_LABELS[model_key])) from exc
    if not checkpoint.is_file():
        raise ValueError(t("missing_weight", language, model=MODEL_LABELS[model_key]))
    progress(0.01, desc=t("checking", language))
    job_id, provenance = job_key(path, window, hop, model_key)
    job = OUTPUTS / job_id
    saved = job / "manifest.json"
    manifest = cached_manifest(job)
    if manifest:
        return job_id, manifest, True
    started = time.perf_counter()
    progress(0.03, desc=t("preparing", language))
    media = prepare_media(path, job)
    media_type = media["media_type"]
    progress(0.12, desc=t("loading", language, model=MODEL_LABELS[model_key]))
    result = separate_audio(media["audio"], window, hop, str(checkpoint),
                            provenance["checkpoint"], model_key, language, progress)
    progress(0.87, desc=t("saving", language))
    export = write_outputs(job, result["stems"], result["sample_rate"], result["metadata"])
    original = "mixture.wav" if media_type == "audio" else "preview.mp4"
    sources = {"mix": {"label": "Original", "url": f"/media/{job_id}/{original}"}}
    for stem, label in (("speech", "Speech"), ("music", "Music"), ("sfx", "SFX")):
        sources[stem] = {"label": label, "url": f"/media/{job_id}/{stem}.wav"}
        if media_type == "video":
            mux_stem(media["video_path"], export["files"][stem], job / f"{stem}.mp4")
    spectrograms = {}
    if media_type == "audio":
        waveforms = {"mix": media["audio"], **{
            key: samples if export["gain"] == 1 else samples * export["gain"]
            for key, samples in result["stems"].items()}}
        spectrograms = write_spectrograms(job, waveforms, result["sample_rate"])
        for spec in spectrograms.values():
            spec["url"] = f"/media/{job_id}/{spec['filename']}"
    manifest = {
        "ready": True, "title": path.stem, "media_type": media_type,
        f"{media_type}_url": sources["mix"]["url"], "spectrograms": spectrograms,
        "duration": media["duration"], "sample_rate": result["sample_rate"],
        "window_seconds": window, "hop_seconds": hop, "sources": sources,
        "partitions": result["windows"], "coarse_boundaries_hz": result["coarse_boundaries_hz"],
        "meta": {**result["metadata"], "checkpoint": weights_source(model_key),
                 "model_key": model_key, "model_label": MODEL_LABELS[model_key],
                 "export_gain": export["gain"], "peak_before": export["peak_before"],
                 "processing_seconds": round(time.perf_counter() - started, 2),
                 "provenance": provenance},
    }
    # Write the manifest only after every export is complete: it is the cache commit.
    (job / "boundaries.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    progress(0.96, desc=t("packaging", language))
    with zipfile.ZipFile(job / "DBS-results.zip", "w", zipfile.ZIP_STORED) as bundle:
        for name in export_names(manifest):
            bundle.write(job / name, name)
    temporary = job / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    temporary.replace(saved)
    progress(1.0, desc=t("complete", language))
    return job_id, manifest, False


def run(video, window, hop, model_key="bsrnn", progress=gr.Progress(), language="en"):
    # Clear old results immediately so new settings never label stale output.
    yield '<div class="empty-preview">' + bi("Separating sound and extracting frequency bands…", "正在分离声音并提取动态频带…") + '</div>', bi("Processing media…", "正在处理媒体…"), gr.update(value="", visible=False)
    try:
        job_id, manifest, cached = process_video(video, window, hop, progress, model_key, language)
    except Exception as exc:
        LOG.exception("Inference failed")
        yield '<div class="empty-preview">' + bi("Processing stopped. Check your file and settings, then try again.", "处理未完成，请检查文件与推理设置后重试。") + '</div>', bi("Processing failed · See the error message", "处理失败 · 请查看错误提示"), gr.update(value="", visible=False)
        if ZERO_GPU and isinstance(exc, gr.Error):
            raise
        raise gr.Error(friendly_error(exc, language)) from exc
    elapsed = manifest["meta"]["processing_seconds"]
    status = bi(
        f"{MODEL_LABELS[model_key]} · {'Matching result loaded' if cached else 'Separation complete'} · {len(manifest['partitions'])} windows · {manifest['window_seconds']:g} s window / {manifest['hop_seconds']:g} s hop · {'Original runtime' if cached else 'Runtime'} {elapsed:.1f} s",
        f"{MODEL_LABELS[model_key]} · {'已载入相同参数的结果' if cached else '分离完成'} · {len(manifest['partitions'])} 个窗口 · {manifest['window_seconds']:g} 秒窗口 / {manifest['hop_seconds']:g} 秒间隔 · {'首次处理' if cached else '处理耗时'} {elapsed:.1f} 秒")
    yield frame(job_id), status, gr.update(value=download_panel(OUTPUTS / job_id), visible=True)


def select_model(model_key, video, window, hop):
    """Never reuse another model's visible result when the selection changes."""
    path = Path(video) if video else SAMPLE
    try:
        window, hop = validate_settings(window, hop)
        key, _ = job_key(path, window, hop, model_key)
        if cached_manifest(OUTPUTS / key):
            return frame(key), bi(f"{MODEL_LABELS[model_key]} · Matching result loaded", f"{MODEL_LABELS[model_key]} · 已载入匹配参数的结果"), gr.update(value=download_panel(OUTPUTS / key), visible=True)
    except (OSError, ValueError):
        pass
    label = html.escape(MODEL_LABELS[model_key])
    return '<div class="empty-preview">' + bi(f"{label} · Select Separate & visualize to generate results", f"{label} · 点击「分离并可视化」生成结果") + '</div>', bi(f"Current model: {label}", f"当前模型：{label}"), gr.update(value="", visible=False)


def preview_input(video, model_key, window, hop, language="en"):
    """Preview the current upload without running a model or exposing stale output."""
    yield '<div class="empty-preview">' + bi("Preparing your preview…", "正在准备预览…") + '</div>', bi("Reading media…", "正在读取媒体…"), gr.update(value="", visible=False)
    try:
        path = Path(video) if video else SAMPLE
        window, hop = validate_settings(window, hop, language)
        key, _ = job_key(path, window, hop, model_key)
        if cached_manifest(OUTPUTS / key):
            yield select_model(model_key, video, window, hop)
            return
        job_id = "preview-" + digest(path)[:24]
        job = OUTPUTS / job_id
        media = prepare_media(path, job)
        kind = media["media_type"]
        original = "mixture.wav" if kind == "audio" else "preview.mp4"
        url = f"/media/{job_id}/{original}"
        specs = write_spectrograms(job, {"mix": media["audio"]}, media["sample_rate"]) if kind == "audio" else {}
        for spec in specs.values():
            spec["url"] = f"/media/{job_id}/{spec['filename']}"
        manifest = {"ready": False, "title": path.stem, "media_type": kind,
                    f"{kind}_url": url, "duration": media["duration"],
                    "sample_rate": media["sample_rate"], "window_seconds": window, "hop_seconds": hop,
                    "sources": {"mix": {"label": "Original", "url": url}}, "spectrograms": specs,
                    "partitions": [], "coarse_boundaries_hz": [], "meta": {}}
        (job / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
        yield frame(job_id), bi("Preview ready · Select Separate & visualize to generate stems and frequency bands", "预览已就绪 · 点击「分离并可视化」生成音轨与频带"), gr.update(value="", visible=False)
    except Exception as exc:
        LOG.exception("Media preview failed")
        yield '<div class="empty-preview">' + bi("Unable to preview this file. Try another audio or video file.", "无法预览此文件，请选择其他音频或视频。") + '</div>', bi("Preview unavailable", "预览不可用"), gr.update(value="", visible=False)
        raise gr.Error(friendly_error(exc, language)) from exc


def initial_view():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        job_id = job_key(SAMPLE, 3.0, 3.0)[0] if SAMPLE.is_file() else None
    except Exception:  # weights not downloadable yet: start with the preview only
        job_id = None
    if job_id is not None:
        if cached_manifest(OUTPUTS / job_id):
            return frame(job_id), bi("Sample ready · 3 s window / 3 s hop · Play and switch stems", "示例已就绪 · 3 秒窗口 / 3 秒间隔 · 可直接播放、切换音轨"), download_panel(OUTPUTS / job_id)
    if SAMPLE.is_file():
        job = OUTPUTS / "preview"
        prepared = prepare_video(SAMPLE, job)
        manifest = {"ready": False, "title": SAMPLE.stem,
                    "video_url": "/media/preview/preview.mp4", "duration": prepared["duration"],
                    "sample_rate": 44100, "window_seconds": 3, "hop_seconds": 3,
                    "sources": {}, "partitions": [], "coarse_boundaries_hz": [], "meta": {}}
        (job / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
        return frame("preview"), bi("Sample loaded · Select Separate & visualize to generate stems and frequency bands", "示例已载入 · 点击「分离并可视化」生成音轨与频带"), ""
    return '<div class="empty-preview">' + bi("Upload audio or video and explore the structure of sound.", "上传音频或视频，探索声音的结构。") + '</div>', bi("Waiting for media", "等待媒体"), ""


def create_app():
    web = FastAPI(docs_url=None, redoc_url=None)

    def job_folder(job_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", job_id):
            raise HTTPException(404)
        return OUTPUTS / job_id

    @web.get("/studio/{job_id}", response_class=HTMLResponse)
    def studio(job_id: str):
        path = job_folder(job_id) / "manifest.json"
        if not path.is_file():
            raise HTTPException(404)
        data = json.loads(path.read_text())
        if data.get("ready"):
            meta = data.setdefault("meta", {})
            key = meta.get("model_key") or ("tiger" if "tiger" in str(meta.get("architecture", "")).lower() else "bsrnn")
            # The player only needs the display name, not training metadata.
            data["meta"] = {"model_key": key, "model_label": MODEL_LABELS.get(key, "Dynamic DBS")}
        payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
        content = (ROOT / "studio.html").read_text().replace("__DBS_MANIFEST__", payload)
        return HTMLResponse(content, headers={"Cache-Control": "no-cache"})

    @web.get("/media/{job_id}/{filename}")
    def media_file(job_id: str, filename: str):
        allowed = {"preview.mp4", "mixture.wav", *ALL_DOWNLOAD_NAMES}
        path = job_folder(job_id) / filename
        if filename not in allowed or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @web.get("/download/{job_id}/{filename}")
    def download_file(job_id: str, filename: str):
        path = job_folder(job_id) / filename
        if filename not in ALL_DOWNLOAD_NAMES or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, filename=filename, content_disposition_type="attachment")

    initial, status_text, initial_downloads = initial_view()
    translations = gr.I18n(**{locale: {key: pair[index] for key, pair in COPY.items()}
                              for locale, index in (("en", 0), ("zh-CN", 1), ("zh", 1))})
    with gr.Blocks(title="DBS · Sound in focus", analytics_enabled=False, fill_width=True) as demo:
        with gr.Row(elem_id="topbar"):
            runtime_label = bi("ZeroGPU", "ZeroGPU") if ZERO_GPU else bi("LOCAL INFERENCE", "本地推理")
            gr.HTML(f'<header class="masthead"><a class="wordmark" href="/" aria-label="DBS home"><span class="mark">▥</span> DBS <span class="wordmark-detail">{bi("LISTENING STUDIO", "声音工作室")}</span></a><span class="local-tag"><i></i>{runtime_label}</span></header>', elem_id="brandbar")
            language = gr.Radio(choices=[("English", "en"), ("中文", "zh")], value="en", label=translations("language"), show_label=False, container=False, scale=0, min_width=160, interactive=True, elem_id="language-switch")
        gr.HTML(f'<section class="hero"><div><div class="eyebrow">{bi("DYNAMIC BAND SPLITTING / DnR", "动态频带划分 / DnR")}</div><h1>{bi("Sound,", "听见，")} <em>{bi("in focus.", "每一层声音。")}</em></h1><p>{bi("Unfold the soundtrack. Explore speech, music, sound effects and the bands between them.", "将声音层层展开，听见人声、音乐与音效，观察模型如何划分频率。")}</p></div><div class="model-stamp"><b>BSRNN / TIGER</b><span>{bi("Dynamic · DnR", "动态频带 · DnR")}</span><span>{bi("3 stems / 44.1 kHz", "3 条音轨 / 44.1 kHz")}</span></div></section>')
        gr.HTML(research_panel(), elem_id="research-info")
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=1, min_width=280, elem_id="control-panel"):
                gr.HTML(f'<div class="section-title"><span>01</span> {bi("Choose a scene", "选择片段")}</div>')
                upload = gr.File(label=translations("upload"), file_types=MEDIA_EXTENSIONS, type="filepath", height=145, elem_id="video-upload")
                gr.Markdown(translations("sample_hint"), elem_classes="hint")
                sample_btn = gr.Button(translations("sample_button"), size="sm")
                gr.HTML(f'<div class="section-title setting-title"><span>02</span> {bi("Inference settings", "推理设置")}</div>')
                model = gr.Dropdown(choices=[(label, key) for key, label in MODEL_LABELS.items()], value="bsrnn", label=translations("model"), interactive=True)
                with gr.Row():
                    window = gr.Number(3.0, label=translations("window"), minimum=0.5, maximum=15, step=0.5, min_width=100)
                    hop = gr.Number(3.0, label=translations("hop"), minimum=0.1, maximum=15, step=0.5, min_width=100)
                gr.Markdown(translations("settings_hint"), elem_classes="hint")
                with gr.Row():
                    reset = gr.Button(translations("reset"), size="sm")
                    overlap = gr.Button(translations("overlap"), size="sm")
                go = gr.Button(translations("run"), variant="primary", elem_id="run-button")
                gr.Markdown(translations("usage"), elem_classes="hint usage-hint")
                with gr.Accordion(translations("about"), open=False):
                    gr.Markdown(translations("about_copy"))
            with gr.Column(scale=4, min_width=540, elem_id="viewer-panel"):
                viewer = gr.HTML(initial, elem_id="studio-host")
                status = gr.HTML(status_text, elem_id="status-line")
        files = gr.HTML(initial_downloads, elem_id="downloads-host", visible=bool(initial_downloads))
        gr.HTML(f'<footer class="studio-footer"><span>{bi("LEARNING WHERE TO SPLIT", "学习频率如何划分")}</span><span>{bi("DYNAMIC FREQUENCY PARTITIONS", "动态频带划分")}</span></footer>')
        reset.click(lambda: (3.0, 3.0), outputs=[window, hop], queue=False)
        overlap.click(lambda: (3.0, 1.5), outputs=[window, hop], queue=False)
        sample_btn.click(lambda: None, outputs=upload, queue=False)
        language.change(None, inputs=[language], outputs=None, js="async (language) => { await window.dbsSetLanguage(language); }", queue=False)
        model.change(select_model, inputs=[model, upload, window, hop], outputs=[viewer, status, files],
                     concurrency_id="media", concurrency_limit=1, trigger_mode="always_last", show_progress="minimal")
        upload.change(preview_input, inputs=[upload, model, window, hop, language], outputs=[viewer, status, files],
                      concurrency_id="media", concurrency_limit=1, trigger_mode="always_last", show_progress="minimal", api_name=False)

        def run_ui(video, window, hop, model, language, progress=gr.Progress()):
            yield from run(video, window, hop, model, progress=progress, language=language)

        go.click(run_ui, inputs=[upload, window, hop, model, language], outputs=[viewer, status, files], concurrency_id="media", concurrency_limit=1, api_name="separate", show_progress="minimal")
    demo.queue(max_size=4, default_concurrency_limit=1)
    theme = gr.themes.Base(primary_hue="emerald", neutral_hue="stone", font=["Avenir Next", "Helvetica Neue", "sans-serif"], font_mono=["Menlo", "monospace"])
    # The studio has a deliberate light palette, including when macOS uses
    # dark mode. Keep native Gradio labels and controls in the same palette.
    palette = theme.to_dict()["theme"]
    theme.set(**{key: palette[key[:-5]] for key in palette
                 if key.endswith("_dark") and key[:-5] in palette})
    # Resolve Gradio's exported locale API from the installed, pinned frontend.
    assets = Path(gr.__file__).parent / "templates" / "frontend" / "assets"
    locale_module = next((p for p in assets.glob("core-*.js") if "as changeLocale" in p.read_text()), None)
    if locale_module is None:
        raise RuntimeError("The installed Gradio frontend does not expose changeLocale.")
    language_js = (ROOT / "language.js").read_text().replace("__GRADIO_LOCALE_MODULE__", f"/assets/{locale_module.name}")
    return gr.mount_gradio_app(web, demo, path="/", allowed_paths=[str(OUTPUTS)],
                              max_file_size="500mb",
                              theme=theme, css=(ROOT / "style.css").read_text(), footer_links=[], i18n=translations, js=language_js,
                              show_error=True, ssr_mode=False, head='''<script>
window.addEventListener('message', event => {
  if (event.origin !== window.location.origin || event.data?.type !== 'dbs-studio-height') return;
  const height = Number(event.data.height);
  if (!Number.isFinite(height)) return;
  document.querySelectorAll('iframe.studio-frame').forEach(frame => {
    if (frame.contentWindow === event.source) frame.style.height = Math.ceil(Math.max(360, Math.min(2000, height))) + 'px';
  });
});
</script>''')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    parser.add_argument("--bsrnn", help="DBS-BSRNN weights: Hub repo id or save_pretrained directory")
    parser.add_argument("--tiger", help="DBS-TIGER weights: Hub repo id or save_pretrained directory")
    parser.add_argument("--sample-video", type=Path, default=SAMPLE)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEVICE)
    args = parser.parse_args()
    for key in ("bsrnn", "tiger"):
        if getattr(args, key):
            os.environ[f"DBS_{key.upper()}_WEIGHTS"] = getattr(args, key)
    SAMPLE, DEVICE = args.sample_video.resolve(), args.device
    logging.basicConfig(level=logging.INFO)
    initialize_gpu_runtime()
    uvicorn.run(create_app(), host=args.host, port=args.port)
