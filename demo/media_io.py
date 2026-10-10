"""Validated, timeline-aligned media I/O for the local DBS demo.

FFmpeg and FFprobe are external executables; no Python video package is needed.
All generated media live in the caller's job directory. Source files are never
modified. Video audio uses the first video timestamp as time zero; audio-only
inputs use the first decoded audio timestamp as time zero.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from fractions import Fraction
import re
import shutil
import subprocess
import uuid
from typing import Any, Mapping

import numpy as np
import soundfile as sf


SAMPLE_RATE = 44_100
MAX_VIDEO_SECONDS = 600.0


class MediaError(ValueError):
    """A media file cannot be processed safely or decoded successfully."""


def _executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise MediaError(f"找不到 {name}。请先安装 FFmpeg，并确保 ffmpeg 和 ffprobe 在 PATH 中。")
    return executable


def _input_file(path: str | os.PathLike[str]) -> Path:
    if not path:
        raise MediaError("请先选择一个包含音轨的视频文件。")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MediaError(f"视频文件不存在：{source}")
    return source


def _run(command: list[str], *, timeout: float, action: str) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{action}超时（{timeout:.0f} 秒）。请尝试较短的视频。") from exc
    except OSError as exc:
        raise MediaError(f"无法执行{action}：{exc}") from exc
    if result.returncode:
        details = result.stderr.decode("utf-8", errors="replace").strip()[-2200:]
        raise MediaError(f"{action}失败：{details or 'FFmpeg 未返回详细错误。'}")
    return result


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _probe_data(source: Path, *, action: str) -> dict[str, Any]:
    result = _run([
        _executable("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe",
        "-show_streams", "-show_format", "-of", "json", str(source),
    ], timeout=30, action=action)
    try:
        data = json.loads(result.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise MediaError("FFprobe 返回了无法解析的媒体信息。") from exc
    if not isinstance(data, dict) or not isinstance(data.get("streams", []), list):
        raise MediaError("FFprobe 返回了无法解析的媒体信息。")
    return data


def probe_video(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return first video/audio stream metadata; reject unusable input.

    Duration is the video stream duration when available, otherwise the format
    duration. Validation rejects missing audio, invalid dimensions, and clips
    longer than ten minutes instead of silently truncating the upload.
    """
    source = _input_file(path)
    data = _probe_data(source, action="读取视频信息")
    streams = data.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"
              and not stream.get("disposition", {}).get("attached_pic")]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not videos:
        raise MediaError("这个文件没有可用的视频轨道。请选择视频文件。")
    if not audios:
        raise MediaError("这个视频没有音轨，无法运行音源分离。")
    video, audio = videos[0], audios[0]
    duration = _finite_number(video.get("duration"))
    if duration is None:
        duration = _finite_number(data.get("format", {}).get("duration"))
    if duration is None or duration <= 0:
        raise MediaError("无法确定视频时长，或视频长度为零。")
    if duration > MAX_VIDEO_SECONDS + 1e-3:
        raise MediaError(f"视频长 {duration:.1f} 秒；本地演示最多支持 {MAX_VIDEO_SECONDS:.0f} 秒。请先选择较短片段。")
    width, height = int(video.get("width", 0)), int(video.get("height", 0))
    if width <= 0 or height <= 0:
        raise MediaError("视频尺寸无效，无法生成浏览器预览。")
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "video_codec": video.get("codec_name", "unknown"),
        "audio_codec": audio.get("codec_name", "unknown"),
        "video_stream_index": int(video["index"]),
        "audio_stream_index": int(audio["index"]),
        "video_start_time": _finite_number(video.get("start_time")) or 0.0,
        "audio_start_time": _finite_number(audio.get("start_time")) or 0.0,
        "audio_sample_rate": int(audio.get("sample_rate", 0)),
        "audio_channels": int(audio.get("channels", 0)),
        "frame_rate": video.get("avg_frame_rate", "0/0"),
    }


def prepare_video(
    input_path: str | os.PathLike[str], job_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Decode mono 44.1 kHz audio and create an H.264 browser preview.

    Audio preceding the video is trimmed; a delayed or short audio stream is
    padded with silence to retain its position and the full video duration.
    The mixture WAV retains floating-point samples. ``audio`` is mono float32.
    """
    source = _input_file(input_path)
    info = probe_video(source)
    destination = Path(job_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    mixture_path, preview_path = destination / "mixture.wav", destination / "preview.mp4"
    if source in (mixture_path, preview_path):
        raise MediaError("输入视频与生成文件路径相同。请使用新的任务目录。")
    token = uuid.uuid4().hex
    temporary_wav = destination / f".mixture-{token}.wav"
    temporary_mp4 = destination / f".preview-{token}.mp4"
    duration = info["duration"]
    try:
        frame_rate = Fraction(info["frame_rate"])
        if not 1 <= frame_rate <= 240:
            frame_rate = Fraction(30)
    except (ValueError, ZeroDivisionError):
        frame_rate = Fraction(30)
    # -copyts keeps original stream offsets; the resampler pads/trims relative
    # to the first video timestamp and corrects discontinuities on that clock.
    audio_filter = (
        f"asetpts=PTS-({info['video_start_time']:.9f})/TB,"
        f"aresample={SAMPLE_RATE}:async=1:first_pts=0,"
        f"apad=whole_dur={duration:.9f},atrim=duration={duration:.9f}"
    )
    ffmpeg = _executable("ffmpeg")
    common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    try:
        _run(common + [
            "-copyts", "-protocol_whitelist", "file,pipe", "-i", str(source),
            "-map", f"0:{info['audio_stream_index']}", "-vn", "-sn", "-dn",
            "-af", audio_filter, "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-c:a", "pcm_f32le", "-t", f"{duration:.9f}", str(temporary_wav),
        ], timeout=max(120, duration * 2), action="解码音频")
        audio, sample_rate = sf.read(temporary_wav, dtype="float32", always_2d=False)
        if audio.ndim != 1 or audio.size == 0 or sample_rate != SAMPLE_RATE:
            raise MediaError("音轨解码结果无效；需要非空的 44.1 kHz 单声道音频。")
        if not np.isfinite(audio).all():
            raise MediaError("音轨含有无效采样值，无法运行推理。")
        expected_samples = round(duration * SAMPLE_RATE)
        # Different container time bases can leave a one-sample rounding error.
        if audio.size < expected_samples:
            audio = np.pad(audio, (0, expected_samples - audio.size))
        elif audio.size > expected_samples:
            audio = audio[:expected_samples]
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        sf.write(temporary_wav, audio, SAMPLE_RATE, subtype="FLOAT")
        _run(common + [
            "-protocol_whitelist", "file,pipe", "-i", str(source),
            "-i", str(temporary_wav), "-map", f"0:{info['video_stream_index']}",
            "-map", "1:a:0", "-vf", "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(frame_rate),
            "-c:a", "aac", "-b:a", "192k",
            "-ar", str(SAMPLE_RATE), "-movflags", "+faststart",
            "-t", f"{duration:.9f}", str(temporary_mp4),
        ], timeout=max(180, duration * 4), action="生成 H.264 视频预览")
        temporary_wav.replace(mixture_path)
        temporary_mp4.replace(preview_path)
    finally:
        temporary_wav.unlink(missing_ok=True)
        temporary_mp4.unlink(missing_ok=True)
    return {
        "video_path": preview_path,
        "mixture_path": mixture_path,
        "audio": audio,
        "sample_rate": SAMPLE_RATE,
        "duration": len(audio) / SAMPLE_RATE,
        "title": source.stem,
        "source_path": source,
        "probe": info,
    }


def prepare_media(
    input_path: str | os.PathLike[str], job_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Prepare video or audio according to its streams, regardless of suffix.

    A video with a soundtrack follows :func:`prepare_video` unchanged. Cover
    art, such as an MP3 ``attached_pic`` stream, is not a video. Audio-only
    sources are decoded to a floating-point mono 44.1 kHz mixture WAV without
    creating a video. Their timeline starts at their first decoded sample;
    FFmpeg removes codec priming/padding, and the actual decoded sample count
    determines duration so short tails are neither dropped nor padded to the
    often approximate compressed-container duration.
    """
    if not input_path:
        raise MediaError("请先选择一个音频或包含音轨的视频文件。")
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise MediaError(f"媒体文件不存在：{source}")
    data = _probe_data(source, action="读取媒体信息")
    streams = data.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"
              and not stream.get("disposition", {}).get("attached_pic")]
    if videos:
        return {**prepare_video(source, job_dir), "media_type": "video"}
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not audios:
        raise MediaError("这个文件没有可用的音轨，无法运行音源分离。")
    stream = audios[0]
    duration = _finite_number(stream.get("duration"))
    stream_duration_known = duration is not None
    start_time = _finite_number(stream.get("start_time")) or 0.0
    if duration is None:
        duration = _finite_number(data.get("format", {}).get("duration"))
    if duration is None or duration <= 0:
        raise MediaError("无法确定音频时长，或音频长度为零。")
    # Some audio-only containers (e.g. Matroska) report the final timestamp as
    # format duration, including a positive starting offset. Permit that case;
    # decoded sample count below remains the authoritative length check.
    duration_hint = duration if stream_duration_known else duration - max(0.0, start_time)
    if duration_hint > MAX_VIDEO_SECONDS + 1e-3:
        raise MediaError(f"音频长 {duration_hint:.1f} 秒；本地演示最多支持 {MAX_VIDEO_SECONDS:.0f} 秒。请先选择较短片段。")
    info = {
        "duration": duration,
        "audio_codec": stream.get("codec_name", "unknown"),
        "audio_stream_index": int(stream["index"]),
        "audio_start_time": start_time,
        "audio_sample_rate": int(stream.get("sample_rate", 0)),
        "audio_channels": int(stream.get("channels", 0)),
    }
    destination = Path(job_dir).expanduser().resolve()
    mixture_path = destination / "mixture.wav"
    if source == mixture_path:
        raise MediaError("输入音频与生成文件路径相同。请使用新的任务目录。")
    destination.mkdir(parents=True, exist_ok=True)
    temporary_wav = destination / f".mixture-{uuid.uuid4().hex}.wav"
    try:
        _run([
            _executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-copyts", "-protocol_whitelist", "file,pipe", "-i", str(source),
            "-map", f"0:{info['audio_stream_index']}", "-vn", "-sn", "-dn",
            "-af", f"asetpts=PTS-STARTPTS,aresample={SAMPLE_RATE}:async=1:first_pts=0",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_f32le",
            # Bound decoding even if a corrupt container underreports duration.
            # The extra samples make an overlong decoded stream fail below.
            "-t", str(MAX_VIDEO_SECONDS + 0.01), str(temporary_wav),
        ], timeout=max(120, min(duration, MAX_VIDEO_SECONDS) * 2), action="解码音频")
        audio, sample_rate = sf.read(temporary_wav, dtype="float32", always_2d=False)
        if audio.ndim != 1 or audio.size == 0 or sample_rate != SAMPLE_RATE:
            raise MediaError("音轨解码结果无效；需要非空的 44.1 kHz 单声道音频。")
        if not np.isfinite(audio).all():
            raise MediaError("音轨含有无效采样值，无法运行推理。")
        if audio.size > round((MAX_VIDEO_SECONDS + 1e-3) * SAMPLE_RATE):
            raise MediaError(f"解码后音频超过 {MAX_VIDEO_SECONDS:.0f} 秒；本地演示最多支持 {MAX_VIDEO_SECONDS:.0f} 秒。请先选择较短片段。")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        temporary_wav.replace(mixture_path)
    finally:
        temporary_wav.unlink(missing_ok=True)
    return {
        "mixture_path": mixture_path,
        "audio": audio,
        "sample_rate": SAMPLE_RATE,
        "duration": len(audio) / SAMPLE_RATE,
        "media_type": "audio",
        "title": source.stem,
        "source_path": source,
        "probe": info,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def write_outputs(
    job_dir: str | os.PathLike[str], stems: Mapping[str, np.ndarray],
    sample_rate: int, metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Save stems with one shared attenuation gain and a JSON manifest.

    The peak ceiling is 0.99. Quiet signals are not amplified, and one shared
    gain preserves the relative loudness of all separated sources.
    """
    if not stems:
        raise MediaError("模型没有返回可导出的音轨。")
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise MediaError("导出采样率必须为正整数。")
    prepared: dict[str, np.ndarray] = {}
    sample_count: int | None = None
    for name, values in stems.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
                or name.lower() in {"mixture", "preview", "results"}):
            raise MediaError(f"无效的音轨名称：{name!r}")
        array = np.asarray(values, dtype=np.float32)
        if array.ndim not in (1, 2) or array.shape[0] == 0:
            raise MediaError(f"音轨 {name} 必须是非空的音频采样数组。")
        if array.ndim == 2 and not 1 <= array.shape[1] <= 8:
            raise MediaError(f"音轨 {name} 的通道数无效；数组格式应为 [采样点, 通道]。")
        if not np.isfinite(array).all():
            raise MediaError(f"音轨 {name} 含有 NaN 或无穷值。")
        if sample_count is not None and array.shape[0] != sample_count:
            raise MediaError("分离音轨长度不一致，无法保持同步。")
        sample_count = array.shape[0]
        prepared[name] = array
    peak = max(float(np.max(np.abs(array))) for array in prepared.values())
    gain = min(1.0, 0.99 / peak) if peak else 1.0
    destination = Path(job_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for name, array in prepared.items():
        path = destination / f"{name}.wav"
        temporary = destination / f".{name}-{uuid.uuid4().hex}.wav"
        try:
            sf.write(temporary, array * gain, int(sample_rate), subtype="FLOAT")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        files[name] = path
    export = {
        "sample_rate": int(sample_rate), "samples": sample_count,
        "gain": gain, "peak_before": peak, "normalized": gain < 1.0,
        "gain_note": "所有分离音轨使用同一个衰减增益，保持相对响度；峰值上限 0.99。",
        "files": {name: path.name for name, path in files.items()},
    }
    manifest = {**dict(metadata), "audio_export": export}
    json_text = json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2, allow_nan=False)
    manifest_path = destination / "results.json"
    temporary_json = destination / f".results-{uuid.uuid4().hex}.json"
    try:
        temporary_json.write_text(json_text + "\n", encoding="utf-8")
        temporary_json.replace(manifest_path)
    finally:
        temporary_json.unlink(missing_ok=True)
    return {"files": files, "gain": gain, "peak_before": peak,
            "normalized": gain < 1.0, "metadata_path": manifest_path}


def mux_stem(
    video_path: str | os.PathLike[str], wav_path: str | os.PathLike[str],
    dest: str | os.PathLike[str],
) -> Path:
    """Replace preview audio with a separated stem without re-encoding video."""
    video, audio = _input_file(video_path), _input_file(wav_path)
    destination = Path(dest).expanduser().resolve()
    if destination in (video, audio):
        raise MediaError("导出路径不能覆盖输入视频或音轨。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".mux-{uuid.uuid4().hex}.mp4"
    try:
        _run([
            _executable("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-protocol_whitelist", "file,pipe", "-i", str(video),
            "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest", str(temporary),
        ], timeout=180, action="导出分离音轨视频")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
