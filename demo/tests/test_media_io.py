"""Small real-FFmpeg checks for timeline preservation and audio export."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

import media_io


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg and FFprobe are required for media integration checks",
)


def ffmpeg(*arguments):
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *map(str, arguments)],
        check=True, stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
    )


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source with spaces.mkv"
    # A delayed, shorter 48 kHz stereo soundtrack must retain both silence
    # regions when converted to the model's 44.1 kHz mono timeline.
    ffmpeg(
        "-f", "lavfi", "-i", "color=c=navy:s=80x48:r=10:d=1.2",
        "-itsoffset", "0.3", "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=48000:duration=0.6",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", "-ac", "2", path,
    )
    return path


def test_prepare_resamples_preserves_offsets_and_input(source, tmp_path):
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    source_info = media_io.probe_video(source)
    assert source_info["audio_sample_rate"] == 48000
    assert source_info["audio_channels"] == 2
    assert source_info["audio_start_time"] == pytest.approx(0.3, abs=0.001)
    result = media_io.prepare_video(source, tmp_path / "job")
    assert result["sample_rate"] == 44100
    assert result["duration"] == pytest.approx(1.2)
    assert result["audio"].shape == (52920,)
    assert result["audio"].dtype == np.float32
    assert np.max(np.abs(result["audio"][:11025])) < 1e-5
    assert np.max(np.abs(result["audio"][17640:22050])) > 0.02
    assert np.max(np.abs(result["audio"][44100:])) < 1e-5
    saved, sample_rate = sf.read(result["mixture_path"], dtype="float32")
    np.testing.assert_array_equal(saved, result["audio"])
    assert sample_rate == 44100
    preview = media_io.probe_video(result["video_path"])
    assert preview["video_codec"] == "h264"
    assert preview["audio_codec"] == "aac"
    assert preview["duration"] == pytest.approx(1.2, abs=0.025)
    assert preview["video_start_time"] == pytest.approx(0.0)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert not list((tmp_path / "job").glob(".*"))


def test_missing_tracks_and_missing_file_are_clear(source, tmp_path):
    silent = tmp_path / "silent.mp4"
    ffmpeg("-i", source, "-map", "0:v:0", "-an", "-c:v", "libx264", silent)
    original = silent.read_bytes()
    with pytest.raises(media_io.MediaError, match="没有音轨"):
        media_io.prepare_video(silent, tmp_path / "bad-job")
    assert silent.read_bytes() == original
    assert not (tmp_path / "bad-job").exists()
    audio_only = tmp_path / "audio.wav"
    sf.write(audio_only, np.zeros(1000, dtype=np.float32), 44100)
    with pytest.raises(media_io.MediaError, match="没有可用的视频轨道"):
        media_io.probe_video(audio_only)
    with pytest.raises(media_io.MediaError, match="不存在"):
        media_io.probe_video(tmp_path / "absent.mp4")


def test_ten_minute_limit_rejects_without_decoding(source, monkeypatch):
    response = {
        "streams": [
            {"index": 0, "codec_type": "video", "width": 80, "height": 48, "duration": "601"},
            {"index": 1, "codec_type": "audio", "sample_rate": "44100", "channels": 1},
        ],
    }
    monkeypatch.setattr(media_io, "_run", lambda *args, **kwargs:
                        subprocess.CompletedProcess([], 0, json.dumps(response).encode(), b""))
    with pytest.raises(media_io.MediaError, match="最多支持 600 秒"):
        media_io.probe_video(source)


def test_shared_gain_preserves_relative_loudness_and_metadata(tmp_path):
    stems = {"speech": np.array([0, 2, -1], dtype=np.float32),
             "music": np.array([0, 0.5, -0.25], dtype=np.float32)}
    result = media_io.write_outputs(tmp_path, stems, 44100, {"windows": np.array([0, 3])})
    assert result["gain"] == pytest.approx(0.495)
    assert result["normalized"]
    speech, _ = sf.read(result["files"]["speech"], dtype="float32")
    music, _ = sf.read(result["files"]["music"], dtype="float32")
    assert np.max(np.abs(speech)) == pytest.approx(0.99)
    np.testing.assert_allclose(speech, music * 4)
    manifest = json.loads(result["metadata_path"].read_text())
    assert manifest["windows"] == [0, 3]
    assert manifest["audio_export"]["gain"] == pytest.approx(0.495)
    quiet = media_io.write_outputs(tmp_path / "quiet", {"sfx": stems["music"]}, 44100, {})
    assert quiet["gain"] == 1.0


@pytest.mark.parametrize("stems", [
    {"../escape": np.zeros(4)},
    {"speech": np.array([np.nan])},
    {"speech": np.zeros(2), "music": np.zeros(3)},
])
def test_invalid_output_never_writes(tmp_path, stems):
    with pytest.raises(media_io.MediaError):
        media_io.write_outputs(tmp_path / "invalid", stems, 44100, {})
    assert not (tmp_path / "invalid").exists()


def test_mux_preserves_video_and_rejects_overwrite(source, tmp_path):
    prepared = media_io.prepare_video(source, tmp_path / "job")
    stems = media_io.write_outputs(
        tmp_path / "job", {"speech": prepared["audio"] * 0.5}, 44100, {},
    )
    before = prepared["video_path"].read_bytes()
    exported = media_io.mux_stem(
        prepared["video_path"], stems["files"]["speech"], tmp_path / "speech.mp4",
    )
    info = media_io.probe_video(exported)
    assert info["video_codec"] == "h264"
    assert info["duration"] == pytest.approx(1.2, abs=0.025)
    assert prepared["video_path"].read_bytes() == before
    with pytest.raises(media_io.MediaError, match="不能覆盖"):
        media_io.mux_stem(prepared["video_path"], stems["files"]["speech"], prepared["video_path"])


def test_prepare_media_dispatches_real_video(source, tmp_path):
    result = media_io.prepare_media(source, tmp_path / "video-job")
    assert result["media_type"] == "video"
    assert result["video_path"].is_file()
    assert result["duration"] == pytest.approx(1.2)
    assert np.max(np.abs(result["audio"][:11025])) < 1e-5


def test_audio_stereo_resampling_preserves_duration_and_short_tail(tmp_path):
    source = tmp_path / "stereo 48k.wav"
    sample_count = 48000 + 37
    phase = np.arange(sample_count) / 48000
    wave = 0.2 * np.sin(2 * np.pi * 440 * phase)
    sf.write(source, np.column_stack([wave, wave]), 48000, subtype="FLOAT")
    before = source.read_bytes()
    result = media_io.prepare_media(source, tmp_path / "audio-job")
    assert result["media_type"] == "audio"
    assert "video_path" not in result
    assert result["sample_rate"] == 44100
    assert result["audio"].ndim == 1
    assert result["audio"].dtype == np.float32
    assert result["duration"] == pytest.approx(sample_count / 48000, abs=1 / 44100)
    assert abs(len(result["audio"]) - round(sample_count * 44100 / 48000)) <= 1
    assert result["probe"]["audio_channels"] == 2
    assert np.max(np.abs(result["audio"][-30:])) > 0.1
    saved, rate = sf.read(result["mixture_path"], dtype="float32")
    np.testing.assert_array_equal(saved, result["audio"])
    assert rate == 44100
    assert sorted(path.name for path in (tmp_path / "audio-job").iterdir()) == ["mixture.wav"]
    assert source.read_bytes() == before


@pytest.mark.parametrize("suffix,codec", [
    ("mp3", "libmp3lame"), ("m4a", "aac"), ("ogg", "libvorbis"),
    ("flac", "flac"), ("opus", "libopus"), ("aiff", "pcm_s16be"),
    ("aac", "aac"),
])
def test_audio_formats_decode_without_video(tmp_path, suffix, codec):
    source = tmp_path / f"input.{suffix}"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=550:sample_rate=48000:duration=0.637",
           "-c:a", codec, source)
    result = media_io.prepare_media(source, tmp_path / "job")
    assert result["media_type"] == "audio"
    # Raw AAC cannot signal gapless priming/padding, unlike MP3 or Opus.
    assert result["duration"] == pytest.approx(0.637, abs=0.05)
    assert np.isfinite(result["audio"]).all()
    assert np.max(np.abs(result["audio"])) > 0.05
    assert not (tmp_path / "job" / "preview.mp4").exists()


def test_mp3_with_attached_cover_is_audio_and_has_no_encoder_delay(tmp_path):
    cover = tmp_path / "cover.png"
    ffmpeg("-f", "lavfi", "-i", "color=c=teal:s=32x32", "-frames:v", "1", cover)
    source = tmp_path / "album.mp3"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=0.7",
           "-i", cover, "-map", "0:a", "-map", "1:v", "-c:a", "libmp3lame",
           "-c:v", "copy", "-disposition:v:0", "attached_pic", source)
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(source),
    ]))
    assert any(stream.get("disposition", {}).get("attached_pic") for stream in data["streams"])
    result = media_io.prepare_media(source, tmp_path / "job")
    assert result["media_type"] == "audio"
    assert len(result["audio"]) == round(0.7 * 44100)
    assert np.max(np.abs(result["audio"][:441])) > 0.05
    assert not (tmp_path / "job" / "preview.mp4").exists()


@pytest.mark.parametrize("offset", [2.5, 700])
def test_audio_timestamp_offset_is_normalized_without_losing_samples(tmp_path, offset):
    source = tmp_path / "offset.mka"
    ffmpeg("-itsoffset", str(offset), "-f", "lavfi", "-i",
           "sine=frequency=440:sample_rate=44100:duration=0.7", "-c:a", "pcm_f32le", source)
    result = media_io.prepare_media(source, tmp_path / "job")
    assert result["probe"]["audio_start_time"] == pytest.approx(offset)
    assert result["duration"] == pytest.approx(0.7, abs=0.001)
    assert np.max(np.abs(result["audio"][:441])) > 0.05


def test_audio_failures_are_clear_and_do_not_publish_outputs(tmp_path):
    invalid = tmp_path / "broken.wav"
    invalid.write_bytes(b"not an audio file")
    with pytest.raises(media_io.MediaError, match="读取媒体信息失败"):
        media_io.prepare_media(invalid, tmp_path / "invalid-job")
    with pytest.raises(media_io.MediaError, match="不存在"):
        media_io.prepare_media(tmp_path / "absent.wav", tmp_path / "missing-job")
    with pytest.raises(media_io.MediaError, match="请先选择"):
        media_io.prepare_media("", tmp_path / "empty-job")
    picture = tmp_path / "picture.png"
    ffmpeg("-f", "lavfi", "-i", "color=c=teal:s=32x32", "-frames:v", "1", picture)
    with pytest.raises(media_io.MediaError, match="没有音轨"):
        media_io.prepare_media(picture, tmp_path / "picture-job")
    nonfinite = tmp_path / "nonfinite.wav"
    sf.write(nonfinite, np.array([0, np.nan, np.inf, 0], dtype=np.float32), 44100, subtype="FLOAT")
    with pytest.raises(media_io.MediaError, match="无效采样值"):
        media_io.prepare_media(nonfinite, tmp_path / "nan-job")
    assert not list(tmp_path.glob("*-job/mixture.wav"))
    assert not list(tmp_path.glob("*-job/.*"))


def test_audio_over_ten_minutes_is_rejected_before_decoding(tmp_path):
    source = tmp_path / "too-long.wav"
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "600.1",
           "-c:a", "pcm_u8", source)
    with pytest.raises(media_io.MediaError, match="最多支持 600 秒"):
        media_io.prepare_media(source, tmp_path / "job")
    assert not (tmp_path / "job").exists()


def test_audio_cannot_overwrite_source(tmp_path):
    source = tmp_path / "mixture.wav"
    sf.write(source, np.zeros(257, dtype=np.float32), 44100, subtype="FLOAT")
    before = source.read_bytes()
    with pytest.raises(media_io.MediaError, match="路径相同"):
        media_io.prepare_media(source, tmp_path)
    assert source.read_bytes() == before
