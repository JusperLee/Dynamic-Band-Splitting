"""Real audio I/O through the app, substituting only neural model inference."""

import io
import json
from pathlib import Path
import re
import shutil
import zipfile

import numpy as np
from PIL import Image
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Real audio integration checks require FFmpeg and FFprobe",
)

STEMS = ("speech", "music", "sfx")
TRACKS = ("mix", *STEMS)
EXPORTS = {
    *(f"{key}.wav" for key in STEMS),
    *(f"{key}-spectrogram.png" for key in TRACKS),
    "boundaries.json", "results.json",
}


@pytest.fixture
def audio_case(tmp_path, monkeypatch):
    source = tmp_path / "Two tones 音频.wav"
    count = 35291  # A non-round duration exercises the final partial window.
    times = np.arange(count) / 48000
    frequency = np.where(times < times[-1] / 2, 880, 4400)
    audio = (0.9 * np.sin(2 * np.pi * frequency * times)).astype(np.float32)
    sf.write(source, np.column_stack([audio, audio]), 48000, subtype="FLOAT")
    weights = [tmp_path / name for name in ("bsrnn.ckpt", "tiger.ckpt")]
    for weight in weights:
        weight.write_bytes(weight.name.encode())
    monkeypatch.setattr(app, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setitem(app.WEIGHTS, "bsrnn", weights[0])
    monkeypatch.setitem(app.WEIGHTS, "tiger", weights[1])
    monkeypatch.setattr(app, "SAMPLE", tmp_path / "no-default-video.mp4")
    calls = []

    class FakeBackend:
        def __init__(self, model):
            self.model = model

        def separate(self, samples, *, window_seconds, hop_seconds, progress):
            calls.append({"model": self.model, "audio": samples.copy(),
                          "window": window_seconds, "hop": hop_seconds})
            sample_rate = 44100
            windows = []
            for index, start in enumerate(range(0, len(samples), round(hop_seconds * sample_rate))):
                end = min(len(samples), start + round(window_seconds * sample_rate))
                windows.append({
                    "index": index, "start": start / sample_rate,
                    "end": end / sample_rate, "nominal_end": start / sample_rate + window_seconds,
                    "valid_samples": end - start,
                    "padding_samples": round(window_seconds * sample_rate) - (end - start),
                    "boundaries_hz": [[100, 500], [1500], [5000], [15000]],
                    "boundary_bins": [[4, 23], [69], [232], [696]],
                    "token_counts": [2, 1, 1, 1],
                })
            progress(1, f"Separating window {len(windows)}/{len(windows)}")
            return {
                # Exercise shared export attenuation as well as silent PNGs.
                "stems": {"speech": samples * 2, "music": samples * 0.25,
                          "sfx": np.zeros_like(samples)},
                "sample_rate": sample_rate, "windows": windows,
                "coarse_boundaries_hz": [0, 1378.125, 2756.25, 12403.125, 22050],
                "metadata": {"architecture": self.model, "window_seconds": window_seconds,
                             "hop_seconds": hop_seconds, "test_backend": True},
            }

    monkeypatch.setattr(app, "get_backend", lambda checkpoint, checksum, model:
                        FakeBackend(model))
    return {"source": source, "count": count, "calls": calls}


@pytest.fixture
def audio_result(audio_case):
    key, manifest, cached = app.process_video(audio_case["source"], 0.5, 0.25)
    assert not cached
    return app.OUTPUTS / key, manifest


def _links(panel):
    value = panel.get("value", "") if isinstance(panel, dict) else panel
    return re.findall(r'href="(/download/[^\"]+)"', value)


def _assert_cleared(update):
    assert "/studio/" not in update[0]
    assert not _links(update[2])
    assert update[2]["visible"] is False


def test_audio_pipeline_preserves_samples_and_exports_no_fake_video(audio_case, audio_result):
    job, manifest = audio_result
    assert len(audio_case["calls"]) == 1
    call = audio_case["calls"][0]
    assert (call["window"], call["hop"]) == (0.5, 0.25)
    assert manifest["media_type"] == "audio"
    assert manifest["sample_rate"] == 44100
    assert manifest["duration"] == pytest.approx(audio_case["count"] / 48000, abs=1 / 44100)
    assert manifest["partitions"][-1]["end"] == manifest["duration"]
    assert manifest["partitions"][-1]["padding_samples"] > 0
    assert manifest["audio_url"] == f"/media/{job.name}/mixture.wav"
    assert "video_url" not in manifest
    assert not list(job.glob("*.mp4"))
    expected_samples = len(call["audio"])
    assert round(manifest["duration"] * 44100) == expected_samples
    gain = manifest["meta"]["export_gain"]
    assert 0 < gain < 1
    for key, factor in (("speech", 2), ("music", 0.25), ("sfx", 0)):
        samples, rate = sf.read(job / f"{key}.wav", dtype="float32")
        assert samples.shape == (expected_samples,)
        assert rate == 44100
        np.testing.assert_allclose(samples, call["audio"] * factor * gain, atol=1e-7)
    assert {Path(path).name for path in app.downloads(job)} == EXPORTS | {"DBS-results.zip"}
    links = _links(app.download_panel(job))
    assert len(links) == 10
    assert set(links) == {f"/download/{job.name}/{name}" for name in EXPORTS | {"DBS-results.zip"}}
    with zipfile.ZipFile(job / "DBS-results.zip") as bundle:
        assert set(bundle.namelist()) == EXPORTS
        assert len(bundle.namelist()) == 9
        assert bundle.testzip() is None
        for name in EXPORTS:
            assert bundle.read(name) == (job / name).read_bytes()
        boundaries = json.loads(bundle.read("boundaries.json"))
        assert boundaries == manifest
        assert json.loads(bundle.read("results.json"))["audio_export"]["gain"] == gain


def test_spectrogram_pngs_show_actual_frequency_and_share_intensity_reference(audio_result):
    job, manifest = audio_result
    specs = manifest["spectrograms"]
    assert set(specs) == set(TRACKS)
    assert len({spec["reference_amplitude"] for spec in specs.values()}) == 1
    assert next(iter(specs.values()))["reference_amplitude"] > 0
    pixels = {}
    for key in TRACKS:
        spec = specs[key]
        assert spec["duration"] == manifest["duration"]
        assert (spec["min_hz"], spec["max_hz"]) == (0, 22050)
        assert (spec["db_min"], spec["db_max"]) == (-80, 0)
        assert spec["url"] == f"/media/{job.name}/{key}-spectrogram.png"
        with Image.open(job / spec["filename"]) as picture:
            assert picture.format == "PNG"
            assert picture.size == (spec["width"], spec["height"])
            pixels[key] = np.asarray(picture.convert("RGB"))
    mix = pixels["mix"]
    height, width, _ = mix.shape
    # Low-to-high tones must move upward in the image, matching the frequency
    # axis instead of being decorative noise or a recycled image.
    first_x, second_x = width // 4, 3 * width // 4
    first_peak = int(mix[:, first_x].mean(axis=1).argmax())
    second_peak = int(mix[:, second_x].mean(axis=1).argmax())
    assert first_peak == pytest.approx(height - 1 - 880 / 22050 * height, abs=3)
    assert second_peak == pytest.approx(height - 1 - 4400 / 22050 * height, abs=3)
    assert second_peak < first_peak
    assert pixels["music"][first_peak, first_x].mean() < mix[first_peak, first_x].mean()
    assert np.unique(pixels["sfx"].reshape(-1, 3), axis=0).shape[0] == 1
    assert pixels["sfx"].mean() < mix.mean()


def test_audio_cache_reuses_matching_result_and_invalidates_missing_outputs(audio_case, audio_result):
    job, manifest = audio_result
    key, reused, cached = app.process_video(audio_case["source"], 0.5, 0.25)
    assert cached and key == job.name and reused == manifest
    assert len(audio_case["calls"]) == 1
    for name in EXPORTS | {"DBS-results.zip", "mixture.wav"}:
        path = job / name
        content = path.read_bytes()
        path.unlink()
        try:
            assert app.cached_manifest(job) is None, name
            assert app.download_panel(job) == "", name
            assert app.downloads(job) == [], name
        finally:
            path.write_bytes(content)
    for key in TRACKS:
        damaged = json.loads(json.dumps(manifest))
        del damaged["spectrograms"][key]["url"]
        (job / "manifest.json").write_text(json.dumps(damaged))
        assert app.cached_manifest(job) is None, key
    (job / "manifest.json").write_text(json.dumps(manifest))
    other_key, _, cached = app.process_video(audio_case["source"], 0.5, 0.5)
    assert not cached and other_key != job.name
    assert len(audio_case["calls"]) == 2


def test_upload_preview_generates_only_original_spectrum_without_inference(audio_case, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Upload preview must not load or run a neural model")

    monkeypatch.setattr(app, "get_backend", forbidden)
    updates = list(app.preview_input(audio_case["source"], "bsrnn", 3, 3))
    assert len(updates) == 2
    _assert_cleared(updates[0])
    viewer, _, panel = updates[-1]
    key = "preview-" + app.digest(audio_case["source"])[:24]
    assert f"/studio/{key}" in viewer
    assert panel["visible"] is False and not _links(panel)
    job = app.OUTPUTS / key
    manifest = json.loads((job / "manifest.json").read_text())
    assert manifest["media_type"] == "audio" and not manifest["ready"]
    assert manifest["partitions"] == []
    assert set(manifest["sources"]) == {"mix"}
    assert set(manifest["spectrograms"]) == {"mix"}
    assert {path.name for path in job.iterdir()} == {"manifest.json", "mixture.wav", "mix-spectrogram.png"}
    assert app.cached_manifest(job) is None


def test_upload_matching_result_reuses_full_audio_cache(audio_case, audio_result, monkeypatch):
    job, _ = audio_result

    def forbidden(*args, **kwargs):
        pytest.fail("A complete audio cache must not decode or infer again")

    monkeypatch.setattr(app, "prepare_media", forbidden)
    monkeypatch.setattr(app, "get_backend", forbidden)
    updates = list(app.preview_input(audio_case["source"], "bsrnn", 0.5, 0.25))
    _assert_cleared(updates[0])
    assert f"/studio/{job.name}" in updates[-1][0]
    assert len(_links(updates[-1][2])) == 10


@pytest.mark.parametrize("language", ["en", "zh"])
def test_bad_audio_preview_clears_stale_results_and_shows_localized_error(audio_case, language):
    bad = audio_case["source"].parent / "broken.mp3"
    bad.write_bytes(b"Not valid audio")
    updates = app.preview_input(bad, "bsrnn", 3, 3, language=language)
    _assert_cleared(next(updates))
    _assert_cleared(next(updates))
    with pytest.raises(app.gr.Error) as failure:
        next(updates)
    assert bool(re.search(r"[\u4e00-\u9fff]", str(failure.value))) == (language == "zh")
    assert audio_case["calls"] == []


def test_audio_media_ranges_png_downloads_and_private_routes(audio_case, audio_result):
    job, _ = audio_result
    (job / "private.ckpt").write_bytes(b"private")
    with TestClient(app.create_app()) as client:
        original = client.get(f"/media/{job.name}/mixture.wav", headers={"Range": "bytes=0-15"})
        assert original.status_code == 206
        assert original.content == (job / "mixture.wav").read_bytes()[:16]
        for key in TRACKS:
            response = client.get(f"/media/{job.name}/{key}-spectrogram.png")
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"
            with Image.open(io.BytesIO(response.content)) as picture:
                assert picture.format == "PNG"
                picture.verify()
        for filename in EXPORTS | {"DBS-results.zip"}:
            response = client.get(f"/download/{job.name}/{filename}")
            assert response.status_code == 200, filename
            assert response.content == (job / filename).read_bytes()
            assert response.headers["content-disposition"].startswith("attachment;")
        for filename in ("private.ckpt", "manifest.json", "other.png", "speech.mp4"):
            assert client.get(f"/media/{job.name}/{filename}").status_code == 404
            assert client.get(f"/download/{job.name}/{filename}").status_code == 404
        assert client.get(f"/download/{job.name}/mixture.wav").status_code == 404
        studio = client.get(f"/studio/{job.name}")
        assert studio.status_code == 200
        assert '"media_type": "audio"' in studio.text
        assert f"/media/{job.name}/mix-spectrogram.png" in studio.text
        assert "epoch=" not in studio.text and ".ckpt" not in studio.text


def test_audio_model_switch_never_relabels_another_models_result(audio_case, audio_result):
    bsrnn_job, _ = audio_result
    viewer, _, panel = app.select_model("tiger", audio_case["source"], 0.5, 0.25)
    _assert_cleared((viewer, "", panel))
    tiger_key, _, cached = app.process_video(audio_case["source"], 0.5, 0.25, model_key="tiger")
    assert not cached and tiger_key != bsrnn_job.name
    viewer, _, panel = app.select_model("tiger", audio_case["source"], 0.5, 0.25)
    assert f"/studio/{tiger_key}" in viewer
    assert all(f"/{tiger_key}/" in link for link in _links(panel))
    assert [call["model"] for call in audio_case["calls"]] == ["bsrnn", "tiger"]
