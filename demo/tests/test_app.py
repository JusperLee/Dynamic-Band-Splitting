"""Integration checks for result identity, incomplete caches and local routes."""
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import zipfile

import pytest
from fastapi.testclient import TestClient

import app


RESULT_FILES = {
    "speech.wav", "music.wav", "sfx.wav",
    "speech.mp4", "music.mp4", "sfx.mp4",
    "boundaries.json", "results.json",
}
DOWNLOAD_FILES = RESULT_FILES | {"DBS-results.zip"}


class _LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href.startswith("/download/"):
                self.links.append(href)


class _VisibleTextParser(HTMLParser):
    """Read displayed text rather than translations stored in data attributes."""
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def _visible_text(value):
    if isinstance(value, dict):
        value = value.get("value", "")
    parser = _VisibleTextParser()
    parser.feed(value or "")
    return " ".join(parser.parts)


def _download_links(value):
    """Accept HTML content or a Gradio component update without UI coupling."""
    if isinstance(value, dict):
        value = value.get("value", "")
    parser = _LinkParser()
    parser.feed(value or "")
    return parser.links


def _expected_links(job):
    return {f"/download/{job.name}/{name}" for name in DOWNLOAD_FILES}


def _complete_job(job, model="bsrnn"):
    job.mkdir(parents=True, exist_ok=True)
    manifest = {
        "ready": True,
        "title": "Sample",
        "duration": 3.0,
        "window_seconds": 3.0,
        "hop_seconds": 3.0,
        "partitions": [{"start": 0, "end": 3}],
        "meta": {"processing_seconds": 1.0, "model_key": model},
    }
    for name in RESULT_FILES:
        (job / name).write_bytes(f"{model}:{name}".encode())
    (job / "preview.mp4").write_bytes(b"preview")
    (job / "manifest.json").write_text(json.dumps(manifest))
    with zipfile.ZipFile(job / "DBS-results.zip", "w") as bundle:
        for name in sorted(RESULT_FILES):
            bundle.write(job / name, name)
    return manifest


@pytest.fixture
def cached_models(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    sample, bsrnn, tiger = [tmp_path / name for name in ("sample.mp4", "bsrnn.ckpt", "tiger.ckpt")]
    for path in (sample, bsrnn, tiger):
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(app, "OUTPUTS", outputs)
    monkeypatch.setattr(app, "SAMPLE", sample)
    monkeypatch.setitem(app.WEIGHTS, "bsrnn", bsrnn)
    monkeypatch.setitem(app.WEIGHTS, "tiger", tiger)

    def inference_must_not_run(*args, **kwargs):
        pytest.fail("An existing complete result must not load or run a model")

    monkeypatch.setattr(app, "get_backend", inference_must_not_run)
    monkeypatch.setattr(app, "prepare_video", inference_must_not_run)
    jobs = {}
    for model in ("bsrnn", "tiger"):
        key, _ = app.job_key(sample, 3.0, 3.0, model)
        jobs[model] = outputs / key
        _complete_job(jobs[model], model)
    return jobs


def test_window_hop_validation():
    assert app.validate_settings(3, 1.5) == (3.0, 1.5)
    for window, hop in ((3, 4), (0, 3), (float("nan"), 3), (3, float("inf"))):
        with pytest.raises(ValueError):
            app.validate_settings(window, hop)


def test_cache_identity_includes_weight_contents_and_windows(tmp_path, monkeypatch):
    video, weight, tiger = [tmp_path / p for p in ("input.mp4", "bsrnn.ckpt", "tiger.ckpt")]
    video.write_bytes(b"same video")
    weight.write_bytes(b"bsrnn weights")
    tiger.write_bytes(b"tiger weights")
    monkeypatch.setitem(app.WEIGHTS, "bsrnn", weight)
    monkeypatch.setitem(app.WEIGHTS, "tiger", tiger)
    a, _ = app.job_key(video, 3, 3, "bsrnn")
    b, _ = app.job_key(video, 3, 3, "tiger")
    c, _ = app.job_key(video, 3, 1.5, "bsrnn")
    assert len({a, b, c}) == 3
    weight.write_bytes(b"new weights!!")
    d, _ = app.job_key(video, 3, 3, "bsrnn")
    assert a != d


def test_incomplete_cache_does_not_mask_missing_exports(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"ready": True, "partitions": [{}]}))
    assert app.cached_manifest(tmp_path) is None


def test_local_media_range_and_private_files(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "OUTPUTS", tmp_path / "results")
    monkeypatch.setattr(app, "SAMPLE", tmp_path / "missing.mp4")
    monkeypatch.setitem(app.WEIGHTS, "bsrnn", tmp_path / "private.ckpt")
    job = app.OUTPUTS / "job1"
    job.mkdir(parents=True)
    (job / "preview.mp4").write_bytes(b"0123456789abcdef")
    (job / "private.ckpt").write_bytes(b"private")
    (job / "manifest.json").write_text(json.dumps({"ready": False, "title": "</script>"}))
    with TestClient(app.create_app()) as client:
        response = client.get("/media/job1/preview.mp4", headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert client.get("/media/job1/private.ckpt").status_code == 404
        assert client.get("/media/job1/missing.wav").status_code == 404
        studio = client.get("/studio/job1")
        assert studio.status_code == 200
        assert '\\u003c/script>' in studio.text


@pytest.mark.parametrize("model", ["bsrnn", "tiger"])
def test_download_panel_exposes_all_exports_once(cached_models, model):
    job = cached_models[model]
    files = [Path(value) for value in app.downloads(job)]
    assert len(files) == 9
    assert {path.name for path in files} == DOWNLOAD_FILES
    assert all(path.parent == job for path in files)
    links = _download_links(app.download_panel(job))
    assert len(links) == 9
    assert set(links) == _expected_links(job)


def test_missing_metadata_invalidates_previously_complete_cache(cached_models):
    job = cached_models["bsrnn"]
    assert app.cached_manifest(job) is not None
    (job / "results.json").unlink()
    assert app.cached_manifest(job) is None


def test_all_downloads_are_attachments_and_private_files_stay_private(cached_models):
    with TestClient(app.create_app()) as client:
        for job in cached_models.values():
            (job / "private.ckpt").write_bytes(b"private checkpoint")
            (job / "mixture.wav").write_bytes(b"original audio")
            for filename in DOWNLOAD_FILES:
                response = client.get(f"/download/{job.name}/{filename}")
                assert response.status_code == 200, filename
                assert response.content == (job / filename).read_bytes()
                disposition = response.headers.get("content-disposition", "")
                assert disposition.startswith("attachment;"), filename
                assert filename in disposition, filename
            for filename in ("private.ckpt", "preview.mp4", "mixture.wav", "manifest.json", "missing.wav"):
                assert client.get(f"/download/{job.name}/{filename}").status_code == 404
            metadata = client.get(f"/media/{job.name}/results.json")
            assert metadata.status_code == 200
            assert metadata.content == (job / "results.json").read_bytes()


@pytest.mark.parametrize("model", ["bsrnn", "tiger"])
def test_model_selection_downloads_match_selected_cached_result(cached_models, model):
    job = cached_models[model]
    viewer, status, panel = app.select_model(model, None, 3.0, 3.0)
    assert f'/studio/{job.name}' in viewer
    assert app.MODEL_LABELS[model] in status
    assert set(_download_links(panel)) == _expected_links(job)
    viewer, status, panel = app.select_model(model, None, 3.0, 1.5)
    assert "/studio/" not in viewer
    assert not _download_links(panel)
    assert app.MODEL_LABELS[model] in status


@pytest.mark.parametrize("model", ["bsrnn", "tiger"])
def test_cached_run_clears_downloads_then_restores_matching_result(cached_models, model):
    job = cached_models[model]
    updates = list(app.run(None, 3.0, 3.0, model, progress=lambda *args, **kwargs: None))
    assert len(updates) >= 2
    assert not _download_links(updates[0][2])
    assert "/studio/" not in updates[0][0]
    viewer, status, panel = updates[-1]
    assert f"/studio/{job.name}" in viewer
    assert app.MODEL_LABELS[model] in status
    assert set(_download_links(panel)) == _expected_links(job)


def test_initial_view_downloads_match_default_model(cached_models):
    job = cached_models["bsrnn"]
    viewer, status, panel = app.initial_view()
    assert f"/studio/{job.name}" in viewer
    assert set(_download_links(panel)) == _expected_links(job)


def test_failed_run_keeps_downloads_clear_and_reports_failure(cached_models, monkeypatch):
    def fail_processing(*args, **kwargs):
        raise ValueError("invalid input video")

    monkeypatch.setattr(app, "process_video", fail_processing)
    updates = app.run(None, 3.0, 3.0, "bsrnn", progress=lambda *args, **kwargs: None)
    processing = next(updates)
    failed = next(updates)
    assert not _download_links(processing[2])
    assert not _download_links(failed[2])
    assert "Processing failed" in _visible_text(failed[1])
    assert "/studio/" not in failed[0]
    with pytest.raises(app.gr.Error, match="Processing failed"):
        next(updates)


@pytest.mark.parametrize("window,hop", [(float("nan"), 3), (0, 3), (3, 4)])
def test_validation_errors_follow_selected_language(window, hop):
    with pytest.raises(ValueError) as english:
        app.validate_settings(window, hop)
    with pytest.raises(ValueError) as chinese:
        app.validate_settings(window, hop, language="zh")
    assert str(english.value)
    assert not re.search(r"[\u4e00-\u9fff]", str(english.value))
    assert re.search(r"[\u4e00-\u9fff]", str(chinese.value))


@pytest.mark.parametrize("model", ["bsrnn", "tiger"])
def test_language_does_not_change_cached_result_or_download_identity(cached_models, model):
    job = cached_models[model]
    for language in ("en", "zh"):
        updates = list(app.run(None, 3.0, 3.0, model,
                               progress=lambda *args, **kwargs: None,
                               language=language))
        viewer, status, panel = updates[-1]
        assert f"/studio/{job.name}" in viewer
        assert set(_download_links(panel)) == _expected_links(job)
        assert app.MODEL_LABELS[model] in _visible_text(status)
    assert set(app.OUTPUTS.iterdir()) == set(cached_models.values())


def test_default_visible_result_and_download_text_is_english(cached_models):
    viewer, status, panel = app.initial_view()
    text = _visible_text(status) + _visible_text(panel)
    assert not re.search(r"[\u4e00-\u9fff]", text)
    assert "Downloads" in text
    assert "Speech" in text
    assert "Music" in text
    assert "Sound effects" in text
    assert "data-en=" in panel and "data-zh=" in panel
    for forbidden in ("epoch=", "best-114", ".ckpt"):
        assert forbidden not in text


def test_language_selector_defaults_to_english_with_complete_native_translations(cached_models):
    with TestClient(app.create_app()) as client:
        response = client.get("/config")
        assert response.status_code == 200
        config = response.json()
    selectors = [component for component in config["components"]
                 if component["type"] == "radio"
                 and {choice[1] for choice in component["props"].get("choices", [])} == {"en", "zh"}]
    assert len(selectors) == 1
    assert selectors[0]["props"]["value"] == "en"
    translations = config["i18n_translations"]
    assert translations["en"]
    assert set(translations["en"]) == set(translations["zh"])
    assert all(translations[locale][key] for locale in ("en", "zh") for key in translations["en"])
    # Every native component's translation key has both locale entries.
    keys = []
    for component in config["components"]:
        for value in component["props"].values():
            if isinstance(value, str) and value.startswith("__i18n__"):
                keys.append(json.loads(value.removeprefix("__i18n__"))["key"])
    assert keys
    assert set(keys) <= set(translations["en"])


@pytest.mark.parametrize("language,expected", [("en", "no audio track"), ("zh", "没有音轨")])
def test_media_errors_are_localized_at_the_ui_boundary(cached_models, monkeypatch, language, expected):
    def fail_processing(*args, **kwargs):
        raise ValueError("这个视频没有音轨，无法运行音源分离。")

    monkeypatch.setattr(app, "process_video", fail_processing)
    updates = app.run(None, 3.0, 3.0, "bsrnn", language=language,
                      progress=lambda *args, **kwargs: None)
    next(updates)
    failed = next(updates)
    assert not _download_links(failed[2])
    with pytest.raises(app.gr.Error, match=expected):
        next(updates)


@pytest.mark.parametrize("language", ["en", "zh"])
def test_internal_error_does_not_expose_checkpoint_details(cached_models, monkeypatch, language):
    def fail_processing(*args, **kwargs):
        raise RuntimeError("Could not load /private/checkpoints/epoch=180.ckpt")

    monkeypatch.setattr(app, "process_video", fail_processing)
    updates = app.run(None, 3.0, 3.0, "bsrnn", language=language,
                      progress=lambda *args, **kwargs: None)
    next(updates)
    next(updates)
    with pytest.raises(app.gr.Error) as failure:
        next(updates)
    message = str(failure.value)
    assert "epoch=" not in message
    assert ".ckpt" not in message
    assert "/private" not in message
    assert bool(re.search(r"[\u4e00-\u9fff]", message)) == (language == "zh")
