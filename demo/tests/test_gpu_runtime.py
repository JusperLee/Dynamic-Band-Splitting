"""GPU dispatch serialization, cache reuse, and mounted-app startup ordering."""

import json
import pickle
import re
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

import app
import gpu_runtime


class ProgressRecorder:
    """A picklable UI progress object, unlike an enclosing local callback."""

    def __init__(self):
        self.updates = []

    def __call__(self, fraction, *, desc):
        self.updates.append((fraction, desc))


@pytest.mark.parametrize("model_key", ["bsrnn", "tiger"])
@pytest.mark.parametrize("language", ["en", "zh"])
def test_uncached_inference_crosses_gpu_boundary_with_serializable_cpu_data(
        tmp_path, monkeypatch, model_key, language):
    source = tmp_path / "input.mp4"
    checkpoint = tmp_path / "weights.safetensors"
    source.write_bytes(b"test video")
    checkpoint.write_bytes(b"released weights")
    audio = np.linspace(-0.1, 0.1, 44100, dtype=np.float32)
    recorder = ProgressRecorder()
    monkeypatch.setattr(app, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(app, "ZERO_GPU", True)
    monkeypatch.setitem(app.WEIGHTS, model_key, checkpoint)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Mock CUDA GPU")

    def prepare_media(path, job):
        assert path == source
        job.mkdir(parents=True)
        preview = job / "preview.mp4"
        preview.write_bytes(b"browser preview")
        return {"media_type": "video", "audio": audio, "duration": 1.0,
                "video_path": preview}

    monkeypatch.setattr(app, "prepare_media", prepare_media)
    monkeypatch.setattr(app, "mux_stem", lambda video, wav, destination:
                        destination.write_bytes(b"separated video"))
    calls = []
    inside_gpu = False

    class FakeBackend:
        model = SimpleNamespace(parameters=lambda: iter([
            SimpleNamespace(device=SimpleNamespace(type="cuda"))]))

        def separate(self, samples, *, window_seconds, hop_seconds, progress):
            assert inside_gpu
            np.testing.assert_array_equal(samples, audio)
            assert (window_seconds, hop_seconds) == (0.5, 0.5)
            progress(0.0, "Separating window 1/2")
            progress(0.5, "Separating window 2/2")
            progress(1.0, "Separation complete")
            return {"stems": {"speech": samples.copy(), "music": samples * 0.5,
                              "sfx": np.zeros_like(samples)},
                    "sample_rate": 44100,
                    "windows": [{"start": 0, "end": 0.5}, {"start": 0.5, "end": 1.0}],
                    "coarse_boundaries_hz": [0, 1000, 2000, 10000, 22050],
                    "metadata": {"architecture": model_key, "device": "cuda"}}

    def get_backend(path, checksum, key):
        assert inside_gpu, "Model loading must occur inside the GPU lease"
        assert (path, checksum, key) == (str(checkpoint), app.digest(checkpoint), model_key)
        calls.append(key)
        return FakeBackend()

    monkeypatch.setattr(app, "get_backend", get_backend)
    helper = app.separate_audio
    boundary_arguments = []

    def dispatch(*arguments):
        nonlocal inside_gpu
        # A callback defined in process_video would fail this pickle boundary.
        restored = pickle.loads(pickle.dumps(arguments))
        assert len(restored) == 8
        assert isinstance(restored[0], np.ndarray)
        assert isinstance(restored[-1], ProgressRecorder)
        assert arguments[-1] is recorder
        boundary_arguments.append(restored)
        inside_gpu = True
        try:
            result = helper(*arguments)
        finally:
            inside_gpu = False
        assert all(isinstance(stem, np.ndarray) for stem in result["stems"].values())
        return pickle.loads(pickle.dumps(result))

    monkeypatch.setattr(app, "separate_audio", dispatch)
    job_id, manifest, cached = app.process_video(source, 0.5, 0.5, recorder, model_key, language)
    assert not cached
    assert calls == [model_key] and len(boundary_arguments) == 1
    assert boundary_arguments[0][5:7] == (model_key, language)
    assert manifest["meta"]["hardware"] == "zerogpu"
    assert manifest["meta"]["cuda_initialized"] is True
    assert manifest["meta"]["gpu_name"] == "Mock CUDA GPU"
    assert manifest["sample_rate"] == 44100 and manifest["duration"] == 1.0
    descriptions = [description for _, description in recorder.updates]
    assert all(bool(re.search(r"[\u4e00-\u9fff]", text)) == (language == "zh")
               for text in descriptions)
    if language == "en":
        assert "Separating window 1/2" in descriptions
        assert "Separating window 2/2" in descriptions
    fractions = [fraction for fraction, _ in recorder.updates]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    job = app.OUTPUTS / job_id
    assert app.cached_manifest(job) is not None
    with zipfile.ZipFile(job / "DBS-results.zip") as bundle:
        assert set(bundle.namelist()) == set(app.EXPORT_NAMES)
        assert bundle.testzip() is None


@pytest.mark.parametrize("model_key", ["bsrnn", "tiger"])
def test_complete_cache_does_not_request_a_gpu_lease(tmp_path, monkeypatch, model_key):
    source = tmp_path / "sample.mp4"
    checkpoint = tmp_path / "weights.safetensors"
    source.write_bytes(b"same video")
    checkpoint.write_bytes(b"same weights")
    monkeypatch.setattr(app, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setitem(app.WEIGHTS, model_key, checkpoint)
    monkeypatch.setattr(app, "ZERO_GPU", True)
    job_id, _ = app.job_key(source, 3.0, 3.0, model_key)
    job = app.OUTPUTS / job_id
    job.mkdir(parents=True)
    manifest = {"ready": True, "media_type": "video", "partitions": [{}],
                "meta": {"model_key": model_key}}
    (job / "manifest.json").write_text(json.dumps(manifest))
    for name in ("preview.mp4", "DBS-results.zip", *app.EXPORT_NAMES):
        (job / name).write_bytes(b"completed export")

    def unexpected_work(*args, **kwargs):
        pytest.fail("A complete cached result must not prepare media or allocate a GPU")

    monkeypatch.setattr(app, "separate_audio", unexpected_work)
    monkeypatch.setattr(app, "get_backend", unexpected_work)
    monkeypatch.setattr(app, "prepare_media", unexpected_work)
    actual_id, actual_manifest, cached = app.process_video(
        source, 3.0, 3.0, ProgressRecorder(), model_key)
    assert cached and actual_id == job_id and actual_manifest == manifest


def test_zero_gpu_startup_preloads_both_cuda_models_before_startup_hook(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "ZERO_GPU", True)
    monkeypatch.setattr(app, "DEVICE", "cuda")
    events = []
    for key in app.MODELS:
        checkpoint = tmp_path / f"{key}.safetensors"
        checkpoint.write_bytes(key.encode())
        monkeypatch.setitem(app.WEIGHTS, key, checkpoint)

    class FakeBackend:
        def __init__(self, key, *, device):
            assert device == "cuda"
            events.append(("preload", key, device))

    def startup():
        assert [event[1] for event in events] == list(app.MODELS)
        events.append(("startup",))

    monkeypatch.setattr(app, "Backend", FakeBackend)
    monkeypatch.setattr(app, "start_zero_gpu", startup)
    app.get_backend.cache_clear()
    try:
        app.initialize_gpu_runtime()
        assert events == [("preload", key, "cuda") for key in app.MODELS] + [("startup",)]
    finally:
        app.get_backend.cache_clear()


def test_cpu_startup_does_not_load_models_or_activate_the_sdk(monkeypatch):
    monkeypatch.setattr(app, "ZERO_GPU", False)

    def unexpected_work(*args, **kwargs):
        pytest.fail("CPU startup must not preload models or activate ZeroGPU")

    monkeypatch.setattr(app, "model_checkpoint", unexpected_work)
    monkeypatch.setattr(app, "get_backend", unexpected_work)
    monkeypatch.setattr(app, "start_zero_gpu", unexpected_work)
    app.initialize_gpu_runtime()


def test_zero_gpu_rejects_cpu_configuration_before_preloading(monkeypatch):
    monkeypatch.setattr(app, "ZERO_GPU", True)
    monkeypatch.setattr(app, "DEVICE", "cpu")
    monkeypatch.setattr(app, "get_backend", lambda *args:
                        pytest.fail("Invalid configuration must fail before model loading"))
    with pytest.raises(ValueError, match="DBS_DEVICE=cuda"):
        app.initialize_gpu_runtime()


@pytest.mark.parametrize("initialized,device", [(False, "cuda"), (True, "cpu")])
def test_inference_rejects_missing_cuda_execution_evidence(monkeypatch, initialized, device):
    metadata = {}
    backend = SimpleNamespace(
        model=SimpleNamespace(parameters=lambda: iter([
            SimpleNamespace(device=SimpleNamespace(type=device))])),
        separate=lambda *args, **kwargs: {"metadata": metadata})
    monkeypatch.setattr(app, "ZERO_GPU", True)
    monkeypatch.setattr(app, "get_backend", lambda *args: backend)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda:
                        pytest.fail("Missing CUDA evidence must fail before reporting a GPU name"))
    with pytest.raises(RuntimeError, match="did not initialize a CUDA model"):
        app.separate_audio(np.zeros(64, dtype=np.float32), 3.0, 1.5,
                           "weights", "checksum", "bsrnn", "en", ProgressRecorder())
    assert "hardware" not in metadata and "cuda_initialized" not in metadata


def test_zero_gpu_startup_hook_runs_once_without_importing_real_sdk(monkeypatch):
    events = []
    sdk = ModuleType("spaces")
    zero = ModuleType("spaces.zero")
    zero.startup = lambda: events.append("startup")
    sdk.zero = zero
    monkeypatch.setitem(sys.modules, "spaces", sdk)
    monkeypatch.setitem(sys.modules, "spaces.zero", zero)
    monkeypatch.setattr(gpu_runtime, "ZERO_GPU", True)
    monkeypatch.setattr(gpu_runtime, "_started", False)
    gpu_runtime.start_zero_gpu()
    gpu_runtime.start_zero_gpu()
    assert events == ["startup"]


def test_cpu_runtime_hook_is_a_noop(monkeypatch):
    monkeypatch.setattr(gpu_runtime, "ZERO_GPU", False)
    monkeypatch.setattr(gpu_runtime, "_started", False)
    gpu_runtime.start_zero_gpu()
    assert not gpu_runtime._started
