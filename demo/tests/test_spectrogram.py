"""Scientific axis, comparison and memory-bound checks for audio heatmaps."""

import numpy as np
from PIL import Image
import pytest

import spectrogram


def tone(frequency=1024, duration=2, sample_rate=8192, amplitude=0.5):
    times = np.arange(round(duration * sample_rate)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * times)).astype(np.float32)


def read_image(path):
    with Image.open(path) as image:
        return np.asarray(image)


def test_tone_frequency_matches_linear_axis(tmp_path):
    metadata = spectrogram.write_spectrograms(tmp_path, {"mix": tone()}, 8192)["mix"]
    pixels = read_image(tmp_path / metadata["filename"])
    brightness = pixels.mean(axis=2)
    peak_row = brightness[:, pixels.shape[1] // 2].argmax()
    estimated_frequency = (1 - (peak_row + 0.5) / pixels.shape[0]) * metadata["max_hz"]
    assert estimated_frequency == pytest.approx(1024, abs=15)
    assert metadata["frequency_scale"] == "linear"
    assert metadata["duration"] == pytest.approx(2)


def test_event_location_matches_full_recording_timeline(tmp_path):
    audio = tone(duration=4)
    audio[:8192 * 2] = 0
    audio[8192 * 3:] = 0
    metadata = spectrogram.write_spectrograms(tmp_path, {"mix": audio}, 8192)["mix"]
    pixels = read_image(tmp_path / metadata["filename"])
    brightness = pixels.mean(axis=2).max(axis=0)
    bright_columns = np.flatnonzero(brightness > 180)
    times = bright_columns / pixels.shape[1] * metadata["duration"]
    assert times[0] == pytest.approx(2, abs=0.15)
    assert times[-1] == pytest.approx(3, abs=0.15)


def test_identical_amplitudes_have_identical_colors_and_quieter_track_is_dimmer(tmp_path):
    audio = tone()
    original = audio.copy()
    metadata = spectrogram.write_spectrograms(
        tmp_path, {"mix": audio, "speech": audio.copy(), "music": audio * 0.01, "sfx": audio * 0}, 8192,
    )
    mix = read_image(tmp_path / metadata["mix"]["filename"])
    speech = read_image(tmp_path / metadata["speech"]["filename"])
    music = read_image(tmp_path / metadata["music"]["filename"])
    np.testing.assert_array_equal(mix, speech)
    np.testing.assert_array_equal(audio, original)
    assert music[:, music.shape[1] // 2].max() < mix[:, mix.shape[1] // 2].max()
    assert len({item["reference_amplitude"] for item in metadata.values()}) == 1
    assert {item["db_min"] for item in metadata.values()} == {-80}


def test_all_silence_is_finite_uniform_floor(tmp_path):
    metadata = spectrogram.write_spectrograms(tmp_path, {"mix": np.zeros(1, dtype=np.float32)}, 44100)["mix"]
    pixels = read_image(tmp_path / metadata["filename"])
    assert metadata["reference_amplitude"] == 1
    assert metadata["width"] == 1
    np.testing.assert_array_equal(pixels, np.broadcast_to(spectrogram.PALETTE[0], pixels.shape))


def test_long_recording_output_and_fft_batches_stay_bounded(tmp_path, monkeypatch):
    # Ten minutes at a low sample rate still exercises > 1600 frames and
    # verifies the actual batching limit without a costly 44.1 kHz fixture.
    audio = np.zeros(600 * 2048, dtype=np.float32)
    original_rfft = spectrogram.rfft
    batch_shapes = []

    def observe(batch, *args, **kwargs):
        batch_shapes.append(batch.shape)
        return original_rfft(batch, *args, **kwargs)

    monkeypatch.setattr(spectrogram, "rfft", observe)
    metadata = spectrogram.write_spectrograms(tmp_path, {"mix": audio}, 2048)["mix"]
    assert metadata["width"] == 1600
    assert metadata["height"] == 400
    assert metadata["duration"] == 600
    assert len(batch_shapes) > 1
    assert max(shape[0] for shape in batch_shapes) <= spectrogram.BATCH_FRAMES
    assert {shape[1] for shape in batch_shapes} == {spectrogram.FFT_SIZE}


@pytest.mark.parametrize("waveforms, sample_rate", [
    ({"../escape": np.zeros(4)}, 44100),
    ({"mix": np.array([np.nan])}, 44100),
    ({"mix": np.array([np.inf])}, 44100),
    ({"mix": np.zeros(2), "music": np.zeros(3)}, 44100),
    ({"mix": np.zeros((2, 1))}, 44100),
    ({"mix": np.zeros(0)}, 44100),
    ({"mix": np.zeros(4)}, 0),
])
def test_invalid_signal_does_not_write(tmp_path, waveforms, sample_rate):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        spectrogram.write_spectrograms(destination, waveforms, sample_rate)
    assert not destination.exists()
