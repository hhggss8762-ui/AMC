from pathlib import Path
import numpy as np
import pytest
from amc.config import CLASSES, load_config
from amc.preprocess import band_power, preprocess_iq
from amc.synthesis import generate_sample, generate_waveform

CONFIG = load_config(Path(__file__).parents[1] / "configs/baseline.yaml")


@pytest.mark.parametrize("name", CLASSES)
@pytest.mark.parametrize("snr", [0, 30])
def test_waveforms_produce_auditable_windows(name, snr):
    subtype = CONFIG["dataset"]["subtypes"].get(name, [name])[0]
    window, metadata, reference = generate_sample(name, subtype, snr, [99, CLASSES.index(name), snr], CONFIG)
    assert window.shape == (2, 2048) and window.dtype == np.float32
    assert np.isfinite(window).all()
    assert np.mean(np.sum(window.astype(float) ** 2, axis=0)) == pytest.approx(1, abs=2e-6)
    assert np.allclose(window, reference["clean"] + reference["noise"], atol=3e-7)
    clean = reference["clean"][0] + 1j * reference["clean"][1]
    noise = reference["noise"][0] + 1j * reference["noise"][1]
    measured = 10 * np.log10(band_power(clean, metadata["output_sample_rate_hz"], metadata["nominal_bandwidth_hz"])
                             / band_power(noise, metadata["output_sample_rate_hz"], metadata["nominal_bandwidth_hz"]))
    assert measured == pytest.approx(snr, abs=0.002)


def test_frequency_translation_preserves_target_and_suppresses_neighbor():
    fs = 320_000
    time = np.arange(32768) / fs
    x = np.exp(2j * np.pi * 49000 * time) + 4 * np.exp(2j * np.pi * 105000 * time)
    y, m = preprocess_iq(x, fs, 25000, CONFIG["preprocess"], center_offset_hz=45000)
    t = np.arange(len(y)) / m["output_sample_rate_hz"]
    target = abs(np.mean(y * np.exp(-2j * np.pi * 4000 * t)))
    alias = abs(np.mean(y * np.exp(2j * np.pi * 40000 * t)))
    assert m["output_sample_rate_hz"] == pytest.approx(100000)
    assert target > 0.98
    assert alias / target < 0.003


def test_ssb_has_only_the_requested_sideband_before_centering():
    source = generate_waveform("AM", "SSB", np.random.default_rng(21), CONFIG)
    f = np.fft.fftfreq(len(source.iq), 1 / source.sample_rate_hz)
    power = abs(np.fft.fft(source.iq)) ** 2
    positive, negative = power[f > 0].sum(), power[f < 0].sum()
    correct, wrong = (positive, negative) if source.parameters["sideband"] == "USB" else (negative, positive)
    assert correct > 10000 * max(wrong, 1e-12)
    window, _, _ = generate_sample("AM", "SSB", 15, [21, 1], CONFIG)
    assert window.shape == (2, 2048)


def test_ask_nonzero_variant_is_not_ook_and_am_carrier_is_kept():
    ook = generate_waveform("ASK", "OOK", np.random.default_rng(11), CONFIG)
    ask = generate_waveform("ASK", "BINARY_NONZERO", np.random.default_rng(11), CONFIG)
    assert ook.parameters["low_to_high_amplitude"] == 0
    assert 0.15 <= ask.parameters["low_to_high_amplitude"] <= 0.6
    _, _, refs = generate_sample("AM", "DSB_LC", 30, [12], CONFIG)
    clean = refs["clean"][0] + 1j * refs["clean"][1]
    assert abs(clean.mean()) > 0.5
