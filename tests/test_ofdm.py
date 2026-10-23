from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from amc.config import load_config
from amc.dataset import validate_dataset
from amc.synthesis import generate_sample, generate_waveform

CONFIG = load_config(Path(__file__).parents[1] / "configs/baseline.yaml")


@pytest.mark.parametrize("fft_size", [64, 128])
@pytest.mark.parametrize("modulation", ["QPSK", "16QAM", "64QAM"])
def test_ofdm_cp_orthogonality_and_constellation(fft_size, modulation):
    config = deepcopy(CONFIG)
    config["synthesis"]["ofdm"]["fft_sizes"] = [fft_size]
    config["synthesis"]["ofdm"]["subcarrier_modulations"] = [modulation]
    wave = generate_waveform("OFDM", "OFDM", np.random.default_rng(5), config)
    p = wave.parameters
    n = fft_size * p["source_oversampling"]
    cp = p["cyclic_prefix_samples"] * p["source_oversampling"]
    blocks = wave.iq.reshape(p["source_ofdm_symbols"], n + cp)
    assert np.array_equal(blocks[:, :cp], blocks[:, -cp:])
    grid = np.fft.fftshift(np.fft.fft(blocks[:, cp:], axis=1), axes=1)
    grid /= n / np.sqrt(p["active_subcarriers"])
    k = p["active_subcarriers"] // 2
    active = np.r_[np.arange(n // 2 - k, n // 2), np.arange(n // 2 + 1, n // 2 + k + 1)]
    unused = np.ones(n, dtype=bool)
    unused[active] = False
    assert np.max(abs(grid[:, unused])) < 1e-10
    symbols = grid[:, active]
    if modulation == "QPSK":
        assert np.max(abs(abs(symbols) - 1)) < 1e-10
    else:
        order = 16 if modulation == "16QAM" else 64
        scaled = symbols * np.sqrt(2 * (order - 1) / 3)
        for component in (scaled.real, scaled.imag):
            assert np.max(abs(component - np.rint(component))) < 1e-10
            assert np.all(np.rint(component).astype(int) % 2 == 1)
            assert np.max(abs(component)) <= np.sqrt(order) - 1 + 1e-10
    sample, m, _ = generate_sample("OFDM", "OFDM", 15, [123, fft_size], config)
    assert sample.shape == (2, 2048)
    assert m["window_ofdm_symbol_durations"] > 3.5


def test_existing_12class_preview_remains_readable():
    preview = Path(__file__).parents[1] / "data/preview_v1"
    if not preview.exists():
        pytest.skip("Original stage-one preview is optional.")
    assert validate_dataset(preview)["passed"]
