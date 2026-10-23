from pathlib import Path
import numpy as np
import pytest
from amc.config import load_config
from amc.long_inference import prepare_long_iq
from amc.preprocess import band_power
from amc.wideband_dataset import generate_wideband_record, load_wideband_settings

ROOT = Path(__file__).parents[1]
CONFIG = load_config(ROOT / "configs/baseline.yaml")
SETTINGS = load_wideband_settings(ROOT / "configs/wideband_eval.yaml")
CASES = [(name, subtype) for name in CONFIG["classes"]
         for subtype in CONFIG["dataset"]["subtypes"].get(name, [name])]


@pytest.mark.parametrize("name,subtype", CASES)
@pytest.mark.parametrize("bandwidth,ratio", [(100_000, 5), (20_000_000, 20)])
def test_physical_bandwidth_acquisition_resampling_and_global_snr(name, subtype, bandwidth, ratio):
    raw, m, refs = generate_wideband_record(
        name, subtype, 15, [992, 4, CASES.index((name, subtype)), bandwidth], CONFIG, SETTINGS,
        bandwidth, ratio)
    assert m["nominal_bandwidth_hz"] == pytest.approx(bandwidth, rel=2e-12)
    assert raw.dtype == np.complex64 and np.isfinite(raw).all()
    p = m["source_parameters"]
    assert p["acquisition_resample_up"] > 0 and p["acquisition_resample_down"] <= 512
    assert m["input"]["sample_rate_hz"] == pytest.approx(
        p["generator_sample_rate_hz"] * p["acquisition_resample_up"] / p["acquisition_resample_down"])
    assert m["input"]["sample_rate_hz"] == pytest.approx(bandwidth * ratio, rel=0.001)
    assert abs(m["input"]["center_offset_hz"]) <= .1 * m["input"]["sample_rate_hz"]
    replay, pre = prepare_long_iq(raw, settings=CONFIG["preprocess"], **m["input"])
    clean, _ = prepare_long_iq(refs["raw_clean"], settings=CONFIG["preprocess"], **m["input"])
    noise, _ = prepare_long_iq(refs["raw_noise"], settings=CONFIG["preprocess"], **m["input"])
    assert len(replay) == 16384 and np.allclose(replay, clean + noise, atol=2e-6, rtol=2e-6)
    measured = 10 * np.log10(band_power(clean, pre["output_sample_rate_hz"], bandwidth)
                             / band_power(noise, pre["output_sample_rate_hz"], bandwidth))
    assert measured == pytest.approx(15, abs=.002)
    assert pre["output_sample_rate_hz"] == pytest.approx(pre["requested_output_sample_rate_hz"], rel=.001)
    assert pre["resample_down"] <= 512
    if "symbol_rate_hz" in p:
        factor = ((1 + (3 if name == "4FSK" else 1) * p["modulation_index"])
                  if name in ("2FSK", "4FSK", "2GFSK") else (1 + CONFIG["synthesis"]["rrc_rolloff"]))
        assert p["symbol_rate_hz"] * factor == pytest.approx(bandwidth, rel=2e-12)
    elif name == "AM":
        assert p["message_bandwidth_hz"] * (2 if subtype == "DSB_LC" else 1) == pytest.approx(bandwidth)
    elif name == "FM":
        assert 2 * (p["message_bandwidth_hz"] + p["peak_deviation_hz"]) == pytest.approx(bandwidth)
    elif name == "OFDM":
        assert (p["active_subcarriers"] + 1) * p["subcarrier_spacing_hz"] == pytest.approx(bandwidth)


def test_intermediate_capture_rate_and_invalid_sweep_settings():
    _, m, _ = generate_wideband_record("QPSK", "QPSK", 20, [992, 4, 42], CONFIG, SETTINGS, 1_000_000, 10)
    assert m["input"]["sample_rate_hz"] == pytest.approx(10_000_000, rel=.001)
    with pytest.raises(ValueError, match="Unexpected acquisition ratio"):
        generate_wideband_record("QPSK", "QPSK", 20, [992, 4, 42], CONFIG, SETTINGS, 1_000_000, 1)
