"""Validated configuration and stable public class identifiers."""
from pathlib import Path
import yaml

CLASSES = (
    "AM", "FM", "ASK", "BPSK", "QPSK", "8PSK", "PI4_DQPSK",
    "16QAM", "64QAM", "2FSK", "4FSK", "2GFSK", "OFDM",
)
SPLITS = ("train", "validation", "test")


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config.get("version") == "amc20-v1":
        from .dataset20 import load_config20
        return load_config20(path)
    if tuple(config["classes"]) not in (CLASSES, CLASSES[:-1]):
        raise ValueError("Baseline class order must match CLASSES.")
    if config["dataset"]["window_samples"] != 2048:
        raise ValueError("This baseline uses exactly 2048 complex samples.")
    if config["dataset"]["storage_dtype"] != "float32":
        raise ValueError("Storage must be float32.")
    if config["preprocess"]["output_rate_over_estimated_bandwidth"] < 2:
        raise ValueError("Output rate must leave a filter transition band.")
    for split in SPLITS:
        count = config["dataset"]["per_class_per_snr"][split]
        if not isinstance(count, int) or count < 2 or count % 2:
            raise ValueError("Split counts must be positive even integers >= 2.")
    if config["synthesis"]["rrc_rolloff"] <= 0:
        raise ValueError("RRC rolloff must be positive.")
    if "OFDM" in config["classes"]:
        ofdm = config["synthesis"]["ofdm"]
        if not ofdm["fft_sizes"] or any(n not in (64, 128) for n in ofdm["fft_sizes"]):
            raise ValueError("This OFDM baseline supports FFT sizes 64/128.")
        if not ofdm["cp_fractions"] or any(f not in (0.125, 0.25) for f in ofdm["cp_fractions"]):
            raise ValueError("OFDM CP must be 1/8 or 1/4 of the useful symbol.")
        if not 0.25 <= ofdm["active_subcarrier_fraction"] <= 0.9:
            raise ValueError("Invalid active subcarrier fraction.")
        if not ofdm["subcarrier_modulations"] or any(
            m not in ("QPSK", "16QAM", "64QAM") for m in ofdm["subcarrier_modulations"]
        ):
            raise ValueError("Unsupported OFDM subcarrier constellation.")
        if ofdm["source_oversampling"] < 4 or ofdm["source_ofdm_symbols"] < 16:
            raise ValueError("Insufficient OFDM source length/oversampling.")
    return config
