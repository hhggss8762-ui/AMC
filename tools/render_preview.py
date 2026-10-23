import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


def read_case(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in ("iq", "clean", "noise")}, json.loads(str(data["metadata_json"]))


def spectrum(ax, data, metadata):
    fs = metadata["output_sample_rate_hz"]
    for key, color in (("clean", "tab:blue"), ("iq", "tab:orange")):
        x = data[key][0] + 1j * data[key][1]
        f, p = welch(x, fs=fs, nperseg=256, return_onesided=False, detrend=False)
        ax.plot(np.fft.fftshift(f) / 1000,
                10 * np.log10(np.maximum(np.fft.fftshift(p), 1e-14)),
                color=color, linewidth=0.8, label=key)
    bw = metadata["nominal_bandwidth_hz"]
    ax.axvline(-bw / 2000, color="gray", linestyle="--", linewidth=0.7)
    ax.axvline(bw / 2000, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Baseband frequency (kHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.grid(alpha=0.2)
    ax.set_title(f"{metadata['class_name']} / {metadata['subtype']} | SNR 15 dB\n"
                 f"Fs={fs/1000:.1f} kS/s, B={bw/1000:.1f} kHz", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/preview_ofdm_v1")
    parser.add_argument("--output", default="reports/preview_ofdm_v1")
    args = parser.parse_args()
    source, output = Path(args.dataset) / "preview_references", Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    manifest = json.loads((Path(args.dataset) / "manifest.json").read_text(encoding="utf-8"))
    for name in manifest["classes"]:
        path = sorted(source.glob(f"{name}_*.npz"))[0]
        selected.append(read_case(path))
    rows = (len(selected) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), constrained_layout=True)
    for ax, (data, metadata) in zip(axes.flat, selected):
        spectrum(ax, data, metadata)
    axes.flat[0].legend(fontsize=8)
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    fig.savefig(output / "spectra.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3 * rows), constrained_layout=True)
    for ax, (data, m) in zip(axes.flat, selected):
        fs = m["output_sample_rate_hz"]
        time_ms = np.arange(256) / fs * 1000
        ax.plot(time_ms, data["clean"][0, :256], label="clean I", linewidth=1)
        ax.plot(time_ms, data["iq"][0, :256], label="noisy I", alpha=0.65, linewidth=0.8)
        ax.set_title(f"{m['class_name']} / {m['subtype']}", fontsize=10)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Normalized I")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)
    fig.savefig(output / "waveforms.png", dpi=150)
    plt.close(fig)
    paths = [source / f"{name}_{subtype}.npz" for name, subtype in
             (("AM", "DSB_LC"), ("AM", "SSB"), ("ASK", "OOK"), ("ASK", "BINARY_NONZERO"))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, path in zip(axes.flat, paths):
        data, metadata = read_case(path)
        spectrum(ax, data, metadata)
    fig.savefig(output / "am_ask_subtypes.png", dpi=150)
    plt.close(fig)
    if "OFDM" in manifest["classes"]:
        data, metadata = read_case(source / "OFDM_OFDM.npz")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        spectrum(axes[0], data, metadata)
        fs = metadata["output_sample_rate_hz"]
        t = np.arange(data["iq"].shape[-1]) / fs * 1000
        clean = data["clean"][0] + 1j * data["clean"][1]
        axes[1].plot(t, abs(clean), linewidth=0.8)
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Clean magnitude")
        p = metadata["source_parameters"]
        axes[1].set_title(f"OFDM: FFT={p['fft_size']}, CP={p['cyclic_prefix_samples']}\n"
                         f"{p['subcarrier_modulation']}, window={metadata['window_ofdm_symbol_durations']:.1f} symbols")
        axes[1].grid(alpha=0.2)
        fig.savefig(output / "ofdm.png", dpi=150)
        plt.close(fig)
    print(f"Saved preview plots to {output.resolve()}")


if __name__ == "__main__":
    main()
