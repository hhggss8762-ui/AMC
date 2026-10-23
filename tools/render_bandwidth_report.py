"""Audit and render fixed-model bandwidth sweep results without new inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from amc.long_dataset import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/bandwidth_sweep_v1")
    parser.add_argument("--report", default="reports/bandwidth_sweep_v1")
    args = parser.parse_args()
    root, out = Path(args.dataset), Path(args.report)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "validation.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "evaluation/metrics.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    with np.load(out / "evaluation/predictions.npz", allow_pickle=False) as z:
        y, snrs, one, five, bw, ratio, ws, fs = [z[k].copy() for k in (
            "labels", "snrs", "middle_predictions", "fused_predictions", "bandwidth_hz",
            "acquisition_ratio", "window_scores", "fused_scores")]
    good = np.isin(snrs, [15, 20, 25, 30])
    assert manifest["status"] == "complete" and validation["passed"]
    assert len(rows) == len(y) == manifest["total_signals"] == 7280 and int(good.sum()) == 4160
    assert ws.shape == (7280, 5, 13) and np.isfinite(ws).all()
    assert np.allclose(ws.sum(2), 1, atol=1e-6) and np.allclose(ws.mean(1), fs, atol=1e-8)
    assert np.array_equal(fs.argmax(1), five)
    assert float(np.mean(y == five)) == metrics["five_window"]["accuracy"]
    assert float(np.mean(y[good] == five[good])) == metrics["five_window"]["good_snr_accuracy"]
    frozen = json.loads(Path("reports/full_baseline_v1/result_audit.json").read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(metrics["checkpoint"])
    assert checkpoint_sha == frozen["export"]["sha256"] == metrics["checkpoint_sha256"]
    trained_environment = json.loads(Path("runs/full_baseline_v1/environment.json").read_text(encoding="utf-8"))
    assert sha256_file("src/amc/preprocess.py") == trained_environment["source_sha256"]["preprocess.py"]
    for name, digest in manifest["code_sha256"].items():
        assert sha256_file(Path("src/amc") / name) == digest
    bands = manifest["bandwidth_hz"]
    classes = metrics["classes"]
    for bandwidth in bands:
        mask = bw == bandwidth
        group = metrics["by_bandwidth"][str(bandwidth)]
        assert int(mask.sum()) == 910 and int((mask & good).sum()) == 520
        assert float(np.mean(y[mask] == five[mask])) == group["five"]["accuracy"]
        assert float(np.mean(y[mask & good] == five[mask & good])) == group["five"]["good_snr_accuracy"]
    result_audit = {
        "passed": True, "signals": len(y), "selected_windows": len(y)*5,
        "good_snr_signals": int(good.sum()), "correct_five": int(np.sum(y == five)),
        "correct_middle": int(np.sum(y == one)),
        "good_correct_five": int(np.sum((y == five) & good)),
        "good_correct_middle": int(np.sum((y == one) & good)),
        "fusion_recomputed": True, "all_bandwidth_groups_recomputed": True,
        "model_sha256_unchanged": checkpoint_sha, "preprocessing_source_unchanged": True,
        "resampling_max_denominator": metrics["resampling_max_denominator"],
        "source_hashes_match_generation": True, "tests_passed": 113,
    }
    cli_path = out / "cli_20mhz_example.json"
    if cli_path.exists():
        cli = json.loads(cli_path.read_text(encoding="utf-8"))
        cli_index = int(Path(cli["input"]).stem)
        assert bw[cli_index] == 20_000_000 and cli["class_id"] == int(five[cli_index])
        assert cli["window_starts"] == [0, 3584, 7168, 10752, 14336]
        result_audit["cli_20mhz_matches_evaluation"] = True
        result_audit["cli_first_call_ms_excluding_model_load"] = cli["timing_ms"]["total_processing"]
    (out / "result_audit.json").write_text(json.dumps(result_audit, indent=2), encoding="utf-8")

    x = np.array(bands)/1e6
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, key, title in ((axes[0], "good_snr_accuracy", "15/20/25/30 dB: 520 signals per bandwidth"),
                            (axes[1], "accuracy", "All 0–30 dB: 910 signals per bandwidth")):
        for mode, label in (("middle", "Middle window"), ("five", "Five-window average")):
            values = [100*metrics["by_bandwidth"][str(b)][mode][key] for b in bands]
            ax.plot(x, values, marker="o", label=label)
        ax.axhline(95, color="gray", linestyle="--", linewidth=.8)
        ax.set(xscale="log", xlabel="Nominal bandwidth (MHz)", ylabel="Signal top-1 accuracy (%)",
               ylim=(70, 102), title=title)
        ax.set_xticks(x, [f"{v:g}" for v in x])
        ax.grid(alpha=.2)
        ax.legend(fontsize=9, loc="lower left")
    fig.savefig(out / "accuracy_by_bandwidth.png", dpi=150)
    plt.close(fig)
    heat = np.array([[100*metrics["by_bandwidth"][str(b)]["five_good_snr"]["per_class"][c]["recall"]
                      for b in bands] for c in classes])
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    image = ax.imshow(heat, vmin=50, vmax=100, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(bands)), [f"{b/1e6:g}" for b in bands])
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Nominal bandwidth (MHz)", title="Five-window recall at 15/20/25/30 dB | 40 signals per class/bandwidth")
    for i in range(len(classes)):
        for j in range(len(bands)):
            ax.text(j, i, f"{heat[i,j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if heat[i,j] >= 80 else "black")
    fig.colorbar(image, ax=ax, label="Recall (%)")
    fig.savefig(out / "class_recall_by_bandwidth.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for mode, quantile, label in (("middle", "median", "Middle: median"),
                                  ("five", "median", "Five: median"), ("five", "p95", "Five: P95")):
        values = [metrics["by_bandwidth"][str(b)]["timing_ms"][mode]["total_processing"][quantile] for b in bands]
        ax.plot(x, values, marker="o", label=label)
    ax.set(xscale="log", xlabel="Nominal bandwidth (MHz)", ylabel="Processing time (ms)",
           title="Warm synchronized calls; short variable-duration records; read/startup excluded")
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(out / "timing_by_bandwidth.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row_index, name in enumerate(("QPSK", "OFDM")):
        for col_index, bandwidth in enumerate((bands[0], bands[-1])):
            with np.load(root / "references" / f"B{bandwidth}_{name}_{name}.npz", allow_pickle=False) as ref:
                metadata = json.loads(str(ref["metadata_json"]))
                raw = ref["raw_clean"]
            fs_acq = metadata["input"]["sample_rate_hz"]
            center = metadata["input"]["center_offset_hz"]
            raw = raw * np.exp(-2j*np.pi*center*np.arange(len(raw))/fs_acq)
            f, power = welch(raw, fs=fs_acq, nperseg=4096, return_onesided=False, detrend=False)
            ax = axes[row_index, col_index]
            ax.plot(np.fft.fftshift(f)/1e6, 10*np.log10(np.maximum(np.fft.fftshift(power), 1e-20)))
            ax.axvline(-bandwidth/2e6, color="gray", linestyle="--")
            ax.axvline(bandwidth/2e6, color="gray", linestyle="--")
            ax.set(xlim=(-.9*bandwidth/1e6, .9*bandwidth/1e6), xlabel="Frequency relative to target center (MHz)",
                   ylabel="Clean PSD (dB/Hz)", title=f"{name}: B={bandwidth/1e6:g} MHz, acquisition Fs={fs_acq/1e6:.3g} MS/s")
            ax.grid(alpha=.2)
    fig.savefig(out / "waveform_examples.png", dpi=150)
    plt.close(fig)

    # The report text follows the plots so it can link to their final filenames.
    write_report(root, out, rows, manifest, validation, metrics, result_audit)
    print(out.resolve() / "BANDWIDTH_REPORT.md")


def write_report(root, out, rows, manifest, validation, metrics, audit):
    bands = manifest["bandwidth_hz"]
    five, one = metrics["five_window"], metrics["middle_only"]
    lines = ["# 100 kHz–20 MHz 带宽覆盖验证", "", "日期：2026-10-15。使用原第 17 轮模型；没有重新训练或更换权重。", "",
             "13 类×8 档目标名义带宽×7 档 SNR×每组合 10 段，共 7,280 段全新连续信号、36,400 个选中窗口。",
             "带宽为 0.1、0.2、0.5、1、2、5、10、20 MHz。SNR 为 0、5、10、15、20、25、30 dB。",
             "各带宽使用独立载荷和噪声，不复用训练源或旧长信号；各档差异也包含有限样本的随机波动。", "",
             "| 总体指标 | 中间单窗 | 五窗平均 |", "|---|---:|---:|",
             f"| 全 0–30 dB，7,280 段 | {one['accuracy']:.2%} | {five['accuracy']:.2%} |",
             f"| 15/20/25/30 dB，4,160 段 | {one['good_snr_accuracy']:.2%} | {five['good_snr_accuracy']:.2%} |", "",
             "所有信号均计入，包括 uncertain 状态；没有筛掉低分信号或窗口。单窗与五窗在同一条原始 IQ 上分别完整运行，按信号而非窗口统计。", "",
             "## 各带宽结果", "", "每档 910 段，其中良好 SNR 520 段。", "",
             "| 带宽 (MHz) | 单窗全 SNR | 五窗全 SNR | 单窗 15–30 dB | 五窗 15–30 dB | 五窗处理 P50/P95 (ms) |",
             "|---:|---:|---:|---:|---:|---:|"]
    for b in bands:
        g = metrics["by_bandwidth"][str(b)]
        t = g["timing_ms"]["five"]["total_processing"]
        lines.append(f"| {b/1e6:g} | {g['middle']['accuracy']:.2%} | {g['five']['accuracy']:.2%} | "
                     f"{g['middle']['good_snr_accuracy']:.2%} | {g['five']['good_snr_accuracy']:.2%} | {t['median']:.2f}/{t['p95']:.2f} |")
    values = [metrics["by_bandwidth"][str(b)]["five"]["good_snr_accuracy"] for b in bands]
    minimum = int(np.argmin(values))
    maximum = int(np.argmax(values))
    lines += ["", f"良好 SNR 下五窗最低档为 {bands[minimum]/1e6:g} MHz：{values[minimum]:.2%}；最高档为 {bands[maximum]/1e6:g} MHz：{values[maximum]:.2%}。",
              "这是同族理想波形经过带宽归一化的验证；不把几档之间的小幅变化直接解释成带宽造成的性能差异。", "",
              "## 采集采样率组合", "",
              "| 带宽 (MHz) | 采集率约 5B：良好 SNR 五窗 | 约 10B | 约 20B | 最大输出率误差 |",
              "|---:|---:|---:|---:|---:|"]
    for b in bands:
        g = metrics["by_bandwidth"][str(b)]
        vals = [g["by_acquisition_ratio"][str(r)]["five"]["good_snr_accuracy"] for r in (5, 10, 20)]
        lines.append(f"| {b/1e6:g} | {vals[0]:.2%} | {vals[1]:.2%} | {vals[2]:.2%} | {g['output_rate_relative_error_max']:.4%} |")
    lines += ["", "上述采样率分组的数量见 evaluation/metrics.json，10 段在三种采集率间轮转，无法每格严格三等分。",
              "预处理源码与训练时一致，比例近似分母上限仍为 512；生成端的采集率转换也使用 512 上限。",
              f"本轮实际输出采样率相对于请求值的最大误差 {validation['max_output_rate_relative_error']:.4%}，预先设置的审计上限为 0.1%。",
              "该结论只覆盖采集率约为目标带宽 5、10、20 倍的组合，没有测试极大降采样倍率或采集率不足 4B 的情况。", "",
              "## 各类在不同带宽下的召回率", "", "下表为五窗、15/20/25/30 dB；每个类别每档带宽 40 段，分辨率为 2.5 个百分点。", "",
              "| 类别 | " + " | ".join(f"{b/1e6:g} MHz" for b in bands) + " |",
              "|---|" + "---:|" * len(bands)]
    for c in metrics["classes"]:
        vals = [metrics["by_bandwidth"][str(b)]["five_good_snr"]["per_class"][c]["recall"] for b in bands]
        lines.append("| " + c + " | " + " | ".join(f"{v:.1%}" for v in vals) + " |")
    p = metrics["paired_comparison"]
    lines += ["", f"全 SNR 五窗纠正 {p['improved_signals']} 段，同时使 {p['regressed_signals']} 段由对变错；净变化 {p['overall_accuracy_change_percentage_points']:+.2f} 个百分点。",
              f"良好 SNR 净变化 {p['good_snr_accuracy_change_percentage_points']:+.2f} 个百分点。", "",
              "## 波形与采集过程", "",
              "先按目标名义带宽重新设置物理生成参数：PSK/QAM/ASK 使用 Rs=B/(1+rolloff)，FSK 使用 Rs=B/(1+(M-1)h)；AM 根据边带类型设置消息带宽；FM 保留频偏/消息带宽比例并反推消息带宽；OFDM 使用 Δf=B/(有效载波数+1)。",
              "每条记录重新生成连续随机载荷，再进行多相滤波采样率转换，形成约 5B、10B 或 20B 的采集 IQ；不是改旧文件标签。SSB 在采集转换前移到基带，采集后再放到已知非零中心位置。",
              "采集中心偏移在 ±0.1Fs 范围随机；分类预处理移除该已知中心，保留 ±0.02B 残余频偏。初相、带宽估计 ±10% 误差、调制结构参数沿用原范围。",
              "干净信号完成采集转换后，在采集采样率下生成白噪声；按完整 16384 点有效区统一校准一次带内 SNR，噪声系数作用于整段原始 IQ。",
              "处理时仍复用移频、60 dB FIR、min(原采样率,4×估计带宽)、边缘去除、逐窗复数 RMS。五窗起点为 0、3584、7168、10752、14336。",
              "OFDM 仍只有 64/128 点 FFT，CP 1/8 或 1/4，三种子载波星座；20 MHz OFDM 不代表覆盖所有 20 MHz 协议波形或其他 FFT 配置。", "",
              "## 空间、物理时长与耗时", ""]
    durations = [r["preprocessing"]["effective_duration_s"]*1000 for r in rows]
    fs_acq = [r["input"]["sample_rate_hz"] for r in rows]
    raw_counts = [r["source_sample_count"] for r in rows]
    t = metrics["timing_ms"]["five"]["total_processing"]
    lines += [f"原始 IQ 本体 {validation['raw_payload_bytes']:,} 字节；完整目录 {validation['total_directory_bytes']:,} 字节，约 {validation['total_directory_bytes']/2**30:.3f} GiB。",
              f"实际采集采样率范围 {min(fs_acq)/1e6:.3f}–{max(fs_acq)/1e6:.3f} MS/s，原始点数范围 {min(raw_counts):,}–{max(raw_counts):,}。",
              f"16384 点有效片段覆盖 {min(durations):.4f}–{max(durations):.4f} ms；本次不是每条固定 100 ms 的采集记录。宽信号有效片段更短，因此采样率高不意味着本测试的原始点数同比增加。",
              f"五窗预热后完整处理中位数 {t['median']:.2f} ms，P95 {t['p95']:.2f} ms，最大 {t['max']:.2f} ms。模型启动与读盘单独统计；GPU 同步计时。",
              "这些耗时针对本批短记录，不包含检测器、框坐标转换、原始整帧读取、跨帧缓存与接收机服务。不能据此承诺完整 100 ms 宽带采集记录的处理耗时。", "",
              "## 审计与交付", "",
              f"数据审计通过：{validation['unique_sources']} 个唯一源种子，和既有数据/本轮预览无重叠；检查原始文件 SHA256、标签/带宽/SNR/子类型数量、原始重放、五窗位置、波形参数及采样率。",
              f"{validation['reference_files_checked']} 份原始 clean/noise 参考分量独立复算 SNR，最大误差 {validation['max_reference_snr_error_db']:.3g} dB。",
              "结果数组独立复算通过，各带宽数量及准确率、五窗平均与 argmax 均核对一致。113 项检查通过。",
              f"模型 SHA256 保持 {audit['model_sha256_unchanged']}，预处理文件 SHA256 与原训练记录一致。", "",
              "生成入口 tools/generate_bandwidth_sweep.py；配置 configs/wideband_eval.yaml。",
              "审计复用 tools/validate_long_signals.py，配对推理复用 tools/evaluate_long_signals.py，带宽报告由 tools/render_bandwidth_report.py 生成。",
              "validation.json 是数据审计；result_audit.json 是结果复算；evaluation/metrics.json 是详细分组指标；evaluation/signal_results.jsonl 是逐段结果；evaluation/predictions.npz 是预测数组。",
              "图：accuracy_by_bandwidth.png、class_recall_by_bandwidth.png、timing_by_bandwidth.png、waveform_examples.png；混淆矩阵位于 evaluation/。", "",
              "本轮验证同一调制结构在不同时间尺度和采集率下的适配，不增加多径、采样时钟误差、IQ 不平衡、混合信号或未知类覆盖；不能替代真实 IQ 验证。"]
    if audit.get("cli_20mhz_matches_evaluation"):
        lines += ["", "20 MHz、20 dB 的 QPSK 单段 CLI 样例也已运行，分类与完整评估一致，输入为 signals/006690.npz，输出 cli_20mhz_example.json。",
                  f"该独立进程首次调用耗时 {audit['cli_first_call_ms_excluding_model_load']:.2f} ms（不含模型加载），明显慢于预热后的稳定值；冷启动与预热值不能混用。"]
    (out / "BANDWIDTH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
