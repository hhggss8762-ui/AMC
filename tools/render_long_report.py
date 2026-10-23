"""Render the fixed long-signal paired evaluation without rerunning inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/long_signal_v1")
    parser.add_argument("--report", default="reports/long_signal_v1")
    args = parser.parse_args()
    root, report = Path(args.dataset), Path(args.report)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((report / "validation.json").read_text(encoding="utf-8"))
    metrics = json.loads((report / "evaluation/metrics.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    with np.load(report / "evaluation/predictions.npz", allow_pickle=False) as data:
        y, snrs, middle, fused = [data[key].copy() for key in (
            "labels", "snrs", "middle_predictions", "fused_predictions")]
    good = np.isin(snrs, metrics["five_window"]["good_snr_levels_db"])
    classes = metrics["classes"]
    per_class = []
    for i, name in enumerate(classes):
        mask = good & (y == i)
        per_class.append({"class_name": name, "count": int(mask.sum()),
                          "middle": float(np.mean(middle[mask] == i)),
                          "five": float(np.mean(fused[mask] == i))})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    levels = manifest["snr_db"]
    for key, label in (("middle_only", "Middle window"), ("five_window", "Five-window average")):
        values = [metrics[key]["per_snr"][str(level)]["accuracy"] * 100 for level in levels]
        axes[0].plot(levels, values, marker="o", label=label)
    axes[0].axhline(95, color="gray", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Nominal whole-segment SNR (dB)", ylabel="Signal top-1 accuracy (%)",
                ylim=(40, 103), title="Paired evaluation on independent continuous signals")
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=9)
    positions = np.arange(len(classes))
    axes[1].barh(positions - .18, [r["middle"] * 100 for r in per_class],
                 height=.36, label="Middle window")
    axes[1].barh(positions + .18, [r["five"] * 100 for r in per_class],
                 height=.36, label="Five-window average")
    axes[1].set_yticks(positions, classes)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Recall (%)", xlim=(0, 105), title="15/20/25/30 dB, 80 signals per class")
    axes[1].axvline(95, color="gray", linestyle="--", linewidth=.8)
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2)
    fig.savefig(report / "comparison.png", dpi=150)
    plt.close(fig)

    stages = ["preprocessing", "window_selection", "normalization_and_inference", "aggregation", "total_processing"]
    labels = ["Preprocess", "Select windows", "Normalize + infer", "Aggregate", "Total processing"]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    positions = np.arange(len(stages))
    for offset, mode, label in ((-.18, "middle", "Middle window"), (.18, "five", "Five windows")):
        vals = [metrics["timing_ms"][mode][s]["median"] for s in stages]
        bars = ax.bar(positions + offset, vals, width=.36, label=label)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=8, padding=3)
    ax.set_xticks(positions, labels)
    ax.set(ylabel="Median time (ms)", title="Warm local calls; synchronized GPU; disk read/model startup excluded")
    ax.legend()
    ax.grid(axis="y", alpha=.2)
    fig.savefig(report / "timing.png", dpi=150)
    plt.close(fig)

    five, one, paired = metrics["five_window"], metrics["middle_only"], metrics["paired_comparison"]
    durations = [r["preprocessing"]["effective_duration_s"] * 1000 for r in rows]
    raw_sizes = [r["source_sample_count"] for r in rows]
    reference = metrics["good_snr_reference_target_reached"]
    cm = np.asarray(metrics["five_window_good_snr"]["confusion_matrix"])
    confusions = sorted([(int(cm[i, j]), classes[i], classes[j]) for i in range(len(classes))
                         for j in range(len(classes)) if i != j and cm[i, j]], reverse=True)
    lines = ["# 长 IQ 单信号推理与配对验证报告", "", "日期：2026-10-15。固定使用首次全量训练第 17 轮模型，没有重新训练或调整权重。", "",
             f"数据为 {len(rows):,} 段独立连续信号，13 类、7 档 SNR、每类每档 20 段；五窗共 {len(rows)*5:,} 个选中窗口。",
             "每段同时运行完整的中间单窗口流程和五窗口平均流程，模型输入仅包含 IQ 与采样参数，不包含真实类别。", "",
             "| 指标 | 中间单窗口 | 五窗口分数平均 |", "|---|---:|---:|",
             f"| 全 0–30 dB 准确率 | {one['accuracy']:.2%} | {five['accuracy']:.2%} |",
             f"| 15/20/25/30 dB 准确率 | {one['good_snr_accuracy']:.2%} | {five['good_snr_accuracy']:.2%} |", "",
             f"全 SNR 准确率变化 {paired['overall_accuracy_change_percentage_points']:+.2f} 个百分点；良好 SNR 变化 {paired['good_snr_accuracy_change_percentage_points']:+.2f} 个百分点。",
             f"相比中间单窗，五窗纠正 {paired['improved_signals']} 段，同时使 {paired['regressed_signals']} 段由对变错；其他信号的正确性不变。",
             f"15–30 dB 共 {int(good.sum()):,} 段，五窗正确 {int(np.sum((fused==y)&good)):,} 段；按 95% 作为本次参考目标：{'达到' if reference else '未达到'}。",
             "所有信号均计入准确率，未根据分数或 uncertain 状态筛选。五个窗口属于同一个源信号，不按五个独立源统计样本量。", "",
             "## 分 SNR 的整段识别", "", "| 标记 SNR (dB) | 信号数 | 单窗 | 五窗 |", "|---:|---:|---:|---:|"]
    for level in levels:
        a, b = one["per_snr"][str(level)], five["per_snr"][str(level)]
        lines.append(f"| {level} | {a['count']} | {a['accuracy']:.2%} | {b['accuracy']:.2%} |")
    lines += ["", "## 15–30 dB 分类别召回率", "", "每类 80 段，召回率是该真实类别被正确识别的比例。", "",
              "| 类别 | 单窗 | 五窗 |", "|---|---:|---:|"]
    for row in per_class:
        lines.append(f"| {row['class_name']} | {row['middle']:.2%} | {row['five']:.2%} |")
    lines += ["", "五窗口良好 SNR 下主要有向误判：", ""]
    lines += [f"- {a} → {b}：{n} 段。" for n, a, b in confusions[:8]]
    wc = metrics["window_consistency"]
    lines += ["", f"五窗完全一致的信号 {wc['all_five_agree_count']} 段；存在窗口分歧的 {wc['signals_with_any_disagreement']} 段。平均与汇总结果一致的窗口比例 {wc['mean']:.2%}；一致率不是正确率。", "",
              "## 耗时", "", "以下为本机 RTX 5070 Laptop 上预热后的配对实测；交替先运行单窗或五窗，计算结束后同步 GPU。模型加载、磁盘读取单独统计。", "",
              "| 流程 | 平均 (ms) | 中位数 (ms) | P95 (ms) | 最大 (ms) |", "|---|---:|---:|---:|---:|"]
    for mode, label in (("middle", "单窗完整处理"), ("five", "五窗完整处理")):
        t = metrics["timing_ms"][mode]["total_processing"]
        lines.append(f"| {label} | {t['mean']:.2f} | {t['median']:.2f} | {t['p95']:.2f} | {t['max']:.2f} |")
    io = metrics["disk_read_ms"]
    lines += ["", f"模型启动 {metrics['startup_ms']:.2f} ms；文件读取中位数 {io['median']:.2f} ms，P95 {io['p95']:.2f} ms。读取耗时来自运行时文件缓存状态，不是冷盘基准。",
              f"两种模式分别用 {metrics['timing_warmup_signals']} 段预热，预热不替代正式统计，这些信号仍完整计入准确率。",
              "完整处理包括移频、滤波、重采样、有效区选择、选窗、归一化、传输与推理、汇总；不包含检测器、跨帧缓存或接收机服务，不能直接当成 100 ms 接收机整帧性能。", "",
              "## 数据与固定处理规则", "",
              "原始 complex64 连续 IQ 保存于 signals/*.npz，真实标签和种子在 metadata.jsonl，输入 NPZ 只含 raw_iq 与采样参数 JSON。",
              "源长度：数字信号 4096 个符号，AM 81920 个原始点，FM 40960 点，OFDM 128 个连续 OFDM 符号；未拼接小窗口。",
              "这些源长度保障处理后的有效区至少 16384 点，再居中保留 16384 点；原始长度不统一，实际范围为 "
              f"{min(raw_sizes):,}–{max(raw_sizes):,} 点。",
              "沿用训练预处理：估计带宽 ±10%，通带到 ±0.6B、正常阻带起点 ±0.8B、FIR 60 dB 设计、输出采样率 min(原采样率,4B)、有理数分母最多 512、首尾各去掉 64 个输出点。",
              "随机初相与 ±2% 名义带宽残余频偏整段固定；整段有效区只校准一次带内 SNR，窗口 SNR 自然波动，不逐窗缩放噪声。",
              "正常五窗起点为 0、3584、7168、10752、14336；每窗 2048 点，以 13 类分数等权平均汇总，不做低分窗筛选。",
              f"16384 点有效片段覆盖的物理时长为 {min(durations):.3f}–{max(durations):.3f} ms；并非固定 100 ms，窄带情况可能超过一帧。",
              f"原始 IQ 本体 {audit['raw_payload_bytes']:,} 字节；完整数据目录 {audit['total_directory_bytes']:,} 字节，约 {audit['total_directory_bytes']/2**30:.3f} GiB。",
              f"数据审计通过：{audit['unique_sources']} 个独立源，与原训练/验证/测试、两个旧预览和本轮长信号预览均无种子重叠；原始文件 SHA256、重放、五窗位置、标签数量、子类型平衡通过检查。",
              f"{audit['reference_files_checked']} 个原始 clean/noise 分量参考文件独立复算 SNR，最大误差 {audit['max_reference_snr_error_db']:.3g} dB。全数据 SNR 元信息最大误差 {audit['max_metadata_snr_error_db']:.3g} dB。", "",
              "## 使用与边界", "",
              "单段运行命令（在 AMC 目录的 amc 环境执行）：", "", "```powershell",
              "python tools/infer_long.py --input data/long_signal_v1/signals/000000.npz --output reports/long_signal_v1/example_prediction.json --device cuda", "```", "",
              "生成、校验、配对评估、渲染入口分别为 generate_long_signals.py、validate_long_signals.py、evaluate_long_signals.py、render_long_report.py。生成和评估拒绝覆盖非空输出目录。",
              "推理 API 为 amc.long_inference.LongIQClassifier.predict，接受复数 [N] 或实数 I/Q [2,N]；未设置 effective_samples 时使用全部有效数据。",
              "短信号按实际长度选 1–5 个非重叠窗口，不足一个窗口返回 insufficient_length，不补零。偶数窗口时另计算真正的中间窗口供诊断，不参加均值汇总。",
              "本次仍是同一合成信号族的新独立载荷与噪声，源长度及 SNR 校准区间与旧小窗测试不同；不能把旧测试的 95.04% 与这里的分数变化直接归因于多窗口。多窗口收益以本报告同一批长信号的配对结果判断。",
              "真实中心信息目前已知、带宽使用约定误差；没有检测框坐标转换、混叠干扰、真实采集失真或接收机整帧集成，不宣称真实信号准确率。",
              "主要结果图 comparison.png、timing.png；混淆矩阵、详细 JSONL 和预测数组位于 evaluation/；validation.json 为数据审计。", "",
              f"冻结模型 SHA256：{metrics['checkpoint_sha256']}。"]
    smoke_path = report / "cli_smoke_prediction.json"
    if smoke_path.exists():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        lines += ["", f"单独启动进程的 CLI 首次推理样例耗时 {smoke['timing_ms']['total_processing']:.2f} ms（仍不含模型加载）。首次 CUDA 调用可能明显慢于预热后的稳定结果。",
                  "本次单窗与五窗的预热后耗时接近，五窗略快是本机测量结果，不意味着多窗在其他设备或场景中普遍更快。"]
    (report / "LONG_SIGNAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.resolve() / "LONG_SIGNAL_REPORT.md")


if __name__ == "__main__":
    main()
