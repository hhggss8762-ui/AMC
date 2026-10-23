"""Render full-capture geometry and a traceable report; does not run classifiers."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib.patches import Rectangle
import numpy as np
from amc.station_capture import load_station_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.report).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    source = summary["source"]
    iq, metadata, checked = load_station_capture(Path(source["payload_file"]).parent)
    if checked["payload_sha256"] != source["payload_sha256"]:
        raise ValueError("Source changed since adaptation.")
    fs, center = metadata["sample_rate_hz"], metadata["center_freq_hz"]
    power, freq, _ = mlab.specgram(iq, NFFT=2048, Fs=fs, noverlap=1024, mode="psd")
    db = 10 * np.log10(np.maximum(power, np.finfo(np.float32).tiny))
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    low, high = (center - fs / 2) / 1e6, (center + fs / 2) / 1e6
    ax.imshow(db, origin="lower", extent=[0, source["duration_s"], low, high],
              aspect="auto", cmap="viridis", vmin=float(db.min()), vmax=float(db.max()))
    for d in diagnostics:
        b = d.get("extraction")
        if not b:
            continue
        start, end = b["start_time_s"], b["end_time_s"]
        flo, fhi = b["freq_low_hz"] / 1e6, b["freq_high_hz"] / 1e6
        ax.add_patch(Rectangle((start, flo), end - start, fhi - flo, fill=False, edgecolor="#ffdc69", linewidth=1))
        ax.text(start + .12, (flo + fhi) / 2,
                f"#{d['detection_index'] + 1}  {d.get('class_name', 'unavailable')}  {d.get('score', 0):.3f}",
                fontsize=8, color="white", va="center", bbox={"facecolor": "black", "alpha": .6, "pad": 2})
    ax.set(xlabel="Sample-relative time (s)", ylabel="RF frequency (MHz)",
           title=f"Entire station IQ: {source['duration_s']:.4f} s | original 0.2.1 boxes + AMC predictions\nUnlabelled capture: predicted types are not verified labels")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.savefig(root / "full_capture_predictions.png", dpi=160)
    plt.close(fig)
    mean_db = 10 * np.log10(np.maximum(power.mean(axis=1), np.finfo(float).tiny))
    rf = freq + center
    rows = max(1, int(np.ceil(len(diagnostics) / 3)))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3 * rows), layout="constrained", squeeze=False)
    for ax, d in zip(axes.flat, diagnostics):
        b = d["extraction"]
        fc = (b["freq_low_hz"] + b["freq_high_hz"]) / 2
        bandwidth = b["estimated_bandwidth_hz"]
        selection = np.abs(rf - fc) < 1.3 * bandwidth
        ax.plot((rf[selection] - fc) / 1000, mean_db[selection], linewidth=.8)
        ax.axvspan(-bandwidth / 2000, bandwidth / 2000, color="#3399ff", alpha=.15, label="Detector band")
        for sign in [-1, 1]:
            ax.axvline(sign * .6 * bandwidth / 1000, color="#d46b08", linestyle="--", linewidth=1)
        ax.set(title=f"#{d['detection_index']+1}: {fc/1e6:.6f} MHz | {d.get('class_name', 'unavailable')}",
               xlabel="Offset from target center (kHz)", ylabel="Mean PSD (dB)")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(diagnostics):]:
        ax.set_visible(False)
    fig.suptitle("Extraction audit: original mean spectrum, blue = box band, dashed = filter pass edges (0.6B)\nThis checks extraction geometry; it does not establish modulation ground truth", fontsize=13)
    fig.savefig(root / "target_frequency_audit.png", dpi=140)
    plt.close(fig)
    lines = ["# 真实 IQ 检测框适配验证", "",
             "使用整条原始记录，无 100 ms 截断。师兄原版 0.2.1 通过 HTTP 检测，本地冻结 AMC 逐框识别；尚未合并为新镜像。", "",
             f"- 输入：`{source['payload_file']}`",
             f"- {source['complex_samples']:,} 个复数点；{fs:,.0f} S/s；中心频率 {center/1e6:.6f} MHz；时长 {source['duration_s']:.4f} s。",
             f"- IQ载荷 {source['payload_bytes']:,} bytes；{source['index_records']} 条索引，所有偏移、点数、参数及序号检查通过。",
             "- 解码：小端 int16，I/Q 交错，I+jQ；沿用既有站点读取约定。",
             f"- 设备索引首末时间戳相差 {source['timestamp_span_s']:.3f} s，有 {source['repeated_timestamp_intervals']} 次重复间隔。它不等同于样本轴；按 N/Fs 建立检测和裁切时间，不把文件中的包时间戳插入 IQ 时间轴。连续的文件偏移不证明射频采样绝对无丢失。",
             "- 默认 adaptive 检测，未设置 specific_band；内部没有触发 zoom。", "",
             "## 逐框预测（没有真实调制标签）", "",
             "下表编号对应两张图；分数为五窗 softmax 分数平均，并非经过校准的真实正确概率。窗口一致性也不是正确率。", "",
             "| 框 | 中心频率 MHz | 框带宽 kHz | 框时长 s | 预测 | 分数 | 同类窗口/总数 | 适配 ms |",
             "|---:|---:|---:|---:|---|---:|---:|---:|"]
    for d in diagnostics:
        b = d["extraction"]
        same = sum(w["class_name"] == d.get("class_name") for w in d.get("windows", []))
        lines.append(f"| {d['detection_index']+1} | {(b['freq_low_hz']+b['freq_high_hz'])/2e6:.6f} | {b['estimated_bandwidth_hz']/1000:.3f} | {b['end_time_s']-b['start_time_s']:.3f} | {d.get('class_name', '不可用')} | {d.get('score', 0):.4f} | {same}/{d.get('window_count', 0)} | {d.get('timing_ms', {}).get('total_processing', 0):.2f} |")
    pre_ms = sum(d.get("timing_ms", {}).get("preprocessing", 0) for d in diagnostics)
    inf_ms = sum(d.get("timing_ms", {}).get("window_selection_normalization_and_inference", 0) for d in diagnostics)
    lines += ["", "## 耗时与审计", "",
              (f"- 本轮复用原版检测响应（读取 {summary['detector_http_or_replay_seconds']:.6f} s，不是检测耗时），来源 `{summary['detector_response_replay']}`。原始 HTTP 运行见 detector_origin.json。"
               if summary.get("detector_response_replay") else
               f"- 检测 HTTP 请求 {summary['detector_http_or_replay_seconds']:.3f} s；包含上传和检测，模型已就绪。"),
              f"- AMC 加载与单窗预热 {summary['amc_load_and_warmup_seconds']:.3f} s；逐框适配 {summary['adaptation_seconds']:.3f} s。",
              f"- 适配内部：预处理合计 {pre_ms/1000:.3f} s；选窗、归一化和网络推理合计 {inf_ms/1000:.3f} s。",
              f"- {len(diagnostics)} 个目标，本次主要为长持续信号，当前实现对每个框的整段目标做预处理，然后只选最多五窗；不能套用短仿真片段约 10 ms 的数据。本次只有一次记录，不是吞吐量或 P95 基准。",
              f"- 原检测响应移除新增两个字段后逐项相等：{summary['original_response_preserved']}；输入 IQ 哈希前后相同：{summary['source_payload_unchanged']}。",
              f"- 冻结模型 SHA256：`{summary['checkpoint_sha256']}`。未训练或调整阈值；512 分母上限保持不变。",
              "- 新增测试覆盖时间/频率方向、正频偏目标及强邻频抑制、滤波上下文排除、分数平均、短框/越界/非法框、捕获边缘的频率余量、空结果、标定不匹配和索引损坏；测试结果见 tests.txt。",
              "", "## 产物", "",
              "- `response.json`：面向接收机的响应，只新增 modulation_type、modulation_score。",
              "- `detector_response.json`：原版检测器响应。",
              "- `diagnostics.json`：每框物理范围、实际重采样率、五窗预测与绝对样本时间、耗时。",
              "- `source.json` / `request_metadata.json`：完整数据来源、哈希、采集与请求参数。",
              "- `detector_container_inspect.json` / `detector_logs.txt`：本次原版检测容器信息与日志。",
              "", "## 发现的实测问题与暂停点", "",
              "频率审计图显示，部分框覆盖强信号之间的弱能量区域或邻频边缘，例如 #7、#8、#10、#11。分类器仍可能返回接近 1 的 AM 分数。仅凭这些图不能断定它们都是误检，也不能确认其中有独立目标。",
              "当前训练主要为单目标仿真，实际强弱邻频、载波分量和检测误差可能影响输出。需先核实这些框的有效性与分类输入，不能将此结果宣称为真实调制识别通过。",
              "按用户要求遇到问题停下确认：已保留完整原响应和全部预测；停止进一步 HTTP 接入及整合镜像制作。没有自行修改检测阈值、类别、频率余量或额外加入能量筛选。",
              "", "![完整数据与预测](full_capture_predictions.png)", "",
              "![频率提取审计](target_frequency_audit.png)", "",
              "流程验证通过不等于真实分类正确。部分窗口存在类别分歧，且模型只在仿真域验证过；AM、64QAM 等输出需有标注实测信号进一步确认。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
