# AMC baseline v1 推理权重

`amc_baseline_v1.pt` 为 2026-10-15 首次全量训练第 17 轮最佳模型的推理检查点。
包含网络张量、固定类别顺序、预处理与模型配置、来源信息，移除了优化器等训练状态。
大小 3,934,603 字节（约 3.75 MiB），加载需要本项目已安装的 `amc` 包与 PyTorch。
完整训练检查点保留在 `runs/full_baseline_v1/best.pt` 和 `last.pt`。

输入必须是已移频、滤波、重采样的实数 I/Q 两通道窗口，float32，形状 `[2,2048]` 或 `[N,2,2048]`。
通道 0 为 I，通道 1 为 Q；推理接口再次进行整体复数 RMS 归一化。
当前模型不直接接收任意长的宽带原始 IQ，不包含前置检测与框选裁剪。

在 AMC 目录和 amc Conda 环境下使用：

```python
import numpy as np
from amc.inference import WindowClassifier

classifier = WindowClassifier("models/amc_baseline_v1.pt", device="cuda")
windows = np.load("data/full_v1/test/iq.npy", mmap_mode="r")
results = classifier.predict(windows[:64], batch_size=64)
print(results[0])
```

结果含 `class_id`、`class_name`、`score`、`class_scores` 与 `status`。
默认 0.6 分数阈值只用于启发式 `classified/uncertain` 状态；不表示可靠未知类检测。
95.04% 验收结果计算所有指定 SNR 测试窗口的 top-1，不按此状态筛选。

类别 ID 依次为 AM、FM、ASK、BPSK、QPSK、8PSK、PI4_DQPSK、16QAM、64QAM、2FSK、4FSK、2GFSK、OFDM。
独立仿真测试在 15/20/25/30 dB 合并后总体准确率 95.0449%，全 0–30 dB 总体 87.1136%。
不是每类都达到 95%；主要薄弱项是 AM/OFDM 与 16QAM/64QAM。
OFDM 只覆盖本项目约定的有限 CP-OFDM 参数范围；本阶段没有真实接收机性能结论。

文件 SHA256：`4f22f777fb859c406fab04b59cfdf67805c1145031ec34a3d10b5af3748d1da0`。
详细报告与验证在 `reports/full_baseline_v1/TRAINING_REPORT.md`、`result_audit.json`。

该权重现已接入 `amc.long_inference.LongIQClassifier`，可提供连续原始单信号 IQ、
采样率、目标中心频偏与估计带宽，自动预处理并汇总最多五个窗口。
使用入口 `tools/infer_long.py`，输入格式详见 `docs/LONG_IQ_INFERENCE.md`。
权重没有改变；独立长信号配对结果见 `reports/long_signal_v1/LONG_SIGNAL_REPORT.md`。
