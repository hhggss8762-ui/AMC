# 连续长 IQ 推理接口

本阶段输入是已经明确目标频段的一段连续单信号原始 IQ。目标在整段内持续存在，
由调用方提供采样率、相对采集中心的目标中心频偏、估计带宽。
接口不使用真实类别、真实 SNR、符号率或其他生成标签进行识别。
检测框坐标转换、宽带多目标检测、框的时间上下文读取及跨帧缓存尚未实现。

## Python 调用

```python
import numpy as np
from amc.long_inference import LongIQClassifier

# 一个进程加载一次模型，再重复调用 predict。
classifier = LongIQClassifier("models/amc_baseline_v1.pt", device="cuda")
raw = np.load("my_continuous_iq.npy", allow_pickle=False)
result = classifier.predict(
    raw,
    sample_rate_hz=20_000_000,
    estimated_bandwidth_hz=500_000,
    center_offset_hz=3_000_000,
)
print(result["class_name"], result["score"], result["windows"])
```

数值仅演示调用方式，必须换成这段 IQ 的实际采样信息。原始输入支持复数 `[N]`
或实数 `[2,N]`，通道顺序 I、Q。输入必须有限，频段要完整处于采集范围内。
中心频偏单位 Hz，是目标中心相对于采集中心的偏移，不是绝对射频频率。

`effective_samples=None`（默认）使用预处理后全部有效点。
本次固定长度仿真测试显式指定 `effective_samples=16384`，从有效区居中截取；
不足指定长度时报错，不补零，不改变采样率硬凑点数。
这是测试设置，不是实际 IQ 必须满足的统一长度。

## 文件入口

```python
import json
import numpy as np

np.savez(
    "my_signal.npz",
    raw_iq=raw.astype(np.complex64),
    input_metadata_json=np.array(json.dumps({
        "sample_rate_hz": 20_000_000,
        "estimated_bandwidth_hz": 500_000,
        "center_offset_hz": 3_000_000,
    })),
)
```

在 AMC 目录及 amc 环境中执行：

```powershell
python tools/infer_long.py --input my_signal.npz --output reports/my_signal_result.json --device cuda
```

文件元信息仅允许三个必要采集参数及可选 `effective_samples`，不接受类别等标签字段。
`np.load` 使用 `allow_pickle=False`，不保存对象数组。推理输出文件已存在时拒绝覆盖。

## 固定处理规则

1. 复用模型检查点内的预处理配置，按中心频偏移频。
2. FIR 通带到 ±0.6B，正常阻带起点 ±0.8B，按 60 dB 设计；接近原采样范围边缘时按现有函数限制并检查。
3. 目标输出采样率 min(原采样率,4B)，有理数近似分母不超过 512；记录实际输出采样率。
4. FIR 使用有效卷积，重采样后首尾各去掉 64 点，避免边缘瞬态。
5. 可选居中保留指定有效长度，再取 K=min(5,floor(L/2048)) 个非重叠窗口。
6. K=1 时取中间；K>=2 时起点为 floor(linspace(0,L-2048,K))。不足 2048 有效点返回 insufficient_length。
7. 每窗整体复数 RMS 归一化，不独立缩放 I/Q，不统一减均值。模型 float32 推理，最多每批 64 个窗口。
8. 所有选中窗口的 13 类分数等权平均，最高分类别为整段结果，不剔除低分窗口。

正常测试的 16384 点对应起点 0、3584、7168、10752、14336。
不足五窗时按实际数量汇总；偶数窗的真正中心通常不在选中列表中，会另算一个中心窗口供诊断，
它不参与所选窗口的均值。`window_count` 是参加汇总的数量，`windows_inferred` 是实际预测窗口数量。

## 结果与时间坐标

结果包括 `class_name`、最高平均分 `score`、完整 `class_scores`、五窗一致率 `consistency`、
每窗结果 `windows`、真正中间窗 `middle_window`、选窗起点和处理参数。
`start_sample`、`end_sample_exclusive`、`start_time_s` 均相对于选定的有效片段。
`preprocessing.effective_origin_in_raw_s` 表示该有效片段相对于输入原始 IQ 的起始时间，
两者相加可以映射到原始记录；不涉及图片像素坐标。

默认最高平均分 >=0.6 标记 classified，否则 uncertain；它仍有 top-1 预测类别。
分数没有校准成正确概率，uncertain 不是可靠的未知调制判断。一致率也不是准确率。

`timing_ms` 分别记录 CPU 预处理、选窗、归一化与模型推理、汇总及总处理时间。
GPU 前后同步；文件读取和模型启动不包含在这个总处理时间中。
调用计时包含诊断中间窗的额外计算（若需要）；本次正常五窗中心已包含在五窗中。

## 本轮仿真验证

新配置 `configs/long_signal_eval.yaml`；种子首项 20261015、命名空间 3，
preview/full 再分命名空间，源信号不与既有数据复用。
数字信号 4096 符号、AM 81920 原始点、FM 40960 原始点、OFDM 128 连续符号。
沿用现有调制参数范围，只增加连续源长度，不重复或拼接小窗。

每条原始 clean/noise 分量通过相同预处理，并在居中的完整 16384 点有效区上校准一次带内 SNR，
一个噪声系数应用于整条原始记录。存储 raw_iq=raw_clean+raw_noise，complex64。
五窗实际 SNR 可以自然波动；名义真实带宽和分量信息只用于生成与审计，不输入分类器。
每类子类型在每档内平分。15 个参考文件保存原始及处理后分量，用于独立重放与 SNR 复算。

正式配对评估对同一条原始 IQ 分别运行 max_windows=1 和 max_windows=5 的完整流程，
交替运行顺序，分别实测耗时。每种模式先预热 5 段；所有 1820 段仍计入正式准确率和统计。
主要统计单位是独立长信号，不把 9100 个窗口当成 9100 个独立源。
真实 SNR 组别按整段名义标记分组；旧小窗测试是逐窗校准 SNR，两份不同数据的分数不可直接用于测量汇总收益。
