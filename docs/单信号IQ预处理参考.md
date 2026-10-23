# 单信号 IQ 预处理参考

接收机按以下流程准备数据：

```text
检测 JSON → 时间/频率范围 → 同帧缓存 IQ → 截取、频移、滤波、重采样 → 每条 IQ＋JSON → 调制种类
```

## 1. 从检测 JSON 换算时间和频率

原版 0.2.1 检测响应节选：

```json
{
  "image_width_px": 560,
  "image_height_px": 560,
  "calibration": {
    "time_window_ms": 200.0,
    "freq_start_hz": 858360000.0,
    "freq_end_hz": 858640000.0
  },
  "detections": [
    {
      "xyxy": [56.0, 205.0, 448.0, 255.0]
    }
  ]
}
```

`xyxy=[x1,y1,x2,y2]` 为像素坐标；向右时间增加，向下频率降低。使用响应中的实际宽高 `W、H`，不能用原始矩阵行列数代替。略微越界的框先限制到图像边界，空框或无效标定不用于提取。设 `T=time_window_ms/1000` 秒，`F_min=freq_start_hz`，`F_max=freq_end_hz`：

```text
t_start = x1 / W × T
t_end   = x2 / W × T
f_high  = F_max - y1 / H × (F_max - F_min)
f_low   = F_max - y2 / H × (F_max - F_min)
B       = f_high - f_low
f_target = (f_high + f_low) / 2
```

时间相对于本次瀑布图输入起点。**必须使用与该帧对应的缓存 IQ**；瀑布图与缓存起点不同时，计入偏移 `Δt=瀑布图起点−缓存起点`。原始采样率 `Fs` 和采集中心频率 `f_capture` 取设备参数，不能把显示带宽当作原始采样率。

## 2. 预处理步骤

**① 时间截取。** 按原始复数采样点索引读取：

```text
n_start = floor((Δt + t_start) × Fs)
n_end   = ceil((Δt + t_end) × Fs)
读取范围：[n_start, n_end)
```

提取范围须落在实际缓存内。前后多取滤波所需的真实样本作为上下文；补偿处理延时、去掉无效边缘后，只提交原框时间内的有效 IQ，不补零凑长度。

**② 频移。** 以 `I+jQ` 为复数约定，将目标搬到零频附近：

```text
Δf = f_target - f_capture
x_shift[n] = x[n] × exp(-j × 2π × Δf × n / Fs)
```

**③ 滤波。** 按原框带宽 `B`，频率上下各留约 `0.1B` 余量，即零频两侧通带约为 `±0.6B`，并抑制邻频干扰。目标及余量须在原始 IQ 覆盖范围内，可采用接收机已有的 DDC/滤波实现。

**④ 重采样。**

```text
Fs_target = min(Fs, 4 × B)
```

`B` 为原框带宽。降低采样率时必须抗混叠滤波，不能简单隔点抽取。JSON 填实际输出采样率，接口要求其大于 `1.2B`；明显偏离目标值时双方确认。

**算例：** 上述框为 `20～160 ms`、`858512500～858537500 Hz`，带宽 `25 kHz`、中心 `858525000 Hz`。若原始采样率为 `320 kS/s`、采集中心为 `858500000 Hz`，则向下频移 `25 kHz`，通带约 `±15 kHz`，目标输出采样率 `100 kS/s`。

## 3. 提交 IQ 和配套 JSON

每框提交一段已完成上述处理的有效复数 IQ：

- 小端 `float32`，按 `I0,Q0,I1,Q1…` 交错排列，数值须有限。
- 保留 I/Q 相对幅度和相位，不分别做 RMS 归一化，不默认去直流或减均值。
- 至少 **2048 个有效复数点**；达到 **10240 点**支持五窗，不要求固定长度或固定时长，也无需接收机切窗。

同帧多条 IQ 可一次批量提交；`frame_id` 在请求顶层表单中提供一次。每条 IQ 的 JSON 有以下五个必填字段：

```json
{
  "signal_id": "signal_0",
  "dtype": "float32",
  "sample_rate_hz": 100000.0,
  "estimated_bandwidth_hz": 25000.0,
  "sample_count": 13900
}
```

`signal_id` 由接收机分配、对应原框且请求内唯一；`estimated_bandwidth_hz` 填原框带宽；`sample_count` 填实际有效复数点数，载荷字节数为 `sample_count × 2 × 4`。可选 `iq_start_time_s` 表示补偿延时后首个有效点相对瀑布图输入起点的时间。

分类服务负责选窗、归一化和推理，按 `frame_id`、`signal_id` 返回 `modulation_type`；过短或无有效功率时返回 `null`。接收机据此关联原框，无需再次提交检测 JSON。

完整接口见 [AMC API](AMC_API.md)，部署见 [整合服务说明](README.md)。
