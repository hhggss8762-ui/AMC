# 100 kHz–20 MHz 带宽覆盖验证约定

本轮固定原有模型和预处理，新增独立测试数据，配置为 `configs/wideband_eval.yaml`。
八档目标名义带宽 100k、200k、500k、1M、2M、5M、10M、20M Hz；13 类、7 档 SNR、
每组合 10 段，共 7280 段，五窗共 36400 个选中窗口。
每档有 910 段，其中 15/20/25/30 dB 共 520 段。

## 实际生成步骤

1. 从目标名义带宽反推符号率、消息带宽、频偏或子载波间隔，再调用物理波形生成器，产生新的连续随机载荷。
2. 调制结构参数沿用训练家族：RRC rolloff=0.35，FSK 指数与 GFSK BT 原范围，AM 子类型与指数原范围，FM 偏移比例原范围，OFDM 仅 64/128 FFT、两种 CP 和三种星座。
3. 将生成的波形实际通过多相采样率转换，形成采集率约 5B、10B 或 20B 的输入；三种条件按样本索引轮转，不额外扩大样本总量。
4. 设置已知目标中心位置（±0.1Fs），加随机初相和 ±0.02B 残余频偏；带宽估计误差 ±10%。
5. 在采集率下生成复数白噪声，按处理后居中 16384 点完整有效区统一设定一次带内 SNR。原始复数 IQ 保存为 complex64。
6. 推理从原始 IQ 开始，沿用原预处理文件和模型检查点配置。单窗、五窗分别完整运行，不输入真实类别或真实 SNR。

这里验证的是同一调制家族改变发送速度后的适配与采集率转换，不是新增 Wi-Fi/LTE 等协议实现。
由于固定有效点数，宽带信号的有效时间更短，原始记录点数不会随着绝对带宽直接增加。
本轮不是固定 100 ms 帧的实时吞吐基准。

## 保留 512 设置

预处理 `resampling_max_denominator=512` 保持原样，采集率转换也使用相同分母上限。
每条记录都记录请求采样率与实际采样率，审计要求输出率相对偏差不超过 0.1%。
采集率至少约 5B，因而不会触及此前讨论的极大降采样比例；本轮也不代表已覆盖那些比例。

## 数据与命令

小批验证使用各带宽、各类、15 dB 每组合 2 段，共 208 段，与正式数据使用不同种子命名空间。
正式数据存入 `data/bandwidth_sweep_v1`，seed_namespace=4；旧数据不改动。
原始输入仍为 NPZ 中的 `raw_iq` 和 `input_metadata_json`，采集参数不包含标签；
真实类别、名义带宽、生成参数、种子和哈希保存在 `metadata.jsonl`。

在 AMC 目录、amc Conda 环境运行（生成与评估拒绝覆盖非空输出目录）：

```powershell
python tools/generate_bandwidth_sweep.py --profile full --output data/bandwidth_sweep_v1
python tools/validate_long_signals.py data/bandwidth_sweep_v1 --report reports/bandwidth_sweep_v1/validation.json
python tools/evaluate_long_signals.py --dataset data/bandwidth_sweep_v1 --output reports/bandwidth_sweep_v1/evaluation --device cuda
python tools/render_bandwidth_report.py
```

详细报告是 `reports/bandwidth_sweep_v1/BANDWIDTH_REPORT.md`，带宽分组与采集率分组指标在
`evaluation/metrics.json` 的 `by_bandwidth`。预测数组中包含带宽与采集率组别，便于独立复算。
逐类/逐带宽召回率只有每格 40 段良好 SNR 信号，2.5 个百分点的分辨率，不能把小幅随机变化解读为确定的带宽效应。

## 本轮完成结果

7,280 段生成、审计、配对推理均已完成。八档合并，良好 SNR 单窗 95.31%、五窗 98.27%；
八档分别的良好 SNR 五窗为 98.27%、98.27%、98.85%、98.46%、98.46%、98.08%、98.08%、97.69%。
全 SNR 五窗为 91.58%。原模型和预处理源码保持不变，输出率最大相对误差约 0.092%。
生成和评估目录已非空，重跑命令应选择新输出目录，避免把拒绝覆盖误认为程序异常。
