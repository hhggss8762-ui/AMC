# AMC20 微调训练

本次使用 `data/amc20_v1` 的训练 154000 条、验证 17500 条，保持 2048 个复数采样点输入。测试集和真实 IQ 不参与本阶段训练、模型选择或报告。

先按数据集说明生成 `data/amc20_v1`，并在本地准备旧 13 类初始化权重 `models/amc_baseline_v1.pt`；GitHub 不提供权重文件。在项目根目录、Anaconda `amc` 环境执行：

```powershell
python tools/finetune_amc20.py --config configs/amc20_v1.yaml --dataset data/amc20_v1 --source models/amc_baseline_v1.pt --output runs/amc20_v1 --report reports/amc20_v1_training
```

## 固定设置

- 加载旧 13 类模型全部兼容骨干参数，原输出行按类别名称迁移；新增 7 类输出按随机种子初始化。全部层参与微调。
- 实际训练设置覆盖数据生成配置中预留的训练默认值：AdamW、初始学习率 0.0003、weight decay 0.0001、Cosine、最多 50 轮。
- GPU 混合精度，batch 64；单批预检查遇到显存不足时降到 32。混合精度初始梯度缩放 1024，并由 GradScaler 动态调整。
- 正常打乱实际训练样本，OFDM 已有三倍数量，没有额外类别权重、重采样或过采样。
- 使用 15、20、25、30 dB 验证准确率选择权重；准确率相同时，以更低的同范围验证损失打破平局。连续 8 轮不改善时提前停止。
- 运行前验证原 13 类迁移输出一致，再用独立模型副本执行一次实际前向、反向和优化器更新检查。副本检查不计入正式训练。

## 文件

- `runs/amc20_v1/best.pt`：唯一保留的最优权重，包含模型参数、20 类名称、配置及必要推理元数据；不保存优化器状态、逐轮权重或 last.pt。改善时原子替换同一文件。
- `history.jsonl`：每轮损失、准确率、OFDM/AM 召回率及相互误判比例。
- `status.json`：运行状态与最新进度。结束后状态为 complete。
- `migration.json`、`preflight.json`、`environment.json`、`config_resolved.yaml`：迁移、检查和实际参数记录。
- `best_validation_metrics.json`、`best_validation_good_snr_metrics.json`、`best_validation_predictions.npz`：最优权重对应的验证结果，改善时覆盖。
- `summary.json`：训练完成、最佳轮次、权重摘要和保留文件数量。
- `reports/amc20_v1_training`：日志、训练曲线、验证混淆矩阵及训练报告；不额外复制权重。

完成后会重新加载实际保留的 best.pt，复核验证集选择指标。已有完整训练目录不允许覆盖；训练只保留最优权重，没有逐批精确续训所需的优化器检查点。

这些验证结果用于模型选择。随机未扩频 BPSK/DBPSK、QPSK/DQPSK 的可辨识性限制见 `AMC20_DATASET.md`。最优权重已冻结为 `models/amc20_v1.pt`，独立测试结果见[模型说明](../models/README.md)；服务已接入 20 类和 0.5 的 unknown 阈值，接口见 [AMC API](AMC_API.md)。
