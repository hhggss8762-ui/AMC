# 项目工具

从仓库根目录、在已安装项目及相应依赖的 Python 3.12 环境下运行。各脚本参数使用 `python tools/<脚本名>.py --help` 查看。

| 用途 | 工具 |
|---|---|
| 仿真窗口生成与校验 | `generate_dataset.py`、`validate_dataset.py`、`render_preview.py` |
| 训练与独立测试 | `train.py`、`evaluate.py`、`render_training.py` |
| 窗口与连续单信号推理 | `infer.py`、`infer_long.py` |
| 长信号生成与配对验证 | `generate_long_signals.py`、`validate_long_signals.py`、`evaluate_long_signals.py`、`render_long_report.py` |
| 带宽覆盖验证 | `generate_bandwidth_sweep.py`、`render_bandwidth_report.py` |
| 批量 HTTP 客户端 | `client_amc_batch.py`，仅需 Python 标准库 |
| GPU 环境自检 | `verify_environment.py`，先创建 `reports/` |
| 整合镜像构建与运行时核对 | `build_signal_recognition.ps1`、`audit_integrated_runtime.py` |
| 历史站点采集本地适配 | `adapt_station_iq.py`、`audit_adapter_result.py`、`render_adapter_report.py` |
| 历史容器全流程回归 | `validate_signal_recognition.py` |

数据集、完整训练输出、真实 IQ 及内部 HTTP 样例未随仓库上传。`validate_signal_recognition.py` 依赖本机保存的 `reports/real_iq_adapter_v2/`、`reports/amc_service_v1/internal_inputs/` 等文件和运行中的容器，不能直接在干净克隆上完成历史回归。`render_bandwidth_report.py` 也需要原训练环境记录 `runs/full_baseline_v1/environment.json` 等输入。相关原始记录只在本地保留。

`audit_integrated_runtime.py` 会运行临时容器，并比较当前源码与镜像内容，适用于按当前源码重新构建的镜像。仓库文档已整理，不能要求它与未重建的已发布镜像逐文件相同。本次建仓不运行 Docker 构建、容器回归或发布操作。

历史离线打包工具仅留在本地，不纳入仓库。接收机部署采用 [Docker Hub 拉取流程](../docs/README.md)。
