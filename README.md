# AMC

单信号 IQ 的 20 类调制识别项目，包含仿真数据生成、轻量一维 ResNet-18 训练与评估、冻结模型推理、长信号选窗融合，以及批量分类 HTTP 服务。

AMC 分类服务与 `umdetree/signal-detector:0.2.1` 整合，检测与分类共用 5555 端口。当前镜像为 `weitaoccc/signal-recognition:0.1.1`，扩展到 20 类，并增加阈值 0.5 的 `unknown` 输出。接收机负责逐框 IQ 预处理和结果关联。

0.1.1 已完成两条实采数据、六个窗口的容器完整流程验证，并发布至 Docker Hub。远端镜像摘要与本地验证版本一致，可按下文命令拉取。

## 快速导航

| 目标 | 入口 |
|---|---|
| 接收机部署及调用流程 | [部署与使用](docs/README.md) |
| 批量分类请求、响应及错误处理 | [AMC API](docs/AMC_API.md) |
| 按检测框准备单信号 IQ | [单信号 IQ 预处理参考](docs/单信号IQ预处理参考.md) |
| 模型类别、权重文件名和哈希 | [模型说明](models/README.md) |
| 新版仿真数据约定 | [20 类数据规范](docs/AMC20_DATASET.md) |
| 连续单信号的本地预处理与推理 | [长 IQ 推理](docs/LONG_IQ_INFERENCE.md) |
| 仿真指标及历史验证记录 | [结果索引](reports/README.md) |
| 工具用途及本地实验所需文件 | [工具说明](tools/README.md) |

## 模型与适用范围

- 输入：I/Q 两通道，每窗 2048 个复数采样点，float32。
- 网络：轻量一维 ResNet-18。
- 类别顺序：`AM、FM、ASK、BPSK、QPSK、8PSK、PI4_DQPSK、16QAM、64QAM、2FSK、4FSK、2GFSK、OFDM、OQPSK、DBPSK、DQPSK、MSK、GMSK、32QAM、256QAM`。
- 冻结权重文件名：`amc20_v1.pt`，约 3.76 MiB，类别顺序与哈希见[模型说明](models/README.md)。GitHub 不提供权重文件，已发布 Docker 镜像内置推理权重。
- 新版数据配置：[amc20_v1.yaml](configs/amc20_v1.yaml)；数据及训练说明见 [AMC20 数据集](docs/AMC20_DATASET.md)和[微调训练](docs/AMC20_TRAINING.md)。

| 独立仿真测试 | 准确率 |
|---|---:|
| 20 类单窗口，15/20/25/30 dB 合并，不拒绝 | 79.4050% |
| 20 类单窗口，0～30 dB 合并，不拒绝 | 71.9029% |

新版独立测试共 35,000 条。采用 0.5 阈值后，15～30 dB 的保留率为 90.12%，保留结果正确率为 83.98%。原 13 类的仿真、长信号和带宽评估作为历史结果保留在[结果索引](reports/README.md)。

以上为仿真结果。新版 OFDM 数据增加了 WLAN 风格、LTE 风格、多尺度、稀疏导频和局部频带结构；`OFDM` 输出表示调制家族，不区分具体协议。`unknown` 表示分数低于阈值。

## 接收机调用约定

```text
原始 IQ 缓存 → 瀑布图强度矩阵 → POST /v1/infer → 检测框
             → 接收机逐框截取、频移、滤波、重采样
             → 批量单信号 IQ＋各自参数 JSON → POST /v1/amc/infer
             → 接收机按 frame_id、signal_id 关联类别并显示
```

分类服务不重复频移、滤波或重采样，不需要原检测 JSON。每条信号最多选 5 个 2048 点窗口，跨信号合批推理，再逐信号平均分数。

公共响应只返回 `frame_id`、`signal_id` 和 `modulation_type`，不返回置信分数。最大平均分数 ≥0.5 返回调制类别，<0.5 返回字符串 `"unknown"`；过短或无有效功率返回 `null`。

接收机升级时增加 7 类名称的映射和显示，并将 `"unknown"` 显示为“未知调制”，与 `null` 分别处理。IQ 预处理、请求参数、接口路径和按编号关联结果的方式保持原样。

IQ 载荷必须是无文件头的小端 float32，按 `I0,Q0,I1,Q1…` 排列。每条 JSON 必填 `signal_id、dtype、sample_rate_hz、estimated_bandwidth_hz、sample_count`。目标输出采样率为 `min(原始采样率, 4×框带宽)`，JSON 填实际输出采样率；修改参数值不会替代实际重采样。

## Docker 部署

GPU 部署需要支持 NVIDIA GPU 容器访问的 Docker 环境。镜像内含两套模型，宿主机无需安装本项目训练环境。

```bash
docker pull weitaoccc/signal-recognition:0.1.1
docker compose up -d --pull never
```

在仓库根目录运行上述 Compose 命令。服务默认映射到主机 5555 端口，可通过 `SERVICE_PORT` 修改。Windows PowerShell 检查：

```powershell
curl.exe --fail http://localhost:5555/health
curl.exe --fail http://localhost:5555/ready
curl.exe --fail http://localhost:5555/v1/amc/ready
```

| 功能 | 接口 |
|---|---|
| 检测 | `POST /v1/infer` |
| 批量调制识别 | `POST /v1/amc/infer` |
| 检测就绪 | `GET /ready` |
| 分类就绪 | `GET /v1/amc/ready` |

0.1.1 的测试、容器验证及 [Docker Hub](https://hub.docker.com/r/weitaoccc/signal-recognition) 摘要见[发布记录](reports/signal_recognition_011_release/REPORT.md)。原 `0.1.0-gpu` 的核验记录继续保留在[历史结果](reports/README.md)。

## 本地开发环境

本地训练与开发采用 Python 3.12、PyTorch 2.9.1/CUDA 12.8。以下为 Windows Conda 示例：

```powershell
conda create -n amc python=3.12 pip -y
conda activate amc
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements.lock.txt
python -m pip install -r requirements-service.lock.txt
python -m pip install -e .
python -m pytest -q
```

如果已经有 `amc` 环境，直接激活即可，无需重复创建。依赖锁文件记录本地环境；GPU 可用性取决于实际硬件和驱动，推理可显式选 `cpu`。需要保存硬件自检结果时，先创建 `reports/` 再运行 `tools/verify_environment.py`。

整合镜像复用检测器的 Python 3.10.21 / PyTorch 2.9.0+cu128，Dockerfile 仅复制推理模块，不执行 `pip install -e .`。不要将本地 Python 3.12 的整套依赖安装进基础检测镜像。

### 启动独立分类服务

```powershell
python -m uvicorn amc.service:app --host 127.0.0.1 --port 5557 --workers 1
curl.exe --fail http://127.0.0.1:5557/v1/amc/ready
```

源码运行前需自行准备匹配的权重，放入 `models/amc20_v1.pt`，或通过 `AMC_CHECKPOINT` 指定绝对路径。以上命令在仓库根目录、按前述 `pip install -e .` 安装后运行；GitHub 和 Python wheel 均不包含权重。

独立服务只提供 AMC，不含检测接口，默认加载 `models/amc20_v1.pt`。`AMC_DEVICE` 可设为 `cpu`、`cuda` 或 `auto`，`AMC_SCORE_THRESHOLD` 默认 `0.5`，其余参数见[部署说明](docs/README.md)。批量调用客户端为 [client_amc_batch.py](tools/client_amc_batch.py)。

### 使用本地权重的推理示例

先准备 `models/amc20_v1.pt`；示例 IQ 由代码生成。

```python
import numpy as np
from amc.config import load_config
from amc.synthesis import generate_sample
from amc.inference import WindowClassifier

config = load_config("configs/baseline.yaml")
iq, metadata, _ = generate_sample("QPSK", "QPSK", 20, [20261023, 0], config)
classifier = WindowClassifier("models/amc20_v1.pt", device="cpu")
result = classifier.predict(iq)[0]
print(result["class_name"])
```

此示例生成一条 QPSK 仿真信号，用于检查新版模型加载。`WindowClassifier` 返回原始首选类别及内部诊断分数；HTTP 服务在多窗融合后应用 0.5 阈值。

## 仿真数据与训练复现

新版 20 类数据生成及微调步骤见 [AMC20 数据集](docs/AMC20_DATASET.md)和[AMC20 微调训练](docs/AMC20_TRAINING.md)。以下命令保留用于复现原 13 类基线。

仓库不含数据集和权重文件。默认先生成小批预览，正式数据需要显式指定 `--profile full`。

```powershell
python tools/generate_dataset.py --config configs/baseline.yaml --profile preview --output data/preview_ofdm_v1
python tools/validate_dataset.py data/preview_ofdm_v1 --report reports/preview_ofdm_validation.json
python tools/train.py --dataset data/preview_ofdm_v1 --output runs/preview_reproduction --epochs 2 --device cuda
```

正式规模为训练 91,000、验证 18,200、测试 27,300 个窗口，共 136,500 个；SNR 为 0、5、10、15、20、25、30 dB。需要复现正式实验时：

```powershell
python tools/generate_dataset.py --config configs/baseline.yaml --profile full --output data/full_v1
python tools/validate_dataset.py data/full_v1 --report reports/full_v1/validation.json
python tools/train.py --config configs/baseline.yaml --dataset data/full_v1 --output runs/full_reproduction --device cuda
python tools/evaluate.py --checkpoint runs/full_reproduction/best.pt --dataset data/full_v1 --output reports/full_reproduction/evaluation --device cuda
```

生成、训练及评估工具会检查输出路径，重跑请选择新的空目录。训练种子与配置保存在项目中，不保证跨硬件训练结果逐位一致。连续长信号和带宽评估入口见[长 IQ 推理](docs/LONG_IQ_INFERENCE.md)与[带宽评估](docs/BANDWIDTH_EVAL.md)。

## 从源码构建整合镜像

使用 [docker/Dockerfile](docker/Dockerfile)。先自行准备匹配的分类权重并放入 `models/amc20_v1.pt`，再准备原检测镜像和构建：

```powershell
docker pull umdetree/signal-detector:0.2.1
.\tools\build_signal_recognition.ps1
```

构建脚本会核对基础镜像身份；不一致时停止。该命令会在本地构建同名镜像，只有需要重新构建时执行。Dockerfile 默认也记录基础镜像摘要；本机校验后的构建脚本使用已核对的标签。

## 仓库内容与来源

- `src/`、`configs/`、`tests/`、`tools/`：项目实现、配置、测试和必要工具。
- `models/`：只纳入模型说明，权重文件保留在本地，不提交 GitHub。
- `reports/`：只纳入精选仿真结果与历史服务验证记录；原报告中提及的完整预测、日志或数据集可能只保留在本地，范围见[结果索引](reports/README.md)。
- `data/`、`runs/`、缓存、真实 IQ 与离线交付文件：不纳入版本管理。
- 检测器通过 `umdetree/signal-detector:0.2.1` 依赖，不在仓库中复制其实现和模型；保留文档的来源见 [SOURCE.md](docs/detector/SOURCE.md)。

当前仓库公开可见，暂未添加开源许可证；第三方检测器及保留文档的来源单独说明。
