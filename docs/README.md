# 信号检测与调制识别服务



## 1. 服务概述

镜像：`weitaoccc/signal-recognition:0.1.0-gpu`，运行平台为 Linux amd64。

本版已完成本机 GPU 容器验证，并于 2026-10-20 发布至 Docker Hub。部署时使用上述固定镜像名称和版本。

服务包含两项功能：

| 功能 | 输入 | 输出 |
|---|---|---|
| 信号检测 | 瀑布图二维强度矩阵及时间、频率信息；也兼容原版 IQ 输入 | 检测框 JSON |
| 调制识别 | 多条已预处理的单信号 IQ，每条配套一条参数 JSON | 各信号的编号和调制类别 |

检测部分以 `umdetree/signal-detector:0.2.1` 为基线，保留原检测接口。调制识别为新增独立接口。当前整合范围为检测与调制识别，不包含频谱占用度计算。

### 文档导航

| 文档 | 使用场景 |
|---|---|
| [批量调制识别 API](AMC_API.md) | 实现分类请求、读取响应、处理错误 |
| [单信号 IQ 预处理参考](单信号IQ预处理参考.md) | 根据检测框截取、频移、滤波和重采样 |
| [原版检测 API](detector/api.md) | 调用 0.2.1 的检测及其他已有接口 |
| [原版检测 README](detector/README.md) | 查看原版服务说明 |
| [原版发布说明](detector/BETA.md) | 查看检测策略及版本背景 |

`detector/` 中保留原版文档，包含历史协议模型示例。当前 0.2.1 检测配置为 `signal-localization-beta`、单类 `SIGNAL`；使用实际响应中的类别名称，不套用历史协议模型类别。

## 2. 接收机调用流程

```text
接收机缓存原始 IQ
    │
    ├─ 将同一时段的瀑布图强度矩阵提交检测接口
    │                         ↓
    │                    获取检测框 JSON
    │                         ↓
    └─ 按框在缓存 IQ 上进行时间截取、频移、滤波和重采样
                              ↓
             同一帧的多组“单信号 IQ＋参数 JSON”批量提交
                              ↓
                     一次返回各信号的调制类别
                              ↓
               接收机按编号关联原框，显示整帧识别结果
```

- 检测与分类由接收机分两次调用，不是检测接口自动完成后续分类。
- 原始 IQ 的缓存、逐框预处理、编号分配和结果显示由接收机负责。
- 分类请求不需要重新上传原检测 JSON、瀑布图或整段宽带 IQ。
- 每条 IQ 对应一条 JSON；分类响应只包含编号和调制类别，不重复返回检测框或检测分数。
- 分类服务一次返回本次请求中所有条目的结果。未能提取并提交的框，由接收机自行保留对应的显示状态。

## 3. 接口选择

整合镜像在同一个 HTTP 服务、同一个端口提供以下入口：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 进程存活检查 |
| GET | `/ready` | 原版检测模型加载、预热与就绪检查 |
| POST | `/v1/infer` | 原版结构化检测接口 |
| GET | `/v1/amc/ready` | 分类模型加载、预热与就绪检查 |
| POST | `/v1/amc/infer` | 新增批量调制识别接口 |

部署后以同一 `BASE_URL` 调用两套功能，例如 `http://192.168.1.100:5555`；地址仅为示例，请替换为实际部署地址。

本镜像默认地址为 `http://<部署主机>:5555`。客户端应允许配置主机地址和映射端口；此前独立分类服务的本地验证端口 `5557` 不属于本次镜像的部署配置。

## 4. Docker 部署方式

### 4.1 运行条件

- 具备 Docker 运行环境；GPU 部署需支持 NVIDIA GPU 容器访问。
- 从 Docker Hub 拉取镜像时，部署机器需能访问 Docker Hub。
- 镜像内置代码、检测权重及分类权重，接收机宿主机无需安装训练环境、Conda 或挂载模型目录。

### 4.2 拉取与启动

先从 Docker Hub 拉取指定版本；本机已有对应镜像时，可跳过此步，直接启动：

```bash
docker pull weitaoccc/signal-recognition:0.1.0-gpu
```

直接启动容器：

```bash
docker run -d --name signal-recognition --gpus all -p 5555:5555 -e MODEL_DEVICE=cuda -e AMC_DEVICE=cuda weitaoccc/signal-recognition:0.1.0-gpu
```

主机端口 `5555` 如已占用，可改为 `-p 5556:5555`，客户端相应改用 `5556`。端口不同不改变接口路径。

已有配套 `docker-compose.yml` 时，也可在该文件所在目录使用 Compose（与上述 `docker run` 方式二选一）：

```bash
docker compose up -d --pull never
docker compose ps
```

Compose 部署可通过同目录 `.env` 中的 `SERVICE_PORT=5556` 修改主机端口。镜像中的 Compose 文件位置见第 7 节。

### 4.3 检查服务

```bash
curl --fail http://localhost:5555/health
curl --fail http://localhost:5555/ready
curl --fail http://localhost:5555/v1/amc/ready
```

Windows PowerShell 中请使用 `curl.exe` 代替别名 `curl`。远程调用请将 `localhost` 替换为部署机器地址。

- `/health` 返回 `{"status":"ok"}`，只表示进程可访问。
- 两个就绪接口均应返回 HTTP 200，且 `status` 为 `ready`，再开始完整检测与分类联调。
- 首次就绪请求会触发对应模型加载与预热，可能比后续请求慢；不要把该次耗时当作稳定推理耗时。
- `/ready` 仅检查检测模型，不能代替 `/v1/amc/ready`。

### 4.4 查看日志、停止与更新

```bash
docker logs --tail 100 signal-recognition
docker logs -f signal-recognition
docker stop signal-recognition
```

Compose 部署可使用 `docker compose logs -f` 和 `docker compose down`。更新时使用新的固定镜像版本，并按交付说明重新创建容器；保留旧版本以便回退，不依赖 `latest` 判断版本。

## 5. 业务请求示例

以下使用 Linux/macOS Shell；PowerShell 的可复制示例见 [分类 API 文档](AMC_API.md)。将 `BASE_URL` 设置为实际运行的服务地址。

### 5.1 检测：瀑布图强度矩阵

将矩阵按原版检测协议保存为二进制 `tf.bin`，并将对应参数保存为 `tf_metadata.json`，参数示例：

```json
{
  "input_type": "tf_intensity",
  "dtype": "float32",
  "value_scale": "db",
  "shape": [2048, 256],
  "time_window_ms": 200.0,
  "freq_start_hz": 858360000.0,
  "freq_end_hz": 858640000.0
}
```

矩阵按 `[频率行, 时间列]` 排列，频率行从低频到高频，按行优先展开。参数与矩阵必须对应。200 ms 仅为示例，不限制实际帧时长。

```bash
BASE_URL=http://192.168.1.100:5555
curl --fail-with-body "$BASE_URL/v1/infer" -F 'data=@tf.bin;type=application/octet-stream' -F 'metadata=<tf_metadata.json;type=application/json'
```

默认沿用原版自适应检测。若业务使用 `specific_band` 策略提示，按原版检测文档约定填写，不因新增分类接口而改变。

### 5.2 分类：提交同一帧的多个单信号

收到框后，接收机先按[预处理参考](单信号IQ预处理参考.md)准备各条有效 IQ 及对应 JSON，例如 `signal_0.iq` 与 `signal_0.json`、`signal_1.iq` 与 `signal_1.json`：

```bash
curl --fail-with-body "$BASE_URL/v1/amc/infer" \
  --form-string 'frame_id=frame_0001' \
  -F 'iq_0=@signal_0.iq;type=application/octet-stream' \
  -F 'metadata_0=<signal_0.json;type=application/json' \
  -F 'iq_1=@signal_1.iq;type=application/octet-stream' \
  -F 'metadata_1=<signal_1.json;type=application/json'
```

`metadata_N` 是 JSON 文本字段（`<文件`），不是上传的 JSON 文件（`@文件`）。IQ 本身是无文件头的小端 float32 I/Q 交错二进制，每个复数点占 8 字节。各条数据可有不同点数和采样率，具体字段见分类 API。

响应示例：

```json
{
  "frame_id": "frame_0001",
  "results": [
    {"signal_id": "signal_0", "modulation_type": "QPSK"},
    {"signal_id": "signal_1", "modulation_type": null}
  ]
}
```

接收机按 `frame_id` 和 `signal_id` 找回原框，不依赖模型类别推断对应关系。`null` 表示该条数据无法完成分类，不是一个已识别的“未知调制”类别。

## 6. 分类配置与输入上限

以下配置已在本版镜像中生效；配套 Compose 显式选择 GPU。

| 配置项 | 默认值 | 用途 |
|---|---|---|
| `MODEL_DEVICE` | 部署示例设为 `cuda` | 原版检测设备 |
| `AMC_DEVICE` | `auto`；GPU 部署示例设为 `cuda` | 分类设备，可选 `cpu`、`cuda` |
| `AMC_WINDOW_BATCH_SIZE` | `64` | 内部每批网络窗口数，不是接收机必须提交的信号数 |
| `AMC_MAX_SIGNALS` | `64` | 单次分类请求的最大信号条数 |
| `AMC_MAX_REQUEST_BYTES` | `268435456` | 分类请求体上限，含 multipart 开销，256 MiB |
| `TF_MAX_ELEMENTS` | `20000000` | 原版 TF 矩阵最大元素数 |

分类单条参数 JSON 上限为 16 KiB。超过上限应先协调配置，不截掉信号或 IQ 数据后冒充完整输入。每帧框数超过当前信号条数上限时，可协调提高上限；如改为多次请求，则由接收机按帧编号收齐结果后显示。

## 7. 镜像内的文档与客户端示例

当前仓库源码构建后的镜像包含以下文件：

```text
/app/
├─ README.md
├─ docker-compose.yml
├─ docs/
│  ├─ README.md
│  ├─ AMC_API.md
│  ├─ 单信号IQ预处理参考.md
│  └─ detector/
│     ├─ README.md
│     ├─ api.md
│     └─ BETA.md
└─ examples/
   └─ client_amc_batch.py
```

已发布的 `0.1.0-gpu` 镜像保留当时的英文文件名 `RECEIVER_IQ_PREPROCESSING_REFERENCE.md`；仓库现统一使用中文文件名。此次文档整理未重建或更新已发布镜像，查看已发布镜像时以容器内实际文件为准。

部署人员不需要项目训练目录。可在 Docker Desktop 的容器文件视图中查看这些文件，也可从已启动的容器复制到选定的本地工作目录：

```bash
docker cp signal-recognition:/app/docs ./docs
docker cp signal-recognition:/app/examples ./examples
docker cp signal-recognition:/app/docker-compose.yml ./docker-compose.yml
```

客户端只使用 Python 标准库，读取接收机自己的数据，不附测试 IQ。在包含 `examples/` 的本地工作目录运行（Windows 可将 `python3` 替换为 `python`）：

```bash
python3 examples/client_amc_batch.py --url http://127.0.0.1:5555/v1/amc/infer --frame-id frame_0001 --signal signal_0.iq signal_0.json --signal signal_1.iq signal_1.json
```

重复 `--signal IQ文件 JSON文件` 即可增加信号。远程调用时替换主机地址，JSON 字段见 [分类 API](AMC_API.md)。

## 8. 首次联调顺序

1. 确认两个就绪接口正常，再用同一段数据的瀑布图和缓存 IQ 进行联调。
2. 检查一个框的时间、频率换算和预处理结果，确认输出采样率、点数、I/Q 顺序正确。
3. 批量提交同帧多个信号，核对每条结果与原框的对应关系；过短信号可返回 null。
4. 分别记录检测、接收机预处理、上传和分类耗时。模型首次加载另计，整帧时延需要在接收机实际流程中测量。
