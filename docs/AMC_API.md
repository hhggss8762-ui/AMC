# AMC 批量调制识别 API



## 1. 接口与服务检查

下文用 `BASE_URL` 表示服务根地址，例如 `http://<服务主机>:<映射端口>`，不包含 `/v1/amc/infer`。整合镜像默认端口为 `5555`，检测与分类共用服务地址。接入时先检查进程，再检查 AMC 就绪状态，最后提交分类请求。部署方法见 [README](README.md)，IQ 准备方法见 [预处理参考](单信号IQ预处理参考.md)。

| 方法 | 路径 | 用途 | 成功响应 |
|---|---|---|---|
| `GET` | `/health` | 检查服务进程是否存活 | HTTP 200，`status` 为 `ok` |
| `GET` | `/v1/amc/ready` | 检查 AMC 是否可用于分类；首次调用会加载并预热 | HTTP 200，`status` 为 `ready` |
| `POST` | `/v1/amc/infer` | 同一帧的多信号批量分类 | HTTP 200，返回逐信号结果 |

### GET /health

```json
{"status": "ok"}
```

`/health` 成功仅说明进程存活，接收机应以 `/v1/amc/ready` 成功作为分类可用的依据。

### GET /v1/amc/ready

成功示例（`device` 随部署配置变化）：

```json
{
  "status": "ready",
  "device": "cuda",
  "classes": [
    "AM", "FM", "ASK", "BPSK", "QPSK", "8PSK", "PI4_DQPSK",
    "16QAM", "64QAM", "2FSK", "4FSK", "2GFSK", "OFDM"
  ]
}
```

未能就绪时返回 HTTP 503，错误格式见第 4 节。首次非空分类请求也会触发加载，但建议接收机先调用就绪接口。

## 2. POST /v1/amc/infer：请求

请求使用 `multipart/form-data`。一帧只提交一个顶层 `frame_id` 文本字段；每条信号提交一对 `iq_N` 文件字段和 `metadata_N` JSON 文本字段。

| 字段 | 类型 | 必填 | 内容 |
|---|---|---|---|
| `frame_id` | 文本 | 是 | 接收机分配的帧编号，1～128 个字符，不能全为空白 |
| `iq_0` | 二进制文件 | 有第 0 条信号时 | 第 0 条信号的 IQ 数据 |
| `metadata_0` | JSON 文本 | 有第 0 条信号时 | 仅描述 `iq_0` 的参数 |
| `iq_1`、`metadata_1` | 文件、JSON 文本 | 有第 1 条信号时 | 第 1 条信号，以此类推 |

索引须从 `0` 连续编号，每个索引恰好包含一份 IQ 和一份元数据。不接受重复的 multipart 字段、其他字段、缺少配对或跳号。仅提交 `frame_id` 表示空帧，返回空结果列表。

### IQ 二进制格式

| 项目 | 约定 |
|---|---|
| 数据类型 | 小端 IEEE 754 `float32` |
| 排列 | `I0, Q0, I1, Q1, …`，复数定义为 `I + jQ` |
| 文件头 | 无 |
| 单个复数采样点 | 8 字节，即一个 I 和一个 Q |
| 载荷长度 | 必须恰好为 `8 × sample_count` 字节 |
| 数值 | 所有 I/Q 值必须有限，不允许 NaN 或 Inf |

上传的是上述原始二进制字节，不是 `.npy/.npz` 文件、文本或 JSON 数组。各信号可以使用不同采样率和不同采样点数；接口不要求固定的信号时长或固定点数。

### 每条信号的 metadata

`metadata_N` 必须是 UTF-8（不带 BOM）JSON **文本字段**，不能作为 JSON 文件附件上传。它只接受以下字段：

| 参数 | JSON 类型 | 必填 | 单位与要求 |
|---|---|---|---|
| `signal_id` | string | 是 | 接收机分配的信号编号，1～128 个字符，不能全为空白；同一请求内唯一，用于对应原检测框 |
| `dtype` | string | 是 | 固定为 `"float32"`，字节序与排列按上表 |
| `sample_rate_hz` | number | 是 | Hz；预处理后**实际输出采样率**，正的有限数值 |
| `estimated_bandwidth_hz` | number | 是 | Hz；原检测框估计带宽 `B`，未加滤波余量，正的有限数值 |
| `sample_count` | integer | 是 | 复数采样点数，非负整数；不是 I/Q 两路浮点数的总数 |
| `iq_start_time_s` | number 或 null | 否 | 秒；相对检测输入起点的 IQ 起始时间，已补偿预处理延时，须为非负有限值；省略或填 `null` 均可 |

不接受额外的 JSON 字段或重复键。`frame_id` 只放在顶层表单中，不放入 `metadata_N`。`iq_start_time_s` 供时间对齐使用，不参与类别判断。

发送端按 `min(源采样率, 4 × B)` 选择目标采样率，并将实际实现的输出采样率填入 `sample_rate_hz`。接口仅校验 `sample_rate_hz > 1.2 × estimated_bandwidth_hz`，不强制比值等于 4；该校验不代表服务替接收机执行了滤波或重采样。

以下两份 JSON 可分别保存为 `signal_0.json` 和 `signal_1.json`，用于第 5 节的双信号请求。数值用于演示格式，实际发送时必须与对应 IQ 一致。

`signal_0.json`：

```json
{
  "signal_id": "signal_0",
  "dtype": "float32",
  "sample_rate_hz": 100000.0,
  "estimated_bandwidth_hz": 25000.0,
  "sample_count": 13900,
  "iq_start_time_s": 0.0205
}
```

`signal_1.json`：

```json
{
  "signal_id": "signal_1",
  "dtype": "float32",
  "sample_rate_hz": 200000.0,
  "estimated_bandwidth_hz": 50000.0,
  "sample_count": 8192,
  "iq_start_time_s": 0.047
}
```

对应 `signal_0.iq` 应为 111200 字节，`signal_1.iq` 应为 65536 字节。点数仅为示例，必须填写实际有效 IQ 长度。

### 请求限制

| 项目 | 默认上限 |
|---|---|
| 每个请求的信号条数 | 64 条 |
| 整个 HTTP 请求体 | 268435456 字节，即 256 MiB，包含 multipart 字段及分隔符开销 |
| 单条 `metadata_N` | 16384 字节，即 16 KiB，按 UTF-8 字节数计算 |

信号条数和请求体上限可由部署方配置，见[部署说明](README.md)。接收机应以部署方确认的限制为准；超限返回 HTTP 413。

## 3. 成功响应与结果关联

HTTP 200 响应格式：

```json
{
  "frame_id": "frame_0001",
  "results": [
    {"signal_id": "signal_0", "modulation_type": "QPSK"},
    {"signal_id": "signal_1", "modulation_type": "FM"}
  ]
}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_id` | string | 原样返回请求中的帧编号 |
| `results` | array | 与提交信号逐条对应，按 `iq_0`、`iq_1` 等编号顺序返回 |
| `results[].signal_id` | string | 原样返回对应元数据中的信号编号 |
| `results[].modulation_type` | string 或 null | 调制类别；合法但无法分类的信号为 `null` |

接收机按 `frame_id` 和 `signal_id` 将结果回填到自己保存的检测框。响应只包含上述编号和类别，不返回原检测框、置信分数或原检测 JSON。示例类别仅用于说明响应结构，实际类别由上传的 IQ 决定。

### 可返回的 13 个类别

`AM`、`FM`、`ASK`、`BPSK`、`QPSK`、`8PSK`、`PI4_DQPSK`、`16QAM`、`64QAM`、`2FSK`、`4FSK`、`2GFSK`、`OFDM`。

`PI4_DQPSK` 表示 π/4-DQPSK。非空结果是这 13 类中的首选类别，不能作为可靠的未知调制拒识结论。

### 单条信号无法分类

信号少于 2048 个有效复数采样点，或选中窗口功率不足、数值幅度不可用时，该条返回 `null`。这是合法输入的分类结果，HTTP 状态仍为 200，其他信号仍可返回类别：

```json
{
  "frame_id": "frame_0001",
  "results": [
    {"signal_id": "signal_0", "modulation_type": "QPSK"},
    {"signal_id": "signal_1", "modulation_type": null}
  ]
}
```

服务从每条信号中均匀选取最多 5 个完整的 2048 点窗口，使用 I/Q 共用的 RMS 归一化，再合并该信号各窗口的结果。可用窗口不足 5 个时减少窗口数，不补零。接收机负责时间截取、频移、滤波和重采样；服务不额外去直流或减均值，发送端按预处理参考保留信号自身的载波分量。

### 空帧

请求只有 `frame_id`、没有任何 IQ/metadata 对时，返回 HTTP 200：

```json
{
  "frame_id": "frame_empty",
  "results": []
}
```

空帧成功仅表示请求已被接受，不代表 AMC 已就绪；就绪状态仍通过 `/v1/amc/ready` 检查。

## 4. 错误响应

错误响应统一使用 JSON 对象，`detail` 为错误说明字符串。例如，`iq_0` 没有对应的 `metadata_0` 时返回 HTTP 400：

```json
{
  "detail": "Each iq_N part must have exactly one metadata_N part."
}
```

| HTTP 状态 | 含义与常见原因 | 接收机处理 |
|---|---|---|
| `400` | 请求格式或数据不合法：缺字段、额外字段、重复字段或 JSON 键、索引跳号、重复 `signal_id`、JSON 类型/数值不合法、采样率不满足约束、IQ 字节数不符、IQ 包含 NaN/Inf 等；也包括非 multipart 请求及不支持的 `Content-Encoding` | 整批拒绝；依据 `detail` 修正请求后重新提交 |
| `413` | 整个请求体、信号条数或单条元数据超过上限 | 与部署方核对限制；若拆分请求，由接收机按帧编号汇总结果，不丢弃信号 |
| `503` | AMC 无法就绪或分类执行失败 | 本次没有分类结果；检查就绪状态后再提交 |

HTTP 413 示例：

```json
{
  "detail": "Request body exceeds the byte limit."
}
```

`GET /v1/amc/ready` 的 HTTP 503 示例：

```json
{
  "detail": "AMC classifier is not ready."
}
```

`POST /v1/amc/infer` 的 HTTP 503 示例：

```json
{
  "detail": "AMC inference failed; no classification results were returned."
}
```

400、413、503 均不返回部分信号结果。客户端应依据 HTTP 状态判断成功与失败，并将 `detail` 作为错误说明；不要依赖固定的错误字符串。`modulation_type: null` 属于第 3 节的成功响应，不属于此处错误。

## 5. 双信号调用示例

准备同一目录下的 `signal_0.iq`、`signal_0.json`、`signal_1.iq`、`signal_1.json`，其中 IQ 由接收机按[预处理参考](单信号IQ预处理参考.md)生成，JSON 可按第 2 节示例填写。

curl 的 `@` 将 IQ 作为文件上传，`<` 将 JSON 文件内容读成**文本字段**。请保留这一区别；`metadata_0=@signal_0.json` 会作为文件上传并被拒绝。让 curl 自动生成 multipart `Content-Type` 及 boundary。

### Linux / Bash

将 `BASE_URL` 替换为部署方提供的实际地址：

```bash
BASE_URL='http://<服务主机>:<映射端口>'

curl --fail-with-body "$BASE_URL/health"
curl --fail-with-body "$BASE_URL/v1/amc/ready"

curl --fail-with-body "$BASE_URL/v1/amc/infer" \
  --form-string 'frame_id=frame_0001' \
  --form 'iq_0=@./signal_0.iq;type=application/octet-stream' \
  --form 'metadata_0=<./signal_0.json;type=application/json' \
  --form 'iq_1=@./signal_1.iq;type=application/octet-stream' \
  --form 'metadata_1=<./signal_1.json;type=application/json'
```

### Windows / PowerShell

使用 `curl.exe`，避免旧版 PowerShell 将 `curl` 解释为其他命令。将 `$BASE_URL` 替换为实际地址：

```powershell
$BASE_URL = 'http://<服务主机>:<映射端口>'

curl.exe --fail-with-body "$BASE_URL/health"
curl.exe --fail-with-body "$BASE_URL/v1/amc/ready"

curl.exe --fail-with-body "$BASE_URL/v1/amc/infer" `
  --form-string 'frame_id=frame_0001' `
  --form 'iq_0=@.\signal_0.iq;type=application/octet-stream' `
  --form 'metadata_0=<.\signal_0.json;type=application/json' `
  --form 'iq_1=@.\signal_1.iq;type=application/octet-stream' `
  --form 'metadata_1=<.\signal_1.json;type=application/json'
```

`--fail-with-body` 在 HTTP 失败时保留服务端错误 JSON，并使 curl 以非零状态退出。空帧调用可保留 `frame_id`，省略全部 `iq_N` 和 `metadata_N` 字段。
