# 批量调制分类 API：步骤 1–3 验证记录

完成范围：接口协议、接收已预处理 IQ 的批量分类模块、独立 HTTP API。没有修改 0.2.1 检测服务，也未构建 Docker。

## 已实现接口

- `GET /health`：进程状态。
- `GET /v1/amc/ready`：加载/预热一次，之后模型常驻。
- `POST /v1/amc/infer`：一帧提交多组 `iq_N` 文件与 `metadata_N` JSON 文本，零起始连续编号。
- 返回仅含 `frame_id`、各 `signal_id` 和 `modulation_type`；不需要原检测 JSON，不返回检测框或公共分数。
- 服务不频移、不滤波、不重采样、不减均值；只做校验、选窗、RMS、批量推理和逐信号融合。

## 验证结果

- 完整测试：212 passed；HTTP 边界测试 52 项，新增预处理后批量分类测试 35 项，原有测试 125 项。结果见 tests.txt。
- 真实冻结权重通过本机 Uvicorn + HTTP 运行，标准库客户端成功调用。
- 五信号混合请求：三条可分类信号返回 AM/FM/QPSK，过短和零功率条目为 null；每条都保留编号、顺序。该例为内部仿真验证，不是实采准确率估计。
- HTTP 输出与本地同样窗口推理输出一致，空帧返回空列表。
- 14 条信号合计 70 窗，默认 64 窗上限自动分成两次网络前向，最终同一响应内返回全部条目。
- 无效元数据、重复编号、字段配对、尺寸不符、NaN/Inf、请求超限、截断 multipart、模型失败和并发模型锁已验证。深层嵌套 JSON 返回400，而非500。
- 内部诊断记录分类分数和耗时，公共响应仅返回类别；正常 INFO 不输出全部类别向量，DEBUG 可查看。
- pip check 无依赖冲突；PyTorch、冻结权重及预处理参数未改变。

## 单次本机性能诊断

- GPU：RTX 5070 Laptop；设备 cuda。
- 五信号（8 个有效推理窗口、另两条返回 null），预热后 5 次 HTTP 请求中位数 23.54 ms，最大 37.45 ms。
- 14 信号 / 70 窗的单次 HTTP 请求 67.67 ms。
- 测量包含本地上传、请求解析、输入校验、选窗、分类和响应；不包含接收机预处理、检测服务、客户端读取文件、模型冷启动。仅为小规模诊断，不能视为接收机整帧时延承诺或稳定吞吐/P95 基准。

## 产物与状态

- API 文档：`docs/AMC_API.md`；短版预处理参考：`docs/RECEIVER_IQ_PREPROCESSING_REFERENCE.md`。
- 客户端：`tools/client_amc_batch.py`，使用接收机自己的 IQ 和各自 JSON 文件。
- `response.json` / `client_response.json`：本次真实 HTTP 的类别响应。
- `local_diagnostics.json` / `bulk_diagnostics.json`：内部窗口与融合结果。
- `summary.json` / `environment.json` / `tests.txt`：计时、环境、完整测试。
- `internal_inputs/` 仅为本地集成测试中从已有独立仿真记录准备的小型测试夹具，不作为给接收机的参考 IQ 数据交付。
- 冻结权重 SHA256：`4f22f777fb859c406fab04b59cfdf67805c1145031ec34a3d10b5af3748d1da0`。
- 临时验证服务器结束后停止；重新启动方式见 API 文档。后续步骤为与原版 0.2.1 合并镜像及部署回归，尚未执行。
