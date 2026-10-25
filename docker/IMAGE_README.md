# 信号检测与调制识别服务

镜像：`weitaoccc/signal-recognition:0.1.1`。

本版扩展到 20 类，新增 `OQPSK、DBPSK、DQPSK、MSK、GMSK、32QAM、256QAM`。分类最大平均分数低于 0.5 时返回字符串 `"unknown"`；过短或无有效功率的数据仍返回 `null`。

接收机增加新增类别和“未知调制”的显示即可，原 IQ 预处理、请求格式、接口路径及结果关联方式保持原样。检测服务继续使用 0.2.1，检测和分类共用 5555 端口。

完整部署和调用说明见 [部署文档](https://github.com/hhggss8762-ui/AMC/blob/main/docs/README.md)。

- [调制识别 API](https://github.com/hhggss8762-ui/AMC/blob/main/docs/AMC_API.md)
- [单信号 IQ 预处理参考](https://github.com/hhggss8762-ui/AMC/blob/main/docs/%E5%8D%95%E4%BF%A1%E5%8F%B7IQ%E9%A2%84%E5%A4%84%E7%90%86%E5%8F%82%E8%80%83.md)
- [原版检测 API](https://github.com/hhggss8762-ui/AMC/blob/main/docs/detector/api.md)

容器本地文档位于 `/app/docs/`，预处理参考为 `/app/docs/单信号IQ预处理参考.md`，启动配置为 `/app/docker-compose.yml`，批量客户端为 `/app/examples/client_amc_batch.py`。
