# 信号检测与调制识别服务

镜像：`weitaoccc/signal-recognition:0.1.0-gpu`。

完整部署和调用说明见 [部署文档](https://github.com/hhggss8762-ui/AMC/blob/main/docs/README.md)。

- [调制识别 API](https://github.com/hhggss8762-ui/AMC/blob/main/docs/AMC_API.md)
- [单信号 IQ 预处理参考](https://github.com/hhggss8762-ui/AMC/blob/main/docs/%E5%8D%95%E4%BF%A1%E5%8F%B7IQ%E9%A2%84%E5%A4%84%E7%90%86%E5%8F%82%E8%80%83.md)
- [原版检测 API](https://github.com/hhggss8762-ui/AMC/blob/main/docs/detector/api.md)

容器本地文档位于 `/app/docs/`，启动配置为 `/app/docker-compose.yml`，批量客户端为 `/app/examples/client_amc_batch.py`。已发布镜像内的预处理文档使用原英文名，当前源码构建使用中文文件名；仓库文档整理没有更新已发布镜像。
