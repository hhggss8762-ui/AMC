# 2026-10-20 文档修正与本地镜像更新

本次范围为修正文档、更新本地镜像并验证。未上传 Docker Hub，未生成离线包，未清理其他文件。

镜像标签保持 `weitaoccc/signal-recognition:0.1.0-gpu`。

- 更新前：`sha256:9f15053a124015470d23da977f474c98ec10bee5b89736b828e784a6c8a4d780`。
- 更新后：`sha256:1a45d1b7824f5dfade99c08681fa7d5f4f1f5d24493c19f302e7dd695ae48d48`。

## 文档修改

`docs/README.md` 删除离线镜像交付、docker load、压缩包/校验清单等说明，
改为镜像发布后从 Docker Hub 拉取及 Docker/Compose 启动；已有本地镜像可直接启动。
镜像目录说明改为实际 `/app/` 内容，补充从运行容器复制文档和客户端的方法。

`docs/RECEIVER_IQ_PREPROCESSING_REFERENCE.md` 删除末尾的离线交付说明，
本地项目 `README.md` 同步更新当前部署说明。
原版检测文档及 AMC API 的字段和协议保持原样；原版的 offline matrix 指预计算矩阵，予以保留。

## 构建与验证

- 使用已有构建脚本与原标签增量构建，推理模块、权重和运行时检查步骤全部命中缓存。
- 新建容器已切换至更新后的镜像 ID，5555 端口服务正常，健康状态为 healthy。
- 对比前后镜像的文件哈希，实际仅两份内置文档变化：`/app/docs/README.md`、
  `/app/docs/RECEIVER_IQ_PREPROCESSING_REFERENCE.md`。镜像运行配置、推理代码及客户端保持一致。
- 原检测源文件、配置和权重共 15 份哈希一致，122 个依赖包版本全部保持一致。
- 17 份镜像内源码/文档/配置/客户端文件与本地构建源一致；15 个 JSON 示例及 14 个本地文档链接检查通过。
- 三种输入的完整检测 JSON 与既有原版 0.2.1 对照记录一致：TF 自适应 14 框、
  TF 指定频段策略 14 框、完整原始 IQ 11 框。
- 五条不同长度与功率的分类输入得到 AM、FM、QPSK、null、null；14 条重复 AM 条目验证 70 窗跨批处理。
  空帧、格式错误 400、体积/信号条数超限 413 检查通过。
- 容器内四份入口/主要文档均无离线交付说明；delivery 目录在操作前后均不存在。

首次探测发生在新容器服务启动完成前，连接被关闭；待两个模型就绪检查通过后重新执行 HTTP 回归，全部通过。
首次探测记录保留在 `http_validation.log`，就绪后的通过记录为 `http_validation_ready.log`。
完整 214 项自动检查结果沿用上一轮；本轮针对文档变化执行文件核验与容器回归。

详细证据：`source_checks.json`、`build.log`、`runtime_audit.json`、
`integrated_summary.json`、`summary.json`、`final_container_inspect.json`。
