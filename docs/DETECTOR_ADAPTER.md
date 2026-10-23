# 检测框到 AMC 的本地适配

> 本文记录 2026-10-16 的本地实验及对应工具。当前接收机接口采用[批量分类 API](AMC_API.md)，预处理由接收机负责；本文的框内分数和本地富化响应不是公共分类接口格式。实采 IQ、原始诊断和实验结果未纳入仓库。

此阶段连接原版 `umdetree/signal-detector:0.2.1` HTTP 服务和本地冻结 AMC。
不是整合后的镜像，也尚未修改原版 HTTP 入口。训练数据和模型不变。

## 输入与返回

- 站点目录：恰好一个 `Iq_*.ix` 和同名 `.md`。50 字节索引记录，参数须一致，
  序号连续，所有偏移/点数与载荷完全对应。读取小端 int16 交错 I/Q，构造 I+jQ。
  这些约定沿用 SpectrumOccupancy/tools/diagnose_921data.py。
- 整条载荷上传，实际时长 N/Fs；不固定 100 ms、不截断、不按设备包时间戳填零。
- 如果索引/载荷不一致、参数发生变化或检测响应标定与原数据不匹配，停止报错。
- 未指定 `--specific-band` 时维持原版自适应检测；不由文件名猜测策略。
- `DetectionIQAdapter.enrich(iq, sample_rate_hz, center_freq_hz, detector_response)`
  返回 `(response, diagnostics)`。
- `response` 深拷贝原响应，每个 detection 仅新增 `modulation_type`、
  `modulation_score`。原 model/class_names/xyxy/confidence/calibration 均不改。
- 无足够采样点、零功率、频带或所需频率余量超出捕获范围等预期数据限制：
  两字段置 null，在 diagnostics 记录原因。低分仍返回候选类别/分数并记录 uncertain。
  未预期的运行时异常会抛出，不伪造正常结果。
- 分类得分未校准；没有可靠的未知调制拒识能力。没有原始 IQ 的 TF/图片输入，
  本地工具不支持分类，后续服务接入需显式返回不可用。

## 坐标与处理

使用 0.2.1 返回的最终整幅图 xyxy 和 calibration；x 代表时间，y 向下频率降低。
不再次应用其内部 STFT 时间投影或内部频率切窗映射。

1. 对裁切用的坐标限制在图像边界内，但不改响应中的原坐标。
2. `t1=x1/W*T; t2=x2/W*T`；
   `fhi=f_end-y1/H*(f_end-f_start); flo=f_end-y2/H*(f_end-f_start)`。
3. `B=fhi-flo; center_offset=(flo+fhi)/2-capture_center`。
4. 根据冻结 FIR 长度与输出边缘保护量，向时间两侧读取原始上下文，受文件边缘限制。
5. 复用 `prepare_long_iq`：频移、FIR、resample_poly、丢弃原定边缘保护点。
   频率每侧余量 0.1B；请求输出率 min(Fs,4B)；有理近似分母上限 512 不变，
   实际输出率/误差记录在 diagnostics。极端采样率比仍受原配置限制。
6. 按实际输出采样率裁回原框时间范围，要求整个分类窗口的起止时间都在框内，
   上下文不进入选窗；左右端分别向内取整。
   文件端点缺上下文时，只用实际存在的有效输出，不补零伪造长度。
7. 最多选 min(5,floor(L/2048)) 个互不重叠窗口，均匀覆盖有效目标时间。
   不强制 16384 点，不额外推理评估用的中间窗口。
8. 复用现有窗口分类器：I/Q 共用 RMS 缩放、两通道 float32，所有类别分数等权平均。
   模型常驻；当前跨框串行，同框窗口批量推理。

公开 Python 用法：

```python
from amc.station_capture import load_station_capture
from amc.detector_adapter import DetectionIQAdapter

iq, metadata, source = load_station_capture(capture_directory)
adapter = DetectionIQAdapter("models/amc_baseline_v1.pt", device="cuda")
response, diagnostics = adapter.enrich(
    iq, metadata["sample_rate_hz"], metadata["center_freq_hz"], detector_response)
```

## 运行

在 AMC 根目录的已激活 amc Conda 环境中：

```powershell
python tools/adapt_station_iq.py --capture 'D:\captures\station_capture' --detector-url http://127.0.0.1:15555 --output reports/real_iq_adapter_v3 --device cuda
```

需有正在运行的原版检测器。构建阶段的临时容器名为
`amc-detector-021-validation`，仅映射本机 15555→5555；验证后已停止。
需要时使用 `docker start amc-detector-021-validation`，等待 `/ready` 返回 ready。
工具不会安装依赖、更改采集源、启动或停止检测服务；urllib 绕过代理访问指定地址。

输出目录不可已存在。原始响应、请求元数据、数据源哈希、富化响应、逐框诊断和摘要独立保存。
也可将 `--detector-url` 换为 `--detector-json` 重放完全对应本条完整 IQ 的已有响应。
标定检查能发现时长/采样率/频带不匹配，但不能证明同采集参数的两个文件身份相同；
重放时必须核对随原响应保存的 source.json 中载荷 SHA256。

```powershell
python tools/audit_adapter_result.py --report reports/real_iq_adapter_v3
python tools/render_adapter_report.py --report reports/real_iq_adapter_v3
python -m pytest -q
```

## 本次结果与交付边界

首条实测完整数据：5,537,792 复数点，320 kS/s，863.5125 MHz，17.3056 s。
默认检测返回 11 框，全部完成五窗分类；原字段和源 IQ 哈希均保持一致。
最终已审计结果见 `reports/real_iq_adapter_v2/REPORT.md`。
v1 为初次 HTTP 记录，收尾审计发现端点取整可多覆盖不足一个输出采样周期；
已改为严格向内取整，v2 复用同一原始响应重跑全部 11 框、55 窗并通过独立边界审计。
原始框、采集数据和模型未改，预测类别分布未变；全部 125 项测试通过。

这条数据没有真实调制标注，不能测真实准确率。不同窗口和不同信号的预测可能不同，
高分不证明类别正确。设备时间戳有抖动/重复，与 N/Fs 轴不同；本次保留完整样本序列，
无法仅据文件索引证明 RF 采样绝对连续。

当前预处理每个框的整段 IQ，11 个长目标初次适配约 9.68 s，最终重放约 7.84 s，主要为 CPU 预处理，
不代表每框恒定 10 ms。未实施跨框批处理、按选窗局部预处理或整合容器部署。
后续接 HTTP 服务时优先复用已解码 IQ，加载一次 AMC，保留相同两字段的返回约定，
再做镜像内 Python/PyTorch 兼容与接收机端验证。

频谱核对发现：部分框位于强信号之间的弱能量区域或邻频边缘，而模型返回高分 AM。
尚不能证明这些框都是独立有效目标，也不能将高分当作正确率。结果已全部保存，
等待用户决定先排查框/邻频影响还是接受当前原样 baseline；未继续接入或制作整合镜像。
