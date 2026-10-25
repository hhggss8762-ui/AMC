"""Single-run 20-class fine tuning, validation selection, best weights only."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from .data_loader import IQWindowDataset
from .dataset import write_json
from .evaluation import evaluate_model, summarize_predictions, plot_confusion_matrix
from .model import build_model, load_classifier
from .training import save_checkpoint


def migrate_classifier(source_payload, config):
    """Copy the entire compatible backbone and match classifier rows by name."""
    source_classes = source_payload["classes"]
    if source_classes != source_payload["config"]["classes"]:
        raise ValueError("Source class mapping is inconsistent")
    if len(set(source_classes)) != len(source_classes) or len(set(config["classes"])) != len(config["classes"]):
        raise ValueError("Duplicate class names")
    model = build_model(config)
    dest, source = model.state_dict(), source_payload["model_state"]
    head = {"classifier.weight", "classifier.bias"}
    if set(dest)-head != set(source)-head:
        raise ValueError("Backbone keys differ")
    copied_tensors = 0
    for key in set(dest)-head:
        if dest[key].shape != source[key].shape:
            raise ValueError(f"Backbone tensor shape mismatch: {key}")
        dest[key].copy_(source[key])
        copied_tensors += 1
    migrated = []
    for old_index, name in enumerate(source_classes):
        if name not in config["classes"]:
            raise ValueError(f"Source class absent in destination: {name}")
        new_index = config["classes"].index(name)
        dest["classifier.weight"][new_index].copy_(source["classifier.weight"][old_index])
        dest["classifier.bias"][new_index].copy_(source["classifier.bias"][old_index])
        migrated.append({"class":name,"source_row":old_index,"destination_row":new_index})
    model.load_state_dict(dest, strict=True)
    return model, {"backbone_tensors_copied":copied_tensors,"migrated_classes":migrated,
                   "new_classes":[name for name in config["classes"] if name not in source_classes]}


def _probe(cpu_model, dataset, batch_size, device, amp, learning_rate, weight_decay):
    # A disposable clone avoids counting this check as unlogged training.
    model = deepcopy(cpu_model).to(device).train()
    indices = np.random.default_rng(20261025).choice(len(dataset),batch_size,replace=False)
    x = torch.stack([dataset[int(i)][0] for i in indices]).to(device)
    y = torch.tensor([int(dataset.labels[i]) for i in indices],device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda",enabled=amp,init_scale=1024.)
    for attempt in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits,y)
        if tuple(logits.shape) != (batch_size,len(dataset.classes)) or not torch.isfinite(loss):
            raise RuntimeError("Invalid preflight logits/loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite_gradients = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        accepted_scale = scaler.get_scale()
        # GradScaler skips an overflowing step and backs off, as intended by AMP.
        scaler.step(optimizer); scaler.update()
        if finite_gradients:
            break
    else:
        raise RuntimeError("Nonfinite preflight gradients after AMP scale backoff")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Nonfinite updated parameters")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"passed":True,"batch_size":batch_size,"logits_shape":list(logits.shape),
            "loss":float(loss.detach()),"finite_gradients":True,"optimizer_step_passed":True,
            "disposable_clone":True,"accepted_scale":accepted_scale,"scale_attempts":attempt+1}


def _metrics(raw, classes, levels):
    good = np.isin(raw["snrs"],levels)
    return summarize_predictions(raw["labels"][good],raw["predictions"][good],raw["snrs"][good],classes,levels)


def _rate(metrics, source, destination):
    classes = metrics["classes"]
    row = metrics["confusion_matrix"][classes.index(source)]
    return row[classes.index(destination)]/sum(row)


def finetune20(config, dataset_root, source_checkpoint, output, report_dir):
    config = deepcopy(config)
    t = config["training"]
    t.update(learning_rate=.0003,batch_size=64,max_epochs=50,early_stopping_patience=8,
             mixed_precision=True,optimizer="AdamW",scheduler="cosine",weight_decay=.0001,
             selection_metric="validation_accuracy_good_snr")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this authorized training run")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    output, report = Path(output), Path(report_dir)
    if output.exists() and any(output.iterdir()):
        if not {p.name for p in output.iterdir()} <= {"migration.json","preflight.json"}:
            raise FileExistsError(f"Refusing to overwrite a training run: {output}")
    output.mkdir(parents=True,exist_ok=True)
    report.mkdir(parents=True,exist_ok=True)
    root = Path(dataset_root)
    validation_audit = json.loads((root/"validation.json").read_text(encoding="utf-8"))
    if not validation_audit["passed"]:
        raise ValueError("Dataset validation did not pass")
    train_data = IQWindowDataset(root,"train",config["classes"],2048)
    valid_data = IQWindowDataset(root,"validation",config["classes"],2048)
    if len(train_data) != 154000 or len(valid_data) != 17500:
        raise ValueError("Unexpected dataset size")
    data_config = yaml.safe_load((root/"config_resolved.yaml").read_text(encoding="utf-8"))
    if data_config["preprocess"] != config["preprocess"]:
        raise ValueError("Dataset and inference preprocessing differ")
    source_sha = hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest()
    source = torch.load(source_checkpoint,map_location="cpu",weights_only=True)
    cpu_model,migration = migrate_classifier(source,config)
    cpu_model.eval()
    source_model = build_model(source["config"]).eval()
    source_model.load_state_dict(source["model_state"],strict=True)
    example = torch.stack([train_data[0][0],train_data[7000][0]])
    with torch.inference_mode():
        old_logits = source_model(example)
        new_logits = cpu_model(example)[:,[config["classes"].index(n) for n in source["classes"]]]
    # GEMM output dimensions can select a slightly different accumulation kernel.
    error = float((old_logits-new_logits).abs().max())
    if not torch.allclose(old_logits,new_logits,atol=2e-5,rtol=2e-5):
        raise AssertionError("Migrated old-class logits differ")
    migration["max_old_class_logit_error"] = error
    del source_model,source
    write_json(output/"migration.json",migration)
    amp = True
    fallback = False
    try:
        probe = _probe(cpu_model,train_data,64,device,amp,t["learning_rate"],t["weight_decay"])
    except torch.OutOfMemoryError:
        fallback = True
        t["batch_size"] = 32
    if fallback:
        # Leave the exception handler first so the failed probe's tensors are released.
        torch.cuda.empty_cache()
        probe = _probe(cpu_model,train_data,32,device,amp,t["learning_rate"],t["weight_decay"])
    probe["batch_size_fallback"] = fallback
    write_json(output/"preflight.json",probe)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = cpu_model.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader_args = dict(batch_size=t["batch_size"],num_workers=0,pin_memory=True)
    train_loader = DataLoader(train_data,shuffle=True,generator=generator,**loader_args)
    valid_loader = DataLoader(valid_data,shuffle=False,**loader_args)
    optimizer = torch.optim.AdamW(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=t["max_epochs"])
    scaler = torch.amp.GradScaler("cuda",enabled=amp,init_scale=probe["accepted_scale"])
    environment = {
        "python_executable":__import__("sys").executable,"torch":str(torch.__version__),
        "cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"device":"cuda",
        "seed":seed,"batch_size":t["batch_size"],"mixed_precision":amp,"num_workers":0,
        "dataset_root":str(root.resolve()),"dataset_manifest_sha256":hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest(),
        "source_checkpoint":str(Path(source_checkpoint).resolve()),"source_checkpoint_sha256":source_sha,
        "train_windows":len(train_data),"validation_windows":len(valid_data),
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "all_layers_trainable":all(p.requires_grad for p in model.parameters()),
        "class_sampling":"shuffle actual counts; OFDM 3x; no extra class weights or oversampling",
        "test_used_for_model_selection":False,"test_evaluated":False,"checkpoint_policy":"best-only weights with inference metadata; no last/epoch/optimizer checkpoint",
        "started_utc":datetime.now(timezone.utc).isoformat(),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
    }
    write_json(output/"environment.json",environment)
    (output/"config_resolved.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
    print(json.dumps({"event":"training_started","gpu":environment["gpu"],"preflight":probe,
                      "migration":migration,"settings":t},ensure_ascii=False),flush=True)
    best_accuracy,best_loss,best_epoch,stale = -1.,float("inf"),0,0
    history=[]
    started=time.perf_counter()
    levels=config["acceptance"]["snr_db"]
    write_json(output/"status.json",{"status":"running","epoch":0,"best_epoch":0})
    with (output/"history.jsonl").open("w",encoding="utf-8") as log:
        for epoch in range(1,t["max_epochs"]+1):
            epoch_started=time.perf_counter(); last_progress=epoch_started
            model.train()
            total_loss=torch.zeros((),device=device,dtype=torch.float64)
            total_correct=torch.zeros((),device=device,dtype=torch.int64)
            seen=0
            for step,(x,y,_) in enumerate(train_loader,1):
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                    logits=model(x)
                    loss=torch.nn.functional.cross_entropy(logits,y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                total_loss+=loss.detach().double()*len(y)
                total_correct+=(logits.detach().argmax(1)==y).sum()
                seen+=len(y)
                now=time.perf_counter()
                if now-last_progress>=20:
                    current_loss=float(total_loss)/seen
                    if not np.isfinite(current_loss):
                        raise RuntimeError("Nonfinite training loss")
                    progress={"status":"running","phase":"train","epoch":epoch,"max_epochs":t["max_epochs"],
                              "step":step,"steps":len(train_loader),"seen":seen,"train_windows":len(train_data),
                              "train_loss_so_far":current_loss,"elapsed_seconds":round(now-started,1),"best_epoch":best_epoch,
                              "best_validation_good_snr_accuracy":best_accuracy}
                    write_json(output/"status.json",progress)
                    print(json.dumps(progress),flush=True)
                    last_progress=now
            validation,raw=evaluate_model(model,valid_loader,device,config["classes"],levels,mixed_precision=amp)
            good=_metrics(raw,config["classes"],levels)
            accuracy,good_loss=validation["good_snr_accuracy"],validation["good_snr_loss"]
            improved=accuracy>best_accuracy or (accuracy==best_accuracy and good_loss<best_loss)
            if improved:
                best_accuracy,best_loss,best_epoch,stale=accuracy,good_loss,epoch,0
                payload={"format_version":1,"epoch":epoch,"classes":list(config["classes"]),"config":config,
                         "model_state":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                         "best_epoch":best_epoch,"best_validation_good_snr_accuracy":best_accuracy,
                         "validation_metrics":validation,"environment":environment,
                         "training_kind":"13-to-20 full-layer fine tune; best validation weights only"}
                save_checkpoint(output/"best.pt",payload)
                write_json(output/"best_validation_metrics.json",validation)
                write_json(output/"best_validation_good_snr_metrics.json",good)
                np.savez_compressed(output/"best_validation_predictions.npz",**raw)
                del payload
            else:
                stale+=1
            row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],"train_loss":float(total_loss)/seen,
                 "train_accuracy":int(total_correct)/seen,"validation_loss":validation["loss"],
                 "validation_accuracy":validation["accuracy"],"validation_good_snr_accuracy":accuracy,
                 "validation_good_snr_loss":good_loss,"validation_OFDM_recall":validation["per_class"]["OFDM"]["recall"],
                 "validation_AM_recall":validation["per_class"]["AM"]["recall"],
                 "validation_OFDM_to_AM":_rate(validation,"OFDM","AM"),"validation_AM_to_OFDM":_rate(validation,"AM","OFDM"),
                 "good_snr_OFDM_recall":good["per_class"]["OFDM"]["recall"],"good_snr_AM_recall":good["per_class"]["AM"]["recall"],
                 "good_snr_OFDM_to_AM":_rate(good,"OFDM","AM"),"good_snr_AM_to_OFDM":_rate(good,"AM","OFDM"),
                 "best_epoch":best_epoch,"improved":improved,"stale_epochs":stale,"seconds":time.perf_counter()-epoch_started}
            if not np.isfinite(row["train_loss"]):
                raise RuntimeError("Nonfinite training loss")
            history.append(row);log.write(json.dumps(row)+"\n");log.flush()
            scheduler.step()
            write_json(output/"status.json",{"status":"running","phase":"epoch_complete",**row})
            print(json.dumps({"event":"epoch_complete",**row}),flush=True)
            if stale>=t["early_stopping_patience"]:
                print(json.dumps({"event":"early_stopping","epoch":epoch,"best_epoch":best_epoch}),flush=True)
                break
    # Load and validate the actual retained artifact on validation only.
    retained,payload=load_classifier(output/"best.pt",device="cuda",expected_classes=config["classes"])
    final_metrics,final_raw=evaluate_model(retained,valid_loader,device,config["classes"],levels,mixed_precision=amp)
    final_good=_metrics(final_raw,config["classes"],levels)
    if final_metrics["good_snr_accuracy"] != best_accuracy or payload["epoch"]!=best_epoch:
        raise AssertionError("Retained best checkpoint does not reproduce selection metric")
    if any(output.glob("last.pt")) or len(list(output.glob("*.pt")))!=1:
        raise AssertionError("Only one best checkpoint may remain")
    summary={"status":"complete","epochs_completed":len(history),"best_epoch":best_epoch,
             "best_validation_good_snr_accuracy":best_accuracy,"best_validation_accuracy":final_metrics["accuracy"],
             "best_validation_loss":final_metrics["loss"],"elapsed_seconds":time.perf_counter()-started,
             "early_stopped":len(history)<t["max_epochs"],"best_checkpoint":str((output/"best.pt").resolve()),
             "best_checkpoint_sha256":hashlib.sha256((output/"best.pt").read_bytes()).hexdigest(),
             "checkpoint_files_retained":1,"peak_allocated_gpu_bytes":torch.cuda.max_memory_allocated(),
             "source_checkpoint_unchanged":hashlib.sha256(Path(source_checkpoint).read_bytes()).hexdigest()==source_sha,
             "test_evaluated":False,"unknown_threshold_calibrated":False,"service_updated":False,
             "retained_checkpoint_validation_reproduced":True,"environment":environment}
    if not summary["source_checkpoint_unchanged"]:
        raise AssertionError("Frozen checkpoint changed")
    write_json(output/"summary.json",summary)
    write_json(output/"status.json",{k:v for k,v in summary.items() if k!="environment"})
    _report(output,report,summary,history,final_metrics,final_good)
    print(json.dumps({k:v for k,v in summary.items() if k!="environment"}),flush=True)
    return summary


def _report(output,report,summary,history,metrics,good):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs=[r["epoch"] for r in history]
    fig,axes=plt.subplots(1,3,figsize=(17,4.5),constrained_layout=True)
    axes[0].plot(epochs,[r["train_loss"] for r in history],label="Train")
    axes[0].plot(epochs,[r["validation_loss"] for r in history],label="Validation")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].plot(epochs,[r["validation_accuracy"] for r in history],label="Validation 0-30 dB")
    axes[1].plot(epochs,[r["validation_good_snr_accuracy"] for r in history],label="Validation 15-30 dB")
    axes[1].set_ylabel("Accuracy")
    axes[2].plot(epochs,[r["validation_OFDM_recall"] for r in history],label="OFDM recall")
    axes[2].plot(epochs,[r["validation_AM_recall"] for r in history],label="AM recall")
    axes[2].plot(epochs,[r["validation_OFDM_to_AM"] for r in history],label="OFDM to AM")
    axes[2].set_ylabel("Validation fraction (0-30 dB)")
    for ax in axes:
        ax.set_xlabel("Epoch");ax.axvline(summary["best_epoch"],color="gray",ls="--",lw=1);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("AMC20 full-layer fine tune | validation selection only")
    fig.savefig(report/"training_curves.png",dpi=150);plt.close(fig)
    plot_confusion_matrix(metrics,report/"validation_confusion_matrix.png")
    plot_confusion_matrix(good,report/"validation_confusion_matrix_good_snr.png")
    write_json(report/"summary.json",summary)
    write_json(report/"validation_metrics.json",metrics)
    write_json(report/"validation_good_snr_metrics.json",good)
    table="\n".join(["| 类别 | 验证 0～30 dB 召回率 | 验证 15～30 dB 召回率 |","|---|---:|---:|",*
        [f"| {name} | {metrics['per_class'][name]['recall']:.2%} | {good['per_class'][name]['recall']:.2%} |" for name in metrics["classes"]]])
    report_text=f"""# AMC20 v1 微调训练结果

训练已完成。只保留一份最优权重：`{(output/'best.pt').resolve()}`。

由旧 13 类冻结模型迁移全部兼容骨干参数，按名称复制原 13 类输出行，新增 7 类输出随机初始化；所有层参与微调。使用独立仿真训练 154000 条、验证 17500 条，I/Q 双通道 2048 点。OFDM 训练量为其他单类的 3 倍，正常随机抽样，没有额外类别权重。未使用测试集和真实 IQ 训练或选择权重。

配置：AdamW，初始学习率 0.0003，weight decay 0.0001，Cosine，batch {summary['environment']['batch_size']}，AMP；最多 50 轮，验证 15～30 dB 准确率连续 8 轮未改善停止，同准确率时以较低验证损失打破平局。

- 完成轮数：{summary['epochs_completed']}；最佳轮次：{summary['best_epoch']}。
- 最佳权重验证 15～30 dB 准确率：**{summary['best_validation_good_snr_accuracy']:.4%}**。
- 最佳权重验证 0～30 dB 准确率：**{summary['best_validation_accuracy']:.4%}**。
- 验证 OFDM→AM：全 SNR {_rate(metrics,'OFDM','AM'):.2%}；15～30 dB {_rate(good,'OFDM','AM'):.2%}。
- 验证 AM→OFDM：全 SNR {_rate(metrics,'AM','OFDM'):.2%}；15～30 dB {_rate(good,'AM','OFDM'):.2%}。
- 训练与最终权重复核耗时：{summary['elapsed_seconds']/60:.2f} 分钟。
- 峰值 GPU 分配显存：{summary['peak_allocated_gpu_bytes']/1024**3:.2f} GiB。

这些是模型选择所用验证集结果，不是独立测试准确率，也不代表真实接收机效果。训练集的未扩频随机 BPSK/DBPSK、QPSK/DQPSK 分布存在重叠，细分能力有先验限制；各类结果如实记录。

{table}

权重 SHA256：`{summary['best_checkpoint_sha256']}`。最终保留的 checkpoint 已重新加载并复核验证指标。旧冻结权重 SHA256 未变。未校准 unknown 阈值，未评估测试集或真实数据，未更新服务、构建/发布 Docker、提交/推送 GitHub。
"""
    (report/"REPORT.md").write_text(report_text,encoding="utf-8")
