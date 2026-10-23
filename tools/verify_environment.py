"""GPU forward/backward sanity check, not a classifier training run."""
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
import torch
from amc.config import CLASSES


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available in this environment.")
    torch.manual_seed(123)
    device = torch.device("cuda")
    module = torch.nn.Sequential(
        torch.nn.Conv1d(2, 32, 7, padding=3), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool1d(1), torch.nn.Flatten(), torch.nn.Linear(32, len(CLASSES))
    ).to(device)
    x = torch.randn(8, 2, 2048, device=device)
    labels = torch.arange(8, device=device) % len(CLASSES)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = module(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(logits).all() or any(
        p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()
    ):
        raise RuntimeError("GPU forward/backward produced invalid results.")
    report = {
        "passed": True, "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "mixed_precision_forward_backward": True, "output_shape": list(logits.shape),
        "finite_loss": float(loss.detach()), "classifier_trained": False,
        "packages": {name: version(name) for name in ("numpy", "scipy", "matplotlib", "PyYAML")},
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/environment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
