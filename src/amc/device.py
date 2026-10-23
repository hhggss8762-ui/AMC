"""Device selection shared by inference and training, with no training imports."""
import torch


def resolve_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA are supported.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device
