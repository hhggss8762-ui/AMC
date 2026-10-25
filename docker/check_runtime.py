"""Build-time compatibility check; uses only the inherited runtime."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import torch


expected = {
    "torch": "2.9.0", "numpy": "2.2.6", "fastapi": "0.121.1",
    "starlette": "0.49.3", "python-multipart": "0.0.20", "uvicorn": "0.38.0",
}
for name, version in expected.items():
    actual = importlib.metadata.version(name).split("+")[0]
    if actual != version:
        raise RuntimeError(f"Unexpected inherited dependency {name}: {actual}, expected {version}")

checkpoint = Path("/models/amc20_v1.pt")
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if digest != "26c2606b81d599426d9786b5c4c12b434425a5dc39c374cfeafeb109d6bffbea":
    raise RuntimeError("AMC checkpoint checksum mismatch")
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
if payload["format_version"] != 1 or payload["config"]["dataset"]["window_samples"] != 2048:
    raise RuntimeError("Unexpected AMC checkpoint configuration")
if payload["classes"] != [
    "AM", "FM", "ASK", "BPSK", "QPSK", "8PSK", "PI4_DQPSK", "16QAM",
    "64QAM", "2FSK", "4FSK", "2GFSK", "OFDM",
    "OQPSK", "DBPSK", "DQPSK", "MSK", "GMSK", "32QAM", "256QAM",
]:
    raise RuntimeError("AMC class mapping mismatch")
if payload["classes"] != payload["config"]["classes"]:
    raise RuntimeError("AMC checkpoint/config class order mismatch")
manifest = {
    "python": sys.version, "torch_cuda": torch.version.cuda,
    "amc_checkpoint_sha256": digest,
    "amc_classes": payload["classes"],
    "packages": dict(sorted(
        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    )),
}
print(json.dumps(manifest, indent=2))
