"""Memory-mapped IQ reader; SNR is an evaluation attribute, never a model feature."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import SPLITS


class IQWindowDataset(Dataset):
    def __init__(self, root, split, expected_classes=None, window_samples=2048):
        self.root = Path(root)
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["status"] != "complete":
            raise ValueError("Dataset generation is incomplete.")
        self.classes = list(self.manifest["classes"])
        if expected_classes is not None and self.classes != list(expected_classes):
            raise ValueError("Dataset/config class order mismatch.")
        self._iq_path = self.root / split / "iq.npy"
        self._label_path = self.root / split / "labels.npy"
        self.iq = np.load(self._iq_path, mmap_mode="r", allow_pickle=False)
        self.labels = np.load(self._label_path, mmap_mode="r", allow_pickle=False)
        actual_counts = self.manifest.get("actual_counts_per_class_per_snr")
        if actual_counts is not None:
            # Version 2 permits intentionally more training windows for OFDM.
            counts = actual_counts[split]
            if set(counts) != set(self.classes) or any(not isinstance(v, int) or v < 1 for v in counts.values()):
                raise ValueError("Invalid per-class sample counts.")
            expected = sum(counts.values()) * len(self.manifest["snr_db"])
        else:
            expected = (len(self.classes) * len(self.manifest["snr_db"])
                        * self.manifest["counts_per_class_per_snr"][split])
        if self.iq.shape != (expected, 2, window_samples) or self.iq.dtype != np.float32:
            raise ValueError("Invalid model input shape or dtype.")
        if self.labels.shape != (expected,) or self.labels.dtype != np.int64:
            raise ValueError("Invalid labels.")
        if np.min(self.labels) < 0 or np.max(self.labels) >= len(self.classes):
            raise ValueError("Class ID outside valid range.")
        snrs = []
        with (self.root / split / "metadata.jsonl").open(encoding="utf-8") as stream:
            for row, line in enumerate(stream):
                m = json.loads(line)
                if row >= expected or m["row"] != row or m["split"] != split:
                    raise ValueError("Metadata rows do not match dataset.")
                if m["class_id"] != int(self.labels[row]) or m["class_name"] != self.classes[m["class_id"]]:
                    raise ValueError("Metadata/label class mismatch.")
                snrs.append(float(m["snr_db"]))
        if len(snrs) != expected or not np.isfinite(snrs).all():
            raise ValueError("Invalid SNR metadata.")
        self.snrs = np.asarray(snrs, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Copy the selected window: PyTorch must not write into a read-only mmap.
        x = torch.from_numpy(np.array(self.iq[index], copy=True))
        return x, int(self.labels[index]), float(self.snrs[index])

    def __getstate__(self):
        state = dict(self.__dict__)
        state["iq"] = state["labels"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.iq = np.load(self._iq_path, mmap_mode="r", allow_pickle=False)
        self.labels = np.load(self._label_path, mmap_mode="r", allow_pickle=False)
