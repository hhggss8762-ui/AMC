"""Window-only classification; raw wideband IQ extraction is outside this stage."""
import numpy as np
import torch
from .model import load_classifier
from .device import resolve_device


class WindowClassifier:
    def __init__(self, checkpoint, device="auto", score_threshold=0.6):
        if not 0 <= score_threshold <= 1:
            raise ValueError("Score threshold must be in [0,1].")
        self.device = resolve_device(device)
        self.model, self.checkpoint = load_classifier(checkpoint, self.device)
        self.classes = self.checkpoint["classes"]
        self.window_samples = self.checkpoint["config"]["dataset"]["window_samples"]
        self.threshold = score_threshold

    @torch.inference_mode()
    def predict(self, windows, batch_size=64):
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        x = np.asarray(windows)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[1:] != (2, self.window_samples) or len(x) == 0:
            raise ValueError(f"Expected [N,2,{self.window_samples}] or [2,{self.window_samples}].")
        if np.iscomplexobj(x) or not np.isfinite(x).all():
            raise ValueError("Input must be finite real I/Q channels.")
        results = []
        for start in range(0, len(x), batch_size):
            batch = np.array(x[start:start + batch_size], dtype=np.float32, copy=True)
            # A single shared complex RMS scale; preserve I/Q relation and carrier.
            power = np.mean(np.sum(batch.astype(np.float64) ** 2, axis=1), axis=1)
            if np.any(power <= 1e-12):
                raise ValueError("Input contains a zero-power window.")
            batch /= np.sqrt(power)[:, None, None].astype(np.float32)
            logits = self.model(torch.from_numpy(batch).to(self.device))
            scores = torch.softmax(logits.float(), dim=1).cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError("Nonfinite inference output.")
            for i, row in enumerate(scores):
                class_id = int(row.argmax())
                score = float(row[class_id])
                results.append({
                    "row": start + i, "class_id": class_id, "class_name": self.classes[class_id],
                    "score": score, "status": "classified" if score >= self.threshold else "uncertain",
                    "class_scores": {name: float(value) for name, value in zip(self.classes, row)},
                })
        return results
