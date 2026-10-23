from pathlib import Path
import numpy as np
import pytest
import torch
from amc.config import load_config
from amc.data_loader import IQWindowDataset
from amc.evaluation import summarize_predictions
from amc.inference import WindowClassifier
from amc.model import build_model, load_classifier

CONFIG = load_config(Path(__file__).parents[1] / "configs/baseline.yaml")


def test_checkpoint_roundtrip_preserves_logits_and_rejects_wrong_class_mapping(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(3)
    model = build_model(CONFIG).eval()
    x = torch.randn(2, 2, 2048)
    with torch.no_grad():
        before = model(x)
    assert before.shape == (2, 13)
    path = tmp_path / "roundtrip.pt"
    torch.save({"format_version": 1, "epoch": 0, "classes": CONFIG["classes"],
                "config": CONFIG, "model_state": model.state_dict()}, path)
    loaded, _ = load_classifier(path)
    with torch.no_grad():
        assert torch.equal(before, loaded(x))
    with pytest.raises(ValueError, match="class order"):
        load_classifier(path, expected_classes=list(reversed(CONFIG["classes"])))
    classifier = WindowClassifier(path, device="cpu")
    a = classifier.predict(x.numpy())
    b = classifier.predict(x.numpy() * 5)
    assert [p["class_id"] for p in a] == [p["class_id"] for p in b]
    assert np.allclose([p["score"] for p in a], [p["score"] for p in b], atol=1e-6)
    with pytest.raises(ValueError, match="zero-power"):
        classifier.predict(np.zeros((2, 2048), dtype=np.float32))


def test_metrics_count_all_samples_and_separate_good_snr():
    metrics = summarize_predictions(
        [0, 1, 1, 2], [0, 0, 1, 2], [0, 15, 20, 30],
        ["A", "B", "C"], [15, 20, 25, 30],
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["good_snr_count"] == 3
    assert metrics["good_snr_accuracy"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert metrics["per_class"]["B"]["recall"] == 0.5
    assert metrics["all_samples_counted"]


def test_reader_keeps_snr_out_of_iq_and_enforces_classes():
    root = Path(__file__).parents[1] / "data/preview_ofdm_v1"
    if not root.exists():
        pytest.skip("Preview dataset is optional.")
    dataset = IQWindowDataset(root, "train", CONFIG["classes"])
    assert len(dataset) == 364
    x, label, snr = dataset[0]
    assert x.shape == (2, 2048) and x.dtype == torch.float32
    assert 0 <= label < 13 and snr in CONFIG["dataset"]["snr_db"]
    with pytest.raises(ValueError, match="class order"):
        IQWindowDataset(root, "train", list(reversed(CONFIG["classes"])))
