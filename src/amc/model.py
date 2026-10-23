"""A small real-valued ResNet-18 for two-channel IQ windows."""
import torch
from torch import nn


class BasicBlock1D(nn.Module):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(outgoing)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(outgoing)
        self.shortcut = (nn.Identity() if incoming == outgoing and stride == 1 else
                         nn.Sequential(nn.Conv1d(incoming, outgoing, 1, stride=stride, bias=False),
                                       nn.BatchNorm1d(outgoing)))

    def forward(self, x):
        residual = self.shortcut(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + residual)


class ResNet18IQ(nn.Module):
    def __init__(self, num_classes, stage_channels=(32, 64, 128, 256), window_samples=2048):
        super().__init__()
        if len(stage_channels) != 4 or min(stage_channels) < 1 or num_classes < 2:
            raise ValueError("Invalid model dimensions.")
        self.window_samples = int(window_samples)
        self.num_classes = int(num_classes)
        self.stem = nn.Sequential(
            nn.Conv1d(2, stage_channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(stage_channels[0]), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        incoming = stage_channels[0]
        stages = []
        for index, outgoing in enumerate(stage_channels):
            stages.append(nn.Sequential(
                BasicBlock1D(incoming, outgoing, stride=1 if index == 0 else 2),
                BasicBlock1D(outgoing, outgoing),
            ))
            incoming = outgoing
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(stage_channels[-1], num_classes)
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] != self.window_samples:
            raise ValueError(f"Expected [batch, 2, {self.window_samples}] IQ windows.")
        return self.classifier(self.pool(self.stages(self.stem(x))).flatten(1))


def build_model(config):
    if config["model"]["architecture"] != "resnet18_1d":
        raise ValueError("Only resnet18_1d is supported.")
    if config["model"]["input_channels"] != 2:
        raise ValueError("Model requires I/Q channels.")
    return ResNet18IQ(
        len(config["classes"]), tuple(config["model"]["stage_channels"]),
        config["dataset"]["window_samples"],
    )


def load_classifier(checkpoint, device="cpu", expected_classes=None):
    """Load our tensor/dict checkpoint with safe weights-only deserialization."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    classes = list(payload["classes"])
    if classes != payload["config"]["classes"]:
        raise ValueError("Checkpoint class mapping is inconsistent.")
    if expected_classes is not None and classes != list(expected_classes):
        raise ValueError("Checkpoint/dataset class order mismatch.")
    model = build_model(payload["config"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload
