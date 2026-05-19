"""Backbone builders for scratch face-recognition experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from facenet_pytorch import InceptionResnetV1
from torch import nn


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class IRBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class IRResNet(nn.Module):
    """InsightFace-style IR-ResNet for 112x112 face crops."""

    def __init__(self, layers: Sequence[int], embedding_size: int = 512, dropout: float = 0.4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.body = nn.Sequential(
            self._make_stage(64, 64, layers[0], stride=2),
            self._make_stage(64, 128, layers[1], stride=2),
            self._make_stage(128, 256, layers[2], stride=2),
            self._make_stage(256, 512, layers[3], stride=2),
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            Flatten(),
            nn.Linear(512 * 7 * 7, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        self._init_weights()

    def _make_stage(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [IRBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(IRBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)


def canonical_backbone_name(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    aliases = {
        "inceptionresnetv1": "inception_resnet_v1",
        "inception_resnet": "inception_resnet_v1",
        "inception_resnet_v1": "inception_resnet_v1",
        "ir18": "ir_resnet18",
        "ir_resnet18": "ir_resnet18",
        "ir34": "ir_resnet34",
        "ir_resnet34": "ir_resnet34",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported backbone: {name}")
    return aliases[normalized]


def build_backbone(name: str, pretrained_model: str | None = None) -> tuple[nn.Module, int, str]:
    canonical = canonical_backbone_name(name)
    if canonical == "inception_resnet_v1":
        model = InceptionResnetV1(pretrained=pretrained_model, classify=False)
    elif canonical == "ir_resnet18":
        model = IRResNet([2, 2, 2, 2])
    elif canonical == "ir_resnet34":
        model = IRResNet([3, 4, 6, 3])
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return model, 512, canonical
