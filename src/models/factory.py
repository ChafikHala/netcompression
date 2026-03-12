from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.models.fcn import FCN
from src.models.mobilenet import MobileNetV1
from src.models.lenet import LeNet
from src.models.resnet20 import ResNet20

def build_model(cfg, num_classes: int) -> nn.Module:
    name = cfg.model.name.lower()

    if name == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)
    if name== "resnet20":
        return ResNet20(num_classes=num_classes)

    if name == "mobilenet_cifar":
        width_mult = float(getattr(cfg.model, "width_mult", 1.0))
        return MobileNetV1(
            num_classes=num_classes,
            width_mult=width_mult,
        )
    if name == "lenet":
        return LeNet(num_classes=num_classes)
    if name == "fcn":
        return FCN(
            input_shape=cfg.model.input_shape,
            hidden_dims=cfg.model.hidden_dims,
            num_classes=num_classes,
            dropout=float(cfg.model.dropout),
        )

    raise ValueError(f"Unsupported model: {cfg.model.name}")
