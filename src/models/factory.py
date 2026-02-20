from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.models.fcn import FCN


def build_model(cfg, num_classes: int) -> nn.Module:
    name = cfg.model.name.lower()

    if name == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)

    if name == "fcn":
        return FCN(
            input_shape=cfg.model.input_shape,
            hidden_dims=cfg.model.hidden_dims,
            num_classes=num_classes,
            dropout=float(cfg.model.dropout),
        )

    raise ValueError(f"Unsupported model: {cfg.model.name}")
