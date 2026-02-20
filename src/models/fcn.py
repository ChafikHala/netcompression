from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class FCN(nn.Module):

    def __init__(
        self,
        input_shape: Sequence[int],  # (C,H,W)
        hidden_dims: Iterable[int],
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        c, h, w = [int(x) for x in input_shape]
        in_dim = c * h * w

        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_dim
        for hd in hidden_dims:
            hd = int(hd)
            layers.append(nn.Linear(prev, hd))
            layers.append(nn.ReLU(inplace=True))
            if dropout and float(dropout) > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = hd

        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
