import torch.nn as nn
import torch
from typing import Optional

def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch),
            ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1, groups=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileNetV1(nn.Module):
    """
    CIFAR-10 friendly MobileNetV1:
    - first conv stride=1 instead of 2
    - small input 32x32 handled naturally
    """
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()

        def c(ch: int) -> int:
            return _make_divisible(ch * width_mult, 8)

        self.features = nn.Sequential(
            ConvBNAct(3, c(32), kernel_size=3, stride=1),
            DepthwiseSeparableConv(c(32), c(64), stride=1),
            DepthwiseSeparableConv(c(64), c(128), stride=2),
            DepthwiseSeparableConv(c(128), c(128), stride=1),
            DepthwiseSeparableConv(c(128), c(256), stride=2),
            DepthwiseSeparableConv(c(256), c(256), stride=1),
            DepthwiseSeparableConv(c(256), c(512), stride=2),
            DepthwiseSeparableConv(c(512), c(512), stride=1),
            DepthwiseSeparableConv(c(512), c(512), stride=1),
            DepthwiseSeparableConv(c(512), c(512), stride=1),
            DepthwiseSeparableConv(c(512), c(512), stride=1),
            DepthwiseSeparableConv(c(512), c(512), stride=1),
            DepthwiseSeparableConv(c(512), c(1024), stride=2),
            DepthwiseSeparableConv(c(1024), c(1024), stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c(1024), num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x