import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """
    ResNet-20 for CIFAR-10/100 (3×32×32 input).

    Architecture:
        stem  : 3→16, 3×3 conv
        stage1: 3 blocks, 16 channels, stride 1
        stage2: 3 blocks, 32 channels, stride 2
        stage3: 3 blocks, 64 channels, stride 2
        head  : global avg-pool → fc

    Total depth  = 1 + (3+3+3)×2 + 1 = 20 layers (counting only conv/fc).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(16, 16, n_blocks=3, stride=1)
        self.stage2 = self._make_stage(16, 32, n_blocks=3, stride=2)
        self.stage3 = self._make_stage(32, 64, n_blocks=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [_ResBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(_ResBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
