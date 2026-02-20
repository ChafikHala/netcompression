from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class EvalResult:
    loss: float
    accuracy: float


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> EvalResult:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    return EvalResult(loss=avg_loss, accuracy=acc)
