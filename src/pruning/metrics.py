from __future__ import annotations
import torch.nn as nn
from src.pruning.controller import PruningController


def current_pruned_count(ctrl: PruningController) -> int:

    if not ctrl.enabled:
        return 0

    nonzero = sum(int(mask.sum().item()) for mask in ctrl.masks.values())
    return ctrl.total_prunable - nonzero


def current_global_sparsity(ctrl: PruningController) -> float:
    if not ctrl.enabled or ctrl.total_prunable <= 0:
        return 0.0
    alive = sum(int(mask.sum().item()) for mask in ctrl.masks.values())
    return 1.0 - (alive / float(ctrl.total_prunable))


def model_parameter_sparsity(model: nn.Module) -> float:

    total = 0
    zeros = 0

    for p in model.parameters():
        total += p.numel()
        zeros += int((p == 0).sum().item())

    return zeros / max(total, 1)