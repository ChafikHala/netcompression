from __future__ import annotations
import torch.nn as nn
from src.pruning.controller import PruningController
from src.pruning.global_magnitude import apply_global_target_sparsity
from src.pruning.layerwise_magnitude import apply_layerwise_target_sparsity

def prune_to_target_sparsity(ctrl: PruningController, model: nn.Module, target_sparsity: float) -> float:
    if not ctrl.enabled:
        return 0.0

    if ctrl.method == "gradual_cubic_layerwise":
        return apply_layerwise_target_sparsity(ctrl, model, target_sparsity)

    if ctrl.method == "global_magnitude":
        return apply_global_target_sparsity(ctrl, model, target_sparsity)

    raise ValueError(f"Unsupported pruning method: {ctrl.method}")