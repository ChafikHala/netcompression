from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from src.pruning.controller import PruningController
from src.pruning.metrics import current_global_sparsity


def enforce_pruning_masks(ctrl: PruningController, model: nn.Module) -> None:

    if not ctrl.enabled:
        return

    with torch.no_grad():
        for name, param in model.named_parameters():
            mask = ctrl.masks.get(name)
            if mask is not None:
                param.data.mul_(mask)


def apply_global_target_sparsity(
    ctrl: PruningController,
    model: nn.Module,
    target_sparsity: float,
) -> float:

    target_sparsity = float(
        min(max(target_sparsity, ctrl.initial_sparsity), ctrl.final_sparsity)
    )

    target_pruned = int(round(target_sparsity * ctrl.total_prunable))
    current_pruned = int(round(current_global_sparsity(ctrl) * ctrl.total_prunable))

    additional_to_prune = target_pruned - current_pruned
    if additional_to_prune <= 0:
        return current_global_sparsity(ctrl)

    params: Dict[str, nn.Parameter] = dict(model.named_parameters())

    flat_scores: List[torch.Tensor] = []
    flat_refs: List[Tuple[str, int]] = []

    for name, mask in ctrl.masks.items():

        p = params[name]

        alive_idx = torch.nonzero(mask.view(-1), as_tuple=False).squeeze(1)
        if alive_idx.numel() == 0:
            continue

        scores = p.data.view(-1).abs()[alive_idx]

        flat_scores.append(scores)
        flat_refs.extend((name, int(i)) for i in alive_idx.tolist())

    if not flat_scores:
        return current_global_sparsity(ctrl)

    all_scores = torch.cat(flat_scores)
    additional_to_prune = min(additional_to_prune, int(all_scores.numel()))

    prune_idx = torch.argsort(all_scores)[:additional_to_prune]

    for j in prune_idx.tolist():
        name, flat_idx = flat_refs[j]
        ctrl.masks[name].view(-1)[flat_idx] = False

    enforce_pruning_masks(ctrl, model)

    return current_global_sparsity(ctrl)