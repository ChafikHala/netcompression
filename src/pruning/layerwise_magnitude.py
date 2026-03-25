from __future__ import annotations
import torch
import torch.nn as nn
from src.pruning.controller import PruningController
from src.pruning.metrics import current_pruned_count, current_global_sparsity

def apply_layerwise_target_sparsity(ctrl: PruningController, model: nn.Module, target_sparsity: float) -> float:
    target_sparsity = float(min(max(target_sparsity, ctrl.initial_sparsity), ctrl.final_sparsity))
    params = dict(model.named_parameters())

    with torch.no_grad():
        for name, mask in ctrl.masks.items():
            p = params[name]
            total = mask.numel()
            keep = int(round((1.0 - target_sparsity) * total))
            keep = max(0, min(keep, total))

            if keep == total:
                continue
            if keep == 0:
                mask.fill_(False)
                p.data.zero_()
                continue

            flat_abs = p.data.abs().view(-1)
            topk_vals, _ = torch.topk(flat_abs, k=keep, largest=True, sorted=True)
            cutoff = topk_vals[-1]

            local_mask = (p.data.abs() >= cutoff)
            if int(local_mask.sum().item()) > keep:
                eq = (p.data.abs() == cutoff).view(-1)
                lm = local_mask.view(-1)
                extra = int(lm.sum().item()) - keep
                idx = torch.nonzero(eq, as_tuple=False).view(-1)
                lm[idx[:extra]] = False
                local_mask = lm.view_as(mask)

            mask.copy_(local_mask)
            p.data.mul_(mask)

    return current_global_sparsity(ctrl)