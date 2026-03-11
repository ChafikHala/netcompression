from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn


@dataclass
class PruningController:
    enabled: bool
    method: str
    masks: Dict[str, torch.Tensor]
    total_prunable: int

    # common target
    initial_sparsity: float
    final_sparsity: float

    # epoch schedule (global magnitude)
    start_epoch: int
    end_epoch: int
    frequency_epochs: int
    total_epoch_events: int

    # step schedule (gradual cubic)
    begin_step: int
    update_frequency_steps: int
    num_updates: int
    end_step: int


def _is_norm_module(m: nn.Module) -> bool:
    return isinstance(
        m,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.LayerNorm,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    )



def _module_name_map(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _is_prunable_param(
    param_name: str,
    module_map: Dict[str, nn.Module],
    *,
    exclude_bias: bool,
    exclude_norm: bool,
) -> bool:
    if exclude_bias and param_name.endswith(".bias"):
        return False

    module_name = param_name.rsplit(".", 1)[0] if "." in param_name else ""
    module = module_map.get(module_name)
    if exclude_norm and module is not None and _is_norm_module(module):
        return False

    return True

def _build_masks(
    model: nn.Module,
    *,
    exclude_bias: bool,
    exclude_norm: bool,
) -> tuple[Dict[str, torch.Tensor], int]:
    module_map = _module_name_map(model)
    masks: Dict[str, torch.Tensor] = {}
    total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not _is_prunable_param(name, module_map, exclude_bias=exclude_bias, exclude_norm=exclude_norm):
            continue

        mask = torch.ones_like(param.data, dtype=torch.bool)
        masks[name] = mask
        total += mask.numel()

    return masks, total

def build_pruning_controller(cfg, model: nn.Module) -> PruningController:
    pruning_cfg = getattr(cfg, "pruning", None)
    if pruning_cfg is None or not bool(getattr(pruning_cfg, "enabled", False)):
        return PruningController(
            enabled=False,
            method="none",
            masks={},
            total_prunable=0,
            initial_sparsity=0.0,
            final_sparsity=0.0,
            start_epoch=0,
            end_epoch=0,
            frequency_epochs=1,
            total_epoch_events=0,
            begin_step=0,
            update_frequency_steps=1,
            num_updates=0,
            end_step=0,
        )

    method = str(getattr(pruning_cfg, "method", "gradual_cubic_layerwise")).lower()
    supported = {"gradual_cubic_layerwise", "global_magnitude"}
    if method not in supported:
        raise ValueError(f"Unsupported pruning method: {method}. Supported: {sorted(supported)}")

    exclude_bias = bool(getattr(pruning_cfg, "exclude_bias", True))
    exclude_norm = bool(getattr(pruning_cfg, "exclude_norm", True))
    masks, total = _build_masks(model, exclude_bias=exclude_bias, exclude_norm=exclude_norm)
    if total == 0:
        raise ValueError("No prunable parameters were found with current pruning settings.")

    final_sparsity = float(getattr(pruning_cfg, "final_sparsity", getattr(pruning_cfg, "amount", 0.5)))
    initial_sparsity = float(getattr(pruning_cfg, "initial_sparsity", 0.0))
    if not (0.0 <= initial_sparsity < 1.0):
        raise ValueError(f"pruning.initial_sparsity must be in [0, 1), got {initial_sparsity}")
    if not (0.0 <= final_sparsity < 1.0):
        raise ValueError(f"pruning.final_sparsity must be in [0, 1), got {final_sparsity}")
    if final_sparsity < initial_sparsity:
        raise ValueError(
            f"pruning.final_sparsity ({final_sparsity}) must be >= pruning.initial_sparsity ({initial_sparsity})"
        )

    # defaults for global magnitude epoch schedule
    start_epoch = int(getattr(pruning_cfg, "start_epoch", 0))
    default_end_epoch = int(cfg.training.epochs) - 1
    end_epoch_raw = getattr(pruning_cfg, "end_epoch", default_end_epoch)
    end_epoch = default_end_epoch if end_epoch_raw is None else int(end_epoch_raw)
    frequency_epochs = max(int(getattr(pruning_cfg, "frequency", 1)), 1)
    if end_epoch < start_epoch:
        raise ValueError(
            f"pruning.end_epoch ({end_epoch}) must be >= pruning.start_epoch ({start_epoch})"
        )
    total_epoch_events = len(range(start_epoch, end_epoch + 1, frequency_epochs))

    # defaults for gradual cubic step schedule
    begin_step = int(getattr(pruning_cfg, "begin_step", 0))
    update_frequency_steps = max(int(getattr(pruning_cfg, "update_frequency_steps", getattr(pruning_cfg, "frequency", 100))), 1)
    num_updates = int(getattr(pruning_cfg, "num_updates", 100))
    if num_updates < 0:
        raise ValueError(f"pruning.num_updates must be >= 0, got {num_updates}")
    end_step = begin_step + num_updates * update_frequency_steps

    return PruningController(
        enabled=True,
        method=method,
        masks=masks,
        total_prunable=total,
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        frequency_epochs=frequency_epochs,
        total_epoch_events=total_epoch_events,
        begin_step=begin_step,
        update_frequency_steps=update_frequency_steps,
        num_updates=num_updates,
        end_step=end_step,
    )