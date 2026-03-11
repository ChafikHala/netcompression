
from __future__ import annotations
from src.pruning.controller import PruningController



def should_prune_this_epoch(ctrl: PruningController, epoch: int) -> bool:
    if not ctrl.enabled or ctrl.final_sparsity <= ctrl.initial_sparsity:
        return False
    if ctrl.method != "global_magnitude":
        return False
    if ctrl.total_epoch_events == 0:
        return False
    if epoch < ctrl.start_epoch or epoch > ctrl.end_epoch:
        return False
    return (epoch - ctrl.start_epoch) % ctrl.frequency_epochs == 0



def should_prune_this_step(ctrl: PruningController, global_step: int) -> bool:
    if not ctrl.enabled or ctrl.final_sparsity <= ctrl.initial_sparsity:
        return False

    if ctrl.method == "gradual_cubic_layerwise":
        if global_step < ctrl.begin_step or global_step > ctrl.end_step:
            return False
        return (global_step - ctrl.begin_step) % ctrl.update_frequency_steps == 0

    # global_magnitude fallback (epoch method only)
    return False

def target_sparsity_for_epoch(ctrl: PruningController, epoch: int) -> float:
    event_idx = ((epoch - ctrl.start_epoch) // ctrl.frequency_epochs) + 1
    progress = min(max(event_idx / max(ctrl.total_epoch_events, 1), 0.0), 1.0)
    return float(ctrl.initial_sparsity + (ctrl.final_sparsity - ctrl.initial_sparsity) * progress)



def target_sparsity_for_step(ctrl: PruningController, global_step: int) -> float:
    # s_t = sf + (si - sf) * (1 - (t - t0)/(n*dt))^3
    if global_step <= ctrl.begin_step:
        return ctrl.initial_sparsity
    if global_step >= ctrl.end_step:
        return ctrl.final_sparsity

    span = max(ctrl.num_updates * ctrl.update_frequency_steps, 1)
    frac = (global_step - ctrl.begin_step) / float(span)
    frac = min(max(frac, 0.0), 1.0)
    return float(ctrl.final_sparsity + (ctrl.initial_sparsity - ctrl.final_sparsity) * ((1.0 - frac) ** 3))