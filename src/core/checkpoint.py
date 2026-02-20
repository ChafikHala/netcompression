from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class CheckpointPayload:
    epoch: int
    best_metric: float
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    config: Dict[str, Any]


def _atomic_save(obj: Any, path: Path) -> None:
    """
    Write checkpoint atomically to avoid corrupt files if interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(
    path: str | Path,
    payload: CheckpointPayload,
) -> None:
    _atomic_save(payload.__dict__, Path(path))


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def save_best_if_needed(
    *,
    path: str | Path,
    epoch: int,
    metric_value: float,
    best_metric: float,
    mode: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    config_dict: Dict[str, Any],
) -> float:
    """
    Save checkpoint if metric improves.

    mode: "max" for accuracy, "min" for loss.
    Returns updated best_metric.
    """
    assert mode in {"max", "min"}

    improved = (metric_value > best_metric) if mode == "max" else (metric_value < best_metric)
    if not improved:
        return best_metric

    payload = CheckpointPayload(
        epoch=epoch,
        best_metric=float(metric_value),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict() if scheduler is not None else None,
        config=config_dict,
    )
    save_checkpoint(path, payload)
    return float(metric_value)
