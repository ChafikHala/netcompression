from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


def make_run_paths(cfg) -> RunPaths:
    output_root = Path(getattr(cfg.experiment, "output_root", "outputs"))
    exp_name = cfg.experiment.name

    run_id = getattr(cfg.experiment, "run_id", "auto")
    if run_id is None or str(run_id).lower() == "auto":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = output_root / exp_name / str(run_id)
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(run_dir=run_dir, checkpoints_dir=ckpt_dir, logs_dir=logs_dir)
