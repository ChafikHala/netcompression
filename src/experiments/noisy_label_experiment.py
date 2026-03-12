"""Run a MNIST + LeNet sweep over label noise and target sparsity."""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

from src.core.trainer import TrainResult, train
from src.data.dataloaders import build_dataloaders
from src.data.datasets import build_datasets
from src.experiments.train import _build_optimizer, _build_scheduler
from src.models.factory import build_model
from src.utils.config import Config, load_config
from src.utils.device import get_device
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/cifar10_resnet20.yaml"),
        help="Base config for CIFAR10+ResNet runs.",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.15, 0.3, 0.6],
        help="Label noise fractions to sweep.",
    )
    parser.add_argument(
        "--target-sparsities",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 0.85, 0.9, 0.95, 0.98, 0.99],
        help="Final pruning sparsities to explore.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds (three per combination).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string passed to get_device() (example: cuda or cpu).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/noisy_label_experiment"),
        help="Root output directory for all runs and the aggregated results.",
    )
    return parser.parse_args()


def set_label_noise_fraction(cfg, noise_fraction: float) -> None:
    noise_cfg = getattr(cfg.dataset, "noise", None)
    if noise_cfg is None:
        cfg.dataset.noise = Config({"label_noise_fraction": float(noise_fraction)})
        return
    noise_cfg.label_noise_fraction = float(noise_fraction)

def build_run_id(noise_fraction: float, sparsity: float, seed: int) -> str:
    noise_pct = int(round(noise_fraction * 100))
    spars_pct = int(round(sparsity * 100))
    return f"cifar10_resnet20_noise{noise_pct:02d}_sp{spars_pct:02d}_seed{seed}"


def run_combo(
    base_config: Path,
    noise_fraction: float,
    sparsity: float,
    seed: int,
    device_str: str,
    output_dir: Path,
    run_id: str,
) -> tuple[TrainResult, float]:
    cfg = load_config(base_config)
    cfg.experiment.seed = seed
    cfg.experiment.device = device_str
    cfg.experiment.output_dir = str(output_dir)
    cfg.experiment.run_id = run_id

    set_label_noise_fraction(cfg, noise_fraction)
    cfg.pruning.final_sparsity = float(sparsity)

    seed_everything(seed)
    device = get_device(device_str)

    bundle = build_datasets(cfg)
    train_loader, val_loader, _ = build_dataloaders(
        cfg,
        bundle.train,
        bundle.val,
        bundle.test,
        device,
    )

    model = build_model(cfg, num_classes=bundle.num_classes)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    start = time.time()
    result = train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    duration = time.time() - start
    return result, duration


def load_summary(run_dir: str) -> dict | None:
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))



def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    base_cfg = load_config(args.base_config)
    monitor_key = getattr(base_cfg.checkpoint, "monitor", None)

    combos = product(
        sorted(args.noise_levels),
        sorted(args.target_sparsities),
        sorted(args.seeds),
    )

    for noise_fraction, sparsity, seed in combos:
        print(
            f"\nRunning noise={noise_fraction:.3f}, sparsity={sparsity:.3f}, "
            f"seed={seed}"
        )
        run_id = build_run_id(noise_fraction, sparsity, seed)
        run_dir = args.output_dir / base_cfg.experiment.name / run_id
        if run_dir.exists():
            print(f"Skipping {run_id}: output folder already exists ({run_dir}).")
            summary = load_summary(str(run_dir))
            summary_key = f"best_{monitor_key}" if monitor_key else None
            record = {
                "noise_fraction": noise_fraction,
                "target_sparsity": sparsity,
                "seed": seed,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "best_metric": summary.get(summary_key) if summary and summary_key else None,
                "best_epoch": summary.get("best_epoch") if summary else None,
                "last_epoch": None,
                "duration_sec": None,
                "status": "skipped",
                "summary": summary,
            }
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            continue

        result, duration = run_combo(
            base_config=args.base_config,
            noise_fraction=noise_fraction,
            sparsity=sparsity,
            seed=seed,
            device_str=args.device,
            output_dir=args.output_dir,
            run_id=run_id,
        )


        record = {
            "noise_fraction": noise_fraction,
            "target_sparsity": sparsity,
            "seed": seed,
            "run_id": run_id,
            "run_dir": result.run_dir,
            "best_metric": result.best_metric,
            "best_epoch": result.best_epoch,
            "last_epoch": result.last_epoch,
            "duration_sec": duration,
            "summary": load_summary(result.run_dir),
            "status": "completed",
        }

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Recorded run in {results_path}")


if __name__ == "__main__":
    main()
