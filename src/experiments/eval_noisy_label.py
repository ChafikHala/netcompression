"""Evaluate each noisy-label run, pick the checkpoint corresponding to the target sparsity, and plot accuracy curves."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.datasets import build_datasets
from src.data.dataloaders import build_dataloaders
from src.utils.config import load_config
from src.utils.device import get_device
from src.experiments.eval import evaluate_checkpoint


RUN_ID_PATTERN = re.compile(r".*noise(\d+)_sp(\d+)_seed(\d+)(?:_\d+)?$")


def parse_run_id(run_id: str) -> tuple[float, float, int] | None:
    match = RUN_ID_PATTERN.match(run_id)
    if match is None:
        return None
    noise = int(match.group(1)) / 100.0
    sparsity = int(match.group(2)) / 100.0
    seed = int(match.group(3))
    return noise, sparsity, seed


def load_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_metrics(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return []
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def choose_checkpoint(
    run_dir: Path,
    target_sparsity: float,
    summary: dict | None,
    metrics: list[dict],
    tolerance: float = 1e-3,
) -> Path | None:
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    best_epoch = None
    if summary:
        best_epoch = summary.get("best_epoch")
    best_sparsity = None
    if isinstance(best_epoch, (int, float)):
        idx = int(best_epoch)
        if 0 <= idx < len(metrics):
            best_sparsity = metrics[idx].get("sparsity")

    if best_sparsity is not None and abs(best_sparsity - target_sparsity) <= tolerance and best_path.exists():
        return best_path
    if last_path.exists():
        return last_path
    if best_path.exists():
        return best_path
    return None


def evaluate_runs(
    run_dirs: list[Path],
    base_cfg,
    test_loader,
    num_classes: int,
    device,
) -> dict[float, dict[float, list[float]]]:
    results = defaultdict(lambda: defaultdict(list))

    for run_dir in sorted(run_dirs):
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            print(f"Skipping {run_dir.name}: cannot parse run_id")
            continue
        noise, sparsity, seed = parsed

        summary = load_summary(run_dir)
        metrics = load_metrics(run_dir)
        checkpoint = choose_checkpoint(run_dir, sparsity, summary, metrics)
        if checkpoint is None:
            print(f"Skipping {run_dir}: no checkpoint found")
            continue

        result = evaluate_checkpoint(
            ckpt_path=checkpoint,
            base_cfg=base_cfg,
            test_loader=test_loader,
            num_classes=num_classes,
            device=device,
        )
        top1 = result["top1"]
        results[noise][sparsity].append(top1)
        print(
            f"noise={noise:.2f} sparsity={sparsity:.2f} seed={seed} "
            f"→ {checkpoint.name} top1={top1:.4f}"
        )

    return results


def aggregate(results: dict[float, dict[float, list[float]]]) -> dict[float, list[dict]]:
    aggregated = {}
    for noise, spars_map in sorted(results.items()):
        entries = []
        for sparsity in sorted(spars_map):
            accuracies = spars_map[sparsity]
            if not accuracies:
                continue
            mean = statistics.mean(accuracies)
            std = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
            entries.append(
                {
                    "sparsity": sparsity,
                    "mean_top1": mean,
                    "std_top1": std,
                    "n_seeds": len(accuracies),
                }
            )
        aggregated[noise] = entries
    return aggregated


def plot_results(aggregated: dict[float, list[dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for noise, data in aggregated.items():
        if not data:
            continue
        xs = [entry["sparsity"] * 100.0 for entry in data]
        ys = [entry["mean_top1"] * 100.0 for entry in data]
        errs = [entry["std_top1"] * 100.0 for entry in data]
        ax.errorbar(
            xs, ys, yerr=errs, marker="o", linewidth=2.0, capsize=4, label=f"noise={noise:.2f}"
        )

    ax.set_xlabel("Target sparsity (%)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Noisy-label sweep: accuracy vs sparsity (MNIST + LeNet)")
    ax.grid(True)
    ax.legend()

    plot_path = output_dir / "noisy_label_accuracy_vs_sparsity.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")


def save_summary(aggregated: dict[float, list[dict]], output_dir: Path) -> None:
    summary_path = output_dir / "noisy_label_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cifar10_resnet20.yaml"),
        help="Base config to rebuild the test loader.",
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("outputs/noisy_label_experiment"),
        help="Directory containing mnist_lenet_noise*_sp*_seed* runs.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("outputs/noisy_label_eval"),
        help="Directory where aggregated plots and JSON go.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string understood by get_device().",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)

    bundle = build_datasets(cfg)
    _, _, test_loader = build_dataloaders(
        cfg,
        bundle.train,
        bundle.val,
        bundle.test,
        device,
    )

    run_dirs = [
        path for path in sorted(args.exp_dir.glob("mnist_lenet_noise*_sp*_seed*")) if path.is_dir()
    ]
    if not run_dirs:
        raise RuntimeError(f"No matching runs under: {args.exp_dir}")

    results = evaluate_runs(run_dirs, cfg, test_loader, bundle.num_classes, device)
    aggregated = aggregate(results)
    plot_results(aggregated, args.plot_dir)
    save_summary(aggregated, args.plot_dir)


if __name__ == "__main__":
    main()
