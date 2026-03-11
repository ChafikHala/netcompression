# ============================================================
# eval.py
# Auto-discover experiments and seeds from outputs directory
# No hardcoded sparse_levels / width_levels / checkpoint paths
# ============================================================
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils.config import load_config
from src.data.datasets import build_datasets
from src.data.dataloaders import build_dataloaders
from src.models.factory import build_model


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / batch_size).item())
    return res


def extract_state_dict(ckpt: dict) -> dict:
    if "model_state" in ckpt:
        return ckpt["model_state"]
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def extract_model_cfg(base_cfg, ckpt: dict):
    cfg_for_model = copy.deepcopy(base_cfg)

    if isinstance(ckpt, dict) and "config" in ckpt and ckpt["config"] is not None:
        saved_cfg = ckpt["config"]
        if isinstance(saved_cfg, dict):
            model_cfg = saved_cfg["model"]
            for k, v in model_cfg.items():
                setattr(cfg_for_model.model, k, v)
        else:
            cfg_for_model.model = saved_cfg.model

    return cfg_for_model


def count_params_from_state_dict(state_dict: dict):
    total_params = 0
    nnz_params = 0

    for _, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        total_params += tensor.numel()
        nnz_params += torch.count_nonzero(tensor).item()

    sparsity = 1.0 - nnz_params / max(total_params, 1)
    return total_params, nnz_params, sparsity


@torch.no_grad()
def evaluate_checkpoint(
    ckpt_path: str | Path,
    base_cfg,
    test_loader,
    num_classes: int,
    device,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)
    model_cfg = extract_model_cfg(base_cfg, ckpt)
    model = build_model(model_cfg, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_params, nnz_params, sparsity = count_params_from_state_dict(state_dict)

    top1 = 0.0
    top5 = 0.0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        bs = labels.size(0)
        top1 += acc1 * bs
        top5 += acc5 * bs
        total += bs

    top1 /= total
    top5 /= total

    return {
        "checkpoint": str(ckpt_path),
        "total_params": total_params,
        "nnz_params": nnz_params,
        "sparsity": sparsity,
        "top1": top1,
        "top5": top5,
    }


def infer_display_name(exp_name: str) -> str:
    raw = exp_name
    if "sparse" in raw:
        level = raw.split("sparse")[-1]
        return f"MobileNet CIFAR-10 Sparsity = {level}%"
    if "width" in raw:
        level = raw.split("width")[-1]
        return f"MobileNet CIFAR-10 Width = {level}%"
    if "widen" in raw:
        level = raw.split("widen")[-1]
        return f"WideResNet CIFAR-10 Widen = {level}"
    return raw


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def discover_experiments(
    base_dir: str | Path,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
):
    """
    Discover experiments automatically from folders like:
        outputs/<exp_name>_seeds/<exp_name>_seed42/
    """
    base_dir = Path(base_dir)
    discovered = {}

    for seeds_dir in sorted(base_dir.glob("*_seeds")):
        if not seeds_dir.is_dir():
            continue

        exp_name = seeds_dir.name[:-6]  # strip "_seeds"

        if include and not any(token in exp_name for token in include):
            continue
        if exclude and any(token in exp_name for token in exclude):
            continue

        seed_runs = []
        for run_dir in sorted(seeds_dir.glob(f"{exp_name}_seed*")):
            if run_dir.is_dir():
                seed_runs.append(run_dir)

        if seed_runs:
            discovered[exp_name] = seed_runs

    return discovered


def infer_checkpoint_name(run_dir: Path, prefer: str = "auto") -> str:
    """
    Rules:
        - prefer == "best" -> best.pt
        - prefer == "last" -> last.pt
        - prefer == "auto" -> use config:
            pruning enabled  -> last.pt
            pruning disabled -> best.pt
    """
    if prefer in {"best", "last"}:
        return f"{prefer}.pt"

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return "best.pt"

    cfg = load_json(cfg_path)
    pruning_cfg = cfg.get("pruning", {})
    pruning_enabled = bool(pruning_cfg.get("enabled", False))
    return "last.pt" if pruning_enabled else "best.pt"


def load_and_average_metrics_for_runs(run_dirs: list[Path]) -> list[dict]:
    all_seed_metrics = []

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"SKIPPING (metrics not found): {metrics_path}")
            continue
        with open(metrics_path, "r") as f:
            all_seed_metrics.append(json.load(f))

    if not all_seed_metrics:
        return []

    n_epochs = min(len(m) for m in all_seed_metrics)
    averaged = []

    for ep_idx in range(n_epochs):
        epoch_data = [m[ep_idx] for m in all_seed_metrics]
        avg_entry = {"epoch": epoch_data[0]["epoch"]}
        for key in ["train_loss", "train_accuracy", "val_loss", "val_accuracy"]:
            values = [e[key] for e in epoch_data]
            avg_entry[f"{key}_mean"] = statistics.mean(values)
            avg_entry[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        averaged.append(avg_entry)

    return averaged


def plot_metrics(averaged_metrics: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("train_loss",     "Train Loss",     "Loss"),
        ("train_accuracy", "Train Accuracy", "Accuracy (%)"),
        ("val_loss",       "Val Loss",       "Loss"),
        ("val_accuracy",   "Val Accuracy",   "Accuracy (%)"),
    ]

    for metric_key, title, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)

        for exp_name, epochs_data in averaged_metrics.items():
            if not epochs_data:
                continue

            epochs = [e["epoch"]              for e in epochs_data]
            means  = [e[f"{metric_key}_mean"] for e in epochs_data]
            stds   = [e[f"{metric_key}_std"]  for e in epochs_data]

            line, = ax.plot(epochs, means, label=infer_display_name(exp_name))
            ax.fill_between(
                epochs,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15,
                color=line.get_color(),
            )

        ax.legend(fontsize=8)
        ax.grid(True)
        plt.tight_layout()

        plot_path = output_dir / f"{metric_key}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_path}")


def plot_accuracy_vs_nnz(summary: dict, output_dir: str | Path):
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    def group_of(name: str) -> str:
        if "sparse" in name:
            return "pruned"
        if "width" in name:
            return "dense"
        return "other"

    groups = {
        "dense":  {"marker": "o", "items": []},
        "pruned": {"marker": "s", "items": []},
        "other":  {"marker": "^", "items": []},
    }

    for exp_name, stats in summary.items():
        groups[group_of(exp_name)]["items"].append((exp_name, stats))

    for group_name, meta in groups.items():
        items = meta["items"]
        if not items:
            continue

        xs   = np.array([stats["nnz_params"]       for _, stats in items], dtype=float)
        ys   = np.array([100.0 * stats["mean_top1"] for _, stats in items], dtype=float)
        yerr = np.array([100.0 * stats["std_top1"]  for _, stats in items], dtype=float)

        order = np.argsort(xs)
        xs    = xs[order]
        ys    = ys[order]
        yerr  = yerr[order]
        ordered_items = [items[i] for i in order]

        ax.errorbar(
            xs, ys, yerr=yerr,
            marker=meta["marker"],
            linewidth=2.0,
            markersize=8,
            capsize=4,
            label=group_name.capitalize(),
        )

        for exp_name, stats in ordered_items:
            ax.annotate(
                infer_display_name(exp_name),
                xy=(stats["nnz_params"], 100.0 * stats["mean_top1"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Non-zero parameters (log scale)")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy vs Effective Model Size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plot_path = output_dir / "accuracy_vs_nnz.png"
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")


def build_summary(all_results: dict[str, list[dict]]):
    summary = {}

    for exp_name, results in all_results.items():
        top1s      = [r["top1"]     for r in results]
        top5s      = [r["top5"]     for r in results]
        sparsities = [r["sparsity"] for r in results]

        summary[exp_name] = {
            "mean_top1":     statistics.mean(top1s),
            "std_top1":      statistics.stdev(top1s)      if len(top1s) > 1      else 0.0,
            "mean_top5":     statistics.mean(top5s),
            "std_top5":      statistics.stdev(top5s)      if len(top5s) > 1      else 0.0,
            "mean_sparsity": statistics.mean(sparsities),
            "n_seeds":       len(results),
            "nnz_params":    results[0]["nnz_params"]    if results else None,
            "total_params":  results[0]["total_params"]  if results else None,
        }

    return summary


def print_summary(summary: dict):
    print("\n===== Aggregated Results (mean ± std across seeds) =====\n")
    for exp_name, stats in summary.items():
        print(f"[{exp_name}]  n={stats['n_seeds']}")
        print(f"  Top-1:    {stats['mean_top1']:.4f} ± {stats['std_top1']:.4f}")
        print(f"  Top-5:    {stats['mean_top5']:.4f} ± {stats['std_top5']:.4f}")
        print(f"  Sparsity: {stats['mean_sparsity']:.4f}")
        print(f"  Non-zero params: {stats['nnz_params']}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--plot-dir", type=str, default="outputs")
    parser.add_argument("--ckpt",     type=str, default="auto", choices=["auto", "best", "last"])
    parser.add_argument("--include",  nargs="*", default=None)
    parser.add_argument("--exclude",  nargs="*", default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = build_datasets(cfg)
    _, _, test_loader = build_dataloaders(
        cfg,
        bundle.train,
        bundle.val,
        bundle.test,
        device,
    )

    experiments = discover_experiments(
        base_dir=args.base_dir,
        include=args.include,
        exclude=args.exclude,
    )

    if not experiments:
        raise RuntimeError(f"No experiments discovered under: {args.base_dir}")

    averaged_metrics = {}
    all_results      = defaultdict(list)

    for exp_name, run_dirs in experiments.items():
        averaged_metrics[exp_name] = load_and_average_metrics_for_runs(run_dirs)

        for run_dir in run_dirs:
            ckpt_name = infer_checkpoint_name(run_dir, prefer=args.ckpt)
            ckpt_path = run_dir / ckpt_name

            if not ckpt_path.exists():
                print(f"SKIPPING (checkpoint not found): {ckpt_path}")
                continue

            result = evaluate_checkpoint(
                ckpt_path=ckpt_path,
                base_cfg=cfg,
                test_loader=test_loader,
                num_classes=bundle.num_classes,
                device=device,
            )
            all_results[exp_name].append(result)
            print(f"evaluated: {ckpt_path}  top1={result['top1']:.4f}")

    plot_metrics(averaged_metrics, args.plot_dir)

    summary = build_summary(all_results)
    print_summary(summary)

    summary_path = Path(args.plot_dir) / "aggregated_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    plot_accuracy_vs_nnz(summary, args.plot_dir)


if __name__ == "__main__":
    main()


