from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.checkpoint import load_checkpoint
from src.data.dataloaders import build_dataloaders
from src.data.datasets import build_datasets
from src.papers.compressibility_adv_robustness.attack_utils_cifar import (
    build_autopgd_for_cifar10,
    build_fgsm_for_cifar10,
    generate_adversarial_batch,
    normalize_cifar10_tensor,
)
from src.papers.compressibility_adv_robustness.train_cifar10_wide_resnet import (
    DenseWideResNet,
    LowRankWideResNet,
)
from src.utils.config import load_config


def float_to_name(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def run_name_neuron(beta: float, seed: int) -> str:
    return f"cifar10_wrn16_2_group_lasso_beta_{float_to_name(beta)}_seed_{seed}"


def run_name_spectral(rank: int, seed: int) -> str:
    return f"cifar10_wrn16_2_low_rank_rank_{int(rank)}_seed_{seed}"


def build_model(compressibility: str, num_classes: int = 10, rank: int | None = None) -> nn.Module:
    if compressibility == "neuron":
        return DenseWideResNet(
            depth=16,
            widen_factor=2,
            num_classes=num_classes,
            bias=False,
        )

    if compressibility == "spectral":
        if rank is None:
            raise ValueError("rank must be provided for spectral compressibility")
        return LowRankWideResNet(
            depth=16,
            widen_factor=2,
            num_classes=num_classes,
            rank=int(rank),
            bias=False,
        )

    raise ValueError(f"Unknown compressibility: {compressibility}")


def get_penultimate_representation(model: nn.Module, x_norm: torch.Tensor) -> torch.Tensor:
    out = model.conv1(x_norm)
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = F.relu(model.bn(out), inplace=False)
    out = F.avg_pool2d(out, kernel_size=8)
    out = out.view(out.size(0), -1)
    return out


def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    compressibility: str,
    rank: int | None = None,
) -> nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_model(
        compressibility=compressibility,
        num_classes=10,
        rank=rank,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def find_checkpoint(
    base_dir: Path,
    compressibility: str,
    value: float | int,
    seed: int,
    ckpt_name: str,
) -> Path:
    if compressibility == "neuron":
        run_name = run_name_neuron(float(value), seed)
        path = (
            base_dir
            / "compressibility_adv_robustness"
            / "wrn16_2"
            / "neuron"
            / run_name
            / "checkpoints"
            / ckpt_name
        )
    elif compressibility == "spectral":
        run_name = run_name_spectral(int(value), seed)
        path = (
            base_dir
            / "compressibility_adv_robustness"
            / "wrn16_2"
            / "spectral"
            / run_name
            / "checkpoints"
            / ckpt_name
        )
    else:
        raise ValueError(f"Unknown compressibility: {compressibility}")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


@torch.no_grad()
def clean_accuracy(model: nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x_raw, y in dataloader:
        x_raw = x_raw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_norm = normalize_cifar10_tensor(x_raw)
        logits = model(x_norm)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return correct / max(total, 1)


def robust_accuracy_and_repr_shift(
    model: nn.Module,
    dataloader,
    device: torch.device,
    attack: str = "fgsm",
    norm: str = "linf",
    eps_linf: float = 8.0 / 255.0,
    eps_l2: float = 0.5,
    eps_step_linf: float | None = None,
    eps_step_l2: float | None = None,
    max_iter: int = 100,
    nb_random_init: int = 5,
    attack_batch_size: int = 128,
    repr_samples_cap: int = 1000,
) -> tuple[float, float]:
    model.eval()

    attack = attack.lower()
    norm = norm.lower()

    if norm not in ["linf", "l2"]:
        raise ValueError("norm must be one of ['linf', 'l2']")

    if norm == "linf":
        eps = float(eps_linf)
        eps_step = eps_step_linf
    else:
        eps = float(eps_l2)
        eps_step = eps_step_l2

    if attack == "fgsm":
        attacker = build_fgsm_for_cifar10(
            model=model,
            device=device,
            norm=norm,
            eps=eps,
            batch_size=attack_batch_size,
        )
    elif attack == "autopgd":
        attacker = build_autopgd_for_cifar10(
            model=model,
            device=device,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            nb_random_init=nb_random_init,
            batch_size=attack_batch_size,
            loss_type="cross_entropy",
            verbose=False,
        )
    else:
        raise ValueError(f"Unsupported attack: {attack}")

    robust_correct = 0
    total = 0
    repr_ratios: list[float] = []
    counted_repr = 0

    for x_raw, y in dataloader:
        x_raw = x_raw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_adv_raw = generate_adversarial_batch(attacker, x_raw, y, device=device)

        with torch.no_grad():
            x_adv_norm = normalize_cifar10_tensor(x_adv_raw)
            logits_adv = model(x_adv_norm)
            pred_adv = logits_adv.argmax(dim=1)

            robust_correct += int((pred_adv == y).sum().item())
            total += int(y.size(0))

            if counted_repr < repr_samples_cap:
                m = min(x_raw.size(0), repr_samples_cap - counted_repr)

                x_norm = normalize_cifar10_tensor(x_raw[:m])
                x_adv_norm_small = normalize_cifar10_tensor(x_adv_raw[:m])

                z = get_penultimate_representation(model, x_norm)
                z_adv = get_penultimate_representation(model, x_adv_norm_small)

                num = torch.linalg.vector_norm(z_adv - z, ord=2, dim=1)
                den = torch.linalg.vector_norm(z, ord=2, dim=1).clamp_min(1e-12)

                ratios = (num / den).detach().cpu().tolist()
                repr_ratios.extend(ratios)
                counted_repr += m

    robust_acc = robust_correct / max(total, 1)
    mean_repr_ratio = float(np.mean(repr_ratios)) if repr_ratios else float("nan")
    return robust_acc, mean_repr_ratio


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def aggregate_over_seeds(raw_results: Dict[str, List[dict]], x_key_name: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    for value_key, rows in raw_results.items():
        clean_vals = [r["clean_test_accuracy"] for r in rows]
        robust_vals = [r["robust_test_accuracy"] for r in rows]
        ratio_vals = [r["robust_over_clean"] for r in rows]
        repr_vals = [r["repr_shift_ratio"] for r in rows]

        out[value_key] = {
            x_key_name: float(value_key) if x_key_name == "beta" else int(float(value_key)),
            "clean_test_accuracy_mean": float(np.mean(clean_vals)),
            "clean_test_accuracy_std": float(np.std(clean_vals, ddof=0)),
            "robust_test_accuracy_mean": float(np.mean(robust_vals)),
            "robust_test_accuracy_std": float(np.std(robust_vals, ddof=0)),
            "robust_over_clean_mean": float(np.mean(ratio_vals)),
            "robust_over_clean_std": float(np.std(ratio_vals, ddof=0)),
            "repr_shift_ratio_mean": float(np.mean(repr_vals)),
            "repr_shift_ratio_std": float(np.std(repr_vals, ddof=0)),
            "n_seeds": len(rows),
        }

    return out


def _apply_elegant_style(ax):
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#CCCCCC", linewidth=0.6, linestyle="-")
    ax.grid(which="minor", color="#E8E8E8", linewidth=0.3, linestyle="-")
    ax.minorticks_on()

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color("#888888")

    ax.tick_params(axis="both", labelsize=11, length=3, width=0.7)
    ax.figure.patch.set_facecolor("white")
    ax.set_facecolor("white")


def _sorted_x_values(agg: Dict[str, dict], compressibility: str) -> np.ndarray:
    if compressibility == "neuron":
        return np.array(sorted([float(k) for k in agg.keys()]))
    return np.array(sorted([int(float(k)) for k in agg.keys()]))


def plot_accuracy_figure(agg: Dict[str, dict], save_path: Path, compressibility: str) -> None:
    xs = _sorted_x_values(agg, compressibility)

    clean_mean = np.array([agg[str(x)]["clean_test_accuracy_mean"] for x in xs])
    clean_std = np.array([agg[str(x)]["clean_test_accuracy_std"] for x in xs])

    robust_mean = np.array([agg[str(x)]["robust_test_accuracy_mean"] for x in xs])
    robust_std = np.array([agg[str(x)]["robust_test_accuracy_std"] for x in xs])

    ratio_mean = np.array([agg[str(x)]["robust_over_clean_mean"] for x in xs])
    ratio_std = np.array([agg[str(x)]["robust_over_clean_std"] for x in xs])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    curves = [
        (clean_mean, clean_std, "o", "#4169E1", "Clean Acc."),
        (robust_mean, robust_std, "D", "#228B22", "Rob. Acc."),
        (ratio_mean, ratio_std, "s", "#B22222", "Rob./Clean"),
    ]

    for mean, std, marker, color, label in curves:
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)
        ax.errorbar(
            xs,
            mean,
            yerr=std,
            color=color,
            marker=marker,
            markersize=10,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            linewidth=1.4,
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            label=label,
            zorder=3,
        )

    if compressibility == "neuron":
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$", fontsize=15, labelpad=8)
    else:
        ax.set_xlabel("Layer Rank", fontsize=15, labelpad=8)

    ax.set_ylabel("Metric Value", fontsize=15, labelpad=10)
    ax.legend(
        fontsize=13,
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        loc="best",
        handlelength=2,
        borderpad=0.9,
        labelspacing=0.6,
    )

    _apply_elegant_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_repr_shift_figure(agg: Dict[str, dict], save_path: Path, compressibility: str) -> None:
    xs = _sorted_x_values(agg, compressibility)

    ratio_mean = np.array([agg[str(x)]["repr_shift_ratio_mean"] for x in xs])
    ratio_std = np.array([agg[str(x)]["repr_shift_ratio_std"] for x in xs])

    color = "#3BB273"
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    ax.fill_between(xs, ratio_mean - ratio_std, ratio_mean + ratio_std, color=color, alpha=0.12, linewidth=0)
    ax.errorbar(
        xs,
        ratio_mean,
        yerr=ratio_std,
        color=color,
        marker="o",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.5,
        linewidth=1.4,
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=3,
    )

    if compressibility == "neuron":
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$", fontsize=15, labelpad=8)
    else:
        ax.set_xlabel("Layer Rank", fontsize=15, labelpad=8)

    ax.set_ylabel(r"$\|z_{\mathrm{adv}} - z\|_2 / \|z\|_2$", fontsize=13, labelpad=10)

    _apply_elegant_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--compressibility", type=str, required=True, choices=["neuron", "spectral"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    parser.add_argument("--n-betas", type=int, default=15)
    parser.add_argument("--beta-min", type=float, default=1e-5)
    parser.add_argument("--beta-max", type=float, default=1e-1)

    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])

    parser.add_argument("--attack", type=str, default="fgsm", choices=["fgsm", "autopgd"])
    parser.add_argument("--norm", type=str, default="linf", choices=["linf", "l2"])

    parser.add_argument("--eps-linf", type=float, default=8.0 / 255.0)
    parser.add_argument("--eps-step-linf", type=float, default=None)

    parser.add_argument("--eps-l2", type=float, default=0.5)
    parser.add_argument("--eps-step-l2", type=float, default=None)

    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--nb-random-init", type=int, default=5)
    parser.add_argument("--attack-batch-size", type=int, default=128)
    parser.add_argument("--repr-samples-cap", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default=None)

    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = (
            f"outputs/compressibility_adv_robustness/wrn16_2/{args.compressibility}/"
            f"robustness_sweep_{args.attack}_{args.norm}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_results_path = save_dir / "raw_results.json"
    agg_results_path = save_dir / "aggregated_results.json"

    raw_results = {}

    if args.compressibility == "neuron":
        sweep_values = [float(x) for x in np.geomspace(args.beta_min, args.beta_max, args.n_betas)]
        x_key_name = "beta"
    else:
        sweep_values = [int(x) for x in args.ranks]
        x_key_name = "rank"

    missing = [
        (v, seed)
        for v in sweep_values
        for seed in args.seeds
        if not any(
            r[x_key_name] == v and r["seed"] == seed
            for r in raw_results.get(str(v), [])
        )
    ]
    missing = True

    if not missing:
        print("All runs already completed. Skipping evaluation and regenerating plots.")
    else:
        cfg = load_config(args.config)
        cfg = copy.deepcopy(cfg)
        cfg.dataset.augmentation.normalize = False
        bundle = build_datasets(cfg)
        _, _, test_loader = build_dataloaders(cfg, bundle.train, bundle.val, bundle.test, device)

        for value in sweep_values:
            value_key = str(value)
            if value_key not in raw_results:
                raw_results[value_key] = []

            for seed in args.seeds:
                already_done = any(
                    r[x_key_name] == value and r["seed"] == seed
                    for r in raw_results[value_key]
                )
                if already_done:
                    print(f"Skipping {x_key_name}={value}, seed={seed} (already in results)")
                    continue

                ckpt_path = find_checkpoint(
                    base_dir=base_dir,
                    compressibility=args.compressibility,
                    value=value,
                    seed=seed,
                    ckpt_name=args.ckpt,
                )

                print(f"\nEvaluating {x_key_name}={value}, seed={seed}")
                print(f"Checkpoint: {ckpt_path}")

                model = load_model_from_checkpoint(
                    ckpt_path=ckpt_path,
                    device=device,
                    compressibility=args.compressibility,
                    rank=int(value) if args.compressibility == "spectral" else None,
                )

                clean_acc = clean_accuracy(model, test_loader, device=device)
                robust_acc, repr_ratio = robust_accuracy_and_repr_shift(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    attack=args.attack,
                    norm=args.norm,
                    eps_linf=args.eps_linf,
                    eps_l2=args.eps_l2,
                    eps_step_linf=args.eps_step_linf,
                    eps_step_l2=args.eps_step_l2,
                    max_iter=args.max_iter,
                    nb_random_init=args.nb_random_init,
                    attack_batch_size=args.attack_batch_size,
                    repr_samples_cap=args.repr_samples_cap,
                )

                robust_over_clean = robust_acc / max(clean_acc, 1e-12)

                row = {
                    x_key_name: value,
                    "seed": int(seed),
                    "attack": args.attack,
                    "norm": args.norm,
                    "checkpoint": str(ckpt_path),
                    "clean_test_accuracy": float(clean_acc),
                    "robust_test_accuracy": float(robust_acc),
                    "robust_over_clean": float(robust_over_clean),
                    "repr_shift_ratio": float(repr_ratio),
                }
                raw_results[value_key].append(row)

                print(
                    f"attack={args.attack} | norm={args.norm} | "
                    f"clean_acc={clean_acc:.4f} | robust_acc={robust_acc:.4f} | "
                    f"robust/clean={robust_over_clean:.4f} | repr_shift_ratio={repr_ratio:.4f}"
                )

                save_json(raw_results_path, raw_results)

    agg = aggregate_over_seeds(raw_results, x_key_name=x_key_name)

    save_json(raw_results_path, raw_results)
    save_json(agg_results_path, agg)

    plot_accuracy_figure(agg, save_dir / "clean_robust_and_ratio_curve.png", compressibility=args.compressibility)
    plot_repr_shift_figure(agg, save_dir / "repr_shift_ratio_curve.png", compressibility=args.compressibility)

    print(f"\nSaved raw results to: {raw_results_path}")
    print(f"Saved aggregated results to: {agg_results_path}")
    print(f"Saved figure to: {save_dir / 'clean_robust_and_ratio_curve.png'}")
    print(f"Saved figure to: {save_dir / 'repr_shift_ratio_curve.png'}")


if __name__ == "__main__":
    main()