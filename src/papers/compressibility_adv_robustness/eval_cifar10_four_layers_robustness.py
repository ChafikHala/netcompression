from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.core.checkpoint import load_checkpoint
from src.data.dataloaders import build_dataloaders
from src.data.datasets import build_datasets
from src.papers.compressibility_adv_robustness.attack_utils import (
    build_autopgd_l2_for_cifar10,
    build_autopgd_linf_for_cifar10,
    build_fgsm_l2_for_cifar10,
    build_fgsm_linf_for_cifar10,
    generate_adversarial_batch,
)
from src.papers.compressibility_adv_robustness.train_cifar10_four_layers_fcn import (
    DenseFCN,
    LowRankFCN,
)
from src.utils.config import load_config


def alpha_to_name(alpha: float) -> str:
    if float(alpha).is_integer():
        return str(int(alpha))
    return str(alpha).replace(".", "p")


def run_name(compressibility: str, alpha: float | None, rank: int | None, seed: int) -> str:
    if compressibility == "neuron":
        return f"cifar10_fcn_group_lasso_alpha_{alpha_to_name(float(alpha))}_seed_{seed}"
    if compressibility == "spectral":
        return f"cifar10_fcn_low_rank_rank_{int(rank)}_seed_{seed}"
    raise ValueError(f"Unknown compressibility: {compressibility}")


def build_model(compressibility: str, rank: int | None = None) -> nn.Module:
    if compressibility == "neuron":
        return DenseFCN(
            input_dim=3 * 32 * 32,
            hidden_dim=2000,
            num_hidden_layers=4,
            num_classes=10,
            bias=False,
        )
    if compressibility == "spectral":
        if rank is None:
            raise ValueError("rank must be provided for spectral compressibility.")
        return LowRankFCN(
            input_dim=3 * 32 * 32,
            hidden_dim=2000,
            num_hidden_layers=4,
            num_classes=10,
            rank=rank,
            bias=False,
        )
    raise ValueError(f"Unknown compressibility: {compressibility}")


def get_hidden_representation(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Return the last hidden representation before the classification layer.
    Assumes Sequential layout alternating Linear/ReLU blocks, with final classifier at the end.
    """
    x = x.view(x.size(0), -1)

    # forward through all but last classification layer
    # DenseFCN / LowRankFCN are both Sequential in .net
    for layer in list(model.net.children())[:-1]:
        x = layer(x)
    return x


def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    compressibility: str,
    rank: int | None = None,
) -> nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_model(compressibility=compressibility, rank=rank)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def find_checkpoint(
    base_dir: Path,
    compressibility: str,
    alpha: float | None,
    rank: int | None,
    seed: int,
    ckpt_name: str,
) -> Path:
    path = (
        base_dir
        / "compressibility_adv_robustness"
        / compressibility
        / run_name(compressibility, alpha, rank, seed)
        / "checkpoints"
        / ckpt_name
    )
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


@torch.no_grad()
def clean_accuracy(model: nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    return correct / max(total, 1)


def robust_accuracy_and_repr_shift(
    model: nn.Module,
    dataloader,
    device: torch.device,
    compressibility: str,
    attack: str = "autopgd",
    eps_l2: float = 0.125,
    eps_step_l2: float | None = None,
    eps_linf: float = 2 / 255,
    eps_step_linf: float | None = None,
    max_iter: int = 100,
    nb_random_init: int = 5,
    attack_batch_size: int = 128,
    repr_samples_cap: int = 1000,
) -> tuple[float, float]:
    model.eval()

    attack = attack.lower()

    # Default norm choice from the paper:
    # neuron compressibility -> Linf
    # spectral compressibility -> L2
    if compressibility == "neuron":
        if attack == "autopgd":
            attacker = build_autopgd_linf_for_cifar10(
                model=model,
                device=device,
                eps_linf_original_space=eps_linf,
                eps_step_linf_original_space=eps_step_linf,
                max_iter=max_iter,
                nb_random_init=nb_random_init,
                batch_size=attack_batch_size,
                loss_type="cross_entropy",
                verbose=False,
            )
        elif attack == "fgsm":
            attacker = build_fgsm_linf_for_cifar10(
                model=model,
                device=device,
                eps_linf_original_space=eps_linf,
                batch_size=attack_batch_size,
            )
        else:
            raise ValueError(f"Unsupported attack: {attack}")

    elif compressibility == "spectral":
        if attack == "autopgd":
            attacker = build_autopgd_l2_for_cifar10(
                model=model,
                device=device,
                eps_l2_original_space=eps_l2,
                eps_step_l2_original_space=eps_step_l2,
                max_iter=max_iter,
                nb_random_init=nb_random_init,
                batch_size=attack_batch_size,
                loss_type="cross_entropy",
                verbose=False,
            )
        elif attack == "fgsm":
            attacker = build_fgsm_l2_for_cifar10(
                model=model,
                device=device,
                eps_l2_original_space=eps_l2,
                batch_size=attack_batch_size,
            )
        else:
            raise ValueError(f"Unsupported attack: {attack}")
    else:
        raise ValueError(f"Unknown compressibility: {compressibility}")

    robust_correct = 0
    total = 0
    repr_ratios: list[float] = []
    counted_repr = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_adv = generate_adversarial_batch(attacker, x, y, device=device)

        with torch.no_grad():
            logits_adv = model(x_adv)
            pred_adv = logits_adv.argmax(dim=1)
            robust_correct += int((pred_adv == y).sum().item())
            total += int(y.size(0))

            if counted_repr < repr_samples_cap:
                m = min(x.size(0), repr_samples_cap - counted_repr)
                z = get_hidden_representation(model, x[:m])
                z_adv = get_hidden_representation(model, x_adv[:m])

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


def aggregate_over_seeds(raw_results: dict[str, list[dict]], sweep_type: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sweep_key, rows in raw_results.items():
        clean_vals = [r["clean_test_accuracy"] for r in rows]
        robust_vals = [r["robust_test_accuracy"] for r in rows]
        repr_vals = [r["repr_shift_ratio"] for r in rows]

        out[sweep_key] = {
            sweep_type: float(sweep_key) if sweep_type == "alpha" else int(float(sweep_key)),
            "clean_test_accuracy_mean": float(np.mean(clean_vals)),
            "clean_test_accuracy_std": float(np.std(clean_vals, ddof=0)),
            "robust_test_accuracy_mean": float(np.mean(robust_vals)),
            "robust_test_accuracy_std": float(np.std(robust_vals, ddof=0)),
            "repr_shift_ratio_mean": float(np.mean(repr_vals)),
            "repr_shift_ratio_std": float(np.std(repr_vals, ddof=0)),
            "n_seeds": len(rows),
        }
    return out


def _apply_elegant_style(ax):
    """Shared style: fine grey grid, clean spines, high-quality look."""
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


def plot_accuracy_figure(agg: dict[str, dict], save_path: Path, sweep_type: str) -> None:
    xs = np.array(sorted([float(k) for k in agg.keys()]))
    key = lambda x: str(int(x)) if sweep_type == "rank" else str(x)

    clean_mean  = np.array([agg[key(x)]["clean_test_accuracy_mean"]  for x in xs])
    clean_std   = np.array([agg[key(x)]["clean_test_accuracy_std"]   for x in xs])
    robust_mean = np.array([agg[key(x)]["robust_test_accuracy_mean"] for x in xs])
    robust_std  = np.array([agg[key(x)]["robust_test_accuracy_std"]  for x in xs])

    # Ratio: robust / clean
    ratio_mean = robust_mean / np.where(clean_mean == 0, np.nan, clean_mean)
    ratio_std  = ratio_mean * np.sqrt(
        (robust_std / np.where(robust_mean == 0, np.nan, robust_mean))**2 +
        (clean_std  / np.where(clean_mean  == 0, np.nan, clean_mean ))**2
    )

    PALETTE = ["#4169E1", "#228B22", "#9B1C31"]   # royal blue, forest/royal green

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    for mean, std, marker, color, label in [
        (ratio_mean, ratio_std,  "o", PALETTE[0], r"Rob. Acc. / Clean. Acc."),
        (robust_mean, robust_std, "D", PALETTE[1], "Rob. Acc."),
        (clean_mean, clean_std, "^", PALETTE[2], "Clean Acc.")
    ]:
        ax.fill_between(xs, mean - std, mean + std,
                        facecolor=color, alpha=0.12, linewidth=0)
        ax.errorbar(
            xs, mean,
            yerr=std,
            color=color,
            marker=marker,
            markersize=7,
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

    if sweep_type == "alpha":
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$", fontsize=18, labelpad=8)
    else:
        ax.set_xlabel("Rank", fontsize=18, labelpad=8)

    ax.set_ylabel("Test Accuracy", fontsize=18, labelpad=10)
    ax.legend(fontsize=8, frameon=True, framealpha=0.95,
              edgecolor="#DDDDDD", loc="best",
              handlelength=2, borderpad=0.9, labelspacing=0.6)

    _apply_elegant_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_repr_shift_figure(agg: dict[str, dict], save_path: Path, sweep_type: str) -> None:
    xs = np.array(sorted([float(k) for k in agg.keys()]))
    key = lambda x: str(int(x)) if sweep_type == "rank" else str(x)

    ratio_mean = np.array([agg[key(x)]["repr_shift_ratio_mean"] for x in xs])
    ratio_std  = np.array([agg[key(x)]["repr_shift_ratio_std"]  for x in xs])

    COLOR = "#3BB273"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    ax.fill_between(xs, ratio_mean - ratio_std, ratio_mean + ratio_std,
                    color=COLOR, alpha=0.12, linewidth=0)
    ax.errorbar(
        xs, ratio_mean,
        yerr=ratio_std,
        color=COLOR,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=COLOR,
        markeredgewidth=1.5,
        linewidth=1.4,
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=3,
    )

    if sweep_type == "alpha":
        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$", fontsize=18, labelpad=8)
    else:
        ax.set_xlabel("Rank", fontsize=18, labelpad=8)

    ax.set_ylabel(
        r"$\||z_{\mathrm{adv}} - z\||_2$" + r"$/ \;\||z\||_2$",
        fontsize=16,
        labelpad=10,
    )

    _apply_elegant_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="outputs_h")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--compressibility", type=str, required=True, choices=["neuron", "spectral"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    # neuron / alpha sweep
    parser.add_argument("--n-alphas", type=int, default=15)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=1e-1)

    # spectral / rank sweep
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])

    # attacks
    parser.add_argument("--attack", type=str, default="autopgd", choices=["autopgd", "fgsm"])
    parser.add_argument("--eps-l2", type=float, default=0.125)
    parser.add_argument("--eps-step-l2", type=float, default=None)
    parser.add_argument("--eps-linf", type=float, default=2 / 255)
    parser.add_argument("--eps-step-linf", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--nb-random-init", type=int, default=5)
    parser.add_argument("--attack-batch-size", type=int, default=128)

    parser.add_argument("--repr-samples-cap", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default=None)

    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"outputs_h/compressibility_adv_robustness/robustness_{args.compressibility}_{args.attack}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_results_path = save_dir / "raw_results.json"
    agg_results_path = save_dir / "aggregated_results.json"

    # Load existing results if available, skipping already-completed runs
    if raw_results_path.exists():
        print(f"Found existing raw results at {raw_results_path}, loading...")
        with open(raw_results_path) as f:
            raw_results = json.load(f)
    else:
        raw_results = {}

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)

    if args.compressibility == "neuron":
        sweep_values = [float(a) for a in np.geomspace(args.alpha_min, args.alpha_max, args.n_alphas)]
        sweep_type = "alpha"
    else:
        sweep_values = [int(r) for r in args.ranks]
        sweep_type = "rank"

    # Check if any runs are missing before loading datasets
    missing = [
        (sv, seed)
        for sv in sweep_values
        for seed in args.seeds
        if not any(
            r[sweep_type] == sv and r["seed"] == seed
            for r in raw_results.get(str(sv), [])
        )
    ]

    if not missing:
        print("All runs already completed. Skipping evaluation and regenerating plots.")
    else:
        bundle = build_datasets(cfg)
        _, _, test_loader = build_dataloaders(cfg, bundle.train, bundle.val, bundle.test, device)

        for sweep_value in sweep_values:
            sweep_key = str(sweep_value)
            if sweep_key not in raw_results:
                raw_results[sweep_key] = []

            for seed in args.seeds:
                already_done = any(
                    r[sweep_type] == sweep_value and r["seed"] == seed
                    for r in raw_results[sweep_key]
                )
                if already_done:
                    print(f"Skipping {sweep_type}={sweep_value}, seed={seed} (already in results)")
                    continue

                alpha = float(sweep_value) if args.compressibility == "neuron" else None
                rank = int(sweep_value) if args.compressibility == "spectral" else None

                ckpt_path = find_checkpoint(
                    base_dir=base_dir,
                    compressibility=args.compressibility,
                    alpha=alpha,
                    rank=rank,
                    seed=seed,
                    ckpt_name=args.ckpt,
                )

                print(f"\nEvaluating {sweep_type}={sweep_value}, seed={seed}")
                print(f"Checkpoint: {ckpt_path}")

                model = load_model_from_checkpoint(
                    ckpt_path,
                    device=device,
                    compressibility=args.compressibility,
                    rank=rank,
                )

                clean_acc = clean_accuracy(model, test_loader, device=device)
                robust_acc, repr_ratio = robust_accuracy_and_repr_shift(
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    compressibility=args.compressibility,
                    attack=args.attack,
                    eps_l2=args.eps_l2,
                    eps_step_l2=args.eps_step_l2,
                    eps_linf=args.eps_linf,
                    eps_step_linf=args.eps_step_linf,
                    max_iter=args.max_iter,
                    nb_random_init=args.nb_random_init,
                    attack_batch_size=args.attack_batch_size,
                    repr_samples_cap=args.repr_samples_cap,
                )

                row = {
                    sweep_type: sweep_value,
                    "seed": int(seed),
                    "compressibility": args.compressibility,
                    "attack": args.attack,
                    "checkpoint": str(ckpt_path),
                    "clean_test_accuracy": float(clean_acc),
                    "robust_test_accuracy": float(robust_acc),
                    "repr_shift_ratio": float(repr_ratio),
                }
                raw_results[sweep_key].append(row)

                print(
                    f"attack={args.attack} | "
                    f"clean_acc={clean_acc:.4f} | "
                    f"robust_acc={robust_acc:.4f} | "
                    f"repr_shift_ratio={repr_ratio:.4f}"
                )

                # Save incrementally after each run so progress isn't lost
                save_json(raw_results_path, raw_results)

    agg = aggregate_over_seeds(raw_results, sweep_type=sweep_type)

    save_json(raw_results_path, raw_results)
    save_json(agg_results_path, agg)

    plot_accuracy_figure(agg, save_dir / "std_and_robust_test_accuracy.png", sweep_type=sweep_type)
    plot_repr_shift_figure(agg, save_dir / "repr_shift_ratio.png", sweep_type=sweep_type)

    print(f"\nSaved raw results to: {raw_results_path}")
    print(f"Saved aggregated results to: {agg_results_path}")
    print(f"Saved figure to: {save_dir / 'std_and_robust_test_accuracy.png'}")
    print(f"Saved figure to: {save_dir / 'repr_shift_ratio.png'}")

if __name__ == "__main__":
    main()