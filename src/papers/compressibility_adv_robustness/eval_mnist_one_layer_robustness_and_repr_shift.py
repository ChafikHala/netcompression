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
from src.models.fcn import FCN
from src.papers.compressibility_adv_robustness.attack_utils import (
    build_autopgd_l2_for_mnist,
    build_fgsm_l2_for_mnist,
    generate_adversarial_batch,
)
from src.utils.config import load_config


def alpha_to_name(alpha: float) -> str:
    if float(alpha).is_integer():
        return str(int(alpha))
    return str(alpha).replace(".", "p")


def run_name(alpha: float, seed: int) -> str:
    return f"model_one_layer_FC_alpha_{alpha_to_name(alpha)}_seed_{seed}"


def build_model() -> FCN:
    return FCN(
        input_shape=[1, 28, 28],
        hidden_dims=[400],
        num_classes=2,
        dropout=0.0,
        bias=False,
    )


def get_hidden_representation(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    flat = model.net[0](x)
    h = model.net[1](flat)
    h = model.net[2](h)
    return h


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def find_checkpoint(base_dir: Path, alpha: float, seed: int, ckpt_name: str) -> Path:
    path = (
        base_dir
        / "compressibility_adv_robustness"
        / run_name(alpha, seed)
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
    attack: str = "autopgd",
    eps_l2: float = 2,
    eps_step_l2: float | None = None,
    max_iter: int = 100,
    nb_random_init: int = 5,
    attack_batch_size: int = 128,
    repr_samples_cap: int = 1000,
) -> tuple[float, float]:
    model.eval()

    attack = attack.lower()
    if attack == "autopgd":
        attacker = build_autopgd_l2_for_mnist(
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
        attacker = build_fgsm_l2_for_mnist(
            model=model,
            device=device,
            eps_l2_original_space=eps_l2,
            batch_size=attack_batch_size,
        )
    else:
        raise ValueError(f"Unsupported attack: {attack}")

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


def aggregate_over_seeds(raw_results: dict[str, list[dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for alpha_key, rows in raw_results.items():
        clean_vals = [r["clean_test_accuracy"] for r in rows]
        robust_vals = [r["robust_test_accuracy"] for r in rows]
        repr_vals = [r["repr_shift_ratio"] for r in rows]

        out[alpha_key] = {
            "alpha": float(alpha_key),
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

def plot_accuracy_figure(agg: dict[str, dict], save_path: Path) -> None:
    alphas = np.array(sorted([float(k) for k in agg.keys()]))

    clean_mean  = np.array([agg[str(a)]["clean_test_accuracy_mean"]  for a in alphas])
    clean_std   = np.array([agg[str(a)]["clean_test_accuracy_std"]   for a in alphas])
    robust_mean = np.array([agg[str(a)]["robust_test_accuracy_mean"] for a in alphas])
    robust_std  = np.array([agg[str(a)]["robust_test_accuracy_std"]  for a in alphas])


    PALETTE = ["#4169E1", "#228B22"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    for mean, std, marker, color, label in [
        (clean_mean,  clean_std,  "o", PALETTE[0], r"Clean Acc."),
        (robust_mean, robust_std, "D", PALETTE[1], "Rob. Acc."),
    ]:
        ax.fill_between(alphas, mean - std, mean + std,
                        color=color, alpha=0.12, linewidth=0)
        ax.errorbar(
            alphas, mean,
            yerr=std,
            color=color,
            marker=marker,
            markersize=5,
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

    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$", fontsize=15, labelpad=8)
    ax.set_ylabel("Test Accuracy", fontsize=15, labelpad=10)
    ax.legend(fontsize=15, frameon=True, framealpha=0.95,
              edgecolor="#DDDDDD", loc="best",
              handlelength=2, borderpad=0.9, labelspacing=0.6)

    _apply_elegant_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_repr_shift_figure(agg: dict[str, dict], save_path: Path) -> None:
    alphas = np.array(sorted([float(k) for k in agg.keys()]))

    ratio_mean = np.array([agg[str(a)]["repr_shift_ratio_mean"] for a in alphas])
    ratio_std  = np.array([agg[str(a)]["repr_shift_ratio_std"]  for a in alphas])

    COLOR = "#3BB273"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    ax.fill_between(alphas, ratio_mean - ratio_std, ratio_mean + ratio_std,
                    color=COLOR, alpha=0.12, linewidth=0)
    ax.errorbar(
        alphas, ratio_mean,
        yerr=ratio_std,
        color=COLOR,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgecolor=COLOR,
        markeredgewidth=1.5,
        linewidth=1.4,
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$", fontsize=15, labelpad=8)

    ax.set_ylabel(
        r"$\||z_{\mathrm{adv}} - z\||_2$" + r"$/ \;\||z\||_2$",
        fontsize=13,
        labelpad=10,
    )
    _apply_elegant_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-alphas", type=int, default=15)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=3e-1)
    parser.add_argument("--eps-l2", type=float, default=2)
    parser.add_argument("--eps-step-l2", type=float, default=None)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--nb-random-init", type=int, default=5)
    parser.add_argument("--attack-batch-size", type=int, default=128)
    parser.add_argument("--repr-samples-cap", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default=None,)
    parser.add_argument("--attack", type=str, default="fgsm", choices=["autopgd", "fgsm"],)

    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = f"outputs/compressibility_adv_robustness/robustness_alpha_sweep_{args.attack}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)

    bundle = build_datasets(cfg)
    _, _, test_loader = build_dataloaders(cfg, bundle.train, bundle.val, bundle.test, device)

    alphas = np.geomspace(args.alpha_min, args.alpha_max, args.n_alphas)

    raw_results: dict[str, list[dict]] = {}

    for alpha in alphas:
        alpha = float(alpha)
        alpha_key = str(alpha)
        raw_results[alpha_key] = []

        for seed in args.seeds:
            ckpt_path = find_checkpoint(base_dir, alpha, seed, args.ckpt)
            print(f"\nEvaluating alpha={alpha:.8f}, seed={seed}")
            print(f"Checkpoint: {ckpt_path}")

            model = load_model_from_checkpoint(ckpt_path, device=device)

            clean_acc = clean_accuracy(model, test_loader, device=device)
            robust_acc, repr_ratio = robust_accuracy_and_repr_shift(
                model=model,
                dataloader=test_loader,
                device=device,
                attack=args.attack,
                eps_l2=args.eps_l2,
                eps_step_l2=args.eps_step_l2,
                max_iter=args.max_iter,
                nb_random_init=args.nb_random_init,
                attack_batch_size=args.attack_batch_size,
                repr_samples_cap=args.repr_samples_cap,
            )

            row = {
                "alpha": alpha,
                "seed": int(seed),
                "attack": args.attack,
                "checkpoint": str(ckpt_path),
                "clean_test_accuracy": float(clean_acc),
                "robust_test_accuracy": float(robust_acc),
                "repr_shift_ratio": float(repr_ratio),
            }
            raw_results[alpha_key].append(row)

            print(
                f"attack={args.attack} | "
                f"clean_acc={clean_acc:.4f} | "
                f"robust_acc={robust_acc:.4f} | "
                f"repr_shift_ratio={repr_ratio:.4f}"
            )

    agg = aggregate_over_seeds(raw_results)

    save_json(save_dir / "raw_results.json", raw_results)
    save_json(save_dir / "aggregated_results.json", agg)

    plot_accuracy_figure(agg, save_dir / "std_and_robust_test_accuracy_vs_alpha.png")
    plot_repr_shift_figure(agg, save_dir / "repr_shift_ratio_vs_alpha.png")

    print(f"\nSaved raw results to: {save_dir / 'raw_results.json'}")
    print(f"Saved aggregated results to: {save_dir / 'aggregated_results.json'}")
    print(f"Saved figure to: {save_dir / 'std_and_robust_test_accuracy_vs_alpha.png'}")
    print(f"Saved figure to: {save_dir / 'repr_shift_ratio_vs_alpha.png'}")


if __name__ == "__main__":
    main()