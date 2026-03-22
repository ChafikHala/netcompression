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
from src.core.evaluator import evaluate
from src.data.dataloaders import build_dataloaders
from src.data.datasets import build_datasets
from src.models.fcn import FCN
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


def get_two_linear_layers(model: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    if not hasattr(model, "net"):
        raise ValueError("Expected FCN model with attribute `net`.")

    linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]
    if len(linear_layers) != 2:
        raise ValueError(f"Expected exactly 2 Linear layers, found {len(linear_layers)}.")

    return linear_layers[0], linear_layers[1]


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def hidden_neuron_scores_l2(model: nn.Module) -> torch.Tensor:
    fc1, _ = get_two_linear_layers(model)
    w1 = fc1.weight.detach()
    return torch.linalg.vector_norm(w1, ord=2, dim=1)


@torch.no_grad()
def prune_hidden_neurons_by_ratio(model: nn.Module, pruning_ratio: float) -> nn.Module:
    if not (0.0 <= pruning_ratio < 1.0):
        raise ValueError(f"pruning_ratio must be in [0,1), got {pruning_ratio}")

    pruned_model = copy.deepcopy(model)
    fc1, _ = get_two_linear_layers(pruned_model)

    scores = hidden_neuron_scores_l2(pruned_model)
    hidden_dim = scores.numel()
    n_prune = int(round(pruning_ratio * hidden_dim))

    if n_prune <= 0:
        return pruned_model
    if n_prune >= hidden_dim:
        raise ValueError(
            f"Pruning ratio {pruning_ratio} would prune all neurons "
            f"(hidden_dim={hidden_dim}, n_prune={n_prune})."
        )

    prune_idx = torch.argsort(scores)[:n_prune]
    fc1.weight[prune_idx, :] = 0.0
    return pruned_model


def find_checkpoint(base_dir: Path, alpha: float, seed: int, ckpt_name: str) -> Path:
    ckpt_path = (
        base_dir
        / "compressibility_adv_robustness"
        / run_name(alpha, seed)
        / "checkpoints"
        / ckpt_name
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def make_ratio_grid(start: float, end: float, step: float) -> list[float]:
    vals = []
    x = start
    while x <= end + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals

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

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.05])
    parser.add_argument("--ratio-start", type=float, default=0.95)
    parser.add_argument("--ratio-end", type=float, default=0.99)
    parser.add_argument("--ratio-step", type=float, default=0.0025)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/compressibility_adv_robustness/neuron_pruning_multi_seed",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    bundle = build_datasets(cfg)
    _, _, test_loader = build_dataloaders(cfg, bundle.train, bundle.val, bundle.test, device)

    ratios = make_ratio_grid(args.ratio_start, args.ratio_end, args.ratio_step)

    raw_results: dict[str, list[dict]] = {}
    aggregated: dict[str, dict] = {}

    for alpha in args.alphas:
        alpha_key = str(alpha)
        raw_results[alpha_key] = []

        seed_acc_matrix = []

        for seed in args.seeds:
            ckpt_path = find_checkpoint(base_dir, alpha, seed, args.ckpt)
            model = load_model_from_checkpoint(ckpt_path, device=device)

            seed_rows = []
            seed_accs = []

            print(f"\n[alpha={alpha}, seed={seed}] checkpoint={ckpt_path}")

            for ratio in ratios:
                pruned_model = prune_hidden_neurons_by_ratio(model, pruning_ratio=ratio)
                res = evaluate(pruned_model, test_loader, nn.CrossEntropyLoss(), device)

                row = {
                    "alpha": float(alpha),
                    "seed": int(seed),
                    "pruning_ratio": float(ratio),
                    "test_loss": float(res.loss),
                    "test_accuracy": float(res.accuracy),
                }
                seed_rows.append(row)
                seed_accs.append(float(res.accuracy))

                print(
                    f"  ratio={ratio:.4f} | "
                    f"test_loss={res.loss:.4f} | "
                    f"test_acc={res.accuracy:.4f}"
                )

            raw_results[alpha_key].extend(seed_rows)
            seed_acc_matrix.append(seed_accs)

        seed_acc_matrix = np.array(seed_acc_matrix, dtype=float)  # [n_seeds, n_ratios]
        mean_acc = seed_acc_matrix.mean(axis=0)
        std_acc = seed_acc_matrix.std(axis=0, ddof=0)

        aggregated[alpha_key] = {
            "alpha": float(alpha),
            "pruning_ratios": ratios,
            "test_accuracy_mean": mean_acc.tolist(),
            "test_accuracy_std": std_acc.tolist(),
            "n_seeds": len(args.seeds),
        }

    save_json(save_dir / "mnist_one_layer_neuron_pruning_raw.json", raw_results)
    save_json(save_dir / "mnist_one_layer_neuron_pruning_aggregated.json", aggregated)
    PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F18F01",
               "#9B59B6", "#E67E22", "#1ABC9C", "#E74C3C"]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    for i, alpha in enumerate(args.alphas):
        stats = aggregated[str(alpha)]
        xs = np.array(stats["pruning_ratios"], dtype=float)
        ys = np.array(stats["test_accuracy_mean"], dtype=float)
        yerr = np.array(stats["test_accuracy_std"], dtype=float) * 0.1

        color = PALETTE[i % len(PALETTE)]

        ax.fill_between(xs, ys - yerr, ys + yerr,
                        color=color, alpha=0.12, linewidth=0)
        ax.errorbar(
            xs, ys,
            yerr=yerr,
            color=color,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            linewidth=1.4,
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            label=rf"$\alpha = {alpha}$",
            zorder=3,
        )

    ax.set_xlabel("Pruning Ratio", fontsize=15, labelpad=8)
    ax.set_ylabel("Test Accuracy", fontsize=15, labelpad=10)
    ax.legend(fontsize=15, frameon=True, framealpha=0.95,
              edgecolor="#DDDDDD", loc="best",
              handlelength=2, borderpad=0.9, labelspacing=0.6)

    _apply_elegant_style(ax)
    plt.tight_layout()
    plt.savefig(save_dir / "mnist_one_layer_neuron_pruning_accuracy_mean_std.png",
                dpi=180, bbox_inches="tight")
    plt.close()

    print(f"\nSaved raw results to: {save_dir / 'mnist_one_layer_neuron_pruning_raw.json'}")
    print(f"Saved aggregated results to: {save_dir / 'mnist_one_layer_neuron_pruning_aggregated.json'}")
    print(f"Saved figure to: {save_dir / 'mnist_one_layer_neuron_pruning_accuracy_mean_std.png'}")

if __name__ == "__main__":
    main()