from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
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


def build_model() -> FCN:
    return FCN(
        input_shape=[1, 28, 28],
        hidden_dims=[400],
        num_classes=10,
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
    """
    Score hidden neurons by L2 norm of rows of the first layer weight.
    Shape: [hidden_dim]
    """
    fc1, _ = get_two_linear_layers(model)
    w1 = fc1.weight.detach()  # [hidden_dim, in_dim]
    scores = torch.linalg.vector_norm(w1, ord=2, dim=1)
    return scores



@torch.no_grad()
def prune_hidden_neurons_by_ratio(model: nn.Module, pruning_ratio: float) -> nn.Module:
    """
    Prune the smallest-magnitude hidden neurons by zeroing only the
    corresponding rows of the first layer weight matrix.

    This matches the simpler operation: set the lowest-row-magnitude rows
    to zero, without touching the first-layer bias or the second-layer weight.
    """
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


# @torch.no_grad()
# def prune_hidden_neurons_by_ratio(model: nn.Module, pruning_ratio: float) -> nn.Module:
#     """
#     Prune the smallest-magnitude hidden neurons.

#     For neuron j, pruning means zeroing:
#       - row j of first-layer weight
#       - bias j of first layer
#       - column j of second-layer weight
#     """
#     if not (0.0 <= pruning_ratio < 1.0):
#         raise ValueError(f"pruning_ratio must be in [0,1), got {pruning_ratio}")

#     pruned_model = copy.deepcopy(model)
#     fc1, fc2 = get_two_linear_layers(pruned_model)

#     scores = hidden_neuron_scores_l2(pruned_model)
#     hidden_dim = scores.numel()
#     n_prune = int(round(pruning_ratio * hidden_dim))

#     if n_prune <= 0:
#         return pruned_model
#     if n_prune >= hidden_dim:
#         raise ValueError(
#             f"Pruning ratio {pruning_ratio} would prune all neurons "
#             f"(hidden_dim={hidden_dim}, n_prune={n_prune})."
#         )

#     prune_idx = torch.argsort(scores)[:n_prune]

#     fc1.weight[prune_idx, :] = 0.0
#     if fc1.bias is not None:
#         fc1.bias[prune_idx] = 0.0

#     fc2.weight[:, prune_idx] = 0.0

#     return pruned_model


def find_checkpoint(base_dir: Path, alpha: float, seed: int, ckpt_name: str) -> Path:
    exp_name = f"model_one_layer_FC_alpha_{alpha_to_name(alpha)}"
    ckpt_path = (
        base_dir
        / "compressibility_adv_robustness"
        / exp_name
        / f"seed_{seed}"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.05],
    )
    parser.add_argument("--ratio-start", type=float, default=0.95)
    parser.add_argument("--ratio-end", type=float, default=0.99)
    parser.add_argument("--ratio-step", type=float, default=0.0025)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/compressibility_adv_robustness/neuron_pruning",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build held-out test set exactly from config
    cfg = load_config(args.config)
    bundle = build_datasets(cfg)
    _, _, test_loader = build_dataloaders(cfg, bundle.train, bundle.val, bundle.test, device)

    ratios = make_ratio_grid(args.ratio_start, args.ratio_end, args.ratio_step)

    all_results: dict[str, list[dict]] = {}

    for alpha in args.alphas:
        ckpt_path = find_checkpoint(base_dir, alpha, args.seed, args.ckpt)
        model = load_model_from_checkpoint(ckpt_path, device=device)

        alpha_key = str(alpha)
        all_results[alpha_key] = []

        print(f"\n[alpha={alpha}] checkpoint={ckpt_path}")

        for ratio in ratios:
            pruned_model = prune_hidden_neurons_by_ratio(model, pruning_ratio=ratio)
            res = evaluate(
                pruned_model,
                test_loader,
                nn.CrossEntropyLoss(),
                device,
            )

            row = {
                "alpha": float(alpha),
                "pruning_ratio": float(ratio),
                "test_loss": float(res.loss),
                "test_accuracy": float(res.accuracy),
            }
            all_results[alpha_key].append(row)

            print(
                f"  ratio={ratio:.4f} | "
                f"test_loss={res.loss:.4f} | "
                f"test_acc={res.accuracy:.4f}"
            )

    save_json(save_dir / "mnist_one_layer_neuron_pruning_results.json", all_results)

    # Plot
    plt.figure(figsize=(8, 6))
    for alpha in args.alphas:
        rows = all_results[str(alpha)]
        xs = [r["pruning_ratio"] for r in rows]
        ys = [r["test_accuracy"] for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, label=rf"$\alpha = {alpha}$")

    plt.xlabel("Pruning Ratio", fontsize=20)
    plt.ylabel("Test Accuracy", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(save_dir / "mnist_one_layer_neuron_pruning_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nSaved results to: {save_dir / 'mnist_one_layer_neuron_pruning_results.json'}")
    print(f"Saved figure to:  {save_dir / 'mnist_one_layer_neuron_pruning_accuracy.png'}")


if __name__ == "__main__":
    main()