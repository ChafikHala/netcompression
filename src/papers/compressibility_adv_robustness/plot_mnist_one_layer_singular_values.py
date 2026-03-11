from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.core.checkpoint import load_checkpoint
from src.models.fcn import FCN
from src.papers.compressibility_adv_robustness.regularization import (
    get_single_hidden_layer_weight,
)


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
    )


def load_weight_from_checkpoint(ckpt_path: Path) -> torch.Tensor:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    weight = get_single_hidden_layer_weight(model).detach().cpu()
    return weight


def top_singular_values(weight: torch.Tensor, k: int = 24) -> torch.Tensor:
    s = torch.linalg.svdvals(weight)
    s = s[:k]
    return s


def find_checkpoint(base_dir: Path, alpha: float, ckpt_name: str) -> Path:
    exp_name = f"model_one_layer_FC_alpha_{alpha_to_name(alpha)}"
    ckpt_path = (
        base_dir
        / "compressibility_adv_robustness"
        / exp_name
        / "seed_42"
        / "checkpoints"
        / ckpt_name
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.05],
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="outputs/compressibility_adv_robustness/figures/mnist_one_layer_top24_singular_values.png",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    for alpha in args.alphas:
        exp_name = f"model_one_layer_FC_alpha_{alpha_to_name(alpha)}"
        ckpt_path = (
            base_dir
            / "compressibility_adv_robustness"
            / exp_name
            / f"seed_{args.seed}"
            / "checkpoints"
            / args.ckpt
        )

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        weight = load_weight_from_checkpoint(ckpt_path)
        s = top_singular_values(weight, k=args.top_k)

        ks = list(range(len(s)))
        plt.plot(ks, s.numpy(), marker="o", linewidth=2, label=rf"$\alpha = {alpha}$")

    plt.xlabel(r"$k$", fontsize=18)
    plt.ylabel(r"$\sigma_k$", fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()