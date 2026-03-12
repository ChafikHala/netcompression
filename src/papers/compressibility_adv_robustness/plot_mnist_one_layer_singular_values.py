from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.core.checkpoint import load_checkpoint
from src.models.fcn import FCN
from src.papers.compressibility_adv_robustness.regularization import get_single_hidden_layer_weight


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


def load_weight_from_checkpoint(ckpt_path: Path) -> torch.Tensor:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return get_single_hidden_layer_weight(model).detach().cpu()


def top_singular_values(weight: torch.Tensor, k: int = 24) -> np.ndarray:
    s = torch.linalg.svdvals(weight)
    return s[:k].numpy()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.05])
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/compressibility_adv_robustness/figures_multi_seed",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_results: dict[str, list[list[float]]] = {}
    aggregated: dict[str, dict] = {}

    plt.figure(figsize=(7, 5))

    for alpha in args.alphas:
        seed_curves = []

        for seed in args.seeds:
            ckpt_path = find_checkpoint(base_dir, alpha, seed, args.ckpt)
            weight = load_weight_from_checkpoint(ckpt_path)
            s = top_singular_values(weight, k=args.top_k)
            seed_curves.append(s.tolist())

        seed_curves_np = np.array(seed_curves, dtype=float)  # [n_seeds, top_k]
        mean_curve = seed_curves_np.mean(axis=0)
        std_curve = seed_curves_np.std(axis=0, ddof=0)

        raw_results[str(alpha)] = seed_curves
        aggregated[str(alpha)] = {
            "alpha": float(alpha),
            "mean": mean_curve.tolist(),
            "std": std_curve.tolist(),
            "n_seeds": len(args.seeds),
        }

        ks = np.arange(args.top_k)
        plt.errorbar(
            ks,
            mean_curve,
            yerr=std_curve,
            marker="o",
            linewidth=2,
            capsize=3,
            label=rf"$\alpha = {alpha}$",
        )

    plt.xlabel(r"$k$", fontsize=18)
    plt.ylabel(r"$\sigma_k$", fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / "mnist_one_layer_top24_singular_values_mean_std.png", dpi=200, bbox_inches="tight")
    plt.close()

    save_json(save_dir / "singular_values_raw.json", raw_results)
    save_json(save_dir / "singular_values_aggregated.json", aggregated)

    print(f"Saved figure to: {save_dir / 'mnist_one_layer_top24_singular_values_mean_std.png'}")
    print(f"Saved raw results to: {save_dir / 'singular_values_raw.json'}")
    print(f"Saved aggregated results to: {save_dir / 'singular_values_aggregated.json'}")


if __name__ == "__main__":
    main()