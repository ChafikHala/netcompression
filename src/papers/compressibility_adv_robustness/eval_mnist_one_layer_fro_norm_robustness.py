from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

from src.core.checkpoint import load_checkpoint
from src.data.dataloaders import build_dataloaders
from src.data.datasets import build_datasets
from src.models.fcn import FCN
from src.utils.config import load_config


_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081


def fro_to_name(fro_norm: float) -> str:
    if float(fro_norm).is_integer():
        return str(int(fro_norm))
    return str(fro_norm).replace(".", "p")


def run_name(fro_norm: float, seed: int) -> str:
    return f"model_one_layer_FC_fro_{fro_to_name(fro_norm)}_seed_{seed}"


def build_model() -> FCN:
    return FCN(
        input_shape=[1, 28, 28],
        hidden_dims=[400],
        num_classes=2,
        dropout=0.0,
        bias=False,
    )


class NormalizedMNISTModelWrapper(nn.Module):
    """
    Wrap a model that expects normalized MNIST inputs, while exposing
    a [0,1]-input interface to ART.
    """
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        mean = torch.tensor([_MNIST_MEAN], dtype=torch.float32).view(1, 1, 1, 1)
        std = torch.tensor([_MNIST_STD], dtype=torch.float32).view(1, 1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def normalize_mnist_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([_MNIST_MEAN], dtype=x.dtype, device=x.device).view(1, 1, 1, 1)
    std = torch.tensor([_MNIST_STD], dtype=x.dtype, device=x.device).view(1, 1, 1, 1)
    return (x - mean) / std


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def find_checkpoint(base_dir: Path, fro_norm: float, seed: int, ckpt_name: str) -> Path:
    path = (
        base_dir
        / "compressibility_adv_robustness"
        / run_name(fro_norm, seed)
        / "checkpoints"
        / ckpt_name
    )
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def build_art_classifier_mnist(
    model: nn.Module,
    device: torch.device,
    nb_classes: int = 2,
) -> PyTorchClassifier:
    wrapped_model = NormalizedMNISTModelWrapper(model).to(device)
    wrapped_model.eval()

    dummy_optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=1.0)
    loss = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=loss,
        optimizer=dummy_optimizer,
        input_shape=(1, 28, 28),
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return classifier


def build_fgsm_for_mnist(
    model: nn.Module,
    device: torch.device,
    norm: str = "l2",
    eps_l2: float = 2.0,
    eps_linf: float = 0.3,
    batch_size: int = 128,
) -> FastGradientMethod:
    classifier = build_art_classifier_mnist(model, device=device, nb_classes=2)

    norm = norm.lower()
    if norm == "l2":
        norm_value = 2
        eps = float(eps_l2)
    elif norm == "linf":
        norm_value = np.inf
        eps = float(eps_linf)
    else:
        raise ValueError("norm must be 'l2' or 'linf'")

    attacker = FastGradientMethod(
        estimator=classifier,
        norm=norm_value,
        eps=eps,
        eps_step=eps,
        targeted=False,
        batch_size=int(batch_size),
    )
    return attacker


def generate_adversarial_batch(attacker, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> torch.Tensor:
    x_np = x.detach().cpu().numpy().astype(np.float32)
    y_np = y.detach().cpu().numpy()

    x_adv_np = attacker.generate(x=x_np, y=y_np)
    x_adv = torch.from_numpy(x_adv_np).to(device=device, dtype=x.dtype)
    return x_adv


@torch.no_grad()
def clean_accuracy(model: nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x_raw, y in dataloader:
        x_raw = x_raw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_norm = normalize_mnist_tensor(x_raw)
        logits = model(x_norm)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return correct / max(total, 1)


def robust_accuracy(
    model: nn.Module,
    dataloader,
    device: torch.device,
    norm: str = "l2",
    eps_l2: float = 2.0,
    eps_linf: float = 0.3,
    attack_batch_size: int = 128,
) -> float:
    model.eval()

    attacker = build_fgsm_for_mnist(
        model=model,
        device=device,
        norm=norm,
        eps_l2=eps_l2,
        eps_linf=eps_linf,
        batch_size=attack_batch_size,
    )

    robust_correct = 0
    total = 0

    for x_raw, y in dataloader:
        x_raw = x_raw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_adv_raw = generate_adversarial_batch(attacker, x_raw, y, device=device)

        with torch.no_grad():
            x_adv_norm = normalize_mnist_tensor(x_adv_raw)
            logits_adv = model(x_adv_norm)
            pred_adv = logits_adv.argmax(dim=1)

            robust_correct += int((pred_adv == y).sum().item())
            total += int(y.size(0))

    return robust_correct / max(total, 1)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def aggregate_over_seeds(raw_results: dict[str, list[dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}

    for fro_key, rows in raw_results.items():
        clean_vals = [r["clean_test_accuracy"] for r in rows]
        robust_vals = [r["robust_test_accuracy"] for r in rows]
        ratio_vals = [r["robust_over_clean"] for r in rows]

        out[fro_key] = {
            "fro_norm": float(fro_key),
            "clean_test_accuracy_mean": float(np.mean(clean_vals)),
            "clean_test_accuracy_std": float(np.std(clean_vals, ddof=0)),
            "robust_test_accuracy_mean": float(np.mean(robust_vals)),
            "robust_test_accuracy_std": float(np.std(robust_vals, ddof=0)),
            "robust_over_clean_mean": float(np.mean(ratio_vals)),
            "robust_over_clean_std": float(np.std(ratio_vals, ddof=0)),
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


def plot_ratio_figure(agg: dict[str, dict], save_path: Path) -> None:
    fro_norms = np.array(sorted([float(k) for k in agg.keys()]))

    ratio_mean = np.array([agg[str(f)]["robust_over_clean_mean"] for f in fro_norms])
    ratio_std = np.array([agg[str(f)]["robust_over_clean_std"] for f in fro_norms])

    color = "#B22222"
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    ax.fill_between(fro_norms, ratio_mean - ratio_std, ratio_mean + ratio_std, color=color, alpha=0.12, linewidth=0)
    ax.errorbar(
        fro_norms,
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

    ax.set_xlabel("Frobenius norm", fontsize=15, labelpad=8)
    ax.set_ylabel("Robust Acc. / Clean Acc.", fontsize=15, labelpad=10)

    _apply_elegant_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="outputs")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    parser.add_argument("--n-fro-norms", type=int, default=10)
    parser.add_argument("--fro-min", type=float, default=20.0)
    parser.add_argument("--fro-max", type=float, default=200.0)
    parser.add_argument("--fro-grid", type=str, default="linear", choices=["linear", "geom"])

    parser.add_argument("--norm", type=str, default="l2", choices=["l2", "linf"])
    parser.add_argument("--eps-l2", type=float, default=2.0)
    parser.add_argument("--eps-linf", type=float, default=0.3)
    parser.add_argument("--attack-batch-size", type=int, default=128)

    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    if args.save_dir is None:
        args.save_dir = (
            f"outputs/compressibility_adv_robustness/fro_norm_sweep_fgsm_{args.norm}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)

    # build raw [0,1] MNIST test loader so eps is interpreted in image space
    cfg.dataset.augmentation.normalize = False
    bundle = build_datasets(cfg)
    _, _, test_loader = build_dataloaders(cfg, bundle.train, bundle.val, bundle.test, device)

    if args.fro_grid == "geom":
        fro_norms = np.geomspace(args.fro_min, args.fro_max, args.n_fro_norms)
    else:
        fro_norms = np.linspace(args.fro_min, args.fro_max, args.n_fro_norms)

    raw_results: dict[str, list[dict]] = {}

    for fro_norm in fro_norms:
        fro_norm = float(fro_norm)
        fro_key = str(fro_norm)
        raw_results[fro_key] = []

        for seed in args.seeds:
            ckpt_path = find_checkpoint(base_dir, fro_norm, seed, args.ckpt)
            print(f"\nEvaluating fro_norm={fro_norm:.8f}, seed={seed}")
            print(f"Checkpoint: {ckpt_path}")

            model = load_model_from_checkpoint(ckpt_path, device=device)

            clean_acc = clean_accuracy(model, test_loader, device=device)
            robust_acc = robust_accuracy(
                model=model,
                dataloader=test_loader,
                device=device,
                norm=args.norm,
                eps_l2=args.eps_l2,
                eps_linf=args.eps_linf,
                attack_batch_size=args.attack_batch_size,
            )
            robust_over_clean = robust_acc / max(clean_acc, 1e-12)

            row = {
                "fro_norm": fro_norm,
                "seed": int(seed),
                "norm": args.norm,
                "checkpoint": str(ckpt_path),
                "clean_test_accuracy": float(clean_acc),
                "robust_test_accuracy": float(robust_acc),
                "robust_over_clean": float(robust_over_clean),
            }
            raw_results[fro_key].append(row)

            print(
                f"norm={args.norm} | clean_acc={clean_acc:.4f} | "
                f"robust_acc={robust_acc:.4f} | robust/clean={robust_over_clean:.4f}"
            )

    agg = aggregate_over_seeds(raw_results)

    save_json(save_dir / "raw_results.json", raw_results)
    save_json(save_dir / "aggregated_results.json", agg)

    plot_ratio_figure(agg, save_dir / "robust_over_clean_vs_fro_norm.png")

    print(f"\nSaved raw results to: {save_dir / 'raw_results.json'}")
    print(f"Saved aggregated results to: {save_dir / 'aggregated_results.json'}")
    print(f"Saved figure to: {save_dir / 'robust_over_clean_vs_fro_norm.png'}")


if __name__ == "__main__":
    main()