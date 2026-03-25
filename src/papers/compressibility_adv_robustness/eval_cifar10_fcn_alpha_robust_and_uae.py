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
from src.papers.compressibility_adv_robustness.train_cifar10_four_layers_fcn import (
    DenseFCN,
    LowRankFCN,
)
from src.utils.config import load_config


_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def alpha_to_name(alpha: float) -> str:
    if float(alpha).is_integer():
        return str(int(alpha))
    return str(alpha).replace(".", "p")


def run_name(compressibility: str, value: float | int, seed: int) -> str:
    if compressibility == "neuron":
        return f"cifar10_fcn_group_lasso_alpha_{alpha_to_name(float(value))}_seed_{seed}"
    if compressibility == "spectral":
        return f"cifar10_fcn_low_rank_rank_{int(value)}_seed_{seed}"
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


class NormalizedCIFARModelWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

        mean = torch.tensor(_CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)


def normalize_cifar_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_CIFAR10_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_CIFAR10_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


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
    value: float | int,
    seed: int,
    ckpt_name: str,
) -> Path:
    path = (
        base_dir
        / "compressibility_adv_robustness"
        / compressibility
        / run_name(compressibility, value, seed)
        / "checkpoints"
        / ckpt_name
    )
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def build_art_classifier_cifar(
    model: nn.Module,
    device: torch.device,
    nb_classes: int = 10,
) -> PyTorchClassifier:
    wrapped_model = NormalizedCIFARModelWrapper(model).to(device)
    wrapped_model.eval()

    dummy_optimizer = torch.optim.SGD(wrapped_model.parameters(), lr=1.0)
    loss = nn.CrossEntropyLoss()

    classifier = PyTorchClassifier(
        model=wrapped_model,
        loss=loss,
        optimizer=dummy_optimizer,
        input_shape=(3, 32, 32),
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
        device_type="gpu" if device.type == "cuda" else "cpu",
    )
    return classifier


def _linf_eps_array(eps: float) -> np.ndarray:
    arr = np.array([eps, eps, eps], dtype=np.float32)
    return arr.reshape(1, 3, 1, 1)

def _scaled_l2_eps(eps: float) -> float:
    return float(eps)

def build_fgsm_for_cifar(
    model: nn.Module,
    device: torch.device,
    norm: str = "linf",
    eps_l2: float = 0.125,
    eps_linf: float = 2 / 255,
    batch_size: int = 128,
) -> FastGradientMethod:
    classifier = build_art_classifier_cifar(model, device=device, nb_classes=10)

    norm = norm.lower()
    if norm == "l2":
        norm_value = 2
        eps = _scaled_l2_eps(float(eps_l2))
        eps_step = eps
    elif norm == "linf":
        norm_value = np.inf
        eps = _linf_eps_array(float(eps_linf))
        eps_step = eps
    else:
        raise ValueError("norm must be 'l2' or 'linf'")

    attacker = FastGradientMethod(
        estimator=classifier,
        norm=norm_value,
        eps=eps,
        eps_step=eps_step,
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

        x_norm = normalize_cifar_tensor(x_raw)
        logits = model(x_norm)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return correct / max(total, 1)


def robust_accuracy(
    model: nn.Module,
    dataloader,
    device: torch.device,
    norm: str = "linf",
    eps_l2: float = 0.125,
    eps_linf: float = 2 / 255,
    attack_batch_size: int = 128,
) -> float:
    model.eval()

    attacker = build_fgsm_for_cifar(
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
            x_adv_norm = normalize_cifar_tensor(x_adv_raw)
            logits_adv = model(x_adv_norm)
            pred_adv = logits_adv.argmax(dim=1)

            robust_correct += int((pred_adv == y).sum().item())
            total += int(y.size(0))

    return robust_correct / max(total, 1)


def _project_linf(u: torch.Tensor, eps: float) -> torch.Tensor:
    return u.clamp(min=-eps, max=eps)


def _project_l2(u: torch.Tensor, eps: float) -> torch.Tensor:
    flat = u.view(u.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(eps / norms, max=1.0)
    flat = flat * scale
    return flat.view_as(u)


def _fgsm_universal_step(grad: torch.Tensor, step_size: float, norm: str) -> torch.Tensor:
    norm = norm.lower()
    if norm == "linf":
        return step_size * grad.sign()

    if norm == "l2":
        flat = grad.view(grad.size(0), -1)
        grad_norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        flat = flat / grad_norm
        return step_size * flat.view_as(grad)

    raise ValueError("norm must be 'linf' or 'l2'")


def compute_fgsm_based_uae(
    model: nn.Module,
    source_loader,
    device: torch.device,
    norm: str = "linf",
    eps_linf: float = 2 / 255,
    eps_l2: float = 0.125,
    step_size_linf: float | None = None,
    step_size_l2: float | None = None,
    num_passes: int = 3,
    max_source_batches: int | None = None,
) -> torch.Tensor:
    model.eval()

    norm = norm.lower()
    if norm == "linf":
        eps = float(eps_linf)
        if step_size_linf is None:
            step_size_linf = eps / max(num_passes, 1)
        step_size = float(step_size_linf)
        projector = _project_linf
    elif norm == "l2":
        eps = float(eps_l2)
        if step_size_l2 is None:
            step_size_l2 = eps / max(num_passes, 1)
        step_size = float(step_size_l2)
        projector = _project_l2
    else:
        raise ValueError("norm must be 'linf' or 'l2'")

    u = torch.zeros(1, 3, 32, 32, device=device)
    criterion = nn.CrossEntropyLoss()

    for _ in range(int(num_passes)):
        for batch_idx, (x_raw, y) in enumerate(source_loader):
            if max_source_batches is not None and batch_idx >= int(max_source_batches):
                break

            x_raw = x_raw.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x_pert = (x_raw + u).clamp(0.0, 1.0).detach().requires_grad_(True)
            x_pert_norm = normalize_cifar_tensor(x_pert)

            logits = model(x_pert_norm)
            loss = criterion(logits, y)

            grad = torch.autograd.grad(loss, x_pert, retain_graph=False, create_graph=False)[0]
            grad_mean = grad.mean(dim=0, keepdim=True)

            delta = _fgsm_universal_step(grad_mean, step_size=step_size, norm=norm)
            u = projector(u + delta, eps=eps).detach()

    return u


@torch.no_grad()
def universal_accuracy(
    model: nn.Module,
    dataloader,
    u: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for x_raw, y in dataloader:
        x_raw = x_raw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_u = (x_raw + u).clamp(0.0, 1.0)
        x_u_norm = normalize_cifar_tensor(x_u)

        logits = model(x_u_norm)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.size(0))

    return correct / max(total, 1)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def aggregate_over_seeds(raw_results: dict[str, list[dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}

    for key, rows in raw_results.items():
        robust_ratio_vals = [r["robust_over_clean"] for r in rows]
        uae_ratio_vals = [r["uae_over_clean"] for r in rows]

        out[key] = {
            "value": float(key),
            "robust_over_clean_mean": float(np.mean(robust_ratio_vals)),
            "robust_over_clean_std": float(np.std(robust_ratio_vals, ddof=0)),
            "uae_over_clean_mean": float(np.mean(uae_ratio_vals)),
            "uae_over_clean_std": float(np.std(uae_ratio_vals, ddof=0)),
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


def plot_ratio_figure(agg: dict[str, dict], save_path: Path, sweep_type: str) -> None:
    xs = np.array(sorted([float(k) for k in agg.keys()]))

    robust_mean = np.array([agg[str(int(x)) if sweep_type == "rank" else str(x)]["robust_over_clean_mean"] for x in xs])
    robust_std = np.array([agg[str(int(x)) if sweep_type == "rank" else str(x)]["robust_over_clean_std"] for x in xs])

    uae_mean = np.array([agg[str(int(x)) if sweep_type == "rank" else str(x)]["uae_over_clean_mean"] for x in xs])
    uae_std = np.array([agg[str(int(x)) if sweep_type == "rank" else str(x)]["uae_over_clean_std"] for x in xs])

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    curves = [
        (robust_mean, robust_std, "o", "#B22222", "Rob. / Clean"),
        (uae_mean, uae_std, "D", "#228B22", "UAE / Clean"),
    ]

    for mean, std, marker, color, label in curves:
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.errorbar(
            xs,
            mean,
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

    if sweep_type == "alpha":
        ax.set_xscale("log")
        ax.set_xlabel(r"$\alpha$", fontsize=15, labelpad=8)
    else:
        ax.set_xlabel("Rank", fontsize=15, labelpad=8)

    ax.set_ylabel("Accuracy ratio", fontsize=15, labelpad=10)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="outputs_h")
    parser.add_argument("--ckpt", type=str, default="best.pt", choices=["best.pt", "last.pt"])
    parser.add_argument("--compressibility", type=str, required=True, choices=["neuron", "spectral"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    parser.add_argument("--n-alphas", type=int, default=15)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=1e-1)
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])

    parser.add_argument("--norm", type=str, default=None, choices=["linf", "l2"])
    parser.add_argument("--eps-linf", type=float, default=2 / 255)
    parser.add_argument("--eps-l2", type=float, default=0.125)
    parser.add_argument("--attack-batch-size", type=int, default=128)

    parser.add_argument("--step-size-linf", type=float, default=None)
    parser.add_argument("--step-size-l2", type=float, default=None)
    parser.add_argument("--num-passes", type=int, default=3)
    parser.add_argument("--max-source-batches", type=int, default=None)

    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    if args.norm is None:
        args.norm = "linf" if args.compressibility == "neuron" else "l2"

    if args.save_dir is None:
        args.save_dir = f"outputs_h/compressibility_adv_robustness/cifar10_rob_and_uae_{args.compressibility}_{args.norm}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)

    cfg.dataset.augmentation.normalize = False

    bundle = build_datasets(cfg)
    train_loader, _, test_loader = build_dataloaders(cfg, bundle.train, bundle.val, bundle.test, device)

    if args.compressibility == "neuron":
        sweep_values = [float(a) for a in np.geomspace(args.alpha_min, args.alpha_max, args.n_alphas)]
        sweep_type = "alpha"
    else:
        sweep_values = [int(r) for r in args.ranks]
        sweep_type = "rank"

    raw_results: dict[str, list[dict]] = {}

    for value in sweep_values:
        key = str(value)
        raw_results[key] = []

        for seed in args.seeds:
            rank = int(value) if args.compressibility == "spectral" else None

            ckpt_path = find_checkpoint(
                base_dir=base_dir,
                compressibility=args.compressibility,
                value=value,
                seed=seed,
                ckpt_name=args.ckpt,
            )

            print(f"\nEvaluating {sweep_type}={value}, seed={seed}")
            print(f"Checkpoint: {ckpt_path}")

            model = load_model_from_checkpoint(
                ckpt_path,
                device=device,
                compressibility=args.compressibility,
                rank=rank,
            )

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

            u = compute_fgsm_based_uae(
                model=model,
                source_loader=train_loader,
                device=device,
                norm=args.norm,
                eps_linf=args.eps_linf,
                eps_l2=args.eps_l2,
                step_size_linf=args.step_size_linf,
                step_size_l2=args.step_size_l2,
                num_passes=args.num_passes,
                max_source_batches=args.max_source_batches,
            )

            uae_acc = universal_accuracy(
                model=model,
                dataloader=test_loader,
                u=u,
                device=device,
            )
            uae_over_clean = uae_acc / max(clean_acc, 1e-12)

            row = {
                sweep_type: value,
                "seed": int(seed),
                "compressibility": args.compressibility,
                "norm": args.norm,
                "checkpoint": str(ckpt_path),
                "clean_test_accuracy": float(clean_acc),
                "robust_test_accuracy": float(robust_acc),
                "robust_over_clean": float(robust_over_clean),
                "uae_test_accuracy": float(uae_acc),
                "uae_over_clean": float(uae_over_clean),
            }
            raw_results[key].append(row)

            print(
                f"norm={args.norm} | clean_acc={clean_acc:.4f} | "
                f"rob_acc={robust_acc:.4f} | rob/clean={robust_over_clean:.4f} | "
                f"uae_acc={uae_acc:.4f} | uae/clean={uae_over_clean:.4f}"
            )

    agg = aggregate_over_seeds(raw_results)

    save_json(save_dir / "raw_results.json", raw_results)
    save_json(save_dir / "aggregated_results.json", agg)

    plot_ratio_figure(agg, save_dir / "robust_and_uae_over_clean.png", sweep_type=sweep_type)

    print(f"\nSaved raw results to: {save_dir / 'raw_results.json'}")
    print(f"Saved aggregated results to: {save_dir / 'aggregated_results.json'}")
    print(f"Saved figure to: {save_dir / 'robust_and_uae_over_clean.png'}")


if __name__ == "__main__":
    main()