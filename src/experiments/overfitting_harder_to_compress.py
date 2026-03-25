from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms




def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true/false")

def setup_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (8.0, 5.2),
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 5.5,
    })



class WideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = float(drop_rate)

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = F.relu(self.bn2(out), inplace=True)
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return out + shortcut


class WideResNet(nn.Module):
    """
    WRN-16-2:
      depth = 16 => n = (16 - 4) / 6 = 2 blocks per group
      widen_factor = 2
    """
    def __init__(self, depth: int = 16, widen_factor: int = 2, num_classes: int = 10, drop_rate: float = 0.0) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"depth must satisfy (depth - 4) % 6 == 0, got {depth}")

        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_group(widths[0], widths[1], n, stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_group(widths[1], widths[2], n, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_group(widths[2], widths[3], n, stride=2, drop_rate=drop_rate)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes, bias=False)

        self._initialize()

    @staticmethod
    def _make_group(in_planes: int, out_planes: int, n_blocks: int, stride: int, drop_rate: float) -> nn.Sequential:
        layers = [WideBasicBlock(in_planes, out_planes, stride=stride, drop_rate=drop_rate)]
        for _ in range(1, n_blocks):
            layers.append(WideBasicBlock(out_planes, out_planes, stride=1, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.avg_pool2d(out, kernel_size=8)
        out = out.view(out.size(0), -1)
        return self.fc(out)



class NoisyLabelDataset(Dataset):
    def __init__(self, base_dataset: Dataset, noisy_targets: np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.noisy_targets = np.asarray(noisy_targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        x, _ = self.base_dataset[idx]
        y = int(self.noisy_targets[idx])
        return x, y


def build_noisy_targets(original_targets: np.ndarray, alpha: float, seed: int) -> np.ndarray:
    """
    Corrupt an alpha fraction of labels by selecting a subset and randomly permuting
    the labels within that subset. This preserves the label multiset on the corrupted
    subset while destroying alignment with the inputs.

    alpha = 0.0 => unchanged labels
    alpha = 1.0 => all labels are randomly permuted
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    rng = np.random.default_rng(seed)
    y = np.asarray(original_targets, dtype=np.int64).copy()
    n = len(y)
    m = int(round(alpha * n))

    if m == 0:
        return y

    idx = rng.choice(n, size=m, replace=False)
    permuted = y[idx].copy()
    rng.shuffle(permuted)
    y[idx] = permuted
    return y




def build_datasets(
    data_root: str,
    *,
    use_random_crop: bool,
    use_horizontal_flip: bool,
) -> Tuple[Dataset, Dataset]:
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )

    train_transforms = []

    if use_random_crop:
        train_transforms.append(transforms.RandomCrop(32, padding=4))

    if use_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.extend([
        transforms.ToTensor(),
        normalize,
    ])

    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_base = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_base, test_set



def build_eval_train_dataset(data_root: str) -> Dataset:
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)




@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
) -> List[dict]:
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history: List[dict] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            preds = logits.argmax(dim=1)
            running_loss += float(loss.item()) * bs
            running_correct += int((preds == y).sum().item())
            total += bs

        scheduler.step()

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(total, 1),
            "train_acc_noisy_labels_online": running_correct / max(total, 1),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_record)

        print(
            f"[Epoch {epoch+1:03d}/{epochs}] "
            f"loss={epoch_record['train_loss']:.4f} | "
            f"train_acc(noisy-online)={epoch_record['train_acc_noisy_labels_online']:.4f} | "
            f"lr={epoch_record['lr']:.5f}"
        )

    return history



def get_prunable_parameters(model: nn.Module):
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, "weight"))
    return params


def prune_model_global_magnitude(model: nn.Module, amount: float) -> nn.Module:
    pruned_model = copy.deepcopy(model)
    parameters_to_prune = get_prunable_parameters(pruned_model)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=float(amount),
    )

    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    return pruned_model


def collect_abs_weights(model: nn.Module) -> np.ndarray:
    weights = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach().cpu().numpy().ravel()
            weights.append(np.abs(w))
    if not weights:
        return np.array([], dtype=np.float64)
    return np.concatenate(weights, axis=0)




def plot_retained_train_acc_curves(results: Dict[float, dict], save_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots()

    for alpha in sorted(results.keys()):
        pruning = results[alpha]["pruning_curve"]["pruning_ratios"]
        retained = results[alpha]["pruning_curve"]["retained_train_acc"]
        ax.plot(
            pruning,
            retained,
            marker="o",
            label=f"noise={alpha:.0%}",
        )

    ax.set_xlabel("Pruning ratio")
    ax.set_ylabel("Retained train accuracy")
    ax.set_title("Retained train accuracy vs pruning ratio")
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.legend(frameon=True, ncol=1)
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_bars(results: Dict[float, dict], save_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots()

    alphas = sorted(results.keys())
    labels = [f"{int(round(100 * a))}%" for a in alphas]
    values = [results[a]["pruning_curve"]["pruning_ratio_for_80pct_loss"] for a in alphas]

    x = np.arange(len(labels))
    clean_values = [0.0 if (v is None or np.isnan(v)) else float(v) for v in values]
    bars = ax.bar(x, clean_values)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Label-noise level")
    ax.set_ylabel("Smallest pruning ratio with retained train accuracy ≤ 0.2")
    ax.set_title("Pruning ratio needed to retain at most 20% of train accuracy")
    ax.set_ylim(0, 1.05)

    for rect, v in zip(bars, values):
        text = "N/A" if (v is None or np.isnan(v)) else f"{v:.2f}"
        ax.annotate(
            text,
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_weight_histogram(abs_weights: np.ndarray, title: str, save_path: Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots()

    ax.hist(abs_weights, bins=120, density=False)
    ax.set_xlabel("|weight|")
    ax.set_ylabel("Count")
    ax.set_title(title)

    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)




@dataclass
class ExperimentConfig:
    data_root: str
    output_dir: str
    epochs: int
    batch_size: int
    eval_batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool
    use_random_crop: bool
    use_horizontal_flip: bool
    seed: int
    alphas: List[float]
    pruning_ratios: List[float]


def run_experiment(cfg: ExperimentConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(cfg.output_dir)
    models_dir = output_dir / "models"
    plots_dir = output_dir / "plots"
    metrics_dir = output_dir / "metrics"
    ensure_dir(models_dir)
    ensure_dir(plots_dir)
    ensure_dir(metrics_dir)

    train_base, test_set = build_datasets(cfg.data_root, use_random_crop=cfg.use_random_crop, use_horizontal_flip=cfg.use_horizontal_flip,)
    eval_train_clean = build_eval_train_dataset(cfg.data_root)
    original_targets = np.asarray(train_base.targets, dtype=np.int64)

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    all_results: Dict[float, dict] = {}

    for alpha in cfg.alphas:
        print("\n" + "=" * 100)
        print(f"Training model for label-noise alpha = {alpha:.3f}")
        print("=" * 100)

        noisy_targets = build_noisy_targets(original_targets, alpha=alpha, seed=cfg.seed + int(1000 * alpha))
        noisy_train_dataset = NoisyLabelDataset(train_base, noisy_targets)

        noisy_train_loader = DataLoader(
            noisy_train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        noisy_eval_train_dataset = NoisyLabelDataset(eval_train_clean, noisy_targets)
        noisy_eval_train_loader = DataLoader(
            noisy_eval_train_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = WideResNet(depth=16, widen_factor=2, num_classes=10, drop_rate=0.0).to(device)

        train_history = train_one_model(
            model,
            noisy_train_loader,
            device,
            epochs=cfg.epochs,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=cfg.nesterov,
        )

        train_acc = accuracy(model, noisy_eval_train_loader, device)
        test_acc = accuracy(model, test_loader, device)
        overfit_rate = (train_acc - test_acc) / max(train_acc, 1e-12)

        alpha_name = f"{alpha:.3f}".replace(".", "p")
        model_path = models_dir / f"wrn16_2_label_noise_{alpha_name}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "alpha": alpha,
            "seed": cfg.seed,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "overfit_rate": overfit_rate,
        }, model_path)

        print(
            f"[alpha={alpha:.3f}] "
            f"train_acc={train_acc:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"overfit_rate={overfit_rate:.4f}"
        )

        pruning_train_accs = []
        retained_train_acc = []

        for ratio in cfg.pruning_ratios:
            pruned_model = prune_model_global_magnitude(model, amount=ratio).to(device)

            pruned_train_acc = accuracy(pruned_model, noisy_eval_train_loader, device)
            pruning_train_accs.append(pruned_train_acc)

            retained = pruned_train_acc / max(train_acc, 1e-12)
            retained_train_acc.append(retained)

            print(
                f"    pruning_ratio={ratio:.3f} | "
                f"train_acc_pruned={pruned_train_acc:.4f} | "
                f"retained_train_acc={retained:.4f}"
            )

        threshold_ratio = None
        for ratio, retained in zip(cfg.pruning_ratios, retained_train_acc):
            if retained <= 0.2:
                threshold_ratio = float(ratio)
                break

        abs_weights = collect_abs_weights(model)

        result = {
            "alpha": float(alpha),
            "seed": int(cfg.seed),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "overfit_rate": float(overfit_rate),
            "model_path": str(model_path),
            "train_history": train_history,
            "pruning_curve": {
                "pruning_ratios": [float(x) for x in cfg.pruning_ratios],
                "train_acc_pruned": [float(x) for x in pruning_train_accs],
                "retained_train_acc": [float(x) for x in retained_train_acc],
                "pruning_ratio_for_80pct_loss": threshold_ratio,
            },
            "weight_stats": {
                "num_weights": int(abs_weights.size),
                "mean_abs_weight": float(abs_weights.mean()),
                "median_abs_weight": float(np.median(abs_weights)),
                "max_abs_weight": float(abs_weights.max()),
            },
        }

        all_results[float(alpha)] = result
        save_json(metrics_dir / f"result_alpha_{alpha_name}.json", result)

    save_json(metrics_dir / "all_results.json", {
        "config": asdict(cfg),
        "results": all_results,
    })

    plot_retained_train_acc_curves(all_results, plots_dir / "retained_train_acc_vs_ratio.png")
    plot_threshold_bars(all_results, plots_dir / "pruning_ratio_for_80pct_loss.png")

    if 0.0 in all_results:
        ckpt = torch.load(all_results[0.0]["model_path"], map_location="cpu")
        model0 = WideResNet(depth=16, widen_factor=2, num_classes=10, drop_rate=0.0)
        model0.load_state_dict(ckpt["model_state"])
        abs_w0 = collect_abs_weights(model0)
        plot_weight_histogram(
            abs_w0,
            title="Histogram of |weights| for model trained with 0% label noise",
            save_path=plots_dir / "hist_abs_weights_noise_0pct.png",
        )

    if 1.0 in all_results:
        ckpt = torch.load(all_results[1.0]["model_path"], map_location="cpu")
        model1 = WideResNet(depth=16, widen_factor=2, num_classes=10, drop_rate=0.0)
        model1.load_state_dict(ckpt["model_state"])
        abs_w1 = collect_abs_weights(model1)
        plot_weight_histogram(
            abs_w1,
            title="Histogram of |weights| for model trained with 100% label noise",
            save_path=plots_dir / "hist_abs_weights_noise_100pct.png",
        )

    print("\nDone.")
    print(f"Results saved in: {output_dir}")



def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/noise_pruning_cifar10_wrn16")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--use-random-crop", type=str2bool, default=False)
    parser.add_argument("--use-horizontal-flip", type=str2bool, default=False)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.1,0.2,0.4,0.6,0.8,1.0",
        help="comma-separated label-noise levels in [0,1]",
    )
    parser.add_argument(
        "--pruning-ratios",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99",
        help="comma-separated global pruning ratios in [0,1)",
    )

    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        eval_batch_size=int(args.eval_batch_size),
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=bool(args.nesterov),
        use_random_crop=bool(args.use_random_crop),
        use_horizontal_flip=bool(args.use_horizontal_flip),
        seed=int(args.seed),
        alphas=parse_float_list(args.alphas),
        pruning_ratios=parse_float_list(args.pruning_ratios),
    )

    run_experiment(cfg)


if __name__ == "__main__":
    main()