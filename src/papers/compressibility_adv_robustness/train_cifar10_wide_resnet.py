from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from src.core.checkpoint import CheckpointPayload, save_checkpoint
from src.core.evaluator import evaluate
from src.data.dataloaders import build_dataloaders
from src.data.datasets import build_datasets
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import seed_everything


@dataclass
class EarlyStoppingState:
    best_val_loss: float = math.inf
    best_epoch: int = -1
    epochs_without_improvement: int = 0


def _float_to_name(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def _set_nested_attr(obj: Any, path: list[str], value: Any) -> None:
    cur = obj
    for key in path[:-1]:
        cur = getattr(cur, key)
    setattr(cur, path[-1], value)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_jsonl_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def load_config_dict_like(cfg):
    if hasattr(cfg, "to_dict"):
        return cfg

    class _SimpleConfig:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    v = _SimpleConfig(v)
                setattr(self, k, v)

        def to_dict(self):
            out = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "to_dict"):
                    v = v.to_dict()
                out[k] = v
            return out

    return _SimpleConfig(cfg)


# ============================================================
# Low-rank layers
# ============================================================

class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(min(rank, in_features, out_features))

        self.U = nn.Parameter(torch.randn(out_features, self.rank) * 0.02)
        self.V = nn.Parameter(torch.randn(self.rank, in_features) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def effective_weight(self) -> torch.Tensor:
        return self.U @ self.V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        out = x @ w.t()
        if self.bias is not None:
            out = out + self.bias
        return out


class LowRankConv2d(nn.Module):
    """
    Factorized conv:
        Conv2d(in -> rank, kernel=k, stride=stride, padding=padding)
        Conv2d(rank -> out, kernel=1)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        rank: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        rank_eff = int(min(rank, in_channels, out_channels))

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.rank = rank_eff
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv_a = nn.Conv2d(
            in_channels,
            rank_eff,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.conv_b = nn.Conv2d(
            rank_eff,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def effective_weight(self) -> torch.Tensor:
        """
        Return equivalent kernel of shape [out_channels, in_channels, k, k].
        """
        wa = self.conv_a.weight.detach()  # [rank, in, k, k]
        wb = self.conv_b.weight.detach()  # [out, rank, 1, 1]
        wb2 = wb[:, :, 0, 0]              # [out, rank]
        return torch.einsum("or,rihw->oihw", wb2, wa)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_b(self.conv_a(x))


# ============================================================
# WRN-16-2
# ============================================================

class DenseWideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, bias: bool = False) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return out + shortcut


class LowRankWideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, rank: int, bias: bool = False) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = LowRankConv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            rank=rank,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = LowRankConv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            rank=rank,
            bias=bias,
        )

        if stride != 1 or in_planes != out_planes:
            self.shortcut = LowRankConv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                rank=rank,
                bias=bias,
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return out + shortcut


class DenseWideResNet(nn.Module):
    """
    WRN-16-2 for CIFAR-10.
    depth = 16 => n = (16 - 4) / 6 = 2 blocks per group
    widen_factor = 2
    """
    def __init__(self, depth: int = 16, widen_factor: int = 2, num_classes: int = 10, bias: bool = False) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WRN depth must satisfy (depth - 4) % 6 == 0, got {depth}")

        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=bias)
        self.layer1 = self._make_group(DenseWideBasicBlock, widths[0], widths[1], n, stride=1, bias=bias)
        self.layer2 = self._make_group(DenseWideBasicBlock, widths[1], widths[2], n, stride=2, bias=bias)
        self.layer3 = self._make_group(DenseWideBasicBlock, widths[2], widths[3], n, stride=2, bias=bias)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes, bias=bias)

    @staticmethod
    def _make_group(block_cls, in_planes: int, out_planes: int, n_blocks: int, stride: int, bias: bool):
        layers = [block_cls(in_planes, out_planes, stride=stride, bias=bias)]
        for _ in range(1, n_blocks):
            layers.append(block_cls(out_planes, out_planes, stride=1, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.avg_pool2d(out, kernel_size=8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


class LowRankWideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 16,
        widen_factor: int = 2,
        num_classes: int = 10,
        rank: int = 32,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError(f"WRN depth must satisfy (depth - 4) % 6 == 0, got {depth}")

        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = LowRankConv2d(3, widths[0], kernel_size=3, stride=1, padding=1, rank=rank, bias=bias)
        self.layer1 = self._make_group(widths[0], widths[1], n, stride=1, rank=rank, bias=bias)
        self.layer2 = self._make_group(widths[1], widths[2], n, stride=2, rank=rank, bias=bias)
        self.layer3 = self._make_group(widths[2], widths[3], n, stride=2, rank=rank, bias=bias)
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes, bias=bias)

    @staticmethod
    def _make_group(in_planes: int, out_planes: int, n_blocks: int, stride: int, rank: int, bias: bool):
        layers = [LowRankWideBasicBlock(in_planes, out_planes, stride=stride, rank=rank, bias=bias)]
        for _ in range(1, n_blocks):
            layers.append(LowRankWideBasicBlock(out_planes, out_planes, stride=1, rank=rank, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.avg_pool2d(out, kernel_size=8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ============================================================
# Regularization / stats
# ============================================================

def _dense_hidden_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            modules[name] = module
    return modules


def _lowrank_modules(model: nn.Module) -> dict[str, LowRankConv2d]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LowRankConv2d)
    }


def _row_norm_vector_from_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.reshape(weight.shape[0], -1).norm(p=2, dim=1)


def scale_invariant_group_lasso_penalty(model: nn.Module, eps: float = 1e-12) -> torch.Tensor:
    device = next(model.parameters()).device
    penalty = torch.tensor(0.0, device=device)

    for _, layer in _dense_hidden_modules(model).items():
        row_norms = _row_norm_vector_from_weight(layer.weight)
        penalty = penalty + row_norms.sum() / row_norms.norm(p=2).clamp_min(eps)

    return penalty


def collect_first_layer_stats(model: nn.Module, compressibility: str) -> dict[str, float]:
    if compressibility == "neuron":
        first_name, first_layer = next(iter(_dense_hidden_modules(model).items()))
        w = first_layer.weight.detach()
    else:
        first_name, first_layer = next(iter(_lowrank_modules(model).items()))
        w = first_layer.effective_weight()

    w_mat = w.reshape(w.shape[0], -1)
    s = torch.linalg.svdvals(w_mat)

    return {
        "first_layer_name": first_name,
        "fro_norm": float(torch.norm(w_mat, p="fro").item()),
        "nuclear_norm": float(s.sum().item()),
        "top_singular_value": float(s[0].item()),
    }


# ============================================================
# Config override
# ============================================================

def _prepare_cfg(cfg, compressibility: str, beta: Optional[float], rank: Optional[int]):
    cfg = load_config_dict_like(cfg)

    if compressibility == "neuron":
        exp_name = f"cifar10_wrn16_2_group_lasso_beta_{_float_to_name(float(beta))}"
    else:
        exp_name = f"cifar10_wrn16_2_low_rank_rank_{int(rank)}"

    _set_nested_attr(cfg, ["experiment", "name"], exp_name)

    _set_nested_attr(cfg, ["dataset", "name"], "cifar10")
    _set_nested_attr(cfg, ["dataset", "num_classes"], 10)
    _set_nested_attr(cfg, ["dataset", "val_fraction"], 0.05)

    _set_nested_attr(cfg, ["optimizer", "type"], "adamw")
    _set_nested_attr(cfg, ["optimizer", "lr"], 1e-3)
    _set_nested_attr(cfg, ["optimizer", "weight_decay"], 1e-2)

    _set_nested_attr(cfg, ["training", "criterion"], "cross_entropy")
    _set_nested_attr(cfg, ["training", "label_smoothing"], 0.0)

    return cfg


# ============================================================
# Builders / checkpoint
# ============================================================

def _build_model(*, compressibility: str, num_classes: int, rank: Optional[int]) -> nn.Module:
    if compressibility == "neuron":
        return DenseWideResNet(
            depth=16,
            widen_factor=2,
            num_classes=num_classes,
            bias=False,
        )

    if compressibility == "spectral":
        if rank is None:
            raise ValueError("rank must be provided for spectral compressibility.")
        return LowRankWideResNet(
            depth=16,
            widen_factor=2,
            num_classes=num_classes,
            rank=int(rank),
            bias=False,
        )

    raise ValueError(f"Unknown compressibility type: {compressibility}")


def _save_best_checkpoint(
    *,
    path: Path,
    epoch: int,
    best_val_loss: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg,
) -> None:
    payload = CheckpointPayload(
        epoch=epoch,
        best_metric=float(best_val_loss),
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        config=cfg.to_dict(),
    )
    save_checkpoint(path, payload)


# ============================================================
# Train
# ============================================================

def train_one_setting(
    cfg,
    *,
    compressibility: str,
    beta: Optional[float] = None,
    rank: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    cfg = load_config_dict_like(cfg)

    if seed is not None:
        _set_nested_attr(cfg, ["experiment", "seed"], int(seed))

    seed_everything(int(cfg.experiment.seed))
    device = get_device(getattr(cfg.experiment, "device", "auto"))
    print("Using device:", device)

    cfg = _prepare_cfg(cfg, compressibility=compressibility, beta=beta, rank=rank)

    bundle = build_datasets(cfg)
    train_loader, val_loader, _ = build_dataloaders(
        cfg, bundle.train, bundle.val, bundle.test, device
    )

    model = _build_model(
        compressibility=compressibility,
        num_classes=bundle.num_classes,
        rank=rank,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
    )
    criterion = nn.CrossEntropyLoss()

    output_root = Path(getattr(cfg.experiment, "output_dir", "outputs"))
    exp_name = cfg.experiment.name
    run_name = f"{exp_name}_seed_{cfg.experiment.seed}"
    run_dir = output_root / "compressibility_adv_robustness" / "wrn16_2" / compressibility / run_name
    ckpt_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    _save_json(run_dir / "config.json", cfg.to_dict())

    best_ckpt_path = ckpt_dir / "best.pt"
    last_ckpt_path = ckpt_dir / "last.pt"
    metrics_jsonl_path = logs_dir / "metrics.jsonl"

    max_epochs = int(cfg.training.epochs)
    patience = int(cfg.training.early_stopping_patience)

    early = EarlyStoppingState()
    history: list[dict] = []

    print(f"[Experiment] {exp_name}")
    print(f"[Compressibility] {compressibility}")
    if beta is not None:
        print(f"[Beta] {beta}")
    if rank is not None:
        print(f"[Rank] {rank}")
    print(f"[Run dir] {run_dir}")
    print(f"[Early stopping patience] {patience}")

    for epoch in range(max_epochs):
        model.train()

        running_total = 0
        running_correct = 0
        running_ce_loss = 0.0
        running_reg = 0.0
        running_total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            ce_loss = criterion(logits, y)

            if compressibility == "neuron":
                reg_term = scale_invariant_group_lasso_penalty(model)
                loss = ce_loss + float(beta) * reg_term
            else:
                reg_term = torch.tensor(0.0, device=device)
                loss = ce_loss

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            preds = logits.argmax(dim=1)

            running_total += bs
            running_correct += int((preds == y).sum().item())
            running_ce_loss += float(ce_loss.item()) * bs
            running_reg += float(reg_term.item()) * bs
            running_total_loss += float(loss.item()) * bs

        train_ce = running_ce_loss / max(running_total, 1)
        train_reg = running_reg / max(running_total, 1)
        train_total_loss = running_total_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_res = evaluate(model, val_loader, criterion, device)
        stats = collect_first_layer_stats(model, compressibility)

        record = {
            "epoch": epoch + 1,
            "compressibility": compressibility,
            "beta": None if beta is None else float(beta),
            "rank": None if rank is None else int(rank),
            "train_ce_loss": train_ce,
            "train_regularizer": train_reg,
            "train_total_loss": train_total_loss,
            "train_accuracy": train_acc,
            "val_loss": float(val_res.loss),
            "val_accuracy": float(val_res.accuracy),
            "fro_norm": stats["fro_norm"],
            "nuclear_norm": stats["nuclear_norm"],
            "top_singular_value": stats["top_singular_value"],
        }
        history.append(record)
        _save_jsonl_line(metrics_jsonl_path, record)

        print(
            f"[Epoch {epoch+1:03d}] "
            f"train_ce={train_ce:.4f} | "
            f"train_reg={train_reg:.4f} | "
            f"train_total={train_total_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_res.loss:.4f} | "
            f"val_acc={val_res.accuracy:.4f} | "
            f"fro={stats['fro_norm']:.4f} | "
            f"nuc={stats['nuclear_norm']:.4f} | "
            f"top_sv={stats['top_singular_value']:.4f}"
        )

        if val_res.loss < early.best_val_loss:
            early.best_val_loss = float(val_res.loss)
            early.best_epoch = epoch
            early.epochs_without_improvement = 0

            _save_best_checkpoint(
                path=best_ckpt_path,
                epoch=epoch,
                best_val_loss=early.best_val_loss,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
            )
        else:
            early.epochs_without_improvement += 1

        last_payload = CheckpointPayload(
            epoch=epoch,
            best_metric=float(early.best_val_loss),
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None,
            config=cfg.to_dict(),
        )
        save_checkpoint(last_ckpt_path, last_payload)

        if early.epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best val_loss={early.best_val_loss:.6f} at epoch {early.best_epoch+1}."
            )
            break

    summary = {
        "experiment_name": exp_name,
        "compressibility": compressibility,
        "beta": None if beta is None else float(beta),
        "rank": None if rank is None else int(rank),
        "seed": int(cfg.experiment.seed),
        "best_val_loss": float(early.best_val_loss),
        "best_epoch": int(early.best_epoch + 1),
        "last_epoch": int(len(history)),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "history": history,
    }

    _save_json(run_dir / "summary.json", summary)

    print(
        f"Done. compressibility={compressibility} | "
        f"best_val_loss={early.best_val_loss:.6f} | "
        f"best_epoch={early.best_epoch+1}"
    )
    return summary


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--compressibility", type=str, required=True, choices=["neuron", "spectral"])
    parser.add_argument("--beta", type=float, required=False, default=None)
    parser.add_argument("--rank", type=int, required=False, default=None)
    parser.add_argument("--seed", type=int, required=False, default=None)
    args = parser.parse_args()

    if args.compressibility == "neuron" and args.beta is None:
        raise ValueError("--beta is required when --compressibility neuron")
    if args.compressibility == "spectral" and args.rank is None:
        raise ValueError("--rank is required when --compressibility spectral")

    cfg = load_config(args.config)

    train_one_setting(
        cfg,
        compressibility=args.compressibility,
        beta=args.beta,
        rank=args.rank,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()