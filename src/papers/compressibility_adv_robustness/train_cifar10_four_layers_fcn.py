from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
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



def _alpha_to_name(alpha: float) -> str:
    if float(alpha).is_integer():
        return str(int(alpha))
    return str(alpha).replace(".", "p")


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



class DenseFCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dim: int = 2000,
        num_hidden_layers: int = 4,
        num_classes: int = 10,
        bias: bool = False,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, in_features) * 0.02)

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


class LowRankFCN(nn.Module):
    def __init__(
        self,
        input_dim: int = 3 * 32 * 32,
        hidden_dim: int = 2000,
        num_hidden_layers: int = 4,
        num_classes: int = 10,
        rank: int = 256,
        bias: bool = False,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(LowRankLinear(in_dim, hidden_dim, rank=rank, bias=bias))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)



def _dense_linear_modules(model: nn.Module) -> dict[str, nn.Linear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}

def _dense_hidden_linear_modules(model: nn.Module) -> dict[str, nn.Linear]:
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    last_key = list(modules.keys())[-1]
    modules.pop(last_key)

    return modules

def _lowrank_modules(model: nn.Module) -> dict[str, LowRankLinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, LowRankLinear)}


def group_lasso_penalty(model: nn.Module) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for _, layer in _dense_hidden_linear_modules(model).items():
        penalty = penalty + torch.norm(layer.weight, p=2, dim=1).sum()
    return penalty


def get_initial_fro_norms_dense(model: nn.Module) -> dict[str, float]:
    out = {}
    for name, layer in _dense_hidden_linear_modules(model).items():
        out[name] = float(torch.norm(layer.weight.detach(), p="fro").item())
    return out


def get_initial_fro_norms_lowrank(model: nn.Module) -> dict[str, float]:
    out = {}
    for name, layer in _lowrank_modules(model).items():
        out[name] = float(torch.norm(layer.effective_weight().detach(), p="fro").item())
    return out


def frobenius_normalize_dense_to_initial(model: nn.Module, target_fro_norms: dict[str, float], eps: float = 1e-12) -> None:
    with torch.no_grad():
        for name, layer in _dense_hidden_linear_modules(model).items():
            w = layer.weight
            cur = torch.norm(w, p="fro")
            target = target_fro_norms[name]
            w.mul_(target / max(float(cur.item()), eps))


def frobenius_normalize_lowrank_to_initial(model: nn.Module, target_fro_norms: dict[str, float], eps: float = 1e-12) -> None:
    with torch.no_grad():
        for name, layer in _lowrank_modules(model).items():
            w = layer.effective_weight()
            cur = torch.norm(w, p="fro")
            target = target_fro_norms[name]

            scale = target / max(float(cur.item()), eps)
            scale_sqrt = math.sqrt(scale)
            layer.U.mul_(scale_sqrt)
            layer.V.mul_(scale_sqrt)


def collect_first_layer_stats(model: nn.Module, compressibility: str) -> dict[str, float]:
    if compressibility == "neuron":
        first_name, first_layer = next(iter(_dense_linear_modules(model).items()))
        w = first_layer.weight.detach()
    else:
        first_name, first_layer = next(iter(_lowrank_modules(model).items()))
        w = first_layer.effective_weight().detach()

    s = torch.linalg.svdvals(w)
    return {
        "first_layer_name": first_name,
        "fro_norm": float(torch.norm(w, p="fro").item()),
        "nuclear_norm": float(s.sum().item()),
        "top_singular_value": float(s[0].item()),
    }



def _prepare_cfg(cfg, compressibility: str, alpha: Optional[float], rank: Optional[int]):
    cfg = load_config_dict_like(cfg)

    if compressibility == "neuron":
        exp_name = f"cifar10_fcn_group_lasso_alpha_{_alpha_to_name(float(alpha))}"
    else:
        exp_name = f"cifar10_fcn_low_rank_rank_{int(rank)}"

    _set_nested_attr(cfg, ["experiment", "name"], exp_name)

    _set_nested_attr(cfg, ["dataset", "name"], "cifar10")
    _set_nested_attr(cfg, ["dataset", "num_classes"], 10)
    _set_nested_attr(cfg, ["dataset", "val_fraction"], 0.05)

    _set_nested_attr(cfg, ["model", "name"], "fcn")
    _set_nested_attr(cfg, ["model", "input_shape"], [3, 32, 32])
    _set_nested_attr(cfg, ["model", "hidden_dims"], [2000, 2000, 2000, 2000])
    _set_nested_attr(cfg, ["model", "dropout"], 0.0)
    _set_nested_attr(cfg, ["model", "bias"], False)

    _set_nested_attr(cfg, ["optimizer", "type"], "adamw")
    _set_nested_attr(cfg, ["optimizer", "lr"], 1e-3)
    _set_nested_attr(cfg, ["optimizer", "weight_decay"], 1e-2)

    _set_nested_attr(cfg, ["training", "criterion"], "cross_entropy")
    _set_nested_attr(cfg, ["training", "label_smoothing"], 0.0)

    return cfg



def _build_model(
    *,
    compressibility: str,
    num_classes: int,
    rank: Optional[int],
) -> nn.Module:
    if compressibility == "neuron":
        return DenseFCN(
            input_dim=3 * 32 * 32,
            hidden_dim=2000,
            num_hidden_layers=4,
            num_classes=num_classes,
            bias=False,
        )
    if compressibility == "spectral":
        if rank is None:
            raise ValueError("rank must be provided for spectral compressibility.")
        return LowRankFCN(
            input_dim=3 * 32 * 32,
            hidden_dim=2000,
            num_hidden_layers=4,
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




def train_one_setting(
    cfg,
    *,
    compressibility: str,
    alpha: Optional[float] = None,
    rank: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    cfg = load_config_dict_like(cfg)

    if seed is not None:
        _set_nested_attr(cfg, ["experiment", "seed"], int(seed))

    seed_everything(int(cfg.experiment.seed))
    device = get_device(getattr(cfg.experiment, "device", "auto"))
    print("Using device:", device)

    cfg = _prepare_cfg(cfg, compressibility=compressibility, alpha=alpha, rank=rank)

    bundle = build_datasets(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(
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

    if compressibility == "neuron":
        target_fro_norms = get_initial_fro_norms_dense(model)
        frobenius_normalize_dense_to_initial(model, target_fro_norms)
    else:
        target_fro_norms = get_initial_fro_norms_lowrank(model)
        frobenius_normalize_lowrank_to_initial(model, target_fro_norms)

    output_root = Path(getattr(cfg.experiment, "output_dir", "outputs"))
    exp_name = cfg.experiment.name
    run_name = f"{exp_name}_seed_{cfg.experiment.seed}"
    run_dir = output_root / "compressibility_adv_robustness" / compressibility / run_name
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
    if alpha is not None:
        print(f"[Alpha] {alpha}")
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

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            ce_loss = criterion(logits, y)

            if compressibility == "neuron":
                reg_term = group_lasso_penalty(model)
                loss = ce_loss + float(alpha) * reg_term
            else:
                reg_term = torch.tensor(0.0, device=device)
                loss = ce_loss

            loss.backward()
            optimizer.step()

            if compressibility == "neuron":
                frobenius_normalize_dense_to_initial(model, target_fro_norms)
            else:
                frobenius_normalize_lowrank_to_initial(model, target_fro_norms)

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
            "alpha": None if alpha is None else float(alpha),
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
        "alpha": None if alpha is None else float(alpha),
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




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--compressibility", type=str, required=True, choices=["neuron", "spectral"])
    parser.add_argument("--alpha", type=float, required=False, default=None)
    parser.add_argument("--rank", type=int, required=False, default=None)
    parser.add_argument("--seed", type=int, required=False, default=None)
    args = parser.parse_args()

    if args.compressibility == "neuron" and args.alpha is None:
        raise ValueError("--alpha is required when --compressibility neuron")
    if args.compressibility == "spectral" and args.rank is None:
        raise ValueError("--rank is required when --compressibility spectral")

    cfg = load_config(args.config)

    train_one_setting(
        cfg,
        compressibility=args.compressibility,
        alpha=args.alpha,
        rank=args.rank,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()