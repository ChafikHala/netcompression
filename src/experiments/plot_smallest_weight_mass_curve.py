from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn



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
        out = self.conv1(torch.relu(self.bn1(x)))
        out = torch.relu(self.bn2(out))
        if self.drop_rate > 0.0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return out + shortcut


class WideResNet(nn.Module):
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

    @staticmethod
    def _make_group(in_planes: int, out_planes: int, n_blocks: int, stride: int, drop_rate: float) -> nn.Sequential:
        layers = [WideBasicBlock(in_planes, out_planes, stride=stride, drop_rate=drop_rate)]
        for _ in range(1, n_blocks):
            layers.append(WideBasicBlock(out_planes, out_planes, stride=1, drop_rate=drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.relu(self.bn(out))
        out = nn.functional.avg_pool2d(out, kernel_size=8)
        out = out.view(out.size(0), -1)
        return self.fc(out)



def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_seed_results(seed_folder: Path) -> dict:
    metrics_dir = seed_folder / "metrics"
    all_results_path = metrics_dir / "all_results.json"

    if all_results_path.exists():
        return load_json(all_results_path)

    per_alpha_files = sorted(metrics_dir.glob("result_alpha_*.json"))
    if per_alpha_files:
        results = {}
        for path in per_alpha_files:
            payload = load_json(path)
            results[str(float(payload["alpha"]))] = payload
        return {"config": None, "results": results}

    raise FileNotFoundError(
        f"Could not find either:\n"
        f"  - {all_results_path}\n"
        f"  - any files matching {metrics_dir / 'result_alpha_*.json'}"
    )


def build_model() -> WideResNet:
    return WideResNet(depth=16, widen_factor=2, drop_rate=0.0)


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def collect_abs_weights(model: nn.Module) -> np.ndarray:
    weights = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach().cpu().numpy().ravel()
            weights.append(np.abs(w))
    if not weights:
        return np.array([], dtype=np.float64)
    return np.concatenate(weights, axis=0)


def compute_abs_weight_cdf(abs_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the empirical CDF of |w|:
        F(t) = (1/M) * sum_i 1(|w_i| <= t)
    """
    if abs_weights.size == 0:
        raise ValueError("No weights found in model.")

    xs = np.sort(abs_weights)
    M = xs.size
    ys = np.arange(1, M + 1, dtype=np.float64) / M
    return xs, ys


def _apply_elegant_style(ax) -> None:
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


def plot_abs_weight_cdf(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    save_path: Path,
    xmax: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    color0 = "#2E86AB"
    color1 = "#E84855"

    ax.plot(
        x0,
        y0,
        color=color0,
        linewidth=2.0,
        label="0% label noise",
    )
    ax.plot(
        x1,
        y1,
        color=color1,
        linewidth=2.0,
        label="100% label noise",
    )

    ax.set_xlabel(r"Threshold $t$", fontsize=15, labelpad=8)
    ax.set_ylabel(r"$F(t)=\frac{1}{M}\sum_{i=1}^M \mathbf{1}(|w_i|\leq t)$", fontsize=15, labelpad=10)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)

    if xmax is not None:
        ax.set_xlim(0.0, xmax)

    ax.legend(
        fontsize=13,
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        loc="best",
        handlelength=2.2,
        borderpad=0.8,
        labelspacing=0.5,
    )

    _apply_elegant_style(ax)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-folder",
        type=str,
        required=True,
        help="One output folder from the experiment code, containing metrics/models/plots.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path of the output figure.",
    )
    parser.add_argument(
        "--xmax",
        type=float,
        default=None,
        help="Optional max x-axis value for the CDF plot.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used only to load checkpoints.",
    )
    args = parser.parse_args()

    seed_folder = Path(args.seed_folder)
    if not seed_folder.exists():
        raise FileNotFoundError(f"Seed folder not found: {seed_folder}")

    results = load_seed_results(seed_folder)["results"]

    if "0.0" not in results:
        raise FileNotFoundError("Could not find alpha=0.0 results in the provided seed folder.")
    if "1.0" not in results:
        raise FileNotFoundError("Could not find alpha=1.0 results in the provided seed folder.")

    ckpt_path_0 = Path(results["0.0"]["model_path"])
    ckpt_path_1 = Path(results["1.0"]["model_path"])

    device = torch.device(args.device)

    model0 = load_model_from_checkpoint(ckpt_path_0, device=device)
    model1 = load_model_from_checkpoint(ckpt_path_1, device=device)

    abs_w0 = collect_abs_weights(model0)
    abs_w1 = collect_abs_weights(model1)

    x0, y0 = compute_abs_weight_cdf(abs_w0)
    x1, y1 = compute_abs_weight_cdf(abs_w1)

    plot_abs_weight_cdf(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        save_path=Path(args.save_path),
        xmax=args.xmax,
    )

    print(f"Saved figure to: {args.save_path}")


if __name__ == "__main__":
    main()