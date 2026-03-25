from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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




PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def collect_weights(model: nn.Module) -> np.ndarray:
    weights = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach().cpu().numpy().ravel()
            weights.append(w)
    if not weights:
        return np.array([], dtype=np.float64)
    return np.concatenate(weights, axis=0)


def build_model() -> WideResNet:
    return WideResNet(depth=16, widen_factor=2, num_classes=10, drop_rate=0.0)


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_model()
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def alpha_key_to_float(k: str) -> float:
    return float(k)


def find_threshold_ratio(pruning_ratios: List[float], retained: List[float], drop_threshold: float) -> float | None:
    retained_threshold = 1.0 - drop_threshold
    for p, r in zip(pruning_ratios, retained):
        if r <= retained_threshold:
            return float(p)
    return None



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
            alpha = payload["alpha"]
            results[str(float(alpha))] = payload

        return {
            "config": None,
            "results": results,
        }

    raise FileNotFoundError(
        f"Could not find either:\n"
        f"  - {all_results_path}\n"
        f"  - any files matching {metrics_dir / 'result_alpha_*.json'}"
    )


def aggregate_across_seed_folders(seed_folders: List[Path], drop_threshold: float) -> tuple[dict, dict]:
    per_seed = []
    for folder in seed_folders:
        per_seed.append(load_seed_results(folder))

    alpha_keys = sorted(
        per_seed[0]["results"].keys(),
        key=alpha_key_to_float,
    )

    aggregated: Dict[str, dict] = {}
    raw: Dict[str, list] = {}

    for alpha_key in alpha_keys:
        raw[alpha_key] = []

        pruning_ratios_ref = per_seed[0]["results"][alpha_key]["pruning_curve"]["pruning_ratios"]

        retained_matrix = []
        train_accs = []
        test_accs = []
        overfit_rates = []
        threshold_ratios = []

        for seed_idx, seed_payload in enumerate(per_seed):
            seed_result = seed_payload["results"][alpha_key]

            pruning_ratios = seed_result["pruning_curve"]["pruning_ratios"]
            retained = seed_result["pruning_curve"]["retained_train_acc"]

            if len(pruning_ratios) != len(pruning_ratios_ref) or any(
                abs(float(a) - float(b)) > 1e-12 for a, b in zip(pruning_ratios, pruning_ratios_ref)
            ):
                raise ValueError(f"Pruning ratio grid mismatch for alpha={alpha_key} between folders.")

            retained_matrix.append(retained)
            train_accs.append(float(seed_result["train_acc"]))
            test_accs.append(float(seed_result["test_acc"]))
            overfit_rates.append(float(seed_result["overfit_rate"]))

            threshold_ratio = find_threshold_ratio(
                pruning_ratios=pruning_ratios,
                retained=retained,
                drop_threshold=drop_threshold,
            )
            threshold_ratios.append(np.nan if threshold_ratio is None else float(threshold_ratio))

            raw[alpha_key].append({
                "seed_folder": str(seed_folders[seed_idx]),
                "alpha": float(alpha_key),
                "train_acc": float(seed_result["train_acc"]),
                "test_acc": float(seed_result["test_acc"]),
                "overfit_rate": float(seed_result["overfit_rate"]),
                "pruning_ratios": [float(x) for x in pruning_ratios],
                "retained_train_acc": [float(x) for x in retained],
                "threshold_ratio_for_drop": None if threshold_ratio is None else float(threshold_ratio),
            })

        retained_matrix = np.array(retained_matrix, dtype=float)
        threshold_ratios = np.array(threshold_ratios, dtype=float)

        aggregated[alpha_key] = {
            "alpha": float(alpha_key),
            "n_seeds": len(seed_folders),
            "pruning_ratios": [float(x) for x in pruning_ratios_ref],
            "retained_train_acc_mean": retained_matrix.mean(axis=0).tolist(),
            "retained_train_acc_std": retained_matrix.std(axis=0, ddof=0).tolist(),
            "train_acc_mean": float(np.mean(train_accs)),
            "train_acc_std": float(np.std(train_accs, ddof=0)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs, ddof=0)),
            "overfit_rate_mean": float(np.mean(overfit_rates)),
            "overfit_rate_std": float(np.std(overfit_rates, ddof=0)),
            "threshold_ratio_mean": float(np.nanmean(threshold_ratios)) if not np.all(np.isnan(threshold_ratios)) else None,
            "threshold_ratio_std": float(np.nanstd(threshold_ratios, ddof=0)) if not np.all(np.isnan(threshold_ratios)) else None,
            "threshold_ratio_values": [None if np.isnan(x) else float(x) for x in threshold_ratios],
            "drop_threshold": float(drop_threshold),
        }

    return raw, aggregated



def plot_retained_curves_mean_std(aggregated: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    alpha_keys = sorted(aggregated.keys(), key=alpha_key_to_float)

    for i, alpha_key in enumerate(alpha_keys):
        stats = aggregated[alpha_key]
        xs = np.array(stats["pruning_ratios"], dtype=float)
        ys = np.array(stats["retained_train_acc_mean"], dtype=float)
        yerr = np.array(stats["retained_train_acc_std"], dtype=float)

        color = PALETTE[i % len(PALETTE)]

        ax.fill_between(xs, ys - yerr, ys + yerr, color=color, alpha=0.12, linewidth=0)
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=color,
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            linewidth=1.4,
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            label=rf"noise = {100 * float(alpha_key):.0f}%",
            zorder=3,
        )

    ax.set_xlabel("Pruning Ratio", fontsize=15, labelpad=8)
    ax.set_ylabel("Retained Training Accuracy", fontsize=15, labelpad=10)
    # ax.set_title("Retained Training Accuracy vs Pruning Ratio", fontsize=15, pad=10)
    ax.set_ylim(0.0, 1.05)
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
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_threshold_bars_mean_std(aggregated: dict, drop_threshold: float, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    alpha_keys = sorted(aggregated.keys(), key=alpha_key_to_float)
    labels = [f"{100 * float(k):.0f}%" for k in alpha_keys]
    means = []
    stds = []

    for k in alpha_keys:
        m = aggregated[k]["threshold_ratio_mean"]
        s = aggregated[k]["threshold_ratio_std"]
        means.append(0.0 if m is None else float(m))
        stds.append(0.0 if s is None else float(s))

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        linewidth=0.8,
        edgecolor="#666666",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Label-Noise Level", fontsize=15, labelpad=8)
    ax.set_ylabel("Threshold Pruning Ratio", fontsize=15, labelpad=10)
    ax.set_title(
        f"Smallest Pruning Ratio Causing at Least {100 * drop_threshold:.0f}% Accuracy Drop",
        fontsize=15,
        pad=10,
    )
    ax.set_ylim(0.0, 1.05)

    for rect, alpha_key, mean_val in zip(bars, alpha_keys, means):
        txt = "N/A" if aggregated[alpha_key]["threshold_ratio_mean"] is None else f"{mean_val:.2f}"
        ax.annotate(
            txt,
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    _apply_elegant_style(ax)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_weight_histogram_overlay_from_checkpoints(
    ckpt_path_0: Path,
    ckpt_path_1: Path,
    save_path: Path,
    xmin: float | None,
    xmax: float | None,
    device: torch.device,
) -> None:
    model0 = load_model_from_checkpoint(ckpt_path_0, device=device)
    model1 = load_model_from_checkpoint(ckpt_path_1, device=device)

    w0 = collect_weights(model0)
    w1 = collect_weights(model1)

    all_w = np.concatenate([w0, w1])

    if xmin is None:
        xmin = float(all_w.min())
    if xmax is None:
        xmax = float(all_w.max())

    bins = np.linspace(xmin, xmax, 121)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)

    color0 = "#2E86AB"
    color1 = "#E84855"

    ax.hist(
        w0,
        bins=bins,
        density=True,
        alpha=0.42,
        color=color0,
        edgecolor=color0,
        linewidth=0.4,
        label="0% label noise",
    )
    ax.hist(
        w1,
        bins=bins,
        density=True,
        alpha=0.35,
        color=color1,
        edgecolor=color1,
        linewidth=0.4,
        label="100% label noise",
    )

    ax.set_xlabel(r"$w$", fontsize=15, labelpad=8)
    ax.set_ylabel("Density", fontsize=15, labelpad=10)
    ax.set_title("Normalized histogram of weights", fontsize=15, pad=10)
    ax.set_xlim(xmin, xmax)

    ax.legend(
        fontsize=13,
        frameon=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        loc="best",
        handlelength=1.8,
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
        "--seed-folders",
        type=str,
        nargs="+",
        required=True,
        help="List of output folders, one per seed. Each must contain metrics/models/plots.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory where aggregated jsons and figures will be saved.",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.8,
        help="Accuracy drop threshold used for the bar plot. Example: 0.8 means 80%% drop.",
    )
    parser.add_argument(
        "--hist-xmin",
        type=float,
        default=None,
        help="Optional common x-axis min for both histograms.",
    )
    parser.add_argument(
        "--hist-xmax",
        type=float,
        default=None,
        help="Optional common x-axis max for both histograms.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used only to load checkpoints for histograms.",
    )

    args = parser.parse_args()

    if not (0.0 < args.drop_threshold < 1.0):
        raise ValueError(f"--drop-threshold must be in (0,1), got {args.drop_threshold}")

    seed_folders = [Path(p) for p in args.seed_folders]
    for folder in seed_folders:
        if not folder.exists():
            raise FileNotFoundError(f"Seed folder not found: {folder}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw, aggregated = aggregate_across_seed_folders(
        seed_folders=seed_folders,
        drop_threshold=float(args.drop_threshold),
    )

    save_json(save_dir / "aggregated_raw.json", raw)
    save_json(save_dir / "aggregated_mean_std.json", aggregated)

    plot_retained_curves_mean_std(
        aggregated=aggregated,
        save_path=save_dir / "retained_train_acc_mean_std.png",
    )

    plot_threshold_bars_mean_std(
        aggregated=aggregated,
        drop_threshold=float(args.drop_threshold),
        save_path=save_dir / "threshold_pruning_ratio_mean_std.png",
    )

    first_seed_results = load_seed_results(seed_folders[0])["results"]

    if "0.0" in first_seed_results and "1.0" in first_seed_results:
        ckpt_path_0 = Path(first_seed_results["0.0"]["model_path"])
        ckpt_path_1 = Path(first_seed_results["1.0"]["model_path"])

        plot_weight_histogram_overlay_from_checkpoints(
            ckpt_path_0=ckpt_path_0,
            ckpt_path_1=ckpt_path_1,
            save_path=save_dir / "hist_weights_overlay_0_vs_100_noise.png",
            xmin=args.hist_xmin,
            xmax=args.hist_xmax,
            device=torch.device(args.device),
        )

    print(f"Saved raw aggregated data to: {save_dir / 'aggregated_raw.json'}")
    print(f"Saved mean/std aggregated data to: {save_dir / 'aggregated_mean_std.json'}")
    print(f"Saved retained-accuracy figure to: {save_dir / 'retained_train_acc_mean_std.png'}")
    print(f"Saved threshold-bar figure to: {save_dir / 'threshold_pruning_ratio_mean_std.png'}")
    if "0.0" in first_seed_results and "1.0" in first_seed_results:
        print(f"Saved overlay histogram to: {save_dir / 'hist_weights_overlay_0_vs_100_noise.png'}")


if __name__ == "__main__":
    main()