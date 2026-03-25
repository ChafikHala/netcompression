from __future__ import annotations

import argparse
import copy
import numpy as np

from src.papers.compressibility_adv_robustness.train_cifar10_four_layers_fcn import (
    train_one_setting,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    parser.add_argument(
        "--compressibility",
        type=str,
        required=True,
        choices=["neuron", "spectral"],
    )

    parser.add_argument("--n-alphas", type=int, default=15)
    parser.add_argument("--alpha-min", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=1e-1)
    parser.add_argument("--alpha-grid", type=str, default="geom", choices=["geom", "linear"])

    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 512, 1024])

    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    args = parser.parse_args()

    base_cfg = load_config(args.config)

    print("Compressibility:", args.compressibility)
    print("Seeds:", args.seeds)

    if args.compressibility == "neuron":
        if args.alpha_grid == "geom":
            sweep_values = np.geomspace(args.alpha_min, args.alpha_max, args.n_alphas)
        else:
            sweep_values = np.linspace(args.alpha_min, args.alpha_max, args.n_alphas)

        print("Alphas:", [float(a) for a in sweep_values])

        for alpha in sweep_values:
            for seed in args.seeds:
                print("\n" + "=" * 80)
                print(f"Training neuron-compressibility | alpha={float(alpha):.8f}, seed={seed}")
                print("=" * 80)

                cfg = copy.deepcopy(base_cfg)
                train_one_setting(
                    cfg,
                    compressibility="neuron",
                    alpha=float(alpha),
                    rank=None,
                    seed=int(seed),
                )

    else:
        print("Ranks:", args.ranks)

        for rank in args.ranks:
            for seed in args.seeds:
                print("\n" + "=" * 80)
                print(f"Training spectral-compressibility | rank={int(rank)}, seed={seed}")
                print("=" * 80)

                cfg = copy.deepcopy(base_cfg)
                train_one_setting(
                    cfg,
                    compressibility="spectral",
                    alpha=None,
                    rank=int(rank),
                    seed=int(seed),
                )


if __name__ == "__main__":
    main()