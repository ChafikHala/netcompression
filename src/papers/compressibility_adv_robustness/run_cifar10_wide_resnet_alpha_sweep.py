from __future__ import annotations

import argparse
import copy
import numpy as np

from src.papers.compressibility_adv_robustness.train_cifar10_wide_resnet import (
    train_one_setting,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--compressibility", type=str, required=True, choices=["neuron", "spectral"],)

    # neuron-compressibility sweep
    parser.add_argument("--n-betas", type=int, default=10)
    parser.add_argument("--beta-min", type=float, default=1e-5)
    parser.add_argument("--beta-max", type=float, default=1e-1)
    parser.add_argument("--beta-grid", type=str, default="geom", choices=["geom", "linear"])

    # spectral-compressibility sweep
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])

    # common
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])

    args = parser.parse_args()

    base_cfg = load_config(args.config)

    print("Compressibility:", args.compressibility)
    print("Seeds:", args.seeds)

    if args.compressibility == "neuron":
        if args.beta_grid == "geom":
            sweep_values = np.geomspace(args.beta_min, args.beta_max, args.n_betas)
        else:
            sweep_values = np.linspace(args.beta_min, args.beta_max, args.n_betas)

        print("Betas:", [float(b) for b in sweep_values])

        for beta in sweep_values:
            for seed in args.seeds:
                print("\n" + "=" * 80)
                print(f"Training neuron-compressibility | beta={float(beta):.8f}, seed={seed}")
                print("=" * 80)

                cfg = copy.deepcopy(base_cfg)
                train_one_setting(
                    cfg,
                    compressibility="neuron",
                    beta=float(beta),
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
                    beta=None,
                    rank=int(rank),
                    seed=int(seed),
                )


if __name__ == "__main__":
    main()