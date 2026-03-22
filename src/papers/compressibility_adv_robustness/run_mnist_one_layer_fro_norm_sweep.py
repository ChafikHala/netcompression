from __future__ import annotations

import argparse
import copy
import numpy as np

from src.papers.compressibility_adv_robustness.train_mnist_one_layer_fixed_fro_norm import (
    train_one_fro_norm,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n-fro-norms", type=int, default=10)
    parser.add_argument("--fro-min", type=float, default=20.0)
    parser.add_argument("--fro-max", type=float, default=200.0)
    parser.add_argument("--fro-grid", type=str, default="linear", choices=["linear", "geom"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    if args.fro_grid == "geom":
        sweep_values = np.geomspace(args.fro_min, args.fro_max, args.n_fro_norms)
    else:
        sweep_values = np.linspace(args.fro_min, args.fro_max, args.n_fro_norms)

    print("Frobenius norms:", [float(x) for x in sweep_values])
    print("Seeds:", args.seeds)

    for fro_norm in sweep_values:
        for seed in args.seeds:
            print("\n" + "=" * 80)
            print(f"Training fixed-Frobenius model | fro_norm={float(fro_norm):.8f}, seed={seed}")
            print("=" * 80)

            cfg = copy.deepcopy(base_cfg)
            train_one_fro_norm(
                cfg,
                fro_norm=float(fro_norm),
                seed=int(seed),
            )


if __name__ == "__main__":
    main()