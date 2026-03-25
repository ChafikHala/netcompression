# Model Compression: Adversarial Robustness and Generalization

> A PyTorch-based project studying compressibility in neural networks — its relationship to adversarial robustness, its role in optimization via overparameterization, and its connection to generalization and memorization.

---

## What this is

`netcompression` provides training pipelines, adversarial evaluation tools, and sweep infrastructure to study three interconnected questions about neural network compressibility:

1. **Adversarial robustness** — does structured compressibility (neuron or spectral) make models more vulnerable to adversarial attacks? Following the theoretical framework of [Barsbey et al. (2025)](#references), we show that compression concentrates predictive information in fewer directions, increasing operator norms and therefore adversarial sensitivity.

2. **Overparameterization and optimization** — does starting from a large overparameterized model and then pruning it produce better solutions than training a small dense model of the same final size directly? We compare MobileNet models pruned to varying sparsity levels against width-scaled dense models at matched parameter budgets, following [Zhu & Gupta (2017)](#references).

3. **Memorization and generalization** — are models that rely on memorization harder to compress? We train WideResNet-16-2 on progressively label-noisy versions of CIFAR-10 and measure how retained training accuracy under post-hoc pruning degrades as noise (and thus memorization) increases.

The codebase supports CIFAR-10 and MNIST, several regularization strategies (nuclear norm, group lasso, neuron-level shrinkage, low-rank factorization), and adversarial evaluation via the ART library.

---

## Installation

**Requirements:** Python ≥ 3.10 (uses union types and strict dataclasses).

```sh
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install torch torchvision matplotlib numpy pyyaml art
```

Optionally install GPU drivers so `torch.cuda.is_available()` returns `True`.

**Data** is downloaded automatically on first run (CIFAR-10 and MNIST are fetched to `data/` via `download=True`). 

---

## Research sweeps

### 1. Compressibility and adversarial robustness

Sweep scripts live under `src/papers/compressibility_adv_robustness/`. These vary regularization strength and measure how adversarial accuracy, clean accuracy, and internal representation shift respond to increasing compressibility.

Each model is trained on CIFAR-10 (mainly, and also Binary MNIST) under both neuron and spectral compressibility regimes across three seeds. Trained checkpoints are saved for downstream adversarial evaluation.

#### CIFAR-10
Spectral Compressibility via Low-rank approximation and Neuron Compressibility via Group Lasso.

**Four Layers FCN training on CIFAR-10**

```sh
python -m src.papers.compressibility_adv_robustness.run_cifar10_four_layers_neuron_spectral \
  --config configs/papers/compressibility_adv_robustness/cifar10_four_layers_fcn.yaml \
  --compressibility spectral \
  --ranks 10 20 40 60 80 100
  --seeds 0 1 2
```

Switch `--compressibility` to `neuron` for the group lasso variant, and specify `--n-alphas`, `alpha-min`, `alpha-max`, `alpha-grid`.


**WideResNet-16-2 training on CIFAR-10** (neuron compressibility via Group Lasso, or spectral via low-rank):

```sh
python -m src.papers.compressibility_adv_robustness.run_cifar10_wide_resnet_neuron_spectral \
  --config configs/papers/compressibility_adv_robustness/cifar10_wrn16_2.yaml \
  --compressibility neuron \
  --n-betas 7 \
  --beta-min 1e-5 \
  --beta-max 3e-1 \
  --seeds 0 1 2
```

Switch `--compressibility` to `spectral` for the spectral norm variant, and specify desired `ranks`.

### Adversarial evaluation

`attack_utils.py` (and the CIFAR-specific `attack_utils.cifar`) wrap ART's AutoPGD and FGSM attacks under both L2 and L∞ norms. Use the `PyTorchClassifier` wrappers whenever you need adversarial gradients — their default budgets are already calibrated to the normalized CIFAR-10/MNIST transforms in `src/data/datasets.py`.

Beyond standard adversarial accuracy, the paper scripts also log the relative representation shift `‖z_adv − z‖₂ / ‖z‖₂` between clean and adversarially perturbed inputs, and vulnerability to universal adversarial examples (UAE) as a function of regularization strength and Frobenius norm scaling.


Here, we run **FGSM** attacks on the saved models, and retrieve **adversarial accuracy** and **representation shift**.

Neuron compressibility --- CIFAR-10 --- WideResNet-16-2 --- FGSM attack example:

```sh
python -m src.papers.compressibility_adv_robustness.eval_cifar10_wrn_robustness_and_repr_shift \
  --config configs/papers/compressibility_adv_robustness/cifar10_wrn16_2.yaml \
  --compressibility neuron \
  --base-dir outputs \
  --ckpt best.pt \
  --n-betas 10 \
  --beta-min 1e-5 \
  --beta-max 3e-1 \
  --attack fgsm \
  --norm linf \
  --eps-linf 0.031 \
  --seeds 0 1 2
```

### 2. Overparameterization and optimization (large-sparse vs. small-dense)

Train MobileNet at different width multipliers, then compare against a full-width model pruned to a matching parameter budget:

# Dense baselines at varying widths
```sh
python -m src.experiments.train --config configs/cifar10_mobilenet_width1.yaml
python -m src.experiments.train --config configs/cifar10_mobilenet_width25.yaml
python -m src.experiments.train --config configs/cifar10_mobilenet_width50.yaml
python -m src.experiments.train --config configs/cifar10_mobilenet_width75.yaml
```
# Pruned from full-width model

```sh
python -m src.experiments.train --config configs/cifar10_mobilenet_sparse_45.yaml
python -m src.experiments.train --config configs/cifar10_mobilenet_sparse_80.yaml
python -m src.experiments.train --config configs/cifar10_mobilenet_sparse_96.yaml
```

The key comparison is accuracy at matched non-zero parameter counts. The gradual cubic pruning schedule follows [Zhu & Gupta (2017)](#references).

### 3. Memorization, label noise, and compressibility

Train WideResNet-16-2 on progressively corrupted CIFAR-10 labels, then apply post-hoc global magnitude pruning to measure how retained training accuracy degrades as memorization increases:

```sh
python src/experiments/overfitting_harder_to_compress.py \
  --data-root ./data \
  --output-dir ./outputs/noise_pruning_cifar10_wrn16 \
  --epochs 100 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --nesterov \
  --seed 0 \
  --alphas 0.0,0.1,0.2,0.4,0.6,0.8,1.0 \
  --pruning-ratios 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95,0.98,0.99
```

`--alphas` controls the fraction of training labels that are randomly permuted (0.0 = clean, 1.0 = fully shuffled). `--pruning-ratios` sets the global magnitude pruning levels applied post-hoc to evaluate retained training accuracy.

Aggregate results afterward:

```sh
python aggregate_noise_pruning_results.py
```


---

## Extending the codebase

**New experiment:** copy a YAML config into `configs/`, adjust `model`, `optimizer`, `training`, and `pruning` fields, then run `python -m src.experiments.train`. Set `experiment.name` to tag the run and keep `experiment.output_dir` consistent so `eval.py` can find it.

**Pruning:** the API in `src/pruning/` supports epoch-based global magnitude pruning and step-based gradual cubic layer-wise schedules (following [Zhu & Gupta, 2017](#references)). Tune `pruning.method`, `final_sparsity`, and `frequency` in your config.

**HPC:** the shell scripts in the root (`run_cifar_10_wide_resnet_layer_rank.sh`, `run_comp_robust_mnist.sh`, etc.) wrap `sbatch` calls, check GPU availability via `nvidia-smi`, and launch the appropriate Python module. Edit the `python -m ...` invocation inside each script to point to a different config or seed range.

---

## References

**[1]** Barsbey, M., Ribeiro, A. H., Şimşekli, U., & Birdal, T. (2025).
*On the Interaction of Compressibility and Adversarial Robustness.*
arXiv:2507.17725 [cs.LG]. https://arxiv.org/abs/2507.17725

**[2]** Zhu, M., & Gupta, S. (2017).
*To prune, or not to prune: exploring the efficacy of pruning for model compression.*
arXiv:1710.01878 [stat.ML]. https://arxiv.org/abs/1710.01878