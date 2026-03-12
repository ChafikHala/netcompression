#!/bin/bash
#SBATCH --job-name=comp_robust_mnist
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=albatros
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -e

cd /users/eleves-a/2022/rida.assalouh/ADL/netcompression || exit 1
mkdir -p slurm_logs

echo "Start time: $(date)"
echo "Host: $(hostname)"

nvidia-smi

python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python -m src.papers.compressibility_adv_robustness.run_mnist_one_layer_four_alpha_three_seeds \
  --config configs/papers/compressibility_adv_robustness/mnist_one_layer_fcn.yaml \
  --seeds 0 1 2 3 4

python -m src.papers.compressibility_adv_robustness.plot_mnist_one_layer_singular_values \
  --base-dir outputs \
  --ckpt best.pt \
  --seeds 0 1 2 3 4

python -m src.papers.compressibility_adv_robustness.eval_mnist_one_layer_neuron_pruning \
  --config configs/papers/compressibility_adv_robustness/mnist_one_layer_fcn.yaml \
  --base-dir outputs \
  --ckpt best.pt \
  --seeds 0 1 2 3 4

python -m src.papers.compressibility_adv_robustness.run_mnist_one_layer_alpha_sweep \
  --config configs/papers/compressibility_adv_robustness/mnist_one_layer_fcn.yaml \
  --n-alphas 15 \
  --alpha-min 1e-4 \
  --alpha-max 3e-1 \
  --seeds 0 1 2 3 4

python -m src.papers.compressibility_adv_robustness.eval_mnist_one_layer_robustness_and_repr_shift \
  --config configs/papers/compressibility_adv_robustness/mnist_one_layer_fcn.yaml \
  --n-alphas 15 \
  --alpha-min 1e-4 \
  --alpha-max 3e-1 \
  --seeds 0 1 2 3 4

echo "End time: $(date)"
echo "Job finished"
