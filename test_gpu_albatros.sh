#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=SallesInfo
#SBATCH --nodelist=albatros
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p slurm_logs
cd /users/eleves-a/2022/rida.assalouh/ADL/netcompression || exit 1

hostname
nvidia-smi
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('current device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
