#!/bin/bash
#SBATCH --job-name=cnn_lopo_debug
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --mem=32000
#SBATCH --output=cnn_lopo_debug_%j.log
#SBATCH --error=cnn_lopo_debug_%j.err

# STAGE 1: Debug run -- pneumonia-holdout folds only (~6 folds)
# Verifies the pipeline runs correctly end-to-end before committing to the
# full 70-fold run. Submit with: sbatch train_cnn_debug.sbatch

module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM

cd /scratch/lustre/users/francisco.ngigi24/icbhi_data

echo "Job started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 train_cnn_lopo.py icbhi_cnn_features.npz --debug

echo "Job finished at $(date)"
