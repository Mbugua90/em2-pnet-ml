#!/bin/bash
#SBATCH --job-name=mlp_pneumonia
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --mem=32000
#SBATCH --output=mlp_train_%j.log
#SBATCH --error=mlp_train_%j.err

# EM2-PNET MLP Branch Training Job
# ---------------------------------
# Submit with: sbatch train_mlp.sbatch
# Monitor with: squeue -u $USER
# Check progress live: tail -f mlp_train_<jobid>.log
# Cancel if needed: scancel <jobid>

module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM

cd /scratch/lustre/users/francisco.ngigi24/mimic_iv_ed

echo "Job started on $(hostname) at $(date)"
echo "GPU assigned:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 train_mlp.py mimic_features_train.csv mimic_features_val.csv mimic_features_test.csv

echo "Job finished at $(date)"
