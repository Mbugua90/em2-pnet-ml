#!/bin/bash
#SBATCH --job-name=mlp_keras_ptq
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --mem=16000
#SBATCH --output=mlp_keras_ptq_%j.log
#SBATCH --error=mlp_keras_ptq_%j.err

# EM2-PNET Objective 3: MLP branch Keras reimplementation + PTQ
# No GPU requested -- this model is tiny (5 input features, <10K parameters),
# CPU training will be fast and this keeps GPU nodes free for other jobs.
# Submit with: sbatch train_mlp_keras.sbatch

# IMPORTANT: activate the isolated tf_env venv, NOT the conda module's own
# Python -- TensorFlow was installed into tf_env specifically to avoid
# conflicting with the PyTorch packages already in the shared conda module.
module load applications/eng/gpu/python/conda-26.1.0-python-3.12-vLLM
source /scratch/lustre/users/francisco.ngigi24/tf_env/bin/activate

cd /scratch/lustre/users/francisco.ngigi24/mimic_iv_ed

echo "Job started on $(hostname) at $(date)"

python3 train_mlp_keras_ptq.py mimic_features_train.csv mimic_features_val.csv mimic_features_test.csv

echo "Job finished at $(date)"
