#!/bin/bash
#SBATCH --job-name=readmission_training
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=training_log_%j.out

# --- Environment Setup ---
echo "Setting up the environment..."
module load conda/base
conda activate readmission_env

# --- Verification ---
echo "Verifying environment..."
echo "Using Python from: $(which python)"
pip show transformers

# --- Run the Python Script ---
echo "Starting the training script..."
python run_training.py
