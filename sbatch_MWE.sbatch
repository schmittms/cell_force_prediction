#!/bin/bash


#SBATCH --output=./out/output.out
#SBATCH --account=pi-vitelli
#SBATCH --partition=vitelli-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --exclude=midway3-0294


module load python
source activate /project/vitelli/ml_venv
cd /project/vitelli/matthew/cell_force_prediction_public

python train_predictor_MWE.py

