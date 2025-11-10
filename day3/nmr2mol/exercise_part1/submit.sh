#!/bin/bash
#SBATCH --partition=AIML4S2025
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB

source conda-env.sh
conda activate NMR_env

nmr_train full_train_config.yaml
