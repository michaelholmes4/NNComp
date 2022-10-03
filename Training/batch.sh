#!/bin/bash
#SBATCH --job-name=4811-nn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michael.holmes1@uqconnect.edu.au

conda activate mlp
python3 ~/NNComp_Training/Training/training.py lstm-4-1 -e 1
