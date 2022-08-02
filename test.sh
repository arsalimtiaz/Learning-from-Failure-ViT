#!/bin/bash
#SBATCH -p gpu -t 18:00:00 -c 4 --mem 8G -C A100|V100|K80 --gres=gpu:2
#SBATCH --output=tmp.log
#SBATCH -a 1-1

module load cuda/11.0
module load anaconda3/2020.11
module load git
source activate llf

python train_vanilla.py with server_user corrupted_cifar10 skewed1 severity4
python train_vanilla.py with server_user corrupted_cifar10 skewed2 severity4
python train_vanilla.py with server_user corrupted_cifar10 skewed3 severity4
python train_vanilla.py with server_user corrupted_cifar10 skewed4 severity4
