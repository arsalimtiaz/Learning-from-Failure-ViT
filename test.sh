#!/bin/bash
#SBATCH -p gpu -t 12:00:00 -c 4 --mem 8G -C A100|V100|K80 --gres=gpu:1
#SBATCH --output=tmp.log
#SBATCH -a 1-1

module load cuda/11.0
module load anaconda3/2020.11
module load git
source activate llf

python train.py with server_user colored_mnist_vit skewed1 severity4
python train.py with server_user colored_mnist_vit skewed2 severity4
python train.py with server_user colored_mnist_vit skewed3 severity4
python train.py with server_user colored_mnist_vit skewed4 severity4