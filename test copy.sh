#!/bin/bash
#SBATCH --partition=debug --output=tmp.log
#SBATCH --job-name=llf-test
#SBATCH -p gpu  --gres=gpu:2

module purge
module load pytorch
python train.py with server_user colored_mnist_vit skewed1 severity4
python train.py with server_user colored_mnist_vit skewed2 severity4
python train.py with server_user colored_mnist_vit skewed3 severity4
python train.py with server_user colored_mnist_vit skewed4 severity4