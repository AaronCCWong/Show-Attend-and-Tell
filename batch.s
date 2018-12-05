#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --mem=150GB
#SBATCH --job-name=rcnn
#SBATCH --mail-type=END
#SBATCH --mail-user=aw2797@nyu.edu
#SBATCH --output=results.out

module purge
module load cudnn/9.0v7.3.0.29

python train.py --epochs 1
