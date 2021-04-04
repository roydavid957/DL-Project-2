#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --job-name=MogLSTM_lr
#SBATCH --mem=4000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=MogLSTM_LearningRate.txt

module load Python/3.6.4-foss-2018a

python3 run.py -M train -E MogLSTM -ED 2 -D MogLSTM -O ADAM -EN 100 -lr 0.001
python3 run.py -M train -E MogLSTM -ED 2 -D MogLSTM -O ADAM -EN 100 -lr 0.002
python3 run.py -M train -E MogLSTM -ED 2 -D MogLSTM -O ADAM -EN 100 -lr 0.003