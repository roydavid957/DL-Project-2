#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --job-name=GRU_drp
#SBATCH --mem=4000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=GRU_Dropout.txt

module load Python/3.6.4-foss-2018a

python3 run.py -M train -E GRU -ED 2 -D GRU -O ADAM -EN 50 -d 0.1
python3 run.py -M train -E GRU -ED 2 -D GRU -O ADAM -EN 50 -d 0.2
python3 run.py -M train -E GRU -ED 2 -D GRU -O ADAM -EN 50 -d 0.3
