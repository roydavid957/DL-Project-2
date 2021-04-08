#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --job-name=gpt2_formatted_model
#SBATCH --mem=8000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --output=gpt2_formatted_model_result.txt

module load Python

python3 gpt2_formatted_model.py
