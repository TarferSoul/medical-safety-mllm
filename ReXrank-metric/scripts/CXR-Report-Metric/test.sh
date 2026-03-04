#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH --job-name="metric"
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --output=./sbatch/test_%j.log
#SBATCH --error=./sbatch/test_%j.err

module load gcc/9.2.0 cuda/11.7 # miniconda3/4.10.3

python test_metric_findings.py 
python test_metric.py 