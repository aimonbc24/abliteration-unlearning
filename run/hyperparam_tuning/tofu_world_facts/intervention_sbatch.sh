#!/bin/bash
#SBATCH --partition=gpu-l40s
#SBATCH --account=ark
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=eval
#SBATCH --output="./slurm/slurm-%J-%x.out"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aimonbc@cs.washington.edu

cat $0
echo "--------------------"

sh run/hyperparam_tuning/tofu_world_facts/intervention.sh