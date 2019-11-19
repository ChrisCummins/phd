#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --time=00:05:00
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --exclude=ault07,ault08

source /users/zfisches/.bash_profile;
cd /users/zfisches/phd;

srun bash ./deeplearning/ml4pl/scripts/devmap_runs/debug_run.sh
