#!/bin/bash
#SBATCH --job-name=debug
#SBATCH --time=00:05:00
#SBATCH --partition=total
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zacharias.vf@gmail.com
#SBATCH --constraint=NV

source /users/zfisches/.bash_profile;
cd /users/zfisches/phd;

srun ./debug_run.sh