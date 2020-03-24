#!/usr/bin/env bash

#SBATCH --job-name=qb-bert
#SBATCH --gres=gpu:1
#SBATCH --chdir=/fs/clip-quiz/entilzha/code/qb-bert/src
#SBATCH --output=/fs/clip-quiz/entilzha/logs/%A.log
#SBATCH --error=/fs/clip-quiz/entilzha/logs/%A.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20g
#SBATCH --partition=gpu
#SBATCH --exclude=materialgpu00

set -x
hostname
nvidia-smi
source /fs/clip-quiz/entilzha/anaconda3/etc/profile.d/conda.sh > /dev/null 2> /dev/null
conda activate qb-bert
export SLURM_LOG_FILE="/fs/clip-quiz/entilzha/logs/${SLURM_JOB_ID}.log"
export MODEL_CONFIG_FILE="$2"
cd $1
pwd
srun python qb/main.py train $2
