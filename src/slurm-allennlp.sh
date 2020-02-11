#!/usr/bin/env bash

#SBATCH --job-name=qb-bert
#SBATCH --gres=gpu:1
#SBATCH --chdir=/fs/clip-quiz/entilzha/code/qb-bert/src
#SBATCH --output=/fs/www-users/entilzha/logs/%A.log
#SBATCH --error=/fs/www-users/entilzha/logs/%A.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20g
#SBATCH --time=1-00:00:00
#SBATCH --qos=gpu-medium
#SBATCH --partition=gpu
#SBATCH --exclude=materialgpu00

set -x
hostname
nvidia-smi
source /fs/clip-quiz/entilzha/anaconda3/etc/profile.d/conda.sh > /dev/null 2> /dev/null
conda activate qb-bert
export SLURM_LOG_FILE="/fs/www-users/entilzha/logs/${SLURM_JOB_ID}.log"
export MODEL_CONFIG_FILE="${@: -1}"
srun allennlp train --include-package qb -f $@
