#!/usr/bin/env bash
export MODEL_CONFIG_FILE="$2"
cd $1
pwd
python qb/main.py train $2
