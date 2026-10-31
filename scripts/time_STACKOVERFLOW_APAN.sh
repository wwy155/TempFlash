#!/usr/bin/env bash
# Epoch time of TempFlash, Stack-Overflow with APAN (settings of train_STACKOVERFLOW_APAN.sh): 2 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_STACKOVERFLOW_APAN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data STACKOVERFLOW --config ../config/APAN_STACKOVERFLOW.yml --batch_size 2000 --lr 0.0001 --dropout 0.1 --att_dropout 0.1 --epochs 2 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/STACKOVERFLOW_APAN.json | tee ../logs/timing/STACKOVERFLOW_APAN.log
