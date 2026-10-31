#!/usr/bin/env bash
# Epoch time of TempFlash, Stack-Overflow with TGAT (settings of train_STACKOVERFLOW_TGAT.sh): 2 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_STACKOVERFLOW_TGAT.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data STACKOVERFLOW --config ../config/TGAT_STACKOVERFLOW.yml --batch_size 5000 --lr 0.0001 --dropout 0.1 --att_dropout 0.1 --epochs 2 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/STACKOVERFLOW_TGAT.json | tee ../logs/timing/STACKOVERFLOW_TGAT.log
