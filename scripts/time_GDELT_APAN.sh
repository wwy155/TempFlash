#!/usr/bin/env bash
# Epoch time of TempFlash, GDELT with APAN (settings of train_GDELT_APAN.sh): 2 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_GDELT_APAN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data GDELT --config ../config/APAN_GDELT.yml --batch_size 2500 --lr 0.0001 --dropout 0.1 --att_dropout 0.1 --epochs 2 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/GDELT_APAN.json | tee ../logs/timing/GDELT_APAN.log
