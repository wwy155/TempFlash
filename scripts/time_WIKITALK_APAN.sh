#!/usr/bin/env bash
# Epoch time of TempFlash, Wiki-Talk with APAN (settings of train_WIKITALK_APAN.sh): 3 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_WIKITALK_APAN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data WIKITALK --config ../config/APAN_WIKITALK.yml --batch_size 1500 --lr 0.0002 --dropout 0.1 --att_dropout 0.1 --epochs 3 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/WIKITALK_APAN.json | tee ../logs/timing/WIKITALK_APAN.log
