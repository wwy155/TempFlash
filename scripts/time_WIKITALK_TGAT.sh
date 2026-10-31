#!/usr/bin/env bash
# Epoch time of TempFlash, Wiki-Talk with TGAT (settings of train_WIKITALK_TGAT.sh): 3 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_WIKITALK_TGAT.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data WIKITALK --config ../config/TGAT_WIKITALK.yml --batch_size 2000 --lr 0.0001 --dropout 0.1 --att_dropout 0.1 --epochs 3 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/WIKITALK_TGAT.json | tee ../logs/timing/WIKITALK_TGAT.log
