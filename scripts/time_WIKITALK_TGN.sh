#!/usr/bin/env bash
# Epoch time of TempFlash, Wiki-Talk with TGN (settings of train_WIKITALK_TGN.sh): 3 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_WIKITALK_TGN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data WIKITALK --config ../config/TGN_WIKITALK.yml --batch_size 600 --lr 0.0001 --dropout 0.2 --att_dropout 0.2 --epochs 3 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/WIKITALK_TGN.json | tee ../logs/timing/WIKITALK_TGN.log
