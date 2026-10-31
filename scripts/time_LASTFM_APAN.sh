#!/usr/bin/env bash
# Epoch time of TempFlash, LastFM with APAN (settings of train_LASTFM_APAN.sh): 3 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_LASTFM_APAN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data LASTFM --config ../config/APAN_LASTFM.yml --batch_size 600 --lr 0.0001 --dropout 0.1 --att_dropout 0.1 --epochs 3 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/LASTFM_APAN.json | tee ../logs/timing/LASTFM_APAN.log
