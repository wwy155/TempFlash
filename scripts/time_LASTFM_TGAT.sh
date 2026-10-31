#!/usr/bin/env bash
# Epoch time of TempFlash, LastFM with TGAT (settings of train_LASTFM_TGAT.sh): 3 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_LASTFM_TGAT.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data LASTFM --config ../config/TGAT_LASTFM.yml --batch_size 5000 --lr 0.0002 --dropout 0.1 --att_dropout 0.1 --epochs 3 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/LASTFM_TGAT.json | tee ../logs/timing/LASTFM_TGAT.log
