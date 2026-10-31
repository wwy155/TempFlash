#!/usr/bin/env bash
# Epoch time of TempFlash, LastFM with TGN (settings of train_LASTFM_TGN.sh): 3 training epochs without evaluation;
# the per-epoch times are written to the JSON file.
# usage: bash scripts/time_LASTFM_TGN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/timing ../logs/timing
python train.py --data LASTFM --config ../config/TGN_LASTFM.yml --batch_size 1000 --lr 0.0002 --dropout 0.2 --att_dropout 0.2 --epochs 3 --seed 0 --skip_eval --gpu $GPU \
  --out_json ../runs/timing/LASTFM_TGN.json | tee ../logs/timing/LASTFM_TGN.log
