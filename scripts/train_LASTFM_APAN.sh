#!/usr/bin/env bash
# TempFlash, LastFM with APAN: nominal batch size 600, learning rate 0.0001, dropout 0.1, five epochs, seeds 0, 1, 2.
# Test AP of the better validation epoch among epochs 3 and 4 (counting from zero), as in ETC's protocol.
# usage: bash scripts/train_LASTFM_APAN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/acc ../logs/acc
for SEED in 0 1 2; do
  python train.py --data LASTFM --config ../config/APAN_LASTFM.yml --batch_size 600 --lr 0.0001 --dropout 0.1 --att_dropout 0.1 --epochs 5 --seed $SEED --val_from 3 --gpu $GPU \
    --out_json ../runs/acc/LASTFM_APAN_s$SEED.json | tee ../logs/acc/LASTFM_APAN_s$SEED.log
done
