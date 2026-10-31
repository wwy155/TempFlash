#!/usr/bin/env bash
# TempFlash, LastFM with TGAT: nominal batch size 5000, learning rate 0.0002, dropout 0.1, five epochs, seeds 0, 1, 2.
# Test AP of the better validation epoch among epochs 3 and 4 (counting from zero), as in ETC's protocol.
# usage: bash scripts/train_LASTFM_TGAT.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/acc ../logs/acc
for SEED in 0 1 2; do
  python train.py --data LASTFM --config ../config/TGAT_LASTFM.yml --batch_size 5000 --lr 0.0002 --dropout 0.1 --att_dropout 0.1 --epochs 5 --seed $SEED --val_from 3 --gpu $GPU \
    --out_json ../runs/acc/LASTFM_TGAT_s$SEED.json | tee ../logs/acc/LASTFM_TGAT_s$SEED.log
done
