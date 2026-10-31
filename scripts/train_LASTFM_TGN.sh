#!/usr/bin/env bash
# TempFlash, LastFM with TGN: nominal batch size 1000, learning rate 0.0002, dropout 0.2, five epochs, seeds 0, 1, 2.
# Test AP of the better validation epoch among epochs 3 and 4 (counting from zero), as in ETC's protocol.
# usage: bash scripts/train_LASTFM_TGN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/acc ../logs/acc
for SEED in 0 1 2; do
  python train.py --data LASTFM --config ../config/TGN_LASTFM.yml --batch_size 1000 --lr 0.0002 --dropout 0.2 --att_dropout 0.2 --epochs 5 --seed $SEED --val_from 3 --gpu $GPU \
    --out_json ../runs/acc/LASTFM_TGN_s$SEED.json | tee ../logs/acc/LASTFM_TGN_s$SEED.log
done
