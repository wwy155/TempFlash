#!/usr/bin/env bash
# TempFlash, Stack-Overflow with TGN: nominal batch size 2000, learning rate 0.0002, dropout 0.2, five epochs, seeds 0, 1, 2.
# Test AP of the better validation epoch among epochs 3 and 4 (counting from zero), as in ETC's protocol.
# usage: bash scripts/train_STACKOVERFLOW_TGN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/acc ../logs/acc
for SEED in 0 1 2; do
  python train.py --data STACKOVERFLOW --config ../config/TGN_STACKOVERFLOW.yml --batch_size 2000 --lr 0.0002 --dropout 0.2 --att_dropout 0.2 --epochs 5 --seed $SEED --val_from 3 --gpu $GPU \
    --out_json ../runs/acc/STACKOVERFLOW_TGN_s$SEED.json | tee ../logs/acc/STACKOVERFLOW_TGN_s$SEED.log
done
