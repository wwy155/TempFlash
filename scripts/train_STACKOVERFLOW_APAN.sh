#!/usr/bin/env bash
# TempFlash, Stack-Overflow with APAN: nominal batch size 2000, learning rate 0.0001, dropout 0.1, five epochs, seeds 0, 1, 2.
# Test AP of the better validation epoch among epochs 3 and 4 (counting from zero), as in ETC's protocol.
# usage: bash scripts/train_STACKOVERFLOW_APAN.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/acc ../logs/acc
for SEED in 0 1 2; do
  python train.py --data STACKOVERFLOW --config ../config/APAN_STACKOVERFLOW.yml --batch_size 2000 --lr 0.0001 --dropout 0.1 --att_dropout 0.1 --epochs 5 --seed $SEED --val_from 3 --gpu $GPU \
    --out_json ../runs/acc/STACKOVERFLOW_APAN_s$SEED.json | tee ../logs/acc/STACKOVERFLOW_APAN_s$SEED.log
done
