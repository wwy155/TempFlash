#!/usr/bin/env bash
# Paper default configuration: LastFM / TGN, batch size 1000, lr 1e-4,
# dropout 0.2, five epochs, seeds 0, 1, 2.
# Usage: bash scripts/train_default.sh [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
GPU=${1:-0}
mkdir -p ../runs/acc ../logs/acc
for SEED in 0 1 2; do
  python train.py --data LASTFM --config ../config/TGN_LASTFM.yml --batch_size 1000 --lr 0.0001 --dropout 0.2 --att_dropout 0.2 --epochs 5 --seed "$SEED" --val_from 3 --gpu "$GPU" \
    --out_json "../runs/acc/LASTFM_TGN_s${SEED}.json" | tee "../logs/acc/LASTFM_TGN_s${SEED}.log"
done
