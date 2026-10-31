#!/usr/bin/env bash
# TempFlash on the twelve additional graphs with the configuration of the paper's appendix: nominal batch
# size 600 for TGN, TGAT and APAN, learning rate 1e-4, the dropout of the YAML configuration, five epochs,
# seeds 0, 1, 2.  usage: bash scripts/train_extra.sh <DATA> <MODEL> [GPU]
set -euo pipefail
cd "$(dirname "$0")/../tempflash"
D=$1; M=$2; GPU=${3:-0}
mkdir -p ../runs/acc ../logs/acc
for SEED in 0 1 2; do
  python train.py --data $D --config ../config/${M}_${D}.yml --batch_size 600 --lr 1e-4 --epochs 5 \
    --seed $SEED --val_from 3 --gpu $GPU --out_json ../runs/acc/${D}_${M}_s$SEED.json \
    | tee ../logs/acc/${D}_${M}_s$SEED.log
done
