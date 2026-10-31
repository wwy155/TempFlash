#!/usr/bin/env bash
# All twelve workloads of the main table (accuracy, then timing) on one GPU.
set -euo pipefail
GPU=${1:-0}
cd "$(dirname "$0")"
bash train_LASTFM_TGN.sh $GPU
bash train_LASTFM_TGAT.sh $GPU
bash train_LASTFM_APAN.sh $GPU
bash train_WIKITALK_TGN.sh $GPU
bash train_WIKITALK_TGAT.sh $GPU
bash train_WIKITALK_APAN.sh $GPU
bash train_STACKOVERFLOW_TGN.sh $GPU
bash train_STACKOVERFLOW_TGAT.sh $GPU
bash train_STACKOVERFLOW_APAN.sh $GPU
bash train_GDELT_TGN.sh $GPU
bash train_GDELT_TGAT.sh $GPU
bash train_GDELT_APAN.sh $GPU
bash time_LASTFM_TGN.sh $GPU
bash time_LASTFM_TGAT.sh $GPU
bash time_LASTFM_APAN.sh $GPU
bash time_WIKITALK_TGN.sh $GPU
bash time_WIKITALK_TGAT.sh $GPU
bash time_WIKITALK_APAN.sh $GPU
bash time_STACKOVERFLOW_TGN.sh $GPU
bash time_STACKOVERFLOW_TGAT.sh $GPU
bash time_STACKOVERFLOW_APAN.sh $GPU
bash time_GDELT_TGN.sh $GPU
bash time_GDELT_TGAT.sh $GPU
bash time_GDELT_APAN.sh $GPU
