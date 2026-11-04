#!/bin/bash
# usage: make_final.sh <full 40-character commit sha> [EP=FILE ...]; writes a new dir under out/ with the current base citing that commit on every row, and prints its sha. Uploads nothing.
set -u
SHA=$1; shift
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "need a full 40-character commit sha"; exit 2; }
cd /mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe
N=final_${SHA:0:8}_$(date -u +%H%M%S); mkdir -p out/$N
S="--swap 1=base_t2.parquet --swap 3=base_t5.parquet --swap 5=base_t8.parquet --swap 8=base_t6.parquet"
for x in "$@"; do S="$S --swap $x"; done
venv-kit/bin/python merge_swaps.py --base base_bob1.parquet $S --out out/$N/$N.parquet --code-commit https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/$SHA > out/$N/merge.log 2>&1 || { tail -3 out/$N/merge.log; exit 1; }
sha256sum out/$N/$N.parquet | cut -c1-64; echo $N
