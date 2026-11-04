#!/bin/bash
[ -f /mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/UPLOAD_HOLD ] && { echo "uploads on hold by the user"; exit 4; }
# usage (token on stdin): submit_probe.sh <name> <message> [--dry-run] <metrics...>; uploads out/<name>/<name>_on_best.parquet with the kit uploader and waits for scores
case " $* " in *" add_auc "*) [ "${ALLOW_AUC:-1}" = 1 ] || { echo "AUC upload blocked: set ALLOW_AUC=1 only for the file the user names"; exit 3; };; esac
read -r T; export KAGGLE_API_TOKEN=$T
N=$1; M=$2; shift 2
D=/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/out/$N; F=$D/${N}_on_best.parquet; [ -f $F ] || F=$D/$N.parquet
X=--wait; [ "$1" = --dry-run ] && { X=--dry-run; shift; }
cd /mnt/v2d/outputs/experiments_20261021/independent_pipeline_v1/packet/submission_kit
/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/venv-kit/bin/python submit/v2d_submit.py $F --track track_3 -m "$M" --only "$@" $X --receipt $D/receipt_$(date +%H%M%S).json 2>&1 | tail -9 | cut -c1-180
