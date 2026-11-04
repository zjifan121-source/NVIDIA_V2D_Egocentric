#!/bin/bash
[ -f /mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe/UPLOAD_HOLD ] && { echo "uploads on hold by the user"; exit 4; }
# usage (token on stdin): submit_base.sh <dir name under out/> <parquet name> <message> ; uploads a merged base to AUC, then the five other boards, retrying only stops where nothing was sent
[ "${ALLOW_AUC:-1}" = 1 ] || { echo "AUC upload blocked: set ALLOW_AUC=1 only for the file the user names"; exit 3; }
read -r T; export KAGGLE_API_TOKEN=$T; unset T
P=/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe; D=$P/out/$1; F=$D/$2; M=$3
cd /mnt/v2d/outputs/experiments_20261021/independent_pipeline_v1/packet/submission_kit
run(){ for i in 1 2 3 4; do $P/venv-kit/bin/python submit/v2d_submit.py $F --track track_3 -m "$M" --only $1 --wait --receipt $2 > $D/try.log 2>&1; rc=$?; tail -8 $D/try.log | cut -c1-170; echo "rc=$rc try=$i"; grep -q -E "Preflight failed|authentication failed" $D/try.log || break; sleep 20; done; }
run add_auc $D/full_auc.receipt.json
run "spider_sr maniptrans_sr rpe_cm mppe_cm cd_o_cm" $D/full_five.receipt.json
echo BASE_DONE
