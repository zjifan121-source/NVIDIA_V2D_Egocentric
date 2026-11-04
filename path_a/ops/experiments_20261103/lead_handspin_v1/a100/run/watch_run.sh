#!/bin/bash
# usage: watch_run.sh <run_name> <episode>  — evaluates checkpoints >=70M every 30 min while training, then a final pass.
NAME=$1; EP=$2; RUN=/root/runs/$NAME; M=/root/v2d/NVIDIA_V2D_Egocentric; PY=/root/venv-flash-lead/bin/python
cd $M
while pgrep -f "logging.run_name=$NAME" > /dev/null; do
  ls -1t $RUN/train/state_*.safetensors 2>/dev/null | tail -n +3 | xargs -r rm -f
  sleep 1800
  if ls $RUN/train/policy_*.safetensors 2>/dev/null | awk -F_ '{gsub(".safetensors","",$NF); if ($NF+0>=70000000) f=1} END{exit !f}'; then
    pgrep -f "logging.run_name=$NAME" > /dev/null && $PY reports/policy_targeted_2026-10-28/evaluate_checkpoints.py --episode $EP --run $RUN >> $RUN/checkpoint_evaluations.log 2>&1
  fi
done
date -u +%FT%TZ > $RUN/TRAIN_EXITED
$PY reports/policy_targeted_2026-10-28/evaluate_checkpoints.py --episode $EP --run $RUN >> $RUN/checkpoint_evaluations.log 2>&1
date -u +%FT%TZ > $RUN/EVAL_DONE
