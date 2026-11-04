#!/bin/bash
# usage: watch_run2.sh <run_name> <episode>
# Prunes old state_*.safetensors, evaluates every checkpoint >=70M (one evaluation at a time on this host), and does a
# final pass after training exits. Writes TRAIN_EXITED (with last checkpoint step) and EVAL_DONE into the run dir.
NAME=$1; EP=$2; RUN=/root/runs/$NAME; M=/root/v2d/NVIDIA_V2D_Egocentric; PY=/root/venv-flash-lead/bin/python
EVAL=$M/reports/policy_targeted_2026-10-28/evaluate_all70m.py
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
cd $M
ev() { flock /root/eval.lock $PY $EVAL --episode $EP --run $RUN >> $RUN/checkpoint_evaluations.log 2>&1; }
has70() { ls $RUN/train/policy_*.safetensors 2>/dev/null | sed 's/.*policy_//; s/.safetensors//' | awk '$1+0>=70000000{f=1} END{exit !f}'; }
while pgrep -f "logging.run_name=$NAME" > /dev/null; do
  ls -1t $RUN/train/state_*.safetensors 2>/dev/null | tail -n +3 | xargs -r rm -f
  sleep 900
  if has70 && pgrep -f "logging.run_name=$NAME" > /dev/null; then ev; fi
done
LAST=$(ls $RUN/train/policy_*.safetensors 2>/dev/null | sed 's/.*policy_//; s/.safetensors//' | sort -n | tail -1)
echo "$(date -u +%FT%TZ) last_checkpoint=$LAST" > $RUN/TRAIN_EXITED
if has70; then ev; fi
date -u +%FT%TZ > $RUN/EVAL_DONE
