#!/bin/bash
# Evaluate every checkpoint from 70M as soon as it appears, for all runs under $RUNS on this host. Create STOP_EVAL next to this script to end.
HERE=$(cd "$(dirname "$0")" && pwd); source $HERE/env.sh
export KIT FC XLA_PYTHON_CLIENT_PREALLOCATE=false
while [ ! -f $HERE/STOP_EVAL ]; do
  for r in $RUNS/ep*; do
    [ -d $r/train ] || continue
    ep=$(basename $r | sed -E 's/^ep([0-9]+).*/\1/')
    want=0
    for p in $r/train/policy_*.safetensors; do
      [ -f "$p" ] || continue
      s=$(basename $p .safetensors); s=${s#policy_}
      [ "$s" -ge 70000000 ] || continue
      [ -f $r/early_$s/eval.done ] || want=1
    done
    [ $want = 1 ] && { cd $FC; $PY $HERE/evaluate_checkpoints.py --episode $ep --run $r >> $r/checkpoint_evaluations.log 2>&1; }
  done
  sleep 60
done
