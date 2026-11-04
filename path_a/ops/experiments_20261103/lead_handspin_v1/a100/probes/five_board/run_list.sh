#!/bin/bash
# Runs the review session list one file at a time. Retries only a preflight stop (nothing sent), and retries a failed board alone after checking it holds no copy.
F=/root/probes/five_board; KIT=/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit; PY=/root/venv-kit/bin/python
for n in "$@"; do
  echo "=== $n $(date -u +%T)"
  for i in 1 2 3 4 5 6; do
    $F/submit_list.sh $n > $F/log_$n.try$i.log 2>&1; rc=$?
    tail -9 $F/log_$n.try$i.log | cut -c1-170
    echo "rc=$rc try=$i"
    grep -q "Preflight failed" $F/log_$n.try$i.log || break
    sleep 25
  done
  /root/venv-cpu/bin/python $F/retry_failed.py $n
done
echo LIST_DONE
