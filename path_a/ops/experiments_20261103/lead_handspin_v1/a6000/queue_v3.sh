#!/bin/bash
# Host-side training queue with slots. MAX_RUNS holds how many trainings may share this GPU (counts every
# train_flash_sac.py on the host, ours or not). Each queue.txt line = arguments of run_train_v3.sh. Create STOP to end.
HERE=$(cd "$(dirname "$0")" && pwd); cd $HERE; touch queue.txt
while [ ! -f STOP ]; do
  MAX=$(cat MAX_RUNS 2>/dev/null || echo 1)
  N=$(pgrep -fc "scripts/train_flash_sac.py")
  if [ "$N" -ge "$MAX" ]; then sleep 60; continue; fi
  LINE=$(head -1 queue.txt)
  if [ -z "$LINE" ]; then sleep 120; continue; fi
  sed -i 1d queue.txt
  echo "$(date -u +%FT%TZ) START $LINE" >> queue.log
  # shellcheck disable=SC2086
  ( ./run_train_v3.sh $LINE >> queue_runs.log 2>&1; echo "$(date -u +%FT%TZ) END rc=$? $LINE" >> queue.log ) &
  sleep 150
done
