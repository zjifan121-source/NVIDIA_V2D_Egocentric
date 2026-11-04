#!/bin/bash
# A100 slot queue with a GPU memory gate. /root/queue_a100.txt lines: <run_name> <scene_dir> <seed> <episode> [hydra overrides...]
# Ken's recipe holds 4M float32 replay on the GPU: about 38.4 GiB per run against 17.5 for DEF, so it launches only with
# 43 GB free (19 GB for other lines). /root/MAX_RUNS_A100 still caps the count across all sessions. No launches after the
# cutoff. Create /root/QUEUE_STOP to end early.
cd /root; touch queue_a100.txt
CUTOFF=$(date -u -d '2026-11-04 13:30:00' +%s)
while [ ! -f /root/QUEUE_STOP ]; do
  if [ "$(date -u +%s)" -ge "$CUTOFF" ]; then echo "$(date -u +%FT%TZ) CUTOFF reached; remaining lines not launched" >> queue_a100.log; exit 0; fi
  MAX=$(cat /root/MAX_RUNS_A100 2>/dev/null || echo 3)
  N=$(pgrep -fc "python scripts/train_flash_sac.py")
  LINE=$(head -1 queue_a100.txt)
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  NEED=19000; case "$LINE" in *replay.capacity=4194304*) NEED=43000;; esac
  if [ "$N" -ge "$MAX" ] || [ -z "$LINE" ] || [ "$FREE" -lt "$NEED" ]; then sleep 120; continue; fi
  sed -i 1d queue_a100.txt
  set -- $LINE; NAME=$1; SCENE=$2; SEED=$3; EP=$4; shift 4
  echo "$(date -u +%FT%TZ) START $NAME seed=$SEED ep=$EP free_mb=$FREE extra=[$*]" >> queue_a100.log
  bash /root/launch_scene.sh $NAME $SCENE $SEED "$*" >> queue_a100.log 2>&1
  setsid nohup /root/watch_run2.sh $NAME $EP > /root/runs/$NAME/watch.log 2>&1 < /dev/null &
  sleep 540   # a new run reaches its full GPU footprint in about nine minutes
done
