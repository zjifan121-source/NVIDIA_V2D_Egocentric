#!/bin/bash
# A100 slot queue. /root/queue_a100.txt lines: <run_name> <scene_dir> <seed> <episode> [extra hydra overrides...]
# /root/MAX_RUNS_A100 = trainings allowed at once (counts every train_flash_sac.py on the host). Create /root/QUEUE_STOP to end.
cd /root; touch queue_a100.txt
while [ ! -f /root/QUEUE_STOP ]; do
  MAX=$(cat /root/MAX_RUNS_A100 2>/dev/null || echo 3)
  N=$(pgrep -fc "python scripts/train_flash_sac.py")
  LINE=$(head -1 queue_a100.txt)
  if [ "$N" -ge "$MAX" ] || [ -z "$LINE" ]; then sleep 120; continue; fi
  sed -i 1d queue_a100.txt
  set -- $LINE; NAME=$1; SCENE=$2; SEED=$3; EP=$4; shift 4
  echo "$(date -u +%FT%TZ) START $NAME seed=$SEED ep=$EP extra=[$*]" >> queue_a100.log
  bash /root/launch_scene.sh $NAME $SCENE $SEED "$*" >> queue_a100.log 2>&1
  setsid nohup /root/watch_run2.sh $NAME $EP > /root/runs/$NAME/watch.log 2>&1 < /dev/null &
  sleep 200
done
