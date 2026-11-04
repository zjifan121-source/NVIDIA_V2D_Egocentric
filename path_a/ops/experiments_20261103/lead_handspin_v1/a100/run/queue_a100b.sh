#!/bin/bash
# A100 slot queue. /root/queue_a100.txt lines: <run_name> <scene_dir> <seed> <episode> [extra hydra overrides...]
# /root/MAX_RUNS_A100 caps trainings on this host (all sessions counted). No launches after the cutoff: a 100M run at
# four concurrent takes ~4.5-6 h, and final rollouts are due 18:00 UTC. Create /root/QUEUE_STOP to end early.
cd /root; touch queue_a100.txt
CUTOFF=$(date -u -d '2026-11-04 13:30:00' +%s)
while [ ! -f /root/QUEUE_STOP ]; do
  if [ "$(date -u +%s)" -ge "$CUTOFF" ]; then echo "$(date -u +%FT%TZ) CUTOFF reached; remaining lines not launched" >> queue_a100.log; exit 0; fi
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
