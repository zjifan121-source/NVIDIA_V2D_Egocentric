#!/bin/bash
# A100 queue for the Ken-termination arms. /root/queue_a100.txt lines: <base_name> <scene_dir> <seed> <episode> <F16|F16FB>
# All use termination 0.1 m / 0.5 rad with float16 replay; none is Ken's exact recipe.
# F16 = replay 4194304, about 27.3 GiB per run, launches with 31 GB free.
# F16FB = as F16 when 29.5 GB is free, otherwise replay 3145728, about 24.3 GiB, with 25.5 GB free. Three 4M runs total 82.2 GB
# against 81.9 GB on this card, so the third arm needs the smaller replay to start at all.
# Run names carry _kenf16_ or _kenf16c3m_. No launches after the cutoff. Create /root/QUEUE_STOP to end early.
cd /root; touch queue_a100.txt
CUTOFF=$(date -u -d '2026-11-04 13:30:00' +%s)
while [ ! -f /root/QUEUE_STOP ]; do
  if [ "$(date -u +%s)" -ge "$CUTOFF" ]; then echo "$(date -u +%FT%TZ) CUTOFF reached; remaining lines not launched" >> queue_a100.log; exit 0; fi
  LINE=$(head -1 queue_a100.txt); [ -z "$LINE" ] && { sleep 120; continue; }
  MAX=$(cat /root/MAX_RUNS_A100 2>/dev/null || echo 3)
  N=$(pgrep -fc "python scripts/train_flash_sac.py")
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  set -- $LINE; BASE=$1; SCENE=$2; SEED=$3; EP=$4; RECIPE=$5
  TAG=""
  if [ "$RECIPE" = "F16" ] && [ "$FREE" -ge 31000 ]; then TAG=kenf16; CAP=4194304
  elif [ "$RECIPE" = "F16FB" ] && [ "$FREE" -ge 29500 ]; then TAG=kenf16; CAP=4194304
  elif [ "$RECIPE" = "F16FB" ] && [ "$FREE" -ge 25500 ]; then TAG=kenf16c3m; CAP=3145728; fi
  if [ "$N" -ge "$MAX" ] || [ -z "$TAG" ]; then sleep 120; continue; fi
  NAME=${BASE}_${TAG}_s${SEED}
  EXTRA="termination.object_position.threshold=0.1 termination.object_orientation.threshold=0.5 training.replay.capacity=$CAP training.replay.observation_storage_dtype=float16"
  sed -i 1d queue_a100.txt
  echo "$(date -u +%FT%TZ) START $NAME seed=$SEED ep=$EP free_mb=$FREE extra=[$EXTRA]" >> queue_a100.log
  bash /root/launch_scene.sh $NAME $SCENE $SEED "$EXTRA" >> queue_a100.log 2>&1
  setsid nohup /root/watch_run2.sh $NAME $EP > /root/runs/$NAME/watch.log 2>&1 < /dev/null &
  sleep 540   # a new run reaches its full GPU footprint in about nine minutes
done
