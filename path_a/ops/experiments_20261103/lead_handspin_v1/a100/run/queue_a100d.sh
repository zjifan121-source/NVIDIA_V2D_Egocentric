#!/bin/bash
# A100 queue for the Ken-recipe arms. /root/queue_a100.txt lines: <base_name> <scene_dir> <seed> <episode> <KEN|KENAUTO>
# KEN = Ken's exact recipe (termination 0.1/0.5, replay 4194304 float32), about 38.4 GiB per run, needs 43 GB free.
# KENAUTO = exact if it fits now, otherwise the float16-replay variant (same termination and capacity), about 27 GiB,
# needs 31 GB free. Run names carry _ken_ or _kenf16_ so the two are never confused. No launches after the cutoff.
# /root/MAX_RUNS_A100 caps the count across all sessions. Create /root/QUEUE_STOP to end early.
cd /root; touch queue_a100.txt
CUTOFF=$(date -u -d '2026-11-04 13:30:00' +%s)
while [ ! -f /root/QUEUE_STOP ]; do
  if [ "$(date -u +%s)" -ge "$CUTOFF" ]; then echo "$(date -u +%FT%TZ) CUTOFF reached; remaining lines not launched" >> queue_a100.log; exit 0; fi
  LINE=$(head -1 queue_a100.txt); [ -z "$LINE" ] && { sleep 120; continue; }
  MAX=$(cat /root/MAX_RUNS_A100 2>/dev/null || echo 3)
  N=$(pgrep -fc "python scripts/train_flash_sac.py")
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  set -- $LINE; BASE=$1; SCENE=$2; SEED=$3; EP=$4; RECIPE=$5
  PICK=""
  if [ "$FREE" -ge 43000 ]; then PICK=ken; DT=float32
  elif [ "$RECIPE" = "KENAUTO" ] && [ "$FREE" -ge 31000 ]; then PICK=kenf16; DT=float16; fi
  if [ "$N" -ge "$MAX" ] || [ -z "$PICK" ]; then sleep 120; continue; fi
  NAME=${BASE}_${PICK}_s${SEED}
  EXTRA="termination.object_position.threshold=0.1 termination.object_orientation.threshold=0.5 training.replay.capacity=4194304 training.replay.observation_storage_dtype=$DT"
  sed -i 1d queue_a100.txt
  echo "$(date -u +%FT%TZ) START $NAME seed=$SEED ep=$EP free_mb=$FREE extra=[$EXTRA]" >> queue_a100.log
  bash /root/launch_scene.sh $NAME $SCENE $SEED "$EXTRA" >> queue_a100.log 2>&1
  setsid nohup /root/watch_run2.sh $NAME $EP > /root/runs/$NAME/watch.log 2>&1 < /dev/null &
  sleep 540   # a new run reaches its full GPU footprint in about nine minutes
done
