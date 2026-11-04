#!/bin/bash
# Waits for ep33_spinroll_kenf16_s202 to exit and for 31 GB free, then launches ep28_holdold_kenf16_s203 (fork 2's order,
# 13:40 UTC): old pot + per-frame hold + COM fix, Ken termination, 4M float16 replay, default njmax. Logs to /root/launch_after.log.
S=/root/v2d/lead_v3/handspin28oldr2com/scenes/ep28_hold
N=ep28_holdold_kenf16_s203
X="termination.object_position.threshold=0.1 termination.object_orientation.threshold=0.5 training.replay.capacity=4194304 training.replay.observation_storage_dtype=float16"
while pgrep -f "run_name=ep33_spinroll_kenf16_s202" > /dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) kenf16 s202 exited" >> /root/launch_after.log
for i in $(seq 1 60); do F=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); [ "$F" -ge 31000 ] && break; sleep 30; done
echo "$(date -u +%FT%TZ) free_mb=$F launching $N" >> /root/launch_after.log
bash /root/launch_scene.sh $N $S 203 "$X" >> /root/launch_after.log 2>&1
setsid nohup /root/watch_run2.sh $N 28 > /root/runs/$N/watch.log 2>&1 < /dev/null &
echo "$(date -u +%FT%TZ) watcher started" >> /root/launch_after.log
