#!/bin/bash
# Pulls every evaluated checkpoint of the A6000 runs into /root/probes/incoming/<run>/early_<step>/results/ for the review session. Read-only on the A6000. One pass every 45 s; a line per new file in sync.log.
D=/root/probes/incoming; SRC=ubuntu@<host>:/mnt/v2d/outputs/experiments_20261103/lead_v3_train/runs/
exec 9>$D/.sync.lock; flock -n 9 || { echo "already running"; exit 0; }
while [ ! -f $D/SYNC_STOP ]; do
  rsync -az --timeout=90 --prune-empty-dirs --itemize-changes \
    --include="ep*/" --include="ep*/checkpoint_selection.json" --include="ep*/early_*/" --include="ep*/early_*/checkpoints.json" --include="ep*/early_*/eval.done" \
    --include="ep*/early_*/results/" --include="ep*/early_*/results/episode_*.json" --include="ep*/early_*/results/episode_*.parquet" --exclude="*" \
    -e "ssh -i /root/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=20" "$SRC" "$D/" 2>>$D/sync.err | grep -E "^>f" | sed "s/^/$(date -u +%FT%TZ) /" >> $D/sync.log
  date -u +%FT%TZ > $D/LAST_SYNC
  sleep 45
done
