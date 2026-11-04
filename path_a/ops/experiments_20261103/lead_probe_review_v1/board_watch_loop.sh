#!/bin/bash
# Read-only Kaggle board watch; ends by itself at the freeze (2026-11-04 22:00 UTC). No upload calls.
while [ "$(date -u +%Y%m%d%H%M)" -lt 202611042200 ]; do
  timeout 420 nice -n 10 /root/venv-kit/bin/python /root/v2d/tools_lead/board_watch.py >> /root/probes/incumbents/watch.log 2>&1
  sleep 180
done
