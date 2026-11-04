#!/bin/bash
python3 prune_runs.py
sleep 60
df -h /data/user_data/edmondwa | tail -1
echo DONE
