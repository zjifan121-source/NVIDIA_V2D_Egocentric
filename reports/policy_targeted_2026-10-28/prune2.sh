#!/bin/bash
python3 prune_today.py
df -h /data/user_data/edmondwa | tail -1
echo DONE
