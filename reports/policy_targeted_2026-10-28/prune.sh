#!/bin/bash
df -h /data/user_data/edmondwa | tail -1
python3 prune_runs.py
du -sh /data/user_data/edmondwa/v2d/stereo_test/seed && rm -rf /data/user_data/edmondwa/v2d/stereo_test/seed && echo removed stereo_test/seed
df -h /data/user_data/edmondwa | tail -1
