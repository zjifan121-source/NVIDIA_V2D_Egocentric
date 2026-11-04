#!/bin/bash
# Prune non-best learner states of finished runs, sparing every queued run (squeue + the w3 queue file).
cd /home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28
{ squeue -u sarveshp,edmondwa -h -o %j; for t in $(cat .w3_queue.txt); do python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"name\"])" $t; done; } > .queued_names_now.txt
python3 prune_keep.py .queued_names_now.txt
df -BG --output=avail /data/user_data/edmondwa | tail -1
