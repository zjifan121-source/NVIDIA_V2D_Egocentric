#!/bin/bash
# Run as sarveshp: submit queued trials (one path per line in .w3_queue.txt, in priority order) while the account has < 46 jobs.
cd /home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28
n=$(squeue -u sarveshp -h -r | wc -l)
for t in $(cat .w3_queue.txt); do
  [ -f "${t%.json}.job.json" ] && continue
  [ "$n" -ge 46 ] && break
  python3 submit_trial.py "$t" 2>&1 | tail -1; n=$((n+1))
done
squeue -u sarveshp -h -t PD -o "%i %j" | python3 w3_order.py | while read id nice; do scontrol update jobid=$id nice=$nice; done
echo "left: $(for t in $(cat .w3_queue.txt); do [ -f "${t%.json}.job.json" ] || echo $t; done | wc -l)"
