# Sourced at the start of the Track 3 GPU jobs, before they write anything. With less than
# V2D_MIN_FREE_GB (default 30) free in /data/user_data/$USER, the job puts itself back in the queue
# on hold and exits: nothing is half-written, nothing is lost. Free space, then resume the held jobs:
#   squeue -u $USER -t PD -o "%i %j %r" | grep JobHeldUser;  scontrol release <jobid,...>
# (A full quota on 2026-10-21 killed every training run mid-checkpoint.)
free_gb=$(df -BG --output=avail "/data/user_data/$USER" | tail -1 | tr -dc 0-9)
if [ "${free_gb:-0}" -lt "${V2D_MIN_FREE_GB:-30}" ]; then
  echo "== $(date) only ${free_gb} GB free in /data/user_data/$USER (< ${V2D_MIN_FREE_GB:-30} GB): holding this job"
  [ -n "${SLURM_JOB_ID:-}" ] && scontrol requeuehold "$SLURM_JOB_ID"
  exit 0
fi
