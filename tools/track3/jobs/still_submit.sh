#!/bin/bash
# Still-object safety net for the scored episodes: stages 2-3 (--merge --still-object, levelled),
# then training + official evaluator, chained with afterok. Usage: still_submit.sh 1 3 5 ...
# A still scene only asks the hands to hold their first pose: the object stays within ~1-3 mm from the
# first 10M-step checkpoint on (ep37, 2026-10-15), so 20M steps instead of the 250M recipe. njmax 1024
# peaks at 45.5 GB, so a 48 GB A6000 is enough (46 GB L40S is not).
set -euo pipefail
J=/data/user_data/$USER/v2d/jobs
OUT=/data/user_data/$USER/v2d/recon_data/outputs/load_retarget
EXTRA="task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3"
for EP in "$@"; do
  EPD=$(printf "episode_%06d" "$EP")
  PQ=$OUT/$EPD/merged_still/assets/human_motion_data/ego_recon/processed/sequence_id=ep${EP}_merged_still/robot_name=sharpa_wave
  LR=$(sbatch --parsable -J lr-ep${EP}-still "$J/load_retarget.sbatch" --episode "$EP" --merge --still-object --skip-video \
       --sequence-id ep${EP}_merged_still --out "$OUT/$EPD/merged_still")
  TR=$(sbatch --parsable -J t-ep${EP}_still20M --dependency=afterok:$LR --constraint="A6000|VRAM_80GB|VRAM_96GB|VRAM_141GB" \
       --export=ALL,NAME=ep${EP}_still20M,EP=$EP,STEPS=20000768,PARQUET="$PQ",EXTRA="$EXTRA" "$J/train_diag.sbatch")
  echo "ep$EP load_retarget=$LR train=$TR"
done
