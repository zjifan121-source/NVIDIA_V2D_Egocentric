#!/bin/bash
# Still-object safety net for the scored episodes: stages 2-3 (--merge --still-object, levelled),
# then training + official evaluator, chained with afterok. Usage: still_submit.sh 1 3 5 ...
#
# Training: NVIDIA's helper-force (VOC) curriculum compressed from 250M to 60M steps, so the policy
# still spends its last ~30M steps unassisted, as the evaluator runs it. 10M/20M-step policies never
# see VOC 0 and knock the objects away in evaluation (ep33 10M: 63 cm, 75 deg). njmax 1024 peaks at
# 45.5 GB: a 48 GB A6000 is enough, a 46 GB L40S is not. babel-v9-20 fails cuInit (2026-10-16).
#
# ep1 (pot + lid): support_recon's phantom filter drops the table disk under the pot because the
# lid rests on it, and FlashCHORD refuses an empty support stage. restore_support_ep1.sh writes the
# disk support_recon computed back (run it after stages 2-3, before training).
set -euo pipefail
J=/data/user_data/$USER/v2d/jobs
OUT=/data/user_data/$USER/v2d/recon_data/outputs/load_retarget
STEPS=60006400
export EXTRA="task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 curriculum.thresholds=[4718592,8258560,11797504,15336448,18876416,22415360,25954304,29494272,33033216,48006144,60006400]"
for EP in "$@"; do
  EPD=$(printf "episode_%06d" "$EP")
  PQ=$OUT/$EPD/merged_still/assets/human_motion_data/ego_recon/processed/sequence_id=ep${EP}_merged_still/robot_name=sharpa_wave
  LR=$(sbatch --parsable -J lr-ep${EP}-still "$J/load_retarget.sbatch" --episode "$EP" --merge --still-object --skip-video \
       --sequence-id ep${EP}_merged_still --out "$OUT/$EPD/merged_still")
  # EXTRA holds commas, so it travels in the environment (--export=ALL), not in the --export list.
  TR=$(sbatch --parsable -J t-ep${EP}_still60M --dependency=afterok:$LR --exclude=babel-v9-20 \
       --constraint="A6000|VRAM_80GB|VRAM_96GB|VRAM_141GB" --time=01:30:00 \
       --export=ALL,NAME=ep${EP}_still60M,EP=$EP,STEPS=$STEPS,PARQUET="$PQ" "$J/train_diag.sbatch")
  echo "ep$EP load_retarget=$LR train=$TR"
done
