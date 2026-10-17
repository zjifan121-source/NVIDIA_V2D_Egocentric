#!/bin/bash
# Still-object safety net for the scored episodes: stages 2-3 (--merge --still-object, levelled),
# then training + official evaluator, chained with afterok. Usage: still_submit.sh 1 3 5 ...
#
# Training: NVIDIA's helper-force (VOC) curriculum compressed from 250M to 60M steps, so the policy
# still spends its last ~30M steps unassisted, as the evaluator runs it. 10M/20M-step policies never
# see VOC 0 and knock the objects away in evaluation (ep33 10M: 63 cm, 75 deg). njmax 1024 peaks at
# 45.5 GB: a 48 GB A6000 is enough, a 46 GB L40S is not. babel-v9-20 fails cuInit (2026-10-16).
#
# Two-object scenes whose table comes out wrong (ep1, 3, 6, 14, 15, 16, 20, 30) are built with --seat:
# floating objects are lowered onto the table, and support_stacked.py drops a table only under an
# object that rests on another one. Without it, ep6's pot falls 4.4 cm and tips 21 deg (AUC 0.57).
# Usage: SEAT=1 still_submit.sh 1 3 6 ... (outputs merged_still_seat).
#
# GAP=0.8 (with SEAT=1) also scales the second object's offset from the first: ours come out
# 1.03-1.45x too far apart, and 0.8 lifts public-dry-run AUC 0.315 -> 0.340. Seating now puts every
# object on the table (ep28's pot was 45 cm up, ep32's spoon 14 cm down); outputs merged_still_seat_gap0.8.
set -euo pipefail
J=/data/user_data/$USER/v2d/jobs
OUT=/data/user_data/$USER/v2d/recon_data/outputs/load_retarget
STEPS=60006400
export EXTRA="task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 curriculum.thresholds=[4718592,8258560,11797504,15336448,18876416,22415360,25954304,29494272,33033216,48006144,60006400]"
TAG=merged_still${SEAT:+_seat}${GAP:+_gap$GAP}; RUN=${SEAT:+seat}${GAP:+gap$GAP}; RUN=${RUN:-still}
for EP in "$@"; do
  EPD=$(printf "episode_%06d" "$EP")
  PQ=$OUT/$EPD/$TAG/assets/human_motion_data/ego_recon/processed/sequence_id=ep${EP}_$TAG/robot_name=sharpa_wave
  LR=$(sbatch --parsable -J lr-ep${EP}-$RUN "$J/load_retarget.sbatch" --episode "$EP" --merge --still-object ${SEAT:+--seat} ${GAP:+--gap-scale $GAP} \
       --skip-video --sequence-id ep${EP}_$TAG --out "$OUT/$EPD/$TAG")
  # EXTRA holds commas, so it travels in the environment (--export=ALL), not in the --export list.
  TR=$(sbatch --parsable -J t-ep${EP}_${RUN}60M --dependency=afterok:$LR --exclude=babel-v9-20 \
       --constraint="A6000|VRAM_80GB|VRAM_96GB|VRAM_141GB" --time=01:30:00 \
       --export=ALL,NAME=ep${EP}_${RUN}60M,EP=$EP,STEPS=$STEPS,PARQUET="$PQ" "$J/train_diag.sbatch")
  echo "ep$EP load_retarget=$LR train=$TR"
done
