#!/usr/bin/env bash
# On the L40S, in tmux: phase B (V3_BUILD_SPEC.md §9): the phase-1 chain on all 40 episodes.
#
#     ssh -t l40s "tmux new -s phaseb 'bash /root/v2d/NVIDIA_V2D_Egocentric/path_b/vm/phase_b.sh'"
#     EPISODES="40" bash path_b/vm/phase_b.sh                 # some episodes only (the smoke test)
#     EPISODES="12" FROM=s3b_track bash path_b/vm/phase_b.sh  # after a fix: that stage onward, forced
#
# 1. run folder path_b/data/outputs/$RUN (default phaseb, a fixed name: fixes pushed mid-run resume in
#    place; every report records its commit); the public episodes' P0-P2 hard-linked from run 4c2eac31
# 2. run.py run --split public, every stage p0_ingest ... s6_select (a stage with a non-ERROR report is kept)
# 3. run.py waterfall --split public
# 4. run.py run --split evaluation, the same stages (it starts as soon as the public half ends)
# No phase A2 stage and no phase 2 (PATHB_PHASE2 unset). Code reaches the VM only through the bare repo.
set -uo pipefail
cd /root/v2d/NVIDIA_V2D_Egocentric
RUN=${RUN:-phaseb}
OUT=path_b/data/outputs/$RUN
SRC=path_b/data/outputs/4c2eac31
STAGES=(p0_ingest p1_calibrate p2_depth p1b_masks p1c_camera p1d_hands p3_scale_bridge p4_export
        s3a_mesh s3b_track s3c_fuse s4a_phases s4b_reference s4c_retarget s5_task s6_train s6_eval s6_select)
LOG=/root/v2d/logs/phase_b_$RUN.log
mkdir -p /root/v2d/logs "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "== phase B $RUN ($(date -u +%FT%TZ), $(git rev-parse --short=8 HEAD)); log $LOG"

for d in "$SRC"/episode_*; do  # the public P0-P2 of run 4c2eac31 (restored by setup.sh with frames)
  e=$(basename "$d")
  mkdir -p "$OUT/$e"
  for s in p0_ingest p1_calibrate p2_depth; do
    if [ -d "$d/$s" ] && [ ! -e "$OUT/$e/$s" ]; then cp -al "$d/$s" "$OUT/$e/$s"; fi
  done
done

set -a; . path_b/.env; set +a  # HF_TOKEN: the gated SAM 3.1 / SAM 3D downloads
export RETARGET_VENV=${RETARGET_VENV:-/root/v2d/venvs/retarget}
RUNPY=(uv run --project path_b python path_b/run.py)

if [ -n "${EPISODES:-}" ]; then
  stages=("${STAGES[@]}") force=()
  if [ -n "${FROM:-}" ]; then
    for i in "${!STAGES[@]}"; do [ "${STAGES[$i]}" = "$FROM" ] && stages=("${STAGES[@]:$i}") && force=(--force); done
  fi
  # shellcheck disable=SC2086
  "${RUNPY[@]}" run --episodes $EPISODES --out "$OUT" --stages "${stages[@]}" "${force[@]}"
  echo "== episodes $EPISODES done ($(date -u +%FT%TZ))"
  exit 0
fi

"${RUNPY[@]}" run --split public --out "$OUT" --stages "${STAGES[@]}"
"${RUNPY[@]}" waterfall --split public --out "$OUT"
echo "== public half done ($(date -u +%FT%TZ)); evaluation half starts"
"${RUNPY[@]}" run --split evaluation --out "$OUT" --stages "${STAGES[@]}"
echo "== phase B done ($(date -u +%FT%TZ))"
