#!/usr/bin/env bash
# On the L40S, in tmux: one Path B run, episode by episode, each episode uploaded to Hugging Face as
# soon as it finishes and before the next starts (V4_BUILD_SPEC.md §6, §9).
#
#     ssh -t l40s "tmux new -d -s v4 'bash /root/v2d/NVIDIA_V2D_Egocentric/path_b/vm/phase_b.sh'"
#     EPISODES="41 13" bash path_b/vm/phase_b.sh               # the dev cache (vm/dev_cache.sh)
#     EPISODES="12" FROM=p1d_hands bash path_b/vm/phase_b.sh    # after a fix: that stage onward, forced
#     STAGES="s4b_reference s5_task s6_train s6_eval s6_select" bash path_b/vm/phase_b.sh  # a later window
#
# RUN       run folder path_b/data/outputs/$RUN (default v4: a fixed name, so fixes pushed mid-run resume in
#           place; every report records its commit); the public episodes' P0-P2 are hard-linked from run
#           4c2eac31 (restored by setup.sh with frames)
# STAGES    default S1 (V4 task 10): p0_ingest ... p4_export (P4 writes the hands' world exports) and
#           s3a2_fuse_scan; a stage with a non-ERROR report is kept (FROM forces that stage onward)
# EPISODES  default: the dev episodes (41, 13), then FIRST (James's priority list, e.g. "0 18 43 31 23"), then
#           the other public ones, then the evaluation ones
# FRAMES    1 (default): the upload includes each episode's per-frame archive (masks, depth, hand stages),
#           which the CPU stages on the Mac and the next instance restore; 0: results only
# UPLOAD    1 (default): upload each episode to Hugging Face; 0: none (the team repo's storage is full)
# After each episode: hf_outputs.py upload --episodes <n> (3 tries); a failed episode or upload is
# logged and the run goes on. Code reaches the VM from GitHub (git pull) or the bare repo; never edited here.
set -uo pipefail
cd /root/v2d/NVIDIA_V2D_Egocentric
RUN=${RUN:-v4}
OUT=path_b/data/outputs/$RUN
SRC=path_b/data/outputs/4c2eac31
S1="p0_ingest p1_calibrate p2_depth p1b_masks p1c_camera p1d_hands p3_scale_bridge p4_export s3a2_fuse_scan"
read -ra STAGES <<< "${STAGES:-$S1}"
LOG=/root/v2d/logs/phase_b_$RUN.log
mkdir -p /root/v2d/logs "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "== run $RUN ($(date -u +%FT%TZ), $(git rev-parse --short=8 HEAD)); stages ${STAGES[*]}; log $LOG"

for d in "$SRC"/episode_*; do  # the public P0-P2 of run 4c2eac31
  e=$(basename "$d")
  mkdir -p "$OUT/$e"
  for s in p0_ingest p1_calibrate p2_depth; do
    if [ -d "$d/$s" ] && [ ! -e "$OUT/$e/$s" ]; then cp -al "$d/$s" "$OUT/$e/$s"; fi
  done
done

set -a; . path_b/.env; set +a  # HF_TOKEN: the gated model downloads; HF_WRITE_TOKEN (if there) for uploads
export RETARGET_VENV=${RETARGET_VENV:-/root/v2d/venvs/retarget}
RUNPY=(uv run --project path_b python path_b/run.py)
if [ -z "${EPISODES:-}" ]; then
  EPISODES=$(FIRST="${FIRST:-}" uv run --project path_b python -c 'import os, sys; sys.path.insert(0, "path_b")
from pathb.paths import DEV_EPISODES, manifest
m = manifest()
first = [int(x) for x in os.environ["FIRST"].split()]
head = [*DEV_EPISODES, *[n for n in first if n not in DEV_EPISODES]]
rest = [n for n in sorted(m) if n not in head]
print(*head, *[n for n in rest if m[n]["split"] == "public"], *[n for n in rest if m[n]["split"] == "evaluation"])')
fi
stages=("${STAGES[@]}") force=()
if [ -n "${FROM:-}" ]; then
  for i in "${!STAGES[@]}"; do [ "${STAGES[$i]}" = "$FROM" ] && stages=("${STAGES[@]:$i}") && force=(--force); done
fi
frames=(); [ "${FRAMES:-1}" = 1 ] && frames=(--frames)

upload() {  # the write token from path_b/.env if it has one, else this machine's hf login
  local n=$1 t
  for t in 1 2 3; do
    if [ -n "${HF_WRITE_TOKEN:-}" ]; then
      HF_TOKEN="$HF_WRITE_TOKEN" uv run --project tools python path_b/vm/hf_outputs.py upload --run "$RUN" --episodes "$n" "${frames[@]}" && return 0
    else
      env -u HF_TOKEN uv run --project tools python path_b/vm/hf_outputs.py upload --run "$RUN" --episodes "$n" "${frames[@]}" && return 0
    fi
    echo "   upload of episode $n failed (try $t)"; sleep 60
  done
  return 1
}

failed=() unsent=()
for n in $EPISODES; do
  echo "== episode $n ($(date -u +%FT%TZ))"
  "${RUNPY[@]}" run --episodes "$n" --out "$OUT" --stages "${stages[@]}" "${force[@]}" || failed+=("$n")
  [ "${UPLOAD:-1}" = 1 ] && { upload "$n" || unsent+=("$n"); }
done
echo "== run $RUN done ($(date -u +%FT%TZ)); run.py failed on: ${failed[*]:-none}; not uploaded: ${unsent[*]:-none}"
