#!/usr/bin/env bash
# On the L40S, after new_instance.sh: phase A's dev cache (V3_BUILD_SPEC.md §9; BUILD_LOG "Order of work").
#
#     ssh -t l40s "tmux new -s dev 'bash /root/v2d/NVIDIA_V2D_Egocentric/path_b/vm/dev_cache.sh'"
#
# 1. a new run folder whose P0-P2 for ep40 and ep26 are run 4c2eac31's (hard links; restored by setup.sh
#    with frames)
# 2. the chain on ep40 and ep26 up to S4b: the GPU stages P1b (SAM 3.1), P1c (DROID), P1d (hands),
#    S3a (SAM 3D), S3b (FoundationPose), and the CPU stages between them; S4c-S6 are tasks 13-15's
#    smoke tests, run on this run folder once it is checked
# 3. S0.1: run.py oracle --episodes 40 26 (ground-truth task, train, eval; about 2 x 0.8 h)
# 4. both to HF (hf_outputs.py: the run with frames, and the S0.1 folder with its summary)
# A rerun keeps every stage with a report and every S0.1 step done, so fix and rerun the same command.
# path_b/.env gives HF_TOKEN (gated SAM 3.1 / SAM 3D downloads); uploads use HF_WRITE_TOKEN when the
# file has it, else the machine's hf login. Nothing secret is printed.
set -euo pipefail
cd /root/v2d/NVIDIA_V2D_Egocentric
RUN=${RUN:-dev_$(git rev-parse --short=8 HEAD)}
OUT=path_b/data/outputs/$RUN
SRC=path_b/data/outputs/4c2eac31
LOG=/root/v2d/logs/dev_cache_$RUN.log
mkdir -p /root/v2d/logs
exec > >(tee -a "$LOG") 2>&1
echo "== dev cache $RUN ($(date -u +%FT%TZ)); log $LOG"

for e in 000040 000026; do
  mkdir -p "$OUT/episode_$e"
  for s in p0_ingest p1_calibrate p2_depth; do
    [ -d "$SRC/episode_$e/$s" ] || { echo "missing $SRC/episode_$e/$s: rerun setup.sh with RESTORE_RUN=4c2eac31 RESTORE_FRAMES=1"; exit 1; }
    [ -e "$OUT/episode_$e/$s" ] || cp -al "$SRC/episode_$e/$s" "$OUT/episode_$e/$s"
  done
done

set -a; . path_b/.env; set +a
export RETARGET_VENV=${RETARGET_VENV:-/root/v2d/venvs/retarget}
echo "== chain to S4b on ep40, ep26"
uv run --project path_b python path_b/run.py run --episodes 40 26 --out "$OUT" --stages \
  p0_ingest p1_calibrate p2_depth p1b_masks p1c_camera p1d_hands p3_scale_bridge p4_export \
  s3a_mesh s3b_track s3c_fuse s4a_phases s4b_reference
echo "== S0.1 oracle on ep40, ep26"
uv run --project path_b python path_b/run.py oracle --episodes 40 26 --out "path_b/data/outputs/${RUN}_s01"
cp path_b/oracles/s01_*.json "path_b/data/outputs/${RUN}_s01/"  # the summary travels with the run

echo "== upload to HF"
for r in "$RUN" "${RUN}_s01"; do
  if [ -n "${HF_WRITE_TOKEN:-}" ]; then
    HF_TOKEN="$HF_WRITE_TOKEN" uv run --project tools python path_b/vm/hf_outputs.py upload --run "$r" --frames
  else
    env -u HF_TOKEN uv run --project tools python path_b/vm/hf_outputs.py upload --run "$r" --frames
  fi
done
echo "== done ($(date -u +%FT%TZ)). On the Mac: hf_outputs.py download --run $RUN --frames; --run ${RUN}_s01"
