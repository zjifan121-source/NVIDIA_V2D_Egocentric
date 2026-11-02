#!/usr/bin/env bash
# The v4 CPU chain on episodes whose S1 is done (V4_BUILD_SPEC.md §6, task 11), on the Mac or the VM:
#
#     EPISODES="0 18 43 31 23" bash path_b/vm/cpu_chain.sh          # from the repo root
#
# 1. DOWNLOAD=1 (default): hf_outputs.py download --run $RUN --frames --episodes $EPISODES (S1 from HF)
# 2. run.py canonical for every object of those episodes, pooled over every clip of it in the run folder
#    (V4 S3a; the meshes change as more clips arrive, so a later call rebuilds them)
# 3. run.py run --stages s3a2_register s3b_segments s3c_invert s3d_settle s4a_phases s4b_reference
#    (--force: a rebuilt canonical mesh changes everything after it); S4c is off, S5 is the policy window's
# 4. run.py waterfall on the public ones: ref_inverted / ref_keyframe / ref_still beside the baselines
# Mac settings (BUILD_LOG decisions): RETARGET_VENV, RETARGET_DEVICE=cpu, MANO_DIR, COACD_PY, SETTLE_PY.
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-v4}
OUT=path_b/data/outputs/$RUN
: "${EPISODES:?set EPISODES}"
if [ "$(uname)" = Darwin ]; then
  export RETARGET_VENV=${RETARGET_VENV:-$HOME/V2D/venvs/retarget} RETARGET_DEVICE=${RETARGET_DEVICE:-cpu}
  export MANO_DIR=${MANO_DIR:-$HOME/V2D/mano_v1_2} COACD_PY=${COACD_PY:-$HOME/V2D/venvs/sim/bin/python}
  export SETTLE_PY=${SETTLE_PY:-$HOME/V2D/venvs/sim/bin/python}
fi
RUNPY=(uv run --project path_b python path_b/run.py)
LOG=${LOG:-path_b/data/cpu_chain_${RUN}.log}
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "== cpu chain $RUN on $EPISODES ($(date -u +%FT%TZ), $(git rev-parse --short=8 HEAD))"

if [ "${DOWNLOAD:-1}" = 1 ]; then
  # shellcheck disable=SC2086
  (set -a; . path_b/.env; set +a  # the read-only HF_TOKEN is enough to download
   uv run --project tools python path_b/vm/hf_outputs.py download --run "$RUN" --frames --episodes $EPISODES)
fi
objects=$(for n in $EPISODES; do jq -r --argjson n "$n" '.episodes[] | select(.episode == $n) | .objects[].name' tools/track3/episodes.json; done | sort -u)
for o in $objects; do
  echo "== canonical $o"
  "${RUNPY[@]}" canonical --object "$o" --split all --out "$OUT" || echo "   canonical $o failed"
done
# shellcheck disable=SC2086
"${RUNPY[@]}" run --episodes $EPISODES --out "$OUT" --force \
  --stages s3a2_register s3b_segments s3c_invert s3d_settle s4a_phases s4b_reference
public=$(for n in $EPISODES; do jq -r --argjson n "$n" '.episodes[] | select(.episode == $n and .split == "public") | .episode' tools/track3/episodes.json; done)
if [ -n "$public" ]; then
  # shellcheck disable=SC2086
  "${RUNPY[@]}" waterfall --episodes $public --out "$OUT"
fi
echo "== cpu chain done ($(date -u +%FT%TZ))"
