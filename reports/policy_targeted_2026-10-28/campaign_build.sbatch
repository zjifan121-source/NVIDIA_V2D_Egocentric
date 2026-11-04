#!/bin/bash
# Build one episode's scene from a reconstruction bundle for a training campaign (see campaign.py build).
# env: EP, BUNDLE (dir with result.npz) or NS [+ OBJ for one-object episodes], OUT (scene dir), SEQ (sequence id), OBJSMOOTH (frames; 0 = off)
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3:00:00
set -uo pipefail
REPO=${V2D_REPO:-/home/edmondwa/NVIDIA_V2D_Egocentric}
HERE=$REPO/reports/policy_targeted_2026-10-28
PY=${V2D_DATA:-/data/user_data/edmondwa/v2d}/venvs/retarget/bin/python
export V2D_REPO=$REPO
rm -rf "$OUT"; mkdir -p "$OUT"
# hub layout (NS = folder under recon data/outputs, one folder per object): merge multi-object episodes
if [ -n "${NS:-}" ]; then
  export V2D_BUNDLE_NS=$NS
  SRC=$([ -n "${OBJ:-}" ] && echo "--object $OBJ" || echo "--merge")
else
  SRC="--bundle $BUNDLE"
fi
$PY $REPO/tools/track3/run_load_retarget.py $SRC --episode $EP \
  --attach-hands --smooth-hands 2 --smooth-object-rotation ${OBJSMOOTH:-0} --skip-video --device cpu \
  --sequence-id "$SEQ" --out "$OUT" > "$OUT.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  # a single-object episode's mesh can come out as None.obj; the packer wants the roster name
  $PY - "$OUT" $EP <<'E'
import json, os, sys
from pathlib import Path
out, ep = Path(sys.argv[1]), int(sys.argv[2])
eps = {e['episode']: e for e in json.loads(Path(os.environ['V2D_REPO']) / 'tools/track3/episodes.json'.read_text())['episodes']}
d = out / 'score_input/meshes' / f'episode_{ep:06d}'
names = [o['name'] for o in eps[ep]['objects']]
if (d / 'None.obj').exists() and len(names) == 1 and not (d / f'{names[0]}.obj').exists():
    os.link(d / 'None.obj', d / f'{names[0]}.obj'); print('linked None.obj ->', names[0])
E
  # an explicit support companion from the reconstruction replaces an auto support with no shapes (always, with FORCE_SPEC=1)
  SPEC=$(dirname "${BUNDLE:-/nonexistent/x}")/support_v1/support_in_bundle_world.json
  SUP=$(ls "$OUT"/assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda 2>/dev/null | head -1)
  if [ -f "$SPEC" ] && [ -n "$SUP" ] && { [ "${FORCE_SPEC:-0}" = 1 ] || ! grep -qE 'def (Cylinder|Cube)\b' "$SUP"; }; then
    PQ=$(ls -d "$OUT"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave | head -1)
    mv "$SUP" "$SUP.empty_auto" && $PY $HERE/ep32_support.py "$SPEC" "$BUNDLE" "$PQ" "$SUP"
  fi
fi
echo "ep$EP smooth=${OBJSMOOTH:-0} exit $rc $(date +%T)"
