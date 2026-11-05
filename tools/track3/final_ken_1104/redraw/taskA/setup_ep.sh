# setup_ep.sh EP STEP : relocated scene copy + path-patched checkpoint for one bob1 episode (writes only under $W)
set -u
EP=$1 STEP=$2
W=/data/user_data/haisenk/v2d/push1104/taskA
H=$W/hf/experiments/edmond/bob1_submission_20261103/episode_$(printf %06d $EP)
FC=/data/user_data/haisenk/v2d/v2d_train/baseline/robotic_grounding/flash_chord; PY=$FC/.venv/bin/python
SC=$W/scenes/ep$EP
if [ ! -d $SC ]; then mkdir -p $W/scenes; cp -a $H/scene $SC && $PY $W/relocate_scene.py $SC | tail -2; fi
SRC=$(ls $H/policy/policy_$STEP.safetensors $H/policy/train/policy_$STEP.safetensors 2>/dev/null | head -1)
[ -n "$SRC" ] || { echo "EP$EP NO POLICY"; exit 1; }
DST=$W/pol/ep$EP/policy_$STEP.safetensors
[ -f $DST ] || $PY ~/v2d_ops/kenc/push1104/taskA/patch_ckpt.py $SRC $SC $DST
PYTHONPATH=/data/user_data/haisenk/v2d/push1104/fc_src_de66 $PY - $DST $SRC <<'PY'
import sys, os, glob, hashlib
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
import pyarrow.parquet as pq
c = checkpoint_config(read_checkpoint_metadata(sys.argv[1])); p = c["task"]["parquet"]
fs = [p] if p.endswith(".parquet") else glob.glob(os.path.join(p, "*.parquet"))
bad = []
for f in fs:
    t = pq.read_table(f)
    for col in ("raw_motion_file", "object_mesh_paths", "object_urdf_paths"):
        if col in t.column_names:
            v = t.column(col)[0].as_py(); v = v if isinstance(v, list) else [v]
            bad += [x for x in v if x and isinstance(x, str) and x.startswith("/") and not os.path.exists(x)]
print("OK" if fs and not bad else "PROBLEM", "parquet", p, "nfiles", len(fs), "missing", bad[:3],
      "src_sha", hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()[:12])
PY
