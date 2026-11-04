#!/bin/bash
# install_scene_b.sh SRC_SCENE_DIR NAME   (run INSIDE a step of job 10774619 = babel-s5-16, e.g.
#   ssh babel "srun --jobid=10774619 --overlap -n1 -c2 --mem=8G bash ~/v2d_ops/install_scene_b.sh /data/.../NAME NAME")
# Copies a built scene (top folder with assets/human_motion_data/...) to /tmp/haisenk_v2d/scenes/NAME, relocates its
# absolute paths (relocate_scene.py), checks that every referenced file exists, prints PARQUET_DIR for kq_add.sh.
# Refuses to overwrite an existing scene. Never deletes anything.
set -u
SRC=$1; NAME=$2; B=/tmp/haisenk_v2d; S=$B/scenes; P=$B/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python
[ -d "$SRC" ] || { echo "no source dir $SRC"; exit 1; }
[ -e "$S/$NAME" ] && { echo "SCENE EXISTS $S/$NAME - not overwriting (pick a new NAME)"; exit 1; }
cp -a "$SRC" "$S/$NAME" || exit 1
$P $B/orchard_home/v2d_orchard/relocate_scene.py "$S/$NAME" | tail -3
PQ=$(ls -d $S/$NAME/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave 2>/dev/null | head -1)
[ -n "$PQ" ] || { echo "NO PARQUET DIR under $S/$NAME"; exit 1; }
$P - "$PQ" <<'PY'
import sys, glob, os, pyarrow.parquet as pq
f = glob.glob(os.path.join(sys.argv[1], "*.parquet"))[0]
t = pq.read_table(f, columns=["raw_motion_file", "object_mesh_paths", "object_urdf_paths"])
bad = []
for c in t.column_names:
    v = t.column(c)[0].as_py(); v = v if isinstance(v, list) else [v]
    bad += [x for x in v if x and not os.path.exists(x)]
print("missing paths:", bad if bad else "none")
PY
echo "PARQUET_DIR $PQ"
