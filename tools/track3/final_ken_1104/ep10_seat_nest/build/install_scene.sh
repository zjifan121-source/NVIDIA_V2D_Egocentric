#!/bin/bash
# install_scene.sh LOCAL_TGZ NAME EP "SEEDS" RECIPE PRIO RUNPREFIX   (local Git Bash)
# Upload a scene tarball (top-level folder NAME) to orchard, unpack into /tmp/haisenk_v2d/scenes/NAME (refuses to overwrite),
# relocate its absolute paths (relocate_scene.py), sanity-check, and queue RUNPREFIX_s<seed> lines (EP parquet seed RECIPE PRIO).
set -u
TGZ=$1 NAME=$2 EP=$3 SEEDS=$4 RECIPE=$5 PRIO=$6 RP=$7
[ -s "$TGZ" ] || { echo "no tgz $TGZ"; exit 1; }
tar tzf "$TGZ" | head -1 | grep -q "^$NAME/" || { echo "tgz top folder is not $NAME/"; tar tzf "$TGZ" | head -2; exit 1; }
timeout 900 ssh orchard "srun --jobid=157612 --overlap -n1 bash -c 'mkdir -p /tmp/haisenk_v2d/incoming && cat > /tmp/haisenk_v2d/incoming/$NAME.tgz'" < "$TGZ" 2>/dev/null
cat > /tmp/_inst_$NAME.sh <<EOS
set -u
B=/tmp/haisenk_v2d; S=\$B/scenes; P=\$B/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python
echo "uploaded: \$(stat -c %s \$B/incoming/$NAME.tgz) bytes (local $(stat -c %s "$TGZ"))"
if [ -e \$S/$NAME ]; then echo "SCENE EXISTS \$S/$NAME - not overwriting"; else tar xzf \$B/incoming/$NAME.tgz -C \$S && \$P \$HOME/v2d_orchard/relocate_scene.py \$S/$NAME | head -3; fi
PQ=\$(ls -d \$S/$NAME/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave | head -1)
echo "parquet dir: \$PQ"; ls \$PQ | head -2
\$P - "\$PQ" <<'PY'
import sys, glob, os, pyarrow.parquet as pq
f = glob.glob(os.path.join(sys.argv[1], "*.parquet"))[0]
t = pq.read_table(f, columns=["raw_motion_file", "object_mesh_paths", "object_urdf_paths"])
bad = []
for c in t.column_names:
    v = t.column(c)[0].as_py(); v = v if isinstance(v, list) else [v]
    bad += [x for x in v if x and not os.path.exists(x)]
print("missing paths:", bad if bad else "none")
PY
grep -h "filename=" \$S/$NAME/assets/human_motion_data/ego_recon/processed/*.urdf 2>/dev/null | grep -o 'filename="[^"]*"' | sed 's/filename=//; s/"//g' | while read p; do [ -e "\$p" ] || [ -e "\$(dirname \$(ls \$S/$NAME/assets/human_motion_data/ego_recon/processed/*.urdf | head -1))/\$p" ] || echo "URDF missing: \$p"; done | head -5
cd \$B
for s in $SEEDS; do n=${RP}_s\$s; grep -q "^\$n " kq.txt && { echo "already queued \$n"; continue; }
  flock kq.txt.lock bash -c "echo '\$n $EP \$PQ \$s $RECIPE $PRIO' >> kq.txt"; echo "queued \$n"; done
EOS
timeout 600 ssh orchard "srun --jobid=157612 --overlap -n1 -c2 bash -s" < /tmp/_inst_$NAME.sh 2>/dev/null
rm -f /tmp/_inst_$NAME.sh
