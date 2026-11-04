#!/bin/bash
# refine1104 ep8 step 3 (inside job 10774619 = babel-s5-16, CPU only): only if gate_results.json says all_pass: install the
# built scene rf_ep8_ds with ~/v2d_ops/install_scene_b.sh, then queue KEN seeds 42 43 44 at prio 10 with ~/v2d_ops/kq_add_b.sh.
set -u
O=/data/user_data/haisenk/v2d/refine1104/ep8; NAME=rf_ep8_ds
B=/tmp/haisenk_v2d; P=$B/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python
cat $O/gate_results.json | head -c 6000; echo
PASS=$($P -c "import json;print(json.load(open('$O/gate_results.json'))['all_pass'])")
[ "$PASS" = "True" ] || { echo "GATES NOT ALL PASS ($PASS) - not installing"; exit 1; }
[ -e $O/FAILED.json ] && { echo "FAILED.json present - not installing"; exit 1; }
bash ~/v2d_ops/install_scene_b.sh $O/$NAME $NAME | tee $O/install.log
PQ=$(grep "^PARQUET_DIR " $O/install.log | awk '{print $2}')
grep -q "missing paths: none" $O/install.log || { echo "install check did not report 'missing paths: none' - not queueing"; exit 1; }
[ -n "$PQ" ] && [ -d "$PQ" ] || { echo "no PARQUET_DIR - not queueing"; exit 1; }
ls -la $PQ
# load test with the training venv: same columns/rows as the adopted scene's parquet
$P - "$PQ" "$B/scenes/sc_ep8" <<'PY'
import sys, glob, pyarrow.parquet as pq
a = pq.ParquetFile(glob.glob(sys.argv[1] + "/*.parquet")[0]).read()
b = pq.ParquetFile(glob.glob(sys.argv[2] + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet")[0]).read()
print("load test: cols", len(a.column_names), "vs adopted", len(b.column_names), "rows", a.num_rows, b.num_rows, "same names", a.column_names == b.column_names)
print("paths:", a.column("object_mesh_paths")[0].as_py()[0], a.column("raw_motion_file")[0].as_py())
PY
printf "rf_ep8_ds_K_s42 8 $PQ 42 KEN 10\nrf_ep8_ds_K_s43 8 $PQ 43 KEN 10\nrf_ep8_ds_K_s44 8 $PQ 44 KEN 10\n" | bash ~/v2d_ops/kq_add_b.sh | tee $O/queue.log
echo "== queue lines"; grep "rf_ep8_ds" $B/kq_babel.txt
echo B3_DONE $(date +%T)
