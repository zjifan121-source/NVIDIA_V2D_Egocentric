#!/bin/bash
# refine1104 ep8 step 2 (inside job 11035728 = babel-t9-28, CPU only): copy the prepared despin build's parquet (read-only
# source /scratch/haisenk-v2d/diag/ep8_plan/despin, NOT despin_seat) into my dir, copy the adopted scene copy
# (src_sc_ep8 -> rf_ep8_ds) and run merge_gates.py (merge + pre-registered gates). merge_gates.py and gates.json arrive
# base64-encoded in the variables below.
set -u
O=/data/user_data/haisenk/v2d/refine1104/ep8; DS=/scratch/haisenk-v2d/diag/ep8_plan/despin
PY=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
mkdir -p $O/despin_src $O/code
echo "$MG_B64" | base64 -d > $O/code/merge_gates.py
echo "$GJ_B64" | base64 -d > $O/gates.json
md5sum $O/code/merge_gates.py $O/gates.json
PQD=$(ls -d $DS/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave | head -1)
echo "despin parquet dir: $PQD"; ls -la $PQD $DS/bundle_cleaned | head -20
[ -e $O/despin_src/despin.parquet ] || cp $PQD/*.parquet $O/despin_src/despin.parquet
[ -e $O/despin_src/bundle_cleaned_result.npz ] || cp $DS/bundle_cleaned/result.npz $O/despin_src/bundle_cleaned_result.npz
md5sum $PQD/*.parquet $O/despin_src/despin.parquet $DS/bundle_cleaned/result.npz $O/despin_src/bundle_cleaned_result.npz
if [ -e $O/rf_ep8_ds ]; then echo "rf_ep8_ds exists - stop (no overwrite)"; exit 1; fi
cp -a $O/src_sc_ep8 $O/rf_ep8_ds
EB=/data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep8/assets/human_motion_data/ego_recon/processed/ep8_ken_c_scored_attach_beige_cup.obj
md5sum $EB $O/src_sc_ep8/bundle_merged/result.npz
timeout 500 $PY -I $O/code/merge_gates.py $O $O/src_sc_ep8/bundle_merged $EB 0.01027 2>&1 | grep -v Warning
echo B2_DONE $(date +%T)
