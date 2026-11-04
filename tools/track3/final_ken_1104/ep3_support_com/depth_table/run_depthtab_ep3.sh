#!/bin/bash
# run_depthtab_ep3.sh - the GT-free table measurement that set the ep3 support height, as run on 3-4 Nov 2026
# (Babel, CPU step of job 11035728 on babel-t9-28; workflow wf_ed1d2237-bda, agent "depthtab:ep3").
# Original commands (verbatim apart from line breaks):
#   cd ~/v2d_ops/kenc/depthtab/ep3 && timeout 900 /home/haisenk/anaconda3/bin/python depthtab.py \
#     /scratch/haisenk-v2d/hand6/bob1_scenes/experiments/haisenk/hand_fixes_v1_20261102/episode_000003/scene \
#     /data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/ken_c_scored/episode_000003 \
#     /home/haisenk/v2d_ops/kenc/depthtab/ep3/out 2>&1 | tee out/run.log | grep -v Warning          (23:51-23:57 EST)
#   timeout 600 /home/haisenk/anaconda3/bin/python followup.py \
#     /scratch/haisenk-v2d/hand6/bob1_scenes/experiments/haisenk/hand_fixes_v1_20261102/episode_000003/scene \
#     /home/haisenk/v2d_ops/kenc/depthtab/ep3/out 2>&1 | grep -v Warning | tee out/followup.log       (23:58-00:00 EST)
# Inputs:
#   SCENE = bob1's ep3 scene (identical to the parent b1_ep3 of sc_ep3, md5-checked file by file; only the parquet's
#           absolute paths differ). Its support_0 top is 0.9878.
#   REC   = our ken_c_scored reconstruction of episode 3: FoundationStereo depth white_pot/depth/%06d.png
#           (uint16 inverse depth, depth_m = 65535/raw - 1, 65535 = hole), SAM 3 masks <obj>/masks/{1,2,3}/%06d.png
#           (1 = object, 2/3 = hands), camera from <scene>/bundle_cleaned/result.npz.
# depthtab.py imports render_review.py from ~/v2d_ops/kenc/review/tool (copy: ./review_tool/render_review.py,
# md5 8050e21708eb6d052964651ba50e6a1e, unchanged since 3 Nov 22:30 EST, i.e. before this run).
# Outputs (copies in ./out): ep3_depthtab.json, ep3_followup.json, run.log, followup.log. The raw table points
# ep3_table_points.npz (57 MB) stay on Babel (~/v2d_ops/kenc/depthtab/ep3/out/), sha256 in the README.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SCENE=${SCENE:-/scratch/haisenk-v2d/hand6/bob1_scenes/experiments/haisenk/hand_fixes_v1_20261102/episode_000003/scene}
REC=${REC:-/data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/ken_c_scored/episode_000003}
OUT=${OUT:-$HERE/out_rerun}
PY=${PY:-/home/haisenk/anaconda3/bin/python}
mkdir -p "$OUT"
# depthtab.py / followup.py put ~/v2d_ops/kenc/review/tool first on sys.path; where that directory does not exist,
# render_review is found through PYTHONPATH instead (if it does exist, check it holds the md5 above).
export PYTHONPATH="$HERE/review_tool${PYTHONPATH:+:$PYTHONPATH}"
"$PY" "$HERE/depthtab.py" "$SCENE" "$REC" "$OUT" 2>&1 | tee "$OUT/run.log" | grep -v Warning
"$PY" "$HERE/followup.py" "$SCENE" "$OUT" 2>&1 | grep -v Warning | tee "$OUT/followup.log"
