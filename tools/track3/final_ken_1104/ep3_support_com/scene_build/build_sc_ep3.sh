#!/bin/bash
# build_sc_ep3.sh - rebuild the ep3 scene sc_ep3 from its parent b1_ep3, with the same tools, arguments and order as the
# original build (4 Nov 2026 00:12 and 00:20 EST on orchard-flame-21:/tmp/haisenk_v2d/scenes; that directory now lives
# unchanged on babel-s5-16:/tmp/haisenk_v2d/scenes).
#
# Parent b1_ep3 = bob1's ep3 scene = our hand-fix v1 build h5_ep3_A1 (Babel /data/user_data/haisenk/v2d/hand5/scenes/
# h5_ep3_A1, 2 Nov 2026; HF rose-tea-cafe/v2d-t3-recon experiments/haisenk/hand_fixes_v1_20261102/episode_000003/scene).
# b1_ep3 differs from h5_ep3_A1 only in the absolute paths inside its parquet (relocate_scene.py).
#
# The two numbers below are measured values, not tuning knobs (see README.md):
#   0.9990 = min(depth-measured table under the resting objects at frame 0 (1.0065-1.0081),
#                lowest frame-0 resting bottom (white_pot 1.0000) - 1 mm)
#   white_pot = the object whose URDF centre of mass lies outside its own base (8.0 cm from its shell centroid)
#
# usage: SCENES=<dir containing b1_ep3> PY=<python with numpy scipy trimesh pyarrow> bash build_sc_ep3.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SCENES=${SCENES:-/tmp/haisenk_v2d/scenes}
PY=${PY:-/tmp/haisenk_v2d/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python}
# the original runs used the default RELOCATE=~/v2d_orchard/relocate_scene.py, byte-identical to ./relocate_scene.py
export RELOCATE=${RELOCATE:-$HERE/relocate_scene.py}
cd "$SCENES"
test -d b1_ep3 || { echo "parent scene b1_ep3 not found in $SCENES"; exit 1; }

# 0) measurement only: frame-0 / end resting bottoms of each object and the current support (prints, writes nothing)
"$PY" "$HERE/restspots.py" b1_ep3 0,30,280,312
grep -A9 'def Cylinder' b1_ep3/assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda | grep -E 'Cylinder|radius|translate'

# 1) raise support_0 (top 0.9878 -> 0.9990). Copies b1_ep3 to sf_ep3, edits only the support USD, repoints paths.
"$PY" "$HERE/scenefix.py" b1_ep3 sf_ep3 --support-top support_0:0.9990 > sf_ep3.log 2>&1
cat sf_ep3/scenefix_report.json

# 2) white_pot centre of mass -> area-weighted shell centroid of its visual mesh. Copies sf_ep3 to sc_ep3, edits only
#    the pot URDF <inertial><origin>, repoints paths. Expected: [0 0 0] -> [-0.008451 -0.062887 0.049392]
"$PY" "$HERE/comfix.py" fix sf_ep3 sc_ep3 white_pot

# 3) checks: COM/footprint report, the two edited values
"$PY" "$HERE/comfix.py" check sc_ep3
grep -A1 '<inertial>' sc_ep3/assets/human_motion_data/ego_recon/processed/*_rigid.urdf | grep origin
grep translate sc_ep3/assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda
# Expected md5 (Linux, LF): support USD bc823a849edfb0339d7ec8730973ec9f, white_pot_rigid.urdf b1577693c7a1fff05442dc2c06149a8f;
# every other file except the relocated parquet (path columns only) identical to the parent.
