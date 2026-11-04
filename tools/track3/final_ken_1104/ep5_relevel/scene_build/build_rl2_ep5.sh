#!/bin/bash
# build_rl2_ep5.sh SCENES_DIR [PYTHON]
# Rebuild the T8 scene rl2_ep5 (scored episode 5) from its parent scene w7_attcom_ep5.
# Writes SCENES_DIR/rl_ep5 (intermediate) and SCENES_DIR/rl2_ep5. Both tools refuse to overwrite an existing output.
#
#   SCENES_DIR/w7_attcom_ep5 = the parent scene (bundle item w7_attcom_ep5.tgz). Extract it into SCENES_DIR, then run
#   `python relocate_scene.py SCENES_DIR/w7_attcom_ep5` so the absolute paths in its parquet point at that copy.
#   PYTHON needs numpy, scipy, pyarrow and trimesh. We used the flash_chord venv on orchard:
#   /tmp/haisenk_v2d/v2d_train/baseline/robotic_grounding/flash_chord/.venv/bin/python
#   Run on Linux: relevel.py and scenefix.py find the sequence id with the regex sequence_id=([^/]+).
#
# This is the two-command form of what ran on 4 Nov 2026 01:47-01:49 EST (exact history in history/). With the parent
# from the bundle it reproduces the archived rl2_ep5: every numeric parquet column bit-identical, meshes vertex-identical,
# support USD text-identical (checks/rebuild_verification/). Only absolute path strings differ, and step 2 also writes
# scenefix_report.json, which the archived scene does not have.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S=$(cd "$1" && pwd); P=${2:-python3}
export RELOCATE=${RELOCATE:-$HERE/relocate_scene.py}
cd "$S"

# 1) World re-level. The table plane measured from our own FoundationStereo depth (depth_table/out/ep5_depthB.json,
#    planes_start.table: unit normal n, plane z = a*x + b*y + c in the parent's sim world) becomes horizontal.
#    Every world-frame field of the processed parquet is rotated by 5.4786 deg about the measured table point under
#    the pot's frame-0 footprint. support_0 is re-centred under the rotation. --drop-other-supports removes support_1,
#    the partial 9.1 cm disk under the spoon handle.
"$P" "$HERE/relevel.py" w7_attcom_ep5 rl_ep5 \
  -0.02000265 -0.09335555 0.99543188 0.020094445 0.093783968 1.015406946 white_pot --drop-other-supports

# 2) Pot mesh 1.0 cm shorter with the rim fixed: the bottom moves up 1.0 cm along the frame-0 vertical, from 13.34 to
#    12.34 cm tall. The pot's frame-0 bottom goes from z 1.0007 to 1.0107, level with the spoon's (1.0107). The top of
#    support_0 is set to 1.0097, 1 mm under both. Poses and hands are unchanged.
"$P" "$HERE/scenefix.py" rl_ep5 rl2_ep5 --obj white_pot:-1.00 --support-top support_0:1.0097
