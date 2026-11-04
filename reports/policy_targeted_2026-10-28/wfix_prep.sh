#!/bin/bash
# ep3: donor = ken_c_scored merged bundle with Ken's corrected left hand; graft onto Wentao's ep3. ep10/20: link Wentao's bundles.
set -euo pipefail
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
PY=/data/user_data/edmondwa/v2d/venvs/retarget/bin/python
W=/data/user_data/edmondwa/v2d/wentao_mix/hub/experiments/wentacc/scored20_alignment_mix_v1_20261030/scenes
F=/data/user_data/edmondwa/v2d/wentao_fix
KM=/data/user_data/edmondwa/v2d/ken_night/night_2026-10-24/scenes/episode_000003/ken_c_scored_attach/bundle_merged
KNPZ=/data/user_data/edmondwa/v2d/ken_handfix/experiments/haisenk/hand_fixes_v1_20261102/episode_000003/inputs/ep3_left_hand_merged.npz
mkdir -p $F/donor_ep3 $F/hub/episode_000003 $F/hub/episode_000010 $F/hub/episode_000020
for f in $KM/*; do [ "$(basename $f)" = result.npz ] || ln -f $f $F/donor_ep3/ 2>/dev/null || cp $f $F/donor_ep3/; done
cp $KNPZ $F/donor_ep3/result.npz
$PY -I - $KM/result.npz $KNPZ <<'X'
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in ('camera_to_world_transform', 'hand_right_wrist_trans_in_camera', 'object_to_world_transform', 'object_1_to_world_transform'):
    print('donor sanity', k, float(np.abs(a[k].astype(float) - b[k].astype(float)).max()))
print('left hand changed by up to', float(np.abs(a['hand_left_wrist_trans_in_camera'] - b['hand_left_wrist_trans_in_camera']).max()), 'm')
X
rm -rf $F/hub/episode_000003/bundle
$PY $REPO/tools/track3/graft_grasps.py --target $W/episode_000003/bundle --donor $F/donor_ep3 --out $F/hub/episode_000003/bundle
cat $F/hub/episode_000003/bundle/graft_info.json | head -c 1500; echo
ln -sfn $W/episode_000010/bundle $F/hub/episode_000010/bundle
ln -sfn $W/episode_000020/bundle $F/hub/episode_000020/bundle
ls -la $F/hub/*/
