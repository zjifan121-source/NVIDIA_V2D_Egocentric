#!/bin/bash
set -e
cd ~
DL=/data/user_data/edmondwa/v2d/ken_handfix
mkdir -p $DL
/home/edmondwa/NVIDIA_V2D_Egocentric/tools/.venv/bin/python -I -c "
from huggingface_hub import snapshot_download
snapshot_download('rose-tea-cafe/v2d-t3-recon', repo_type='dataset', local_dir='$DL',
  allow_patterns=['experiments/haisenk/hand_fixes_v1_20261102/SUMMARY.json', 'experiments/haisenk/hand_fixes_v1_20261102/episode_000003/inputs/*'])"
K=$DL/experiments/haisenk/hand_fixes_v1_20261102
ls -la $K $K/episode_000003/inputs
/data/user_data/edmondwa/v2d/venvs/retarget/bin/python -I - $K <<'X'
import sys, json, numpy as np
K = sys.argv[1]
print(json.dumps(json.load(open(f'{K}/SUMMARY.json')), indent=1)[:2500])
k = np.load(f'{K}/episode_000003/inputs/ep3_left_hand_merged.npz')
w = np.load('/data/user_data/edmondwa/v2d/wentao_mix/hub/experiments/wentacc/scored20_alignment_mix_v1_20261030/scenes/episode_000003/bundle/result.npz')
print('frames', len(k['camera_to_world_transform']), len(w['camera_to_world_transform']))
def cmp(key):
    a, b = k[key].astype(float), w[key].astype(float)
    if a.shape != b.shape: return f'shape {a.shape} vs {b.shape}'
    return f'max abs diff {np.abs(a - b).max():.2e}'
for key in ['camera_to_world_transform', 'camera_intrinsics', 'hand_right_wrist_trans_in_camera', 'hand_right_finger_pose',
            'hand_right_wrist_orient_in_camera', 'hand_left_wrist_trans_in_camera', 'hand_left_is_valid']:
    print(key, cmp(key))
print('left-hand keys in ken npz:', [x for x in k.files if x.startswith('hand_left')])
X
