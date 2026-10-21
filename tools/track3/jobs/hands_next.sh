#!/bin/bash
# After an episode's late-hand reruns (namespace fs_hands) finish: apply the camera rule (refined
# camera path > 6x the DROID-SLAM path -> bundle without refinement, ns fs_hands_noref), then submit
# the scene (CPU) and a 250M-step training run named ep<EP>_v2_250M. Usage: hands_next.sh EP [EP ...]
O=/data/user_data/$USER/v2d/recon_data/outputs
REPO=/home/$USER/NVIDIA_V2D_Egocentric
for EP in "$@"; do
  E6=$(printf %06d $EP)
  R=$("$REPO/tools/.venv/bin/python" - "$O/fs_hands/episode_$E6" <<'PY'
import sys, numpy as np
from pathlib import Path
rs = []
for run in sorted(Path(sys.argv[1]).iterdir()):
    c = np.load(run / 'result_slam_gravity_aligned/result.npz')['camera_to_world_transform']
    t = np.loadtxt(run / 'slam_trajectory.txt')
    rs.append(np.linalg.norm(np.diff(c[:, :3, 3], axis=0), axis=1).sum() / np.linalg.norm(np.diff(t[:, 1:4], axis=0), axis=1).sum())
print(max(rs))
PY
)
  if python3 -c "import sys; sys.exit(0 if float('$R') > 6 else 1)"; then
    ARGS=$(for o in $(ls $O/fs_hands/episode_$E6); do echo -n "$EP:$o "; done)
    (cd $REPO && baseline/reconstruction/.venv/bin/python tools/track3/write_unrefined_bundle.py fs_hands fs_hands_noref $ARGS | grep -E "hands L=")
    echo "ep$EP camera ratio $R -> no refinement"
    LR_CPU=1 NS=fs_hands_noref TAG=v2_250M LR_EXTRA="--smooth-hands 2" SCORE=0 $REPO/tools/track3/jobs/scene_train.sh $EP
  else
    echo "ep$EP camera ratio $R -> refined"
    LR_CPU=1 NS=fs_hands TAG=v2_250M SCORE=0 $REPO/tools/track3/jobs/scene_train.sh $EP
  fi
done
