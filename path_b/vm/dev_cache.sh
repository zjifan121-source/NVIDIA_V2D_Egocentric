#!/usr/bin/env bash
# On the L40S, after new_instance.sh: the v4 dev cache (V4_BUILD_SPEC.md §6, task 2's GPU smoke test):
# S1 on ep41 and ep13 in run v4 (P0-P2 from run 4c2eac31; P1b SAM 3.1, P1c DROID-SLAM, P1d hands, P3,
# P4 with the hands' world exports, s3a2_fuse_scan), each episode uploaded with its per-frame archive.
#
#     ssh -t l40s "tmux new -d -s v4 'bash /root/v2d/NVIDIA_V2D_Egocentric/path_b/vm/dev_cache.sh'"
#
# On the Mac afterwards:
#     uv run --project tools python path_b/vm/hf_outputs.py download --run v4 --frames --episodes 41 13
# Once it passes, phase_b.sh without EPISODES runs the other 38 episodes in the same run folder (41 and 13
# are kept). S0.1 is a separate job (run.py oracle).
exec env EPISODES="${EPISODES:-41 13}" bash "$(dirname "$0")/phase_b.sh"
