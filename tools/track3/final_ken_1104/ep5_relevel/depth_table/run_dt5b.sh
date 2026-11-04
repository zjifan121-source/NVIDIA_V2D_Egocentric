#!/bin/bash
# The measurement that produced the re-level plane, as it ran on 4 Nov 2026 ~00:08-00:16 EST on Babel (babel-t9-28, CPU step of
# job 11035728). The original wrapper (run_dt5b_full.sh) wrote dt5b.py via a heredoc and then ran the three lines below;
# dt5b.py here is that exact file (sha256 82ae9593..., fetched from ~/v2d_ops/kenc/diag/ep5_objA/dt5b.py).
#   S = the parent scene w7_attcom_ep5 (Babel /data copy; numerically identical to the orchard copy the re-level ran on)
#   E = our own reconstruction of episode 5 (ken_c_scored) that holds the FoundationStereo depth PNGs per object
#       (metres = 65535 / raw - 1). No ground truth is read.
# dt5b.py loads the review tool from the hard-coded path /home/haisenk/v2d_ops/kenc/review/tool/render_review.py
# (the same file is staged in review_tool/render_review.py, sha256 864350cf...).
# Output: dt5b_base/ep5_depthB.json (staged in out/ep5_depthB.json, sha256 2b6225d9...). The re-level uses
# planes_start.table: normal (-0.02000265, -0.09335555, 0.99543188), a 0.020094445, b 0.093783968, c 1.015406946
# (frames 0-24, 260000 points, inlier fraction 0.948, residual std 1.64 mm, tilt 5.4786 deg vs the scene's +z).
set -u
D=~/v2d_ops/kenc/diag/ep5_objA
S=/data/user_data/haisenk/v2d/hand6/scenes/w7_attcom_ep5
E=/data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/ken_c_scored/episode_000005
cd $D && timeout 1500 /home/haisenk/anaconda3/bin/python -I $D/dt5b.py $S $E $D/dt5b_base 2>&1 | tail -60
