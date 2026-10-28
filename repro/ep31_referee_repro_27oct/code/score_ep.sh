#!/bin/bash
# Score runs of any episode: unrefined bundle + official score (local metric code), and with a CoTracker track file
# also the referee-corrected (frame-0 anchor) track of <OBJ> as namespace babel_fs_c0_<ns>.
#   ep/score_ep.sh <tag> <EP> "<EP:obj EP:obj>" <ns,...> [<cotrack.npz> <OBJ>]      (login node; starts its own step)
source ~/v2d_ops/env.sh 2>/dev/null
TAG=$1 EP=$2 OBJS=$3 LIST=$4 NPZ=${5:-} OBJ=${6:-}
srun --jobid=${HJ:-10847124} --overlap --ntasks=1 --cpus-per-task=6 --mem=24G bash -c "
source ~/v2d_ops/env.sh 2>/dev/null; cd \$REPO_REAL; O=baseline/reconstruction/data/outputs; E=\$(printf episode_%06d $EP)
lnrun() { mkdir -p \$2; for f in \$1/*; do [ \${f##*/} = frames ] || cp -al \$f \$2/; done; }
# untextured meshes (FoundationPose model-free) have no material.mtl, which the scorer's bundle merge reads: add a plain one
mtlfix() { for m in \$O/\$1/\$E/*/result_slam_gravity_aligned; do [ -e \$m/material.mtl ] || printf 'newmtl material_0\nKd 0.8 0.8 0.8\n' > \$m/material.mtl; done; }
NS=''
for ns in \$(echo $LIST | tr , ' '); do
  ls \$O/\$ns/\$E/*/result_slam_gravity_aligned/result.npz > /dev/null 2>&1 || { echo \"missing \$ns\"; continue; }
  [ -d \$O/\${ns}_noref/\$E ] || \$VENVS/recon_host/bin/python tools/track3/write_unrefined_bundle.py \$ns \${ns}_noref $OBJS > /dev/null || echo \"noref FAIL \$ns\"
  mtlfix \${ns}_noref; NS=\$NS,\${ns}_noref
  if [ -n '$NPZ' ]; then
    C=\$D/diag/cotrack; c0=babel_fs_c0_\${ns#babel_fs_}; S=\$O/\$ns/\$E/$OBJ
    [ -d \$C/\${ns}_c0 ] || \$VENVS/stereo/bin/python ~/v2d_ops/cotrack_eval.py \$S $NPZ \$C/\${ns}_c0 | sed \"s/^/\$ns c0: /\"
    if [ -f \$C/\${ns}_c0/000000.json ] && [ ! -d \$O/\$c0 ]; then
      for r in \$O/\$ns/\$E/*; do lnrun \$r \$O/\$c0/\$E/\${r##*/}; done
      rm -rf \$O/\$c0/\$E/$OBJ/poses_smoothed; mkdir \$O/\$c0/\$E/$OBJ/poses_smoothed; cp \$C/\${ns}_c0/*.json \$O/\$c0/\$E/$OBJ/poses_smoothed/
    fi
    [ -d \$O/\$c0 ] && { [ -d \$O/\${c0}_noref/\$E ] || \$VENVS/recon_host/bin/python tools/track3/write_unrefined_bundle.py \$c0 \${c0}_noref $OBJS > /dev/null || echo \"noref FAIL \$c0\"; mtlfix \${c0}_noref; NS=\$NS,\${c0}_noref; }
  fi
done; NS=\${NS#,}
rm -rf \$D/stereo_test/bundles_ep_$TAG
SCORE_BUNDLES=\$D/stereo_test/bundles_ep_$TAG V2D_DATA=\$D uv run --project tools --with networkx python tools/track3/score_playback.py \$D/score/ep_$TAG.json \$NS $EP 2>&1 | grep recon_ourmesh | grep -v -i warn
echo SCORE_EP_OK_$TAG"
