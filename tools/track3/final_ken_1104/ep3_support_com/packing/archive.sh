#!/bin/bash
# archive.sh CAND SCENE EP STEP "NOTE"   (local Git Bash)
# Keep everything needed to re-pack / re-submit / continue a candidate without re-running anything:
#   orchard node: /tmp/haisenk_v2d/archive/CAND/{scene.tgz, policy_<STEP>.safetensors, results/, sweep.json, best.txt,
#                 recipe.txt, train.log, manifest.json}   (STEP = checkpoint step, or "final" = old driver's results/)
#   local PC:     C:\v2d_demo\archive\CAND\  (full copy of the above)
#   Babel:        ~/v2d_ops/kenc/tests/CAND/ (already: rollout + meshes + one.parquet + track3_submission.parquet +
#                 Kaggle receipt) -> also copied into the local archive (babel/).
set -u
CAND=$1 SCENE=$2 EP=$3 STEP=$4 NOTE=${5:-}
E6=$(printf %06d $EP); L=/c/v2d_demo/archive/$CAND; mkdir -p $L/babel
timeout 900 ssh orchard "srun --jobid=157612 --overlap -n1 -c2 bash -s" > /dev/null 2>&1 <<EOS
set -u
B=/tmp/haisenk_v2d; A=\$B/archive/$CAND; R=\$B/runs/$CAND; mkdir -p \$A
[ -s \$A/scene.tgz ] || tar czf \$A/scene.tgz -C \$B/scenes $SCENE
if [ "$STEP" = final ]; then RES=\$R/results; PSTEP=\$(ls -1 \$R/train/policy_*.safetensors | sort -V | tail -1 | sed 's/.*policy_//; s/.safetensors//'); else RES=\$R/early_$STEP/results; PSTEP=$STEP; fi
cp -n \$R/train/policy_\$PSTEP.safetensors \$A/ 2>/dev/null
rm -rf \$A/results.tmp && cp -a \$RES \$A/results.tmp && rm -rf \$A/results && mv \$A/results.tmp \$A/results
for f in sweep.json best.txt recipe.txt train.log checkpoints.json; do [ -f \$R/\$f ] && cp \$R/\$f \$A/; done
python3 - \$A "$CAND" "$SCENE" "$EP" "$STEP" "\$PSTEP" "\$RES" "$NOTE" <<'PY'
import json, sys, glob, hashlib, os
A, cand, scene, ep, step, pstep, res, note = sys.argv[1:9]
m = json.load(open(glob.glob(f"{A}/results/episode_*.json")[0]))["metrics"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = dict(candidate=cand, episode=int(ep), scene=scene, scene_tgz="scene.tgz", checkpoint_step=int(pstep), step_arg=step,
           results_source=res, own_ref=m.get("add_auc"), own_ref_per_body=m.get("add_auc_per_body"),
           policy_file=f"policy_{pstep}.safetensors", policy_sha256=sha(f"{A}/policy_{pstep}.safetensors"),
           rollout_sha256=sha(glob.glob(f"{A}/results/episode_*.parquet")[0]), recipe=open(f"{A}/recipe.txt").read().strip() if os.path.exists(f"{A}/recipe.txt") else "KEN (old driver gpu_queue.sh defaults)",
           note=note)
json.dump(man, open(f"{A}/manifest.json", "w"), indent=1)
PY
EOS
timeout 900 ssh orchard "srun --jobid=157612 --overlap -n1 -c1 tar cf - -C /tmp/haisenk_v2d/archive $CAND" 2>/dev/null | tar xf - -C /c/v2d_demo/archive
timeout 300 ssh -o BatchMode=yes -o ConnectTimeout=30 -o ConnectionAttempts=5 babel "cd ~/v2d_ops/kenc/tests/$CAND 2>/dev/null && tar cf - --exclude=in *.parquet *receipt.json *.log SUBMITTED 2>/dev/null" | tar xf - -C $L/babel 2>/dev/null
ls -la $L $L/babel | grep -v "^total"; cat $L/manifest.json 2>/dev/null | head -30
