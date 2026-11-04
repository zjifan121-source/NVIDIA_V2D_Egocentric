#!/bin/bash
# anch_run.sh SCENE EP ANCHOR RUN [RUN...]: fetch every evaluated checkpoint rollout of each RUN (+ SCENE's score meshes) from
# orchard and print plain / anchored own-ref per checkpoint (anch.py). Local copies in C:\v2d_demo\diag\anch\{runs,scenes}.
SCENE=$1 EP=$2 ANCHOR=$3; shift 3
E6=$(printf %06d $EP); D=/c/v2d_demo/diag/anch; P=/c/v2d_demo/venv/Scripts/python.exe
cd $D
L=""; for r in "$@"; do L="$L runs/$r/early_*/results/episode_$E6.parquet runs/$r/results/episode_$E6.parquet"; done
timeout 600 ssh orchard "srun --jobid=157612 --overlap -n1 bash -c 'cd /tmp/haisenk_v2d && tar czf - scenes/$SCENE/score_input/meshes $L 2>/dev/null'" 2>/dev/null > fetch_$$.tgz
tar xzf fetch_$$.tgz 2>/dev/null; rm -f fetch_$$.tgz
M=scenes/$SCENE/score_input/meshes/episode_$E6
for r in "$@"; do
  for f in $(ls -d runs/$r/early_*/results/episode_$E6.parquet 2>/dev/null | sort -V); do
    $P anch.py $f $M $ANCHOR | $P -c "
import json,sys; d=json.loads(sys.stdin.read()); s=d['rollout'].split('/')
print(f\"{s[1]:22s} {s[2]:18s} rot0 {d['anchor_f0_rot_deg']:5.2f}  plain {d['plain']['auc']:.4f} {d['plain']['per_body']}  anchored {d['anchored']['auc']:.4f} {d['anchored']['per_body']}\")"
  done
done
