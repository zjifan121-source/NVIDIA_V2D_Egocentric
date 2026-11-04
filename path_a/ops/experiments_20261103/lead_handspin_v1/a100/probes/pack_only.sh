#!/bin/bash
# Single-episode swap probe: bob1 with one episode replaced by the best evaluated checkpoint of one run.
# usage: pack_only.sh <run_name> <episode> <scene_dir> "<label>"   (check + pack + merge onto the current base; no upload, no Kaggle API call)
# Without --submit nothing is uploaded: plan, check, pack and a dry run only. check_probe_plan exit 0 is required.
set -u
NAME=$1; EP=$2; SCENE=$3; LABEL=$4; MODE=${5:-}
M=/root/v2d/NVIDIA_V2D_Egocentric; TL=/root/v2d/tools_lead; K=$M/assets/data/v2d_submission_kit
COMMIT=https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/f377f3b235f4ba8188d94f0025090ea65ab85ffc
REF=/root/bob1_meta/experiments/edmond/bob1_submission_20261103/candidate_bob1.plan.json
read STEP AUC EARLY PB <<< $(/root/venv-cpu/bin/python -c "
import json
rows=json.load(open('/root/runs/$NAME/checkpoint_selection.json'))
import os
st=os.environ.get('PACK_STEP')
d=next((r for r in rows if st and int(r['step'])==int(st)),None) if st else rows[0]
assert d is not None, 'PACK_STEP %s not evaluated yet' % st
print(d['step'],d['auc'],d['run'],','.join(str(round(x,3)) for x in d['per_body']))")
ID=${NAME}_${STEP}; W=/root/probes/$ID; mkdir -p $W
/root/venv-cpu/bin/python - <<PY
import json
p=json.load(open("$TL/bob1_local.plan.json"))
old=p["$EP"]; p["$EP"]=dict(run="$EARLY",scene="$SCENE",kind="policy",tracking=$AUC,name="$NAME/early_$STEP $LABEL",source="wentao")
json.dump(p,open("$W/plan.json","w"),indent=1)
print("swap ep$EP: bob1 tracking",round(old["tracking"],4),"-> own-ref",round($AUC,4),"per_body $PB step $STEP")
PY
V2D_ROOT=/root/v2d /root/venv-cpu/bin/python -I $TL/check_probe_plan.py $W/plan.json --ref $REF --expect-changes 1 --code-commit $COMMIT --out $W/plan.check.json > $W/check.log 2>&1; RC=$?
grep -E "warn: ep$EP\b|warn: ep$EP:|PASS|FAIL|summary|expected" $W/check.log | cut -c1-230 | tail -10
[ $RC -eq 0 ] || { echo "CHECK_FAILED rc=$RC"; exit 3; }
cd $M && /root/venv-cpu/bin/python tools/track3/pack_submission_handoff.py --plan $W/plan.json --work $W/work --code-commit $COMMIT --python /root/venv-kit/bin/python -o $W/$ID.parquet 2>&1 | tail -3 | cut -c1-170
[ -f $W/$ID.parquet ] || { echo "PACK_FAILED"; exit 4; }
# Merge from raw bob1 with every current incumbent, per merge_swaps.py: a swap must be a one-episode change of its base.
BOB1=/root/probes/incumbents/raw/57934736/57934736.raw
SWAPS=$(/root/venv-cpu/bin/python /root/probes/base_swaps.py $EP)
echo "base swaps: $(echo $SWAPS | sed 's#/root/probes/incumbents/raw/[0-9]*/##g; s#--swap ##g')"
OUT=$W/${ID}_merged.parquet
/root/venv-kit/bin/python /root/v2d/tools_lead/merge_swaps.py --base $BOB1 $SWAPS --swap $EP=$W/$ID.parquet --out $OUT --code-commit $COMMIT > $W/merge.log 2>&1; RC=$?
tail -3 $W/merge.log | cut -c1-200
[ $RC -eq 0 ] && [ -f $OUT ] || { echo "MERGE_FAILED rc=$RC"; exit 5; }
MSG="Wentao probe: current base + ep$EP <- $NAME@$STEP ($LABEL, own-ref $(printf %.3f $AUC))"
echo "MSG=$MSG"
echo "PACK_ONLY: merged file ready, nothing uploaded: $OUT"
echo PROBE_END $ID
