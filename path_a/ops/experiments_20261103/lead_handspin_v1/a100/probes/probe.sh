#!/bin/bash
[ -f /root/probes/UPLOAD_HOLD ] && { echo "uploads on hold by the user"; exit 4; }
# Single-episode swap probe: bob1 with one episode replaced by the best evaluated checkpoint of one run.
# usage: probe.sh <run_name> <episode> <scene_dir> "<kaggle message label>" [--submit]
# Without --submit nothing is uploaded: plan, check, pack and a dry run only. check_probe_plan exit 0 is required.
set -u
NAME=$1; EP=$2; SCENE=$3; LABEL=$4; MODE=${5:-}
M=/root/v2d/NVIDIA_V2D_Egocentric; TL=/root/v2d/tools_lead; K=$M/assets/data/v2d_submission_kit
COMMIT=https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/f377f3b235f4ba8188d94f0025090ea65ab85ffc
REF=/root/bob1_meta/experiments/edmond/bob1_submission_20261103/candidate_bob1.plan.json
read STEP AUC EARLY PB <<< $(/root/venv-cpu/bin/python -c "
import json
d=json.load(open('/root/runs/$NAME/checkpoint_selection.json'))[0]
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
cd $K
if [ "$MODE" = "--submit" ]; then
  # A preflight stop sends nothing (another session uploading can trip the identity check), so only that case is retried.
  up(){ for i in 1 2 3 4 5 6; do
      /root/venv-kit/bin/python -u submit/v2d_submit.py $OUT --track track_3 --only $1 -m "$MSG" --receipt $2 --wait > $3.try$i.log 2>&1; rc=$?
      tail -12 $3.try$i.log | cut -c1-170; echo "upload rc=$rc try=$i"
      grep -q -E "Preflight failed|identity is not the same" $3.try$i.log || break
      sleep 25
    done; }
  if [ "${ALLOW_AUC:-1}" = 1 ]; then up add_auc $W/receipt_auc.json $W/up_auc; else echo "AUC upload skipped: set ALLOW_AUC=1 only for the file the user names"; fi
  up "spider_sr maniptrans_sr rpe_cm mppe_cm cd_o_cm" $W/receipt_rest.json $W/up_rest
else
  /root/venv-kit/bin/python -u submit/v2d_submit.py $OUT --track track_3 --only add_auc --dry-run -m "$MSG" 2>&1 | tail -6 | cut -c1-170
fi
echo PROBE_END $ID
