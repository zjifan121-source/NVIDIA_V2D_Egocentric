#!/bin/bash
# Auto-packer for ep28_holdold_kenf16_s203 (fork 2's request, 17:05 UTC). Every 2 min: read the review proxy records for the run,
# keep checkpoints whose three gains (gated, camera, emulator) are all positive, pick the largest emulator gain; if it is a new
# step, pack it on the current base (pack_only.sh with PACK_STEP), run the kit uploader in --dry-run (validation and preflight
# only, nothing submitted), record the merged parquet sha256, and write PACK_REPORT.json. Stops after the 100M row is judged or
# at 19:45 UTC (fork 2, 17:20). Never uploads: UPLOAD_HOLD stands.
RUN=ep28_holdold_kenf16_s203; EP=28; SCENE=/root/v2d/lead_v3/handspin28oldr2com/scenes/ep28_hold
OUT=/root/probes/ken28_packs; mkdir -p $OUT; LAST=$(/root/venv-cpu/bin/python -c "import json;print(json.load(open('$OUT/PACK_REPORT.json'))['step'])" 2>/dev/null)   # resume without re-packing
K=/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit
while true; do
  PICK=$(/root/venv-cpu/bin/python - <<PY
import json
best=None
for l in open('/root/probes/incoming/proxy_scores.jsonl'):
    r=json.loads(l)
    if r.get('run')!='$RUN': continue
    g=[r.get('gated_gain'),r.get('camera_gain'),r.get('emulator_gain')]
    if any(x is None for x in g): continue
    ok=all(x>0 for x in g)
    if ok and (best is None or r['emulator_gain']>best['emulator_gain']): best=r
if best: print(best['step'],best['gated_gain'],best['camera_gain'],best['emulator_gain'],best.get('own_ref'),best.get('proxy'),best.get('emulator_auc'))
PY
)
  STEP=$(echo "$PICK" | cut -d" " -f1)
  if [ -n "$STEP" ] && [ "$STEP" != "$LAST" ]; then
    echo "$(date -u +%FT%TZ) packing step $STEP gains: $PICK" >> $OUT/packer.log
    PACK_STEP=$STEP /root/probes/pack_only.sh $RUN $EP $SCENE "old pot + hold + COM, KENf16 s203, judged yes" >> $OUT/pack_$STEP.log 2>&1
    D=/root/probes/${RUN}_${STEP}; M=$D/${RUN}_${STEP}_merged.parquet
    if [ -f "$M" ]; then
      SHA=$(sha256sum $M | cut -d" " -f1)
      (cd $K && timeout 600 /root/venv-kit/bin/python -u submit/v2d_submit.py $M --track track_3 --dry-run -m "dry-run only" > $D/dryrun_six.log 2>&1; echo "dryrun rc=$?" >> $D/dryrun_six.log)
      DR=$(grep -E "looks valid|dry-run: nothing submitted|Preflight failed|identity is not the same|rc=" $D/dryrun_six.log | tr "\n" " " | cut -c1-300)
      /root/venv-cpu/bin/python - <<PY
import json,datetime
p="$PICK".split()
rep=dict(run="$RUN",step=int(p[0]),gated_gain=float(p[1]),camera_gain=float(p[2]),emulator_gain=float(p[3]),own_ref=float(p[4]),proxy=float(p[5]),emulator_auc=float(p[6]),
         merged_parquet="$M",merged_sha256="$SHA",single_swap_parquet="$D/${RUN}_${STEP}.parquet",dryrun="$DR",uploaded=False,
         base="raw bob1 + incumbents from INCUMBENTS.json (t2/t5/t8/t6), only ep28 replaced",written_utc=datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
json.dump(rep,open("$OUT/PACK_REPORT.json","w"),indent=1)
open("$OUT/PACK_HISTORY.jsonl","a").write(json.dumps(rep)+"\n")
PY
      LAST=$STEP; echo "$(date -u +%FT%TZ) packed step $STEP sha $SHA dryrun: $DR" >> $OUT/packer.log
    else
      echo "$(date -u +%FT%TZ) PACK FAILED for step $STEP (see pack_$STEP.log)" >> $OUT/packer.log
    fi
  fi
  grep -q "\"run\": \"$RUN\", \"step\": 100003840" /root/probes/incoming/proxy_scores.jsonl && [ -n "$LAST" ] && { echo "$(date -u +%FT%TZ) 100M judged; stopping" >> $OUT/packer.log; exit 0; }
  [ "$(date -u +%H%M)" -ge 1945 ] && { echo "$(date -u +%FT%TZ) 19:45 stop" >> $OUT/packer.log; exit 0; }
  sleep 120
done
