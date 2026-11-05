#!/bin/bash
# reeval.sh LISTFILE NDRAWS GPU... : re-evaluate fixed checkpoints NDRAWS times each (rollout selection), round-robin over GPUs.
# LISTFILE lines: RUN EP STEP   (policy = /tmp/haisenk_v2d/runs/RUN/train/policy_STEP.safetensors)
# Output: /tmp/haisenk_v2d/runs/RUN/reeval_STEP_d<k>/results/episode_<ep6>.json (+ parquet rollout)
B=/tmp/haisenk_v2d; FC=$B/v2d_train/baseline/robotic_grounding/flash_chord; KIT=$B/v2d_train/assets/data/v2d_submission_kit
FC_PY=$FC/.venv/bin/python; export PYTHONPATH=$B/fc_src_de66 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
L=$1; N=$2; shift 2; GPUS=("$@"); jobs=()
while read -r RUN EP STEP; do [ -z "$RUN" ] && continue
  for k in $(seq 1 $N); do jobs+=("$RUN $EP $STEP $k"); done; done < $L
run_one() { set -- $1; RUN=$1 EP=$2 STEP=$3 K=$4 G=$5; OUT=$B/runs/$RUN/reeval_${STEP}_d$K
  [ -f $OUT/eval.done ] && return; mkdir -p $OUT
  echo "{\"$EP\": \"$B/runs/$RUN/train/policy_$STEP.safetensors\"}" > $OUT/checkpoints.json
  CUDA_VISIBLE_DEVICES=$G "$FC_PY" "$KIT/examples/run_policy_evaluation.py" --checkpoints $OUT/checkpoints.json \
     --results $OUT/results --runtime-root "$FC" > $OUT/eval.log 2>&1 && touch $OUT/eval.done; }
i=0; for j in "${jobs[@]}"; do g=${GPUS[$((i % ${#GPUS[@]}))]}; ( run_one "$j $g" ) & i=$((i+1))
  if [ $((i % (2*${#GPUS[@]}))) -eq 0 ]; then wait; fi; done; wait
python3 - "$L" <<'PY'
import json, glob, sys, os
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) < 3: continue
    run, ep, step = p; vals = []
    for d in sorted(glob.glob(f"/tmp/haisenk_v2d/runs/{run}/reeval_{step}_d*")):
        f = f"{d}/results/episode_{int(ep):06d}.json"
        if os.path.exists(f):
            m = json.load(open(f))["metrics"]; vals.append((round(m["add_auc"], 4), os.path.basename(d)))
    vals.sort(reverse=True)
    print(run, step, "n=%d" % len(vals), "best", vals[:3], "worst", vals[-1:] if vals else None)
PY
