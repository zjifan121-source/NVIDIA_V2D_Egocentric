#!/bin/bash
# drv.sh JOBS_FILE TAG GPU... : 2 eval workers per GPU pull lines "EP POLICY DRAW" from JOBS_FILE (node-local counter)
W=/data/user_data/haisenk/v2d/push1104/taskA
FC=/data/user_data/haisenk/v2d/v2d_train/baseline/robotic_grounding/flash_chord; PY=$FC/.venv/bin/python
KIT=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
export PYTHONPATH=/data/user_data/haisenk/v2d/push1104/fc_src_de66 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
[ "${NOPP:-0}" = 1 ] && unset PYTHONPATH
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE
J=$1 TAG=$2; shift 2
mkdir -p /tmp/haisenk_push1104; C=/tmp/haisenk_push1104/$TAG.ctr; [ -f $C ] || echo 0 > $C
N=$(grep -c . $J); PER=${PER:-2}
worker() { local G=$1 i line EP POL D OUT rc
  while true; do
    [ -f $W/STOP_$TAG ] && return
    i=$(flock $C.lock bash -c "i=\$(cat $C); echo \$((i+1)) > $C; echo \$i")
    [ "$i" -ge "$N" ] && return
    line=$(sed -n "$((i+1))p" $J); set -- $line; EP=$1 POL=$2 D=$3
    OUT=$W/draws/ep$EP/$D; [ -f $OUT/eval.done ] && continue
    mkdir -p $OUT; echo "{\"$EP\": \"$POL\"}" > $OUT/checkpoints.json
    SECONDS=0
    CUDA_VISIBLE_DEVICES=$G $PY $KIT/examples/run_policy_evaluation.py --checkpoints $OUT/checkpoints.json \
      --results $OUT/results --runtime-root $FC > $OUT/eval.log 2>&1; rc=$?
    [ $rc -eq 0 ] && touch $OUT/eval.done
    echo "$(date +%T) $(hostname -s) gpu$G ep$EP $D rc=$rc ${SECONDS}s auc=$($PY -c "import json;print(round(json.load(open('$OUT/results/episode_$(printf %06d $EP).json'))['metrics']['add_auc'],4))" 2>/dev/null)"
  done; }
echo "== $(date) drv2 PYTHONPATH=${PYTHONPATH:-none} $TAG jobs=$N gpus=$* per=$PER"
for g in "$@"; do for k in $(seq 1 $PER); do worker $g & sleep 2; done; done; wait; echo "ALL_DONE $TAG $(date +%T)"
