#!/bin/bash
# Early check of training runs: roll out the policy checkpoint nearest below STEP (one deterministic
# rollout, helper force off, as the evaluator does) and score it like train_diag.sbatch. Public episodes
# get the official metric against ground truth; half-speed runs are converted with score_rollout.py.
# Results go to <run>/early_<step>/. Compare runs at the same STEP (the curriculum is the same per step).
# Usage: STEP=150000000 sbatch eval_ckpt.sbatch <episode> <scene dir> <run name> [<run name> ...]
#SBATCH -J eval-ckpt
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|6000Ada|VRAM_80GB|VRAM_141GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/%x-%j.log
set -uo pipefail
EP=${1:?episode}; SCENE=${2:?scene dir}; shift 2
REPO=/home/$USER/NVIDIA_V2D_Egocentric
FC=$REPO/baseline/robotic_grounding/flash_chord
KIT=$REPO/assets/data/v2d_submission_kit
PY=/data/user_data/$USER/v2d/venvs/flash_chord/bin/python
TOOLS=$REPO/tools/.venv/bin/python
STEP=${STEP:?set STEP}
cd "$FC"
for NAME in "$@"; do
  RUN=/data/user_data/$USER/v2d/flash_runs/diag/$NAME
  CKPT=$(ls -1 "$RUN"/train/policy_*.safetensors | sort -V | awk -F'policy_|[.]safetensors' -v s="$STEP" '$2+0 <= s+0' | tail -1)
  [ -z "$CKPT" ] && { echo "$NAME: no checkpoint at or below $STEP yet"; continue; }
  K=$(basename "$CKPT" .safetensors); OUT=$RUN/early_${K#policy_}
  if [ ! -f "$OUT/done" ]; then
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "{\"$EP\": \"$CKPT\"}" > "$OUT/checkpoints.json"
    $PY "$KIT/examples/run_policy_evaluation.py" --checkpoints "$OUT/checkpoints.json" --results "$OUT/results" \
      --runtime-root "$FC" > "$OUT/eval.log" 2>&1 && touch "$OUT/done"
  fi
  if [[ "$NAME" == *half* ]]; then  # half-speed runs (named *half*): keep the even rows, then score
    S=$(cd "$REPO" && $TOOLS tools/track3/score_rollout.py "$OUT" "$EP" "$(basename "$SCENE")" 2>&1 | tail -1)
  else
    S=$($TOOLS "$REPO/tools/track3/score_local.py" --episodes "$EP" --results "$OUT/results" --meshes "$SCENE/score_input/meshes" 2>&1 | grep -A1 '^case' | tail -1)
  fi
  echo "EARLY ep$EP $NAME @${K#policy_}: $S"
done
