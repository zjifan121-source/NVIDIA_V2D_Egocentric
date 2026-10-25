#!/bin/bash
# Train + evaluate FlashCHORD policies on a machine without Slurm: one worker per GPU pulls runs from a queue file.
# Usage: RUNS=<run folders> gpu_queue.sh <queue file> <gpu id> [<gpu id> ...]
# Queue lines (append at any time; '#' starts a comment):
#   NAME EP PARQUET SEED [SCORE] [NJMAX]
#   SCORE=1 also scores the rollout against public ground truth (public episodes only); NJMAX is the contact
#   buffer (default 1024; 2048 for scenes whose train.log reports 'nefc overflow').
# A worker takes the unstarted line (names in <queue>.taken are started) with the lowest SEED, earliest in the
# file on a tie, so every episode gets its first seed before any gets a second. It trains with the default
# recipe (curriculum tracking_100m_object2x, 100,003,840 steps), runs the official evaluator, and moves on.
# An empty queue is polled every 60 s until <queue>.stop exists. A run that crashes resumes from its newest
# state when its name is removed from <queue>.taken; finished stages (train.done, eval.done) are skipped.
# Per run, under $RUNS/NAME: train/ (policies, 2 newest resume states), train.log, results/ (evaluator output:
# episode_XXXXXX.parquet + .json, whose metrics.add_auc is the own-reference score), eval.log, score.txt.
# Env: FC_PY (FlashCHORD python, default <flash_chord>/.venv/bin/python), STEPS, CURRICULUM, DRY=1 (print only).
set -uo pipefail
Q=$(realpath "${1:?queue file}"); shift
[ $# -ge 1 ] || { echo "give at least one GPU id"; exit 1; }
: "${RUNS:?set RUNS to the folder for run outputs}"
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
FC=$REPO/baseline/robotic_grounding/flash_chord
KIT=$REPO/assets/data/v2d_submission_kit
FC_PY=${FC_PY:-$FC/.venv/bin/python}
STEPS=${STEPS:-100003840}
CURRICULUM=${CURRICULUM:-tracking_100m_object2x}
[ $((STEPS % 4096)) -eq 0 ] || { echo "STEPS must be a multiple of 4096 (the parallel environments), got $STEPS"; exit 1; }
RECIPE="curriculum=$CURRICULUM task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 training.checkpoint.save_interval_environment_steps=5000192"
mkdir -p "$RUNS"; touch "$Q.taken"

run() {  # run a command, or print it with DRY=1
  if [ "${DRY:-0}" = 1 ]; then echo "    [dry] $*"; sleep 2; return 0; fi
  "$@"
}

next_task() {  # print the next unstarted queue line and mark it taken (under a lock shared by all workers)
  (
    flock 9
    local best="" best_seed="" name ep parquet seed rest
    while read -r name ep parquet seed rest; do
      { [ -z "$name" ] || [[ "$name" == \#* ]]; } && continue
      grep -qxF "$name" "$Q.taken" && continue
      if [ -z "$best" ] || [ "$seed" -lt "$best_seed" ]; then best="$name $ep $parquet $seed $rest"; best_seed=$seed; fi
    done < "$Q"
    [ -n "$best" ] && echo "${best%% *}" >> "$Q.taken" && echo "$best"
  ) 9>"$Q.lock"
}

prune_states() { ls -1 "$1"/train/state_*.safetensors 2>/dev/null | sort -V | head -n -2 | xargs -r rm -f; }

task() {  # NAME EP PARQUET SEED SCORE NJMAX GPU
  local NAME=$1 EP=$2 PARQUET=$3 SEED=$4 SCORE=${5:-0} NJ=${6:-1024} GPU=$7
  local RUN=$RUNS/$NAME
  mkdir -p "$RUN"
  echo "$(date '+%F %T') gpu$GPU start $NAME (ep$EP seed $SEED)"
  [ -d "$PARQUET" ] || { echo "  $NAME: no scene at $PARQUET"; touch "$RUN/FAILED"; return; }
  export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_DIR=$RUN WANDB_SILENT=true
  if [ ! -f "$RUN/train.done" ]; then
    prune_states "$RUN"
    local RESUME=() STATE
    STATE=$(ls -1 "$RUN"/train/state_*.safetensors 2>/dev/null | sort -V | tail -1)
    [ -n "$STATE" ] && RESUME=(training.resume=true "training.checkpoint_path='$STATE'") && echo "  resuming from $STATE"
    SECONDS=0
    # shellcheck disable=SC2086
    (cd "$FC" && run "$FC_PY" scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$PARQUET'" \
      reset.seed=$SEED logging.mode=offline logging.upload_checkpoints=false logging.run_name=$NAME \
      training.total_environment_steps=$STEPS output_dir="$RUN/train" $RECIPE sim.njmax=$NJ sim.nconmax=$((NJ / 2)) \
      "${RESUME[@]}" >> "$RUN/train.log" 2>&1) &
    local CHILD=$!
    while kill -0 $CHILD 2>/dev/null; do prune_states "$RUN"; sleep 30; done
    wait $CHILD; local rc=$?
    prune_states "$RUN"
    echo "  $NAME train exit=$rc minutes=$((SECONDS / 60)) nefc_overflow_lines=$(grep -c 'nefc overflow' "$RUN/train.log" 2>/dev/null)"
    [ $rc -eq 0 ] || { tail -5 "$RUN/train.log"; touch "$RUN/FAILED"; return; }
    touch "$RUN/train.done"
  fi
  if [ ! -f "$RUN/eval.done" ]; then
    rm -rf "$RUN/results"
    local CKPT
    CKPT=$(ls -1 "$RUN"/train/policy_*.safetensors 2>/dev/null | sort -V | tail -1)
    [ "${DRY:-0}" = 1 ] && CKPT=$RUN/train/policy_dry.safetensors
    echo "{\"$EP\": \"$CKPT\"}" > "$RUN/checkpoints.json"
    run "$FC_PY" "$KIT/examples/run_policy_evaluation.py" --checkpoints "$RUN/checkpoints.json" \
      --results "$RUN/results" --runtime-root "$FC" >> "$RUN/eval.log" 2>&1
    local rc=$?
    [ $rc -eq 0 ] || { echo "  $NAME eval exit=$rc"; tail -5 "$RUN/eval.log"; touch "$RUN/FAILED"; return; }
    touch "$RUN/eval.done"
  fi
  local OWN
  OWN=$(python3 -c "import json,sys; print(round(json.load(open(sys.argv[1]))['metrics']['add_auc'], 3))" \
        "$RUN/results/episode_$(printf %06d "$EP").json" 2>/dev/null || echo "?")
  echo "  $NAME own-reference add_auc=$OWN"
  if [ "$SCORE" = 1 ]; then
    (cd "$REPO" && run uv run --project tools python tools/track3/score_local.py --episodes "$EP" \
      --results "$RUN/results" --meshes "${PARQUET%%/assets/*}/score_input/meshes" > "$RUN/score.txt" 2>&1)
    echo "  $NAME public score: $(tail -2 "$RUN/score.txt" | tr '\n' ' ')"
  fi
  rm -f "$RUN/FAILED"
  echo "$(date '+%F %T') gpu$GPU done $NAME"
}

worker() {
  local GPU=$1 LINE
  while true; do
    LINE=$(next_task)
    if [ -z "$LINE" ]; then
      [ -f "$Q.stop" ] && { echo "$(date '+%F %T') gpu$GPU: queue empty and $Q.stop exists, exiting"; return; }
      sleep "${POLL:-60}"; continue
    fi
    # shellcheck disable=SC2086
    set -- $LINE
    task "$1" "$2" "$3" "$4" "${5:-0}" "${6:-1024}" "$GPU"
  done
}

echo "== $(date) queue $Q, runs in $RUNS, GPUs $*, recipe: $STEPS steps, $CURRICULUM"
for G in "$@"; do worker "$G" & done
wait
echo "== $(date) all workers finished"
