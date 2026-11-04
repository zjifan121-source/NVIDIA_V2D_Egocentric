#!/bin/bash
# kq.sh - orchard V2D queue, 3-4 Nov night. Runs Edmond's training protocol (TRAINING_GUIDE.md / campaign.py @ de66fea3):
#   recipes KEN / DEF / WIDE exactly as campaign.py RECIPES (ep10 DEF/WIDE: njmax 2048), 100,003,840 steps,
#   curriculum tracking_100m_object2x, flash_chord code = de66fea3 (non-finite replay guard) via PYTHONPATH,
#   then evaluates every checkpoint from 70M (official evaluator, own-reference add_auc) and records the best.
# Usage (inside an srun --overlap step of job 157612): bash kq.sh QUEUE GPU [GPU...]
# Queue line: NAME EP PARQUET SEED RECIPE [PRIO]   lowest PRIO first (default 50), file order on ties.
#   RECIPE=SWEEP: no training, only the checkpoint sweep of the existing run $RUNS/NAME (PARQUET/SEED ignored);
#   SWEEP lines go to one sweeper that uses whichever GPU has the most free memory (evals take ~20 s each).
# A training worker starts a task only when its GPU has < 3 GB in use, so it shares GPUs safely with the old driver.
# Per run $RUNS/NAME: train/ (policies every 5M), train.log, early_<step>/results (evaluator output),
#   sweep.json (all evaluated checkpoints, best first), best.txt ("step auc per_body"), recipe.txt.
set -uo pipefail
Q=$(realpath "$1"); shift
B=/tmp/haisenk_v2d
REPO=$B/v2d_train
FC=$REPO/baseline/robotic_grounding/flash_chord
KIT=$REPO/assets/data/v2d_submission_kit
FC_PY=$FC/.venv/bin/python
export PYTHONPATH=$B/fc_src_de66
RUNS=${RUNS:-$B/runs}
STEPS=100003840
CUR=tracking_100m_object2x
COMMON="task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 training.checkpoint.save_interval_environment_steps=5000192"
DEF="termination.object_position.threshold=0.3 termination.object_orientation.threshold=1.5 training.replay.capacity=1048576 training.replay.observation_storage_dtype=float16"
KEN="termination.object_position.threshold=0.1 termination.object_orientation.threshold=0.5 training.replay.capacity=4194304 training.replay.observation_storage_dtype=float32"
WIDEX="action.normalized_mapping=linear training.network.initial_normalized_std=0.3 training.network.log_std_max=-0.5108256237659907 training.exploration.target_sigma=0.3"
recipe() {  # RECIPE EP -> Hydra overrides (campaign.py: DEFAULT, BIG={10}, KEN subs, WIDE extra)
  local nj="sim.njmax=1024 sim.nconmax=512"; [ "$2" = 10 ] && nj="sim.njmax=2048 sim.nconmax=1024"
  case $1 in
    KEN) echo "$COMMON $KEN sim.njmax=1024 sim.nconmax=512";;
    DEF) echo "$COMMON $DEF $nj";;
    WIDE) echo "$COMMON $DEF $nj $WIDEX";;
    *) echo "";;
  esac
}
mkdir -p "$RUNS"; touch "$Q.taken"
next_task() {  # MODE (train|sweep): print the next unstarted line of that kind and mark it taken
  (
    flock 9
    local best="" bp="" name ep pq seed rec prio
    while read -r name ep pq seed rec prio; do
      if [ -z "$name" ] || [ "${name:0:1}" = "#" ]; then continue; fi
      if [ "$1" = sweep ]; then [ "$rec" = SWEEP ] || continue; else [ "$rec" = SWEEP ] && continue; fi
      grep -qxF "$name" "$Q.taken" && continue
      prio=${prio:-50}
      if [ -z "$best" ] || [ "$prio" -lt "$bp" ]; then best="$name $ep $pq $seed $rec"; bp=$prio; fi
    done < "$Q"
    [ -n "$best" ] && echo "${best%% *}" >> "$Q.taken" && echo "$best"
  ) 9>"$Q.lock"
}
gpu_used() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
gpu_free() { local u; u=$(gpu_used "$1"); [ -n "$u" ] && [ "$u" -lt 3000 ]; }
roomiest_gpu() {  # the listed GPU with the least memory in use, if under 60 GB
  local g u bu=999999 bg=""
  for g in "$@"; do u=$(gpu_used "$g"); if [ -n "$u" ] && [ "$u" -lt "$bu" ]; then bu=$u; bg=$g; fi; done
  if [ -n "$bg" ] && [ "$bu" -lt 60000 ]; then echo "$bg"; fi
}
prune_states() { ls -1 "$1"/train/state_*.safetensors 2>/dev/null | sort -V | head -n -2 | xargs -r rm -f; }
sweep() {  # RUN EP : evaluate every policy from 70M, write sweep.json + best.txt
  local RUN=$1 EP=$2 P S OUT
  for P in $(ls -1 "$RUN"/train/policy_*.safetensors 2>/dev/null | sort -V); do
    S=${P##*policy_}; S=${S%%.*}
    [ "$S" -ge 70000000 ] || continue
    OUT=$RUN/early_$S; mkdir -p "$OUT"
    [ -f "$OUT/eval.done" ] && continue
    echo "{\"$EP\": \"$P\"}" > "$OUT/checkpoints.json"
    "$FC_PY" "$KIT/examples/run_policy_evaluation.py" --checkpoints "$OUT/checkpoints.json" \
      --results "$OUT/results" --runtime-root "$FC" > "$OUT/eval.log" 2>&1 && touch "$OUT/eval.done"
  done
  python3 - "$RUN" "$EP" <<'PY'
import glob, json, os, sys
run, ep = sys.argv[1], int(sys.argv[2])
rows = []
for d in glob.glob(f"{run}/early_*"):
    f = f"{d}/results/episode_{ep:06d}.json"
    if os.path.exists(f):
        m = json.load(open(f))["metrics"]
        rows.append(dict(step=int(d.rsplit("_", 1)[1]), auc=m["add_auc"], per_body=m.get("add_auc_per_body"), dir=d))
rows.sort(key=lambda r: -r["auc"])
json.dump(rows, open(f"{run}/sweep.json", "w"), indent=1)
if rows:
    b = rows[0]
    last = max(rows, key=lambda r: r["step"])
    open(f"{run}/best.txt", "w").write(f"{b['step']} {b['auc']:.4f} {[round(x, 3) for x in (b['per_body'] or [])]}\n")
    print(f"  sweep {os.path.basename(run)}: best {b['auc']:.3f} @ {b['step']} {[round(x, 3) for x in (b['per_body'] or [])]}"
          f" (last {last['auc']:.3f} @ {last['step']}, {len(rows)} ckpts)")
PY
}
task() {  # NAME EP PARQUET SEED RECIPE GPU
  local NAME=$1 EP=$2 PARQUET=$3 SEED=$4 REC=$5 GPU=$6 RUN=$RUNS/$1
  export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_DIR=$RUN WANDB_SILENT=true
  mkdir -p "$RUN"
  echo "$(date '+%F %T') gpu$GPU start $NAME (ep$EP seed $SEED $REC)"
  if [ "$REC" != SWEEP ]; then
    local OV; OV=$(recipe "$REC" "$EP")
    [ -n "$OV" ] || { echo "  $NAME: unknown recipe $REC"; touch "$RUN/FAILED"; return; }
    [ -d "$PARQUET" ] || { echo "  $NAME: no scene at $PARQUET"; touch "$RUN/FAILED"; return; }
    echo "$REC $OV" > "$RUN/recipe.txt"
    if [ ! -f "$RUN/train.done" ]; then
      prune_states "$RUN"
      local RESUME=() STATE
      STATE=$(ls -1 "$RUN"/train/state_*.safetensors 2>/dev/null | sort -V | tail -1)
      [ -n "$STATE" ] && RESUME=(training.resume=true "training.checkpoint_path='$STATE'") && echo "  resuming from $STATE"
      SECONDS=0
      # shellcheck disable=SC2086
      (cd "$FC" && "$FC_PY" scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$PARQUET'" \
        reset.seed=$SEED curriculum=$CUR training.checkpoint.save_replay=false logging.mode=offline \
        logging.upload_checkpoints=false logging.run_name=$NAME output_dir="$RUN/train" \
        training.total_environment_steps=$STEPS $OV "${RESUME[@]}" >> "$RUN/train.log" 2>&1) &
      local CHILD=$!
      while kill -0 $CHILD 2>/dev/null; do prune_states "$RUN"; sleep 30; done
      wait $CHILD; local rc=$?
      prune_states "$RUN"
      echo "  $NAME train exit=$rc minutes=$((SECONDS / 60)) nefc_overflow_lines=$(grep -c 'nefc overflow' "$RUN/train.log" 2>/dev/null)"
      [ $rc -eq 0 ] || { tail -5 "$RUN/train.log"; touch "$RUN/FAILED"; return; }
      touch "$RUN/train.done"
    fi
  fi
  sweep "$RUN" "$EP"
  rm -f "$RUN/FAILED"
  echo "$(date '+%F %T') gpu$GPU done $NAME"
}
worker() {  # one per GPU: training lines
  local GPU=$1 LINE
  while true; do
    if ! gpu_free "$GPU"; then sleep 60; continue; fi
    LINE=$(next_task train)
    if [ -z "$LINE" ]; then
      [ -f "$Q.stop" ] && { echo "$(date '+%F %T') gpu$GPU: queue empty, $Q.stop exists, exiting"; return; }
      sleep 60; continue
    fi
    # shellcheck disable=SC2086
    set -- $LINE
    task "$1" "$2" "$3" "$4" "$5" "$GPU"
  done
}
sweeper() {  # SWEEP lines only, on whichever listed GPU has the most free memory
  local LINE G
  while true; do
    G=$(roomiest_gpu $ALLG); [ -n "$G" ] || { sleep 60; continue; }
    LINE=$(next_task sweep)
    if [ -z "$LINE" ]; then [ -f "$Q.stop" ] && return; sleep 60; continue; fi
    # shellcheck disable=SC2086
    set -- $LINE
    task "$1" "$2" "$3" "$4" "$5" "$G"
  done
}
echo "== $(date) kq queue $Q runs $RUNS GPUs $* PYTHONPATH=$PYTHONPATH"
ALLG="$*"
for G in "$@"; do worker "$G" & sleep 2; done
sweeper &
wait
echo "== $(date) all kq workers finished"
