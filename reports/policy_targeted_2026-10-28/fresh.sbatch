#!/bin/bash
# Fixed Edmond data/code roots allow a selected job to run as either authorized account.
#SBATCH --job-name=v2d-fresh-train
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|6000Ada|H100|H200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append
#SBATCH --output=/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/train-%j.log
set -euo pipefail
: "${NAME:?set NAME}" "${EP:?set EP}" "${PARQUET:?set PARQUET}" "${EXTRA:?set EXTRA}"
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
DATA=/data/user_data/edmondwa/v2d
PY=$DATA/venvs/flash_chord/bin/python
HERE=$REPO/reports/policy_targeted_2026-10-28
RUN=$DATA/flash_runs/diag/$NAME
PAIR=$REPO/reports/submission_candidate_2026-10-28/checkpoint_pairs.py
free_gb=$(df -BG --output=avail /data/user_data/edmondwa | tail -1 | tr -dc 0-9)
if [ "${free_gb:-0}" -lt 30 ]; then
  echo "Less than 30GB free on Edmond data; holding this new job."
  scontrol requeuehold "$SLURM_JOB_ID"
  exit 0
fi
mkdir -p "$RUN/train"
exec 9>"$RUN/.writer.lock"
flock -n 9 || { echo 'Another worker owns this run'; exit 1; }
cd "$REPO/baseline/robotic_grounding/flash_chord"
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_DIR=$RUN WANDB_SILENT=true
CHILD=
PRUNER=
stop_worker() {
  trap '' TERM USR1
  if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2>/dev/null || true; wait "$CHILD" || true; fi
  if [ -n "$PRUNER" ]; then kill "$PRUNER" 2>/dev/null || true; fi
}
on_term() { stop_worker; exit 143; }
on_time() { stop_worker; scontrol requeue "$SLURM_JOB_ID"; exit 0; }
trap on_term TERM
trap on_time USR1
echo "== $(date) account=$USER node=$(hostname) name=$NAME extra=$EXTRA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ ! -f "$RUN/train.done" ]; then
  STATE=$(ls -1 "$RUN"/train/state_*.safetensors 2>/dev/null | sort -V | tail -1 || true)
  REPLAY=${STATE/state_/replay_}
  [ -n "$STATE" ] && [ -f "$REPLAY" ] && REPLAY_ARG="training.checkpoint.load_replay_path='$REPLAY'" || REPLAY_ARG=training.checkpoint.load_replay_path=null
  RESUME=()
  if [ -n "$STATE" ]; then
    RESUME=(training.resume=true "training.checkpoint_path='$STATE'" "$REPLAY_ARG")
  elif [ -n "${INIT_STATE:-}" ]; then
    
    RESUME=(training.resume=true "training.checkpoint_path='$INIT_STATE'" training.checkpoint.load_replay_path=null)
  fi
  # EXTRA is an explicit list of Hydra overrides stored in the immutable trial manifest.
  # shellcheck disable=SC2086
  "$PY" scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$PARQUET'" \
    reset.seed=${SEED:-42} curriculum=${CURRICULUM:-tracking_100m_object2x} \
    training.checkpoint.save_replay=${SAVE_REPLAY:-false} logging.mode=offline logging.upload_checkpoints=false \
    "logging.run_name=$NAME" "output_dir=$RUN/train" training.total_environment_steps=${STEPS:-100003840} \
    $EXTRA "${RESUME[@]}" >> "$RUN/train.log" 2>&1 &
  CHILD=$!
  # keep only the two newest learner states (policies are kept); a run otherwise grows by ~165 MB per 5M steps
  ( while kill -0 "$CHILD" 2>/dev/null; do "$PY" "$PAIR" prune "$RUN/train"; ls -1t "$RUN"/train/state_*.safetensors 2>/dev/null | tail -n +3 | xargs -r rm -f; ls -1t "$RUN"/train/replay_*.safetensors 2>/dev/null | tail -n +2 | xargs -r rm -f; sleep 60; done ) &
  PRUNER=$!
  rc=0
  wait "$CHILD" || rc=$?
  CHILD=
  kill "$PRUNER" 2>/dev/null || true
  PRUNER=
  "$PY" "$PAIR" prune "$RUN/train"
  [ "$rc" -eq 0 ] || exit "$rc"
  touch "$RUN/train.done"
fi
"$PY" "$HERE/evaluate_checkpoints.py" --episode "$EP" --run "$RUN" >> "$RUN/checkpoint_evaluations.log" 2>&1 &
CHILD=$!
wait "$CHILD"
CHILD=
echo "== $(date) training and checkpoint evaluations complete"
