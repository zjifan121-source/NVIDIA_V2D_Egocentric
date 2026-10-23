#!/bin/bash
# Diagnostic FlashCHORD run: train + official evaluator, keeping training curves (W&B offline).
# Usage: sbatch --export=ALL,NAME=ep41_contacts,PARQUET=<partition>,EXTRA="sim.njmax=2048 ..."[,EP=41][,STEPS=...] train_diag.sbatch
# Preemption-safe (--requeue): training resumes from the newest state_*.safetensors (written
# atomically) with NVIDIA's compact resume (no replay buffer; it refills); logs append; each
# stage leaves a .done marker. A resume starts a new W&B offline run; read_curves.py merges them.
# EP set, SCORE!=0 and motion_speed 1.0 -> also scores with tools/track3/score_local.py (public episodes only).
#SBATCH -J train-diag
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S|A6000|6000Ada|VRAM_80GB|VRAM_141GB
# Contact buffer by card unless EXTRA sets sim.njmax: >= 48 GB -> njmax 1024 (no overflow on our
# meshes); 46 GB L40S -> njmax 768, which fits at 43 GB with no overflow on ep41 (l40s_probe,
# 2026-10-19). 1024 runs out of memory on L40S; 512 drops contacts. Check nefc_overflow_lines.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/%x-%j.log
# Short walltime, one job per run: 4 min before the limit the job stops its step and requeues
# itself (a plain TIMEOUT would not requeue), then resumes from the newest checkpoint.
set -uo pipefail
source /home/$USER/NVIDIA_V2D_Egocentric/tools/track3/jobs/disk_guard.sh  # hold the job when the quota is nearly full
: "${NAME:?set NAME}" "${PARQUET:?set PARQUET}"
STEPS=${STEPS:-250003456}
EXTRA=${EXTRA:-}
REPO=/home/$USER/NVIDIA_V2D_Egocentric
FC=$REPO/baseline/robotic_grounding/flash_chord
KIT=$REPO/assets/data/v2d_submission_kit
PY=/data/user_data/$USER/v2d/venvs/flash_chord/bin/python
TOOLS=$REPO/tools/.venv/bin/python
RUN=/data/user_data/$USER/v2d/flash_runs/diag/$NAME
mkdir -p "$RUN"; cd "$FC"
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_DIR=$RUN WANDB_SILENT=true
if [[ "$EXTRA" != *sim.njmax* ]]; then
  GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if [ "$GPU_MIB" -ge 48000 ]; then EXTRA="$EXTRA sim.njmax=1024 sim.nconmax=512"
  else EXTRA="$EXTRA sim.njmax=768 sim.nconmax=384"; fi
fi
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0} name=$NAME extra=[$EXTRA]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CHILD=
requeue_self() {  # time-limit warning (USR1) or a terminate (TERM, e.g. some preemptions)
  echo "== $(date) signal: stopping and requeueing (restart=${SLURM_RESTART_COUNT:-0})"
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null && wait "$CHILD" 2>/dev/null
  # scancel also sends TERM: do not undo a cancel (state reads CANCELLED then)
  scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -q 'JobState=CANCELLED' && { echo "== cancelled: not requeueing"; exit 0; }
  [ "${SLURM_RESTART_COUNT:-0}" -lt 60 ] && scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_self USR1 TERM
# Keep only the 2 newest resume states (166 MB each, one per save interval; resuming reads the newest).
# Unpruned, a 250M-step run holds ~8 GB of them; policy_*.safetensors (5 MB, used by eval) are all kept.
prune_states() { ls -1 "$RUN"/train/state_*.safetensors 2>/dev/null | sort -V | head -n -2 | xargs -r rm -f; }

if [ ! -f "$RUN/train.done" ]; then
  prune_states
  RESUME=()
  STATE=$(ls -1 "$RUN"/train/state_*.safetensors 2>/dev/null | sort -V | tail -1)
  [ -n "$STATE" ] && RESUME=(training.resume=true "training.checkpoint_path='$STATE'") && echo "== resuming from $STATE"
  SECONDS=0
  # shellcheck disable=SC2086
  $PY scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$PARQUET'" reset.seed=${SEED:-42} \
    logging.mode=offline logging.upload_checkpoints=false logging.run_name=$NAME \
    training.total_environment_steps=$STEPS output_dir="$RUN/train" $EXTRA "${RESUME[@]}" >> "$RUN/train.log" 2>&1 &
  CHILD=$!
  ( while kill -0 "$CHILD" 2>/dev/null; do prune_states; sleep 60; done ) &
  wait "$CHILD"; rc=$?; CHILD=; prune_states
  echo "== train exit=$rc wall_seconds=$SECONDS nefc_overflow_lines=$(grep -c 'nefc overflow' "$RUN/train.log")"
  grep -v 'nefc overflow' "$RUN/train.log" | tail -15
  [ $rc -eq 0 ] && touch "$RUN/train.done" || exit $rc
fi

if [ ! -f "$RUN/eval.done" ]; then
  rm -rf "$RUN/results"  # a preempted evaluation restarts from scratch
  CKPT=$(ls -1 "$RUN"/train/policy_*.safetensors | sort -V | tail -1)
  echo "{\"${EP:-0}\": \"$CKPT\"}" > "$RUN/checkpoints.json"
  $PY "$KIT/examples/run_policy_evaluation.py" --checkpoints "$RUN/checkpoints.json" --results "$RUN/results" \
    --runtime-root "$FC" >> "$RUN/eval.log" 2>&1 &
  CHILD=$!; wait "$CHILD"; rc=$?; CHILD=; echo "== eval exit=$rc nefc_overflow_lines=$(grep -c 'nefc overflow' "$RUN/eval.log")"
  grep -v 'nefc overflow' "$RUN/eval.log" | grep -E '"evaluation/(add_auc|chord_sr|object_position_error_m|object_orientation_error_deg|mppe_cm)"'
  [ $rc -eq 0 ] && touch "$RUN/eval.done" || exit $rc
fi

if [ -n "${EP:-}" ] && [ "${SCORE:-1}" = 1 ] && [[ "$EXTRA" != *motion_speed=0.5* ]]; then
  # the meshes of the scene this run trained on (one episode can have several scene variants)
  $TOOLS "$REPO/tools/track3/score_local.py" --episodes "$EP" --results "$RUN/results" \
    --meshes "${PARQUET%%/assets/*}/score_input/meshes" 2>&1 | tail -2
fi
echo "== $(date) done"
