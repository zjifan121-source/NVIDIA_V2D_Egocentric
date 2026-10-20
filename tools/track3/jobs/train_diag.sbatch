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
#SBATCH --constraint=VRAM_80GB|VRAM_96GB|VRAM_141GB
# >= 80 GB: sim.njmax=1024 needs ~46 GB at startup and runs out of memory on 46 GB L40S cards
# (oom_probe.sbatch, 2026-10-15); njmax 512 peaks at 40 GB but drops contacts on our meshes.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/%x-%j.log
set -uo pipefail
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
echo "== $(date) $(hostname) restart=${SLURM_RESTART_COUNT:-0} name=$NAME extra=[$EXTRA]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$RUN/train.done" ]; then
  RESUME=()
  STATE=$(ls -1 "$RUN"/train/state_*.safetensors 2>/dev/null | sort -V | tail -1)
  [ -n "$STATE" ] && RESUME=(training.resume=true "training.checkpoint_path='$STATE'") && echo "== resuming from $STATE"
  SECONDS=0
  # shellcheck disable=SC2086
  $PY scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$PARQUET'" reset.seed=42 \
    logging.mode=offline logging.upload_checkpoints=false logging.run_name=$NAME \
    training.total_environment_steps=$STEPS output_dir="$RUN/train" $EXTRA "${RESUME[@]}" >> "$RUN/train.log" 2>&1
  rc=$?
  echo "== train exit=$rc wall_seconds=$SECONDS nefc_overflow_lines=$(grep -c 'nefc overflow' "$RUN/train.log")"
  grep -v 'nefc overflow' "$RUN/train.log" | tail -15
  [ $rc -eq 0 ] && touch "$RUN/train.done" || exit $rc
fi

if [ ! -f "$RUN/eval.done" ]; then
  rm -rf "$RUN/results"  # a preempted evaluation restarts from scratch
  CKPT=$(ls -1 "$RUN"/train/policy_*.safetensors | sort -V | tail -1)
  echo "{\"${EP:-0}\": \"$CKPT\"}" > "$RUN/checkpoints.json"
  $PY "$KIT/examples/run_policy_evaluation.py" --checkpoints "$RUN/checkpoints.json" --results "$RUN/results" \
    --runtime-root "$FC" >> "$RUN/eval.log" 2>&1
  rc=$?; echo "== eval exit=$rc nefc_overflow_lines=$(grep -c 'nefc overflow' "$RUN/eval.log")"
  grep -v 'nefc overflow' "$RUN/eval.log" | grep -E '"evaluation/(add_auc|chord_sr|object_position_error_m|object_orientation_error_deg|mppe_cm)"'
  [ $rc -eq 0 ] && touch "$RUN/eval.done" || exit $rc
fi

if [ -n "${EP:-}" ] && [ "${SCORE:-1}" = 1 ] && [[ "$EXTRA" != *motion_speed=0.5* ]]; then
  LR=$REPO/baseline/reconstruction/data/outputs/load_retarget/$(printf "episode_%06d" "$EP")
  $TOOLS "$REPO/tools/track3/score_local.py" --episodes "$EP" --results "$RUN/results" \
    --meshes "$(ls -d "$LR"/*/score_input/meshes | head -1)" 2>&1 | tail -2
fi
echo "== $(date) done"
