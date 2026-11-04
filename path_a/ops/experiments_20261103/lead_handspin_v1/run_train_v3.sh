#!/bin/bash
# Train one FlashCHORD policy on a lead_v3 scene with Edmond's standard 100M object2x recipe, then evaluate checkpoints.
# usage: run_train_v3.sh <ep> [seed=42] [curriculum=tracking_100m_object2x] [steps=100003840] [extra hydra overrides...]
set -uo pipefail
EP=$1; SEED=${2:-42}; CUR=${3:-tracking_100m_object2x}; STEPS=${4:-100003840}; shift 4 2>/dev/null || shift $#
EXTRA_USER="$*"
REPO=/root/v2d/NVIDIA_V2D_Egocentric; FC=$REPO/baseline/robotic_grounding/flash_chord; PY=/root/venv-flash-lead/bin/python; CPUPY=/root/venv-cpu/bin/python
STAGE=/root/v2d/lead_v3/stage; SCENE=$STAGE/scenes/episode_$(printf %06d $EP)/scene
NAME=ep${EP}_v3_${CUR#tracking_}_s${SEED}; RUN=/root/v2d/runs/$NAME; mkdir -p $RUN/train
[ -f $SCENE/.relocated ] || { $CPUPY $STAGE/tools/relocate_scene.py $SCENE > $SCENE/relocate.log 2>&1 && touch $SCENE/.relocated; }
PARQUET=$(ls $SCENE/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet | head -1)
EXTRA="task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 termination.object_position.threshold=0.3 termination.object_orientation.threshold=1.5 training.checkpoint.save_interval_environment_steps=5000192 training.replay.capacity=1048576 training.replay.observation_storage_dtype=float16 $EXTRA_USER"
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline WANDB_DIR=$RUN WANDB_SILENT=true
cd $FC
echo "== $(date -u +%FT%TZ) start $NAME parquet=$PARQUET extra=[$EXTRA]" | tee -a $RUN/run.log
if [ ! -f $RUN/train.done ]; then
  SECONDS=0
  # shellcheck disable=SC2086
  $PY scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$PARQUET'" reset.seed=$SEED curriculum=$CUR \
    training.checkpoint.save_replay=false logging.mode=offline logging.upload_checkpoints=false "logging.run_name=$NAME" \
    "output_dir=$RUN/train" training.total_environment_steps=$STEPS $EXTRA >> $RUN/train.log 2>&1 &
  CHILD=$!
  ( while kill -0 $CHILD 2>/dev/null; do ls -1t $RUN/train/state_*.safetensors 2>/dev/null | tail -n +3 | xargs -r rm -f; sleep 60; done ) &
  PRUNER=$!; wait $CHILD; rc=$?; kill $PRUNER 2>/dev/null
  echo "== $(date -u +%FT%TZ) train exit=$rc wall_seconds=$SECONDS nefc_overflow_lines=$(grep -c 'nefc overflow' $RUN/train.log)" | tee -a $RUN/run.log
  grep -v 'nefc overflow' $RUN/train.log | tail -8 >> $RUN/run.log
  [ $rc -eq 0 ] && touch $RUN/train.done || exit $rc
fi
$PY $REPO/reports/lead_v3/evaluate_checkpoints.py --episode $EP --run $RUN >> $RUN/checkpoint_evaluations.log 2>&1
echo "== $(date -u +%FT%TZ) eval done; best:" | tee -a $RUN/run.log; head -12 $RUN/checkpoint_selection.json | tee -a $RUN/run.log
