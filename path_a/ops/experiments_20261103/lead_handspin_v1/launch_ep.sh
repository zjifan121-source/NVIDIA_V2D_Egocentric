set -e
M=/root/v2d/NVIDIA_V2D_Egocentric
H=/root/edmond_handoff_de66fea
# bring Edmond's curricula + evaluator into our tree (no AppleDouble files)
for f in $H/baseline/robotic_grounding/flash_chord/src/flash_chord/configs/curriculum/tracking_100m*.yaml; do cp -n "$f" $M/baseline/robotic_grounding/flash_chord/src/flash_chord/configs/curriculum/; done
mkdir -p $M/reports/policy_targeted_2026-10-28
cp -n $H/reports/policy_targeted_2026-10-28/evaluate_checkpoints.py $H/reports/policy_targeted_2026-10-28/TRAINING_GUIDE.md $M/reports/policy_targeted_2026-10-28/
ls $M/assets/data/v2d_submission_kit/examples/run_policy_evaluation.py
EP=$1; MODE=$2; SEED=$3; NAME=ep$(printf %02d $EP)_${MODE}_def_s$SEED
SCENE=$M/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes/ep$(printf %02d $EP)_$MODE
RUN=/root/runs/$NAME; mkdir -p $RUN
cd $M/baseline/robotic_grounding/flash_chord
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
date -u +%FT%TZ > $RUN/START
setsid nohup /root/venv-flash-lead/bin/python scripts/train_flash_sac.py experiment=sharpa_flash_sac \
 "task.parquet='$SCENE/assets/human_motion_data/ego_recon/processed'" \
 reset.seed=$SEED curriculum=tracking_100m_object2x training.checkpoint.save_replay=false logging.mode=offline \
 logging.upload_checkpoints=false "logging.run_name=$NAME" "output_dir=$RUN/train" training.total_environment_steps=100003840 \
 task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 \
 termination.object_position.threshold=0.3 termination.object_orientation.threshold=1.5 \
 training.checkpoint.save_interval_environment_steps=5000192 training.replay.capacity=1048576 \
 training.replay.observation_storage_dtype=float16 > $RUN/train.log 2>&1 < /dev/null &
echo LAUNCHED $!
