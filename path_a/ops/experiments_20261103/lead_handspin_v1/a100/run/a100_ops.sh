set -u
M=/root/v2d/NVIDIA_V2D_Egocentric
# 1. stop the old-pot ep28 run
for p in $(pgrep -f "logging.run_name=ep28_rawexcl_def_s200"); do kill $p; done; sleep 3; pgrep -fc "logging.run_name=ep28_rawexcl_def_s200" || echo OLD_EP28_STOPPED
mv /root/runs/ep28_rawexcl_def_s200 /root/runs/ABANDONED_ep28_oldpot_def_s200 2>/dev/null
# 2. relocate scenes (in place; already-loaded runs unaffected)
SC=$M/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes
/root/venv-cpu/bin/python /root/edmond_handoff_de66fea/tools/track3/relocate_scene.py $SC/ep33_rawexcl $SC/ep25_rawexcl $SC/ep30_rawexcl $SC/ep05_rawexcl $SC/ep01_rawexcl $SC/ep28_rawexcl /root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl 2>&1 | sed 's#/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/##'
/root/venv-cpu/bin/python - <<'PY'
import pyarrow.parquet as pq, glob, os
for s in ["/root/v2d/NVIDIA_V2D_Egocentric/path_a/ops/experiments_20261103/lead_hand_reattach_v1/scenes/ep33_rawexcl","/root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl"]:
    f=glob.glob(s+"/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet")[0]
    for c in ["object_mesh_paths","object_urdf_paths"]:
        for p in pq.read_table(f).column(c)[0].as_py(): print("VERIFY", os.path.exists(p), p.split("/scenes/")[-1])
PY
# 3. explicit-scene launcher
cat > /root/launch_scene.sh <<'L'
NAME=$1; SCENE=$2; SEED=$3; EXTRA="${4:-}"
M=/root/v2d/NVIDIA_V2D_Egocentric; RUN=/root/runs/$NAME; mkdir -p $RUN
cd $M/baseline/robotic_grounding/flash_chord
export XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
date -u +%FT%TZ > $RUN/START; echo "$SCENE" > $RUN/SCENE
setsid nohup /root/venv-flash-lead/bin/python scripts/train_flash_sac.py experiment=sharpa_flash_sac \
 "task.parquet='$SCENE/assets/human_motion_data/ego_recon/processed'" \
 reset.seed=$SEED curriculum=tracking_100m_object2x training.checkpoint.save_replay=false logging.mode=offline \
 logging.upload_checkpoints=false "logging.run_name=$NAME" "output_dir=$RUN/train" training.total_environment_steps=100003840 \
 task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 \
 termination.object_position.threshold=0.3 termination.object_orientation.threshold=1.5 \
 training.checkpoint.save_interval_environment_steps=5000192 training.replay.capacity=1048576 \
 training.replay.observation_storage_dtype=float16 $EXTRA > $RUN/train.log 2>&1 < /dev/null &
echo LAUNCHED $NAME $!
L
bash /root/launch_scene.sh ep28_v8pot_def_s200 /root/v2d/lead_winners_v8/mini_stage/hand_stage/scenes/ep28_rawexcl 200
du -sh $SC/ep30_rawexcl $SC/ep05_rawexcl | cut -f1 | tr "\n" " "; echo
for r in ep33_def_s200 ep25_rawexcl_def_s200; do echo $r $(grep "training progress" /root/runs/$r/train.log | tail -1 | cut -c20-110); done
