#!/bin/bash
# One L40S (46 GB): which setting lets the current ep41 scene train without running out of memory
# or overflowing the contact buffer? njmax 1024 @ 4096 worlds needs ~45.5 GB and fails (oom_probe, 10-15).
#SBATCH -J l40s-probe
#SBATCH -p preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH -o /data/user_data/%u/v2d/jobs/logs/%x-%j.log
REPO=/home/$USER/NVIDIA_V2D_Egocentric; FC=$REPO/baseline/robotic_grounding/flash_chord
PY=/data/user_data/$USER/v2d/venvs/flash_chord/bin/python
P=/data/user_data/$USER/v2d/recon_data/outputs/load_retarget/episode_000041/fs_track_attach/assets/human_motion_data/ego_recon/processed/sequence_id=ep41_fs_track_attach/robot_name=sharpa_wave
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $FC; hostname; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
for case in "768 384 4096" "1024 512 3584" "1024 512 3072"; do
  set -- $case; out=/data/user_data/$USER/v2d/flash_runs/oom_probe/fs41_nj$1_w$3; rm -rf $out
  ( while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done ) > $out.mem &
  mon=$!
  timeout 600 $PY scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$P'" reset.seed=42 logging.mode=disabled \
    task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=$1 sim.nconmax=$2 scene.world_count=$3 \
    termination.wrist_position.threshold=0.3 training.total_environment_steps=4014080 output_dir=$out > $out.log 2>&1
  rc=$?; kill $mon
  echo "== njmax=$1 worlds=$3 exit=$rc peak_MiB=$(sort -n $out.mem | tail -1) oom=$(grep -c 'out of memory' $out.log) nefc_overflow=$(grep -c 'nefc overflow' $out.log) $(grep -o 'training progress: [0-9,/]* environment steps ([0-9.]*%); [0-9,]* env steps/s' $out.log | tail -1)"
done
