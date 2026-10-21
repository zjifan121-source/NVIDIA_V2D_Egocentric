#!/bin/bash
# FoundationStereo dense depth for a frame range of one episode, on CPU (~20-30 s/frame).
# Usage: sbatch fs.sbatch <episode> <start> <stop>
#SBATCH -J t3-fs
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH -o /data/user_data/%u/v2d/stereo_test/logs/%x-%j.log
set -euo pipefail
EP=${1:?episode}; A=${2:?start}; B=${3:?stop}
cd /home/$USER/NVIDIA_V2D_Egocentric
/data/user_data/$USER/v2d/venvs/stereo/bin/python tools/track3/stereo_seed.py --episode "$EP" \
  --out /data/user_data/$USER/v2d/stereo_test/seed/ep$EP --frames "$A" "$B" \
  --foundation-stereo /data/user_data/$USER/v2d/recon_data/weights/foundation_stereo/deployable_foundationstereo_small_576x960_v2.0.onnx 2>&1 | grep -v affinity
