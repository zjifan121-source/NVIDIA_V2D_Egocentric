#!/bin/bash
# Stage 1 of the stereo seed for one episode (calibrated video + SGBM stereo depth), on CPU (~5-10 min).
# Usage: sbatch seed.sbatch <episode>
#SBATCH -J t3-seed
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH -o /data/user_data/%u/v2d/stereo_test/logs/%x-%j.log
set -euo pipefail
EP=${1:?episode}
cd /home/$USER/NVIDIA_V2D_Egocentric
/data/user_data/$USER/v2d/venvs/stereo/bin/python tools/track3/stereo_seed.py --episode "$EP" \
  --out /data/user_data/$USER/v2d/stereo_test/seed/ep$EP --preview reports/stereo_check/ep${EP}_seed.png
