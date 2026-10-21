#!/bin/bash
# Download the weights that need their module image (FoundationPose, WiLoR, GeoCalib, AnyCalib).
#SBATCH -J t3-weights
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /data/user_data/%u/v2d/stereo_test/logs/%x-%j.log
set -uo pipefail
cd /home/$USER/NVIDIA_V2D_Egocentric/baseline/reconstruction
export V2D_CONTAINER_RUNTIME=apptainer V2D_APPTAINER_IMAGE_DIR=/data/user_data/$USER/v2d/sif V2D_APPTAINER_FLAGS="--no-mount /opt"
PY=.venv/bin/python
ls -lh $V2D_APPTAINER_IMAGE_DIR
[ -e data/weights/foundation_pose/.done ] || { $PY -m v2d.foundation_pose.docker.run_download_weights --output_dir data/weights/foundation_pose --accept_nvidia_model_eula && touch data/weights/foundation_pose/.done; }
[ -e data/weights/wilor/.done ] || { $PY -m v2d.wilor.docker.run_download_weights --weights_dir data/weights/wilor && touch data/weights/wilor/.done; }
[ -e data/weights/geocalib/.done ] || { $PY -m v2d.geocalib.docker.run_download_weights --output_dir data/weights/geocalib && touch data/weights/geocalib/.done; }
[ -e data/weights/anycalib/.done ] || { $PY -m v2d.anycalib.docker.run_download_weights --output_dir data/weights/anycalib && touch data/weights/anycalib/.done; }
du -sh data/weights/*
