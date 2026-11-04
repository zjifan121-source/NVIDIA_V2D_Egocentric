#!/bin/bash
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 0:40:00
set -e
cd ~
DL=/data/user_data/edmondwa/v2d/pathb_p4s_dl
mkdir -p $DL
/home/edmondwa/NVIDIA_V2D_Egocentric/tools/.venv/bin/python -I -c "
from huggingface_hub import snapshot_download
snapshot_download('rose-tea-cafe/v2d-t3-recon', repo_type='dataset', local_dir='$DL', allow_patterns=['path_b/runs/v4_eval_p4s_bundles/*'])"
NS=/data/user_data/edmondwa/v2d/recon_data/outputs/pathb_p4s
[ -e $NS ] || ln -s $DL/path_b/runs/v4_eval_p4s_bundles $NS
ls $NS; ls $NS/episode_000003/*; du -sh $DL
