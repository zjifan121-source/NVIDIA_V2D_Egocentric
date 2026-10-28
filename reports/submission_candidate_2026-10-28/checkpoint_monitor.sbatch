#!/bin/bash
#SBATCH --job-name=v2d-checkpoint-monitor
#SBATCH --partition=maxlab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/checkpoint_monitor.log
set -euo pipefail
exec python /home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/checkpoint_monitor.py
