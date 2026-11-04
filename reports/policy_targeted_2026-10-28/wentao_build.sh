#!/bin/bash
# Download Wentao's scored20 alignment mix and build one training scene per episode (Ken's recipe: attach hands, smooth 2, CPU).
#SBATCH -J v2d-wentao-build
#SBATCH -p maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 6:00:00
#SBATCH -o /home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28/wentao_build.out
set -uo pipefail
REPO=/home/edmondwa/NVIDIA_V2D_Egocentric
W=/data/user_data/edmondwa/v2d/wentao_mix
PY=/data/user_data/edmondwa/v2d/venvs/retarget/bin/python
mkdir -p $W
$REPO/tools/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('rose-tea-cafe/v2d-t3-recon', repo_type='dataset', revision='57bb019a1aa3968b08c2703ff9eb320915ec0655',
  allow_patterns=['experiments/wentacc/scored20_alignment_mix_v1_20261030/**'], local_dir='$W/hub')" || exit 1
S=$W/hub/experiments/wentacc/scored20_alignment_mix_v1_20261030/scenes
build() {
  ep=$1; E6=$(printf %06d $ep); out=$W/scenes/ep$ep
  mkdir -p $out
  $PY $REPO/tools/track3/run_load_retarget.py --bundle $S/episode_$E6/bundle --episode $ep \
    --attach-hands --smooth-hands 2 --skip-video --device cpu \
    --sequence-id ep${ep}_wentao_mix_attach --out $out > $out.log 2>&1
  echo "ep$ep exit $? $(date +%T)"
}
export -f build; export W S PY REPO
ls $S | sed 's/episode_0*//' | xargs -P 4 -I{} bash -c 'build {}'
echo ALL_DONE
