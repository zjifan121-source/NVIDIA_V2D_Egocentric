#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install lightweight host orchestration only; all ML stays in pinned containers.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mountpoint -q /mnt/v2d
[[ "$repo" == /mnt/v2d/code/NVIDIA_V2D_Egocentric ]]
mkdir -p /mnt/v2d/cache/{huggingface,pip,torch} /mnt/v2d/{tmp,weights,outputs,datasets/track_3}
cat > /mnt/v2d/code/v2d.env <<'ENV'
export XDG_CACHE_HOME=/mnt/v2d/cache
export HF_HOME=/mnt/v2d/cache/huggingface
export HF_HUB_CACHE=/mnt/v2d/cache/huggingface/hub
export PIP_CACHE_DIR=/mnt/v2d/cache/pip
export TORCH_HOME=/mnt/v2d/cache/torch
export TMPDIR=/mnt/v2d/tmp
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHON=/mnt/v2d/venv/bin/python
export PATH=/mnt/v2d/venv/bin:$PATH
ENV
source /mnt/v2d/code/v2d.env
for pair in 'baseline/reconstruction/data/weights weights' 'baseline/reconstruction/data/outputs outputs' 'assets/data/track_3 datasets/track_3'; do
    read -r relative target <<< "$pair"
    if [[ -L "$repo/$relative" ]]; then
        [[ $(readlink -f "$repo/$relative") == /mnt/v2d/"$target" ]]
    else
        [[ ! -e "$repo/$relative" ]] || { echo "Existing path requires inspection: $repo/$relative" >&2; exit 1; }
        mkdir -p "$(dirname "$repo/$relative")"
        ln -s /mnt/v2d/"$target" "$repo/$relative"
    fi
done
[[ -x "$PYTHON" ]] || python3 -m venv /mnt/v2d/venv
"$PYTHON" -m pip install --upgrade pip
cd "$repo/baseline/reconstruction"
bash scripts/install_ego_reconstruction_packages.sh
"$PYTHON" -m pip install huggingface_hub
"$PYTHON" modules/v2d_pipelines/run_ego_reconstruction.py --help >/mnt/v2d/logs/pipeline_help.txt
"$PYTHON" -m pip freeze >/mnt/v2d/logs/host_pip_freeze.txt
printf 'Host setup complete on %s\n' "$(date -u +%FT%TZ)"
