#!/bin/bash
# orchard V2D (3 Nov night): FlashCHORD training env on orchard-flame-21 node /tmp, same code as Babel
# (team repo origin/edm_training @ 1a6bae7b), uv 0.13.4, locked deps + train extra. Robot assets come from Babel (LFS).
set -u
B=/tmp/haisenk_v2d; mkdir -p $B; cd $B; date; hostname
[ -x $B/uvenv/bin/uv ] || { python3 -m venv $B/uvenv && $B/uvenv/bin/pip install -q uv==0.13.4; }
$B/uvenv/bin/uv --version
if [ ! -d $B/v2d_train/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --filter=blob:none --no-checkout https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric.git $B/v2d_train
  cd $B/v2d_train && git sparse-checkout set --no-cone /baseline/robotic_grounding/flash_chord/ /tools/track3/ /assets/data/v2d_submission_kit/
  GIT_LFS_SKIP_SMUDGE=1 git checkout -q 1a6bae7b2682a40087cd70e5d68c6aa3e5f21b77
fi
cd $B/v2d_train && git log -1 --format="%H %s"; du -sh $B/v2d_train
cd baseline/robotic_grounding/flash_chord
export UV_PYTHON_INSTALL_DIR=$B/uv-python UV_CACHE_DIR=$B/uv-cache UV_LINK_MODE=copy
$B/uvenv/bin/uv sync --frozen --extra train 2>&1 | tail -4
.venv/bin/python -c "import jax, warp, flash_chord, newton; print('jax', jax.__version__, jax.devices()); print('warp', warp.__version__)" 2>&1 | grep -v Warning | tail -3
ls $B/v2d_train/assets/data/v2d_submission_kit/examples/run_policy_evaluation.py
echo ORCH_SETUP_DONE
