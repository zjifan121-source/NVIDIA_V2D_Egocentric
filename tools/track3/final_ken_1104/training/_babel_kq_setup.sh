# run inside a step of 10774619 (babel-s5-16): link Babel's repo + venv where kq.sh expects them, then smoke-test the env
set -u
B=/tmp/haisenk_v2d
[ -e $B/v2d_train ] || ln -s /data/user_data/haisenk/v2d/v2d_train $B/v2d_train
ls -la $B/v2d_train
FC=$B/v2d_train/baseline/robotic_grounding/flash_chord
export PYTHONPATH=$B/fc_src_de66
cd $FC && timeout 300 $FC/.venv/bin/python -c "
import flash_chord, jax, mujoco
print('flash_chord', flash_chord.__file__); print('jax', jax.__version__, jax.devices()[:2]); print('mujoco', mujoco.__version__)
" 2>&1 | grep -v Warning | tail -5
