#!/bin/bash
source /root/v2d/q/env.sh; export KIT FC XLA_PYTHON_CLIENT_PREALLOCATE=false
R=/root/v2d/runs/ep33_v3_100m_object2x_s42
while pgrep -f "scripts/train_flash_sac.py.*ep33_v3" >/dev/null; do sleep 1800; pgrep -f "scripts/train_flash_sac.py.*ep33_v3" >/dev/null && $PY /root/v2d/q/evaluate_checkpoints.py --episode 33 --run $R >> $R/checkpoint_evaluations.log 2>&1; done
