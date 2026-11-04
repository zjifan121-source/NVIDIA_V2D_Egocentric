"""Free space from today's runs: drop resume files of finished dead ends; keep only the newest state/replay pair elsewhere."""
import re, subprocess
from pathlib import Path
D = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
DEAD = ['ep10_WIDE_s42', 'ep10_WIDE_s43', 'ep10_WIDE_s44', 'ep10_NEST2_OT_s42', 'ep3_LOWLR2', 'ep33_LOWLR2', 'ep25_LOWLR2',
        'ep3_LOWLR2b', 'ep33_LOWLR2b', 'ep19_LOWLR_from_kv_ep19_OT_s43', 'ep20_LOWLR_from_kv_ep20_OT_s42', 'ep5_LOWLR_from_kv_ep5_OT_s42',
        'ep3_LOWLR_from_ep3_grip_OT1M_s43', 'ep33_LOWLR_from_ep33_WIDE_s42', 'ep5_TARGET_BEST_FC_s42']
running = set(subprocess.run('squeue -h -u edmondwa,sarveshp -o %j', shell=True, capture_output=True, text=True).stdout.split())
freed = 0
for run in sorted(D.iterdir()):
    train = run / 'train'
    if not train.is_dir() or run.name in running:
        continue
    files = [f for f in train.iterdir() if f.name.startswith(('state_', 'replay_'))]
    if run.name in DEAD:
        drop = files
    elif re.search(r'(_DEF_s\d+|LOWLR|_OBJ4X_|_CAVITY_|_NEST|_WIDE_s\d+|_GRIP30_)', run.name):
        steps = sorted({int(f.stem.split('_')[1]) for f in files if f.name.startswith('state_')} &
                       {int(f.stem.split('_')[1]) for f in files if f.name.startswith('replay_')})
        keep = steps[-1:]  # newest matched pair
        drop = [f for f in files if int(f.stem.split('_')[1]) not in keep]
    else:
        continue
    for f in drop:
        freed += f.stat().st_size
        f.unlink()
    if drop:
        print(f'{run.name}: removed {len(drop)}')
print(f'freed {freed / 1024**3:.1f} GB')
