"""Keep only the newest state and newest matched state/replay pair in this effort's runs, running or not."""
import re
import time
from pathlib import Path

D = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
freed = 0
for run in D.iterdir():
    t = run / 'train'
    if not re.search(r'(_DEF_s\d+|LOWLR|_OBJ4X_|_CAVITY_|_NEST|_WIDE_s\d+|_GRIP30_)', run.name) or not t.is_dir():
        continue
    st = {int(p.stem.split('_')[1]): p for p in t.glob('state_*.safetensors')}
    rp = {int(p.stem.split('_')[1]): p for p in t.glob('replay_*.safetensors')}
    pair = max(set(st) & set(rp), default=None)
    doomed = [p for s, p in st.items() if s not in (max(st), pair)] + [p for s, p in rp.items() if s != pair]
    for p in doomed:
        if time.time() - p.stat().st_mtime > 600:  # never touch a file still being written
            freed += p.stat().st_size
            p.unlink()
print(f'freed {freed / 1e9:.1f} GB')
