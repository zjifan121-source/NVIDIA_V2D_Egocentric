"""Low-LR continuation of a Wentao run from its best checkpoint (actor + learner state at the same step).

python whc_cont.py RUN STEP [LR]
"""
import json, os, re, shutil, sys
from pathlib import Path
H = Path(__file__).resolve().parent
D = Path(os.environ.get('V2D_DATA', '/data/user_data/edmondwa/v2d')) / 'flash_runs/diag'
src, step = sys.argv[1], int(sys.argv[2]); LR = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-5
ep = int(re.match(r'ep(\d+)_', src)[1])
pol = D / src / 'train' / f'policy_{step}.safetensors'
st = next(p for p in (D / src / 'keep' / f'state_{step}.safetensors', D / src / 'train' / f'state_{step}.safetensors') if p.is_file())
name = f'ep{ep}_WHCc_{step // 1000000}M_{src.split("_", 1)[1]}'
keep = D / name / 'base'; keep.mkdir(parents=True, exist_ok=True); dst = keep / st.name
if not dst.exists(): shutil.copy2(st, dst)
trial = next(H.glob(f'{src.lower()}*_trial.json'))
base = json.loads(trial.read_text())
t = dict(base, name=name, launcher='refine_x_keep.sbatch', warm_actor=str(pol), init_state=str(dst), save_replay=False,
         account=os.environ.get('V2D_USER', 'edmondwa'), partition=os.environ.get('V2D_PARTITION', 'preempt'),
         qos=os.environ.get('V2D_QOS', 'preempt_qos'),
         total_environment_steps=step + 25001984, time='01:30:00',
         extra=base['extra'] + f' training.checkpoint.save_replay=false training.optimization.learning_rate_initial={LR} training.optimization.learning_rate_peak={LR} training.optimization.learning_rate_end={LR}',
         hypothesis=f'Hill-climb: continue {src}@{step} (its best) for 25M at constant LR {LR}, fresh replay.')
t.pop('slurm_account', None)
if os.environ.get('V2D_SLURM_ACCOUNT'): t['slurm_account'] = os.environ['V2D_SLURM_ACCOUNT']
p = H / f'{name.lower()}_trial.json'; p.write_text(json.dumps(t, indent=2)); print(p.name)
