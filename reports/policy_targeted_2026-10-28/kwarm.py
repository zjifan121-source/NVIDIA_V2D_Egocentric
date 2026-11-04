"""Warm-start Wentao-scene runs from Ken's best policy state for the same episode."""
import json,sys
from pathlib import Path
H=Path.home()/'NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28'
plan=json.loads((H.parent/'submission_candidate_2026-10-28/plan.json').read_text())
EPS=[int(x) for x in sys.argv[1].split(',')]; LR=float(sys.argv[2])
for ep in EPS:
  r=Path(plan[str(ep)]['run']); parent=r.parent if r.name.startswith('early_') else r
  states=sorted(parent.glob('train/state_*.safetensors'),key=lambda p:int(p.stem.split('_')[1]))
  if not states: print(ep,'NO STATE',parent); continue
  st=states[-1]; step=int(st.stem.split('_')[1])
  base=json.loads((H/f'ep{ep}_wmix_s80g_trial.json').read_text())
  name=f'ep{ep}_KWARM_s80'
  t=dict(base,name=name,init_state=str(st),save_replay=False,total_environment_steps=step+40_000_000,
    extra=base['extra']+f' training.optimization.learning_rate_initial={LR} training.optimization.learning_rate_peak={LR} training.optimization.learning_rate_end={LR}',
    hypothesis=f'Warm start from Ken {parent.name}@{step} on Wentao scene, constant LR {LR}, +40M')
  (H/f'ep{ep}_kwarm_s80g_trial.json').write_text(json.dumps(t,indent=2)); print(ep,parent.name,step)
