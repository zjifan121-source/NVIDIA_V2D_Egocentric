"""Ken's best actor + newest Wentao-scene learner state (critic), fine-tuned on Wentao's scene."""
import json,os,re,shutil,sys
from pathlib import Path
H=Path.home()/'NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28'
plan=json.loads((H.parent/'submission_candidate_2026-10-28/plan.json').read_text())
D=Path('/data/user_data/edmondwa/v2d/flash_runs/diag'); K=Path('/data/user_data/edmondwa/v2d/ken_night/night_2026-10-24/runs')
LR=5e-5
for ep in [int(x) for x in sys.argv[1].split(',')]:
  name=plan[str(ep)]['name']; r=Path(plan[str(ep)]['run'])
  m=re.search(r'early_(\d+)',name); parent=(r.parent if r.name.startswith('early_') else r).name; orig=re.sub(r'_RS$','',parent)
  pols=[x/'train'/f'policy_{m[1]}.safetensors' for x in (D/orig,D/parent,K/orig)] if m else [p for x in (D/orig,K/orig) for p in sorted((x/'train').glob('policy_*.safetensors'),key=lambda p:int(p.stem.split('_')[1]))[-1:]]
  pol=next((p for p in pols if p.is_file()),None)
  sts=[s for w in D.glob(f'ep{ep}_WMIX_s*') for s in w.glob('train/state_*.safetensors')]+[s for w in D.glob(f'ep{ep}_WHC_s*') for s in w.glob('keep/state_*.safetensors')]
  if not pol or not sts: print(ep,'skip',pol,len(sts)); continue
  st=max(sts,key=lambda p:int(p.stem.split('_')[1])); step=int(st.stem.split('_')[1])
  run=f'ep{ep}_KXW_s80'; keep=D/run/'base'; keep.mkdir(parents=True,exist_ok=True); dst=keep/st.name
  if not dst.exists(): shutil.copy2(st,dst)
  base=json.loads(sorted(H.glob(f'ep{ep}_wmix_s8*_trial.json'))[0].read_text())
  t=dict(base,name=run,launcher='refine_x.sbatch',warm_actor=str(pol),init_state=str(dst),save_replay=False,
    total_environment_steps=step+40_001_536,memory=base.get('memory','64G'),time='02:00:00',
    extra=base['extra']+f' training.checkpoint.save_replay=false training.optimization.learning_rate_initial={LR} training.optimization.learning_rate_peak={LR} training.optimization.learning_rate_end={LR}',
    hypothesis=f"Ken actor {pol.parent.parent.name}/{pol.name} + Wentao critic {st.parent.parent.name}@{step}; +40M at LR {LR} on Wentao scene")
  (H/f'ep{ep}_kxw_s80g_trial.json').write_text(json.dumps(t,indent=2)); print(ep,pol.parent.parent.name,'<-',st.parent.parent.name,step)
