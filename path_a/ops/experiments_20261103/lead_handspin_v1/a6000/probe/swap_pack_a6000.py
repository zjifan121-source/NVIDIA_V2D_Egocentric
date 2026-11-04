"""Pack a Track 3 submission = bob1's 20 real rollouts with named episodes swapped for other real rollouts.
Every row stays a real policy rollout with its own checkpoint digest; the kit packer enforces uniform evaluator digests.
usage: swap_pack_a6000.py <name> <ep>=<run_dir>,<scene_dir> [...]   -> probe/out/<name>/{plan.json,<name>.parquet}"""
import json,subprocess,sys
from pathlib import Path
HERE=Path('/mnt/v2d/outputs/experiments_20261103/lead_v3_train/probe'); BASE=HERE/'probe_inputs/experiments/edmond/bob1_submission_20261103'
COMMIT='https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/de66fea3ae7c99ba96458f3cd9d1b77f5f94b4ff'
PY=str(HERE/'venv-kit/bin/python')
name=sys.argv[1]; out=HERE/'out'/name; out.mkdir(parents=True,exist_ok=True)
plan={}
for d in sorted(BASE.glob('episode_*')):
    ep=int(d.name.split('_')[1]); plan[str(ep)]=dict(run=str(d/'policy'),scene=str(d/'scene'))
for a in sys.argv[2:]:
    ep,rest=a.split('='); run,scene=rest.split(','); assert ep in plan,ep
    assert (Path(run)/'results'/f'episode_{int(ep):06d}.parquet').is_file(),run
    plan[ep]=dict(run=run,scene=scene)
assert len(plan)==20,len(plan)
(out/'plan.json').write_text(json.dumps(plan,indent=1)+'\n')
cmd=[PY,str(HERE/'track3_pack.py'),'--plan',str(out/'plan.json'),'--work',str(out/'work'),'--code-commit',COMMIT,'--python',PY,'-o',str(out/f'{name}.parquet')]
print(' '.join(cmd),flush=True)
raise SystemExit(subprocess.run(cmd).returncode)
