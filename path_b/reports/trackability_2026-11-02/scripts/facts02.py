"""Re-match F2's trained references by S5's input (the VM's s4b_reference md5), check the S5 bundle's object
trajectory equals the matched Mac run's, take playback from that run's waterfall."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
OUT = Path('/Users/jzzzzz79/V2D/NVIDIA_V2D_Egocentric/path_b/data/outputs')
VM = OUT / 'v4_h1_vm'
rows = json.load(open(sys.argv[1]))
vm = {}
for line in open(sys.argv[2]):
    h, p = line.split(None, 1); vm[p.strip()] = h
RUNS = ['v4_k7', 'v4_j10', 'v4_h1', 'v4_k6', 'v4_k5', 'v4_j3', 'v4_j9', 'v4_i1', 'v4_g3']
def mac_md5(run, ep, obj, cand):
    p = OUT / run / f'episode_{ep:06d}/s4b_reference/{obj}/{cand}/result.npz'
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None
for r in rows:
    if 'error' in r:
        continue
    ep, task = r['episode'], r['candidate'].split('_250m')[0]
    suffix = '_pre_k7' if r['folder'].endswith('_pre_k7') else ''
    s4b = {o: vm.get(f'episode_{ep:06d}/s4b_reference{suffix}/{o}/{task}/result.npz')
           or vm.get(f'episode_{ep:06d}/s4b_reference/{o}/{task}/result.npz') for o in r['objects']}
    matches = [run for run in RUNS if all(s4b[o] and mac_md5(run, ep, o, task) == s4b[o] for o in r['objects'])]
    src = 'v4_k7' if 'v4_k7' in matches else (matches[0] if matches else None)
    diff = None
    if src:
        diff = 0.0
        for o in r['objects']:
            a = np.load(VM / f'episode_{ep:06d}/s5_task{suffix}/{task}/run/bundles/{o}/result.npz')['object_to_world_transform']
            b = np.load(OUT / src / f'episode_{ep:06d}/s4b_reference/{o}/{task}/result.npz')['object_to_world_transform']
            diff = max(diff, float(np.abs(a.astype(float) - b.astype(float)).max()))
    wf = OUT / src / f'episode_{ep:06d}/waterfall.json' if src else None
    play = json.loads(wf.read_text())['scores'].get(f'ref_{task}') if wf and wf.is_file() else None
    r.update(trained_md5={o: (s4b[o] or '')[:8] for o in r['objects']}, md5_runs=matches, playback_run=src,
             is_k7=('v4_k7' in matches), object_traj_maxdiff=diff, playback=play,
             tax=(play['AUC'] - r['rollout']['AUC']) if play else None)
json.dump(rows, open(sys.argv[3], 'w'), indent=1, default=str)
for r in rows:
    if 'error' in r:
        print(f"ep{r['episode']:>2} {r['folder']:16} {r['candidate']:9} {r['error']}"); continue
    pb = r['playback']['AUC'] if r['playback'] else float('nan')
    print(f"ep{r['episode']:>2} {r['folder'][9:] or '-':8} {r['candidate']:9} src={str(r['playback_run']):7} runs={','.join(m[3:] for m in r['md5_runs'])[:24]:24} "
          f"k7={int(r['is_k7'])} trajdiff={r['object_traj_maxdiff']} play={pb:.3f} roll={r['rollout']['AUC']:.3f} tax={(r['tax'] if r['tax'] is not None else float('nan')):+.3f} "
          f"pick={r['pick'][:3]}/{str(r['gt_free_pick'])[:3]}")
