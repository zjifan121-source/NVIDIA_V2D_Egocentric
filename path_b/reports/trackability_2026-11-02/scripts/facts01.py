"""Step 0.1-0.3 facts: F2's pairs (VM v4_h1, copied to path_b/data/outputs/v4_h1_vm), matched to the Mac runs by md5."""
import sys, json, hashlib
from pathlib import Path
REPO = Path('/Users/jzzzzz79/V2D/NVIDIA_V2D_Egocentric')
sys.path.insert(0, str(REPO / 'path_b'))
import numpy as np, pandas as pd
from pathb import paths, gt
from pathb.team import scorer

OUT = REPO / 'path_b/data/outputs'
VM = OUT / 'v4_h1_vm'
MAC_RUNS = ['v4_k7', 'v4_j10', 'v4_h1', 'v4_k6', 'v4_k5', 'v4_j3', 'v4_j9', 'v4_i1', 'v4_g3']

def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()

mac = {}
def mac_md5(run, ep, obj, cand):
    k = (run, ep, obj, cand)
    if k not in mac:
        p = OUT / run / f'episode_{ep:06d}/s4b_reference/{obj}/{cand}/result.npz'
        mac[k] = md5(p) if p.is_file() else None
    return mac[k]

def waterfall(run, ep):
    p = OUT / run / f'episode_{ep:06d}/waterfall.json'
    return json.loads(p.read_text())['scores'] if p.is_file() else {}

rows = []
for epdir in sorted(VM.glob('episode_*')):
    ep = int(epdir.name.split('_')[1])
    episode = paths.episode_info(ep)
    for suffix in ('', '_pre_k7'):
        sel_p = epdir / f's6_select{suffix}/selection.json'
        if not sel_p.is_file():
            continue
        sel = json.loads(sel_p.read_text())
        for cand, roll in sel.get('scores', {}).items():
            task = cand.split('_250m')[0]
            bundles = epdir / f's5_task{suffix}/{task}/run/bundles'
            if not bundles.is_dir():
                rows.append(dict(episode=ep, folder=f's6_select{suffix}', candidate=cand, error=f'no {bundles.relative_to(VM)}',
                                 rollout=roll, pick=sel.get('candidate')))
                continue
            objs = sorted(p.name for p in bundles.iterdir() if p.is_dir())
            trained = {o: md5(bundles / o / 'result.npz') for o in objs}
            matches = [r for r in MAC_RUNS if all(mac_md5(r, ep, o, task) == trained[o] for o in objs)]
            src = 'v4_k7' if 'v4_k7' in matches else (matches[0] if matches else None)
            play = waterfall(src, ep).get(f'ref_{task}') if src else None
            ev = json.loads((epdir / f's6_eval{suffix}/{cand}/results/episode_{ep:06d}.json').read_text())
            pq = epdir / f's6_eval{suffix}/{cand}/results/episode_{ep:06d}.parquet'
            t = pd.read_parquet(pq)
            t = t[t['environment_id'] == 0]
            err = np.linalg.norm(t[[f'achieved_position_{a}' for a in 'xyz']].to_numpy(float)
                                 - t[[f'reference_position_{a}' for a in 'xyz']].to_numpy(float), axis=1)
            t = t.assign(err=err)
            per = t.groupby('reference_frame')['err'].max()
            first_ref = int(per.index[np.argmax(per.to_numpy() > 0.10)]) if (per > 0.10).any() else None
            poses = scorer().evaluator_poses(pq, episode)
            meshes = {o['name']: epdir / f's5_task{suffix}/{task}/run/score_input/meshes/episode_{ep:06d}/{o["name"]}.obj'
                      for o in episode['objects']}
            fe = gt.official_frame_errors(episode, poses, meshes, [])
            pos = np.asarray(fe['position_m']).max(axis=1)
            first_gt = int(np.argmax(pos > 0.10)) if (pos > 0.10).any() else None
            rows.append(dict(
                episode=ep, folder=f's6_select{suffix}', candidate=cand, objects=objs,
                pick=sel['candidate'], gt_free_pick=sel.get('evaluation_rule_pick'),
                trained_md5={o: trained[o][:8] for o in objs}, md5_runs=matches, playback_run=src,
                is_k7=('v4_k7' in matches), playback=play, rollout=roll,
                tax=(play['AUC'] - roll['AUC']) if play else None,
                failure_cause_fraction=ev['reference_tracking']['failure_cause_fraction'],
                reference_end_fraction=ev['reference_tracking']['reference_end_fraction'],
                non_finite_worlds=ev['non_finite_worlds'], world_count=ev['world_count'], reset_mode=ev['reset_mode'],
                steps=int(ev['step_count']), first_ref_err10_frame=first_ref, first_gt_err10_frame=first_gt,
                cost=sel['costs'].get(cand)))
json.dump(rows, open(sys.argv[1], 'w'), indent=1, default=str)
for r in rows:
    if 'error' in r:
        print(f"ep{r['episode']:>2} {r['folder']} {r['candidate']}: {r['error']}"); continue
    pb = r['playback']['AUC'] if r['playback'] else float('nan')
    print(f"ep{r['episode']:>2} {r['folder'][9:] or '-':8} {r['candidate']:9} pick={r['pick'][:4]} gtfree={str(r['gt_free_pick'])[:4]} "
          f"src={r['playback_run']} k7={int(r['is_k7'])} play={pb:.3f} roll={r['rollout']['AUC']:.3f} "
          f"fail={json.dumps(r['failure_cause_fraction'])} ref10={r['first_ref_err10_frame']} gt10={r['first_gt_err10_frame']}")
