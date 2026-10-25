"""Pick the candidate to track. Candidates skip the held-out frames near their own source frame, so plain means average
different frame sets: each frame's iou + depth is taken relative to that frame's median over the candidates scored on
it, then averaged per candidate. Only placed-pose scores are accepted (see score_fp.py)."""
import json
import os
import shlex
import statistics


def normalised(cands):
    """{id: mean over its frames of (iou + depth - that frame's median over the candidates)}"""
    per = {k: {t: v['iou'] + v['depth'] for t, v in c['per_frame'].items()} for k, c in cands.items()}
    frames = {t for p in per.values() for t in p}
    med = {t: statistics.median([p[t] for p in per.values() if t in p]) for t in frames}
    return {k: sum(s - med[t] for t, s in p.items()) / len(p) for k, p in per.items() if p}


def track_command(P, mesh):
    """The launcher command that tracks the pick with the settings of the run the plan was made from."""
    rec_path = os.path.join(P['run'], 'run.json')
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    episode = P.get('episode') if P.get('episode') is not None else rec.get('episode')
    obj = P.get('object') or rec.get('object')
    if episode is None or not obj:
        return None
    cmd = ['python', 'tools/track3/run_baseline_reconstruction.py', '--episode', str(episode), '--object', obj,
           '--reference-frame', str(P['key_frame']), '--object-mesh', mesh, '--keep-mesh-size', '--no-refinement']
    if rec.get('prompt'):
        cmd += ['--prompt', f'{obj}={rec["prompt"]}']
    if rec.get('seed'):
        cmd += ['--seed', rec['seed']]
    if rec.get('object_segmenter') and rec['object_segmenter'] != 'dino_sam2':
        cmd += ['--object-segmenter', rec['object_segmenter']]
    if '--dev' in rec.get('command', []):
        cmd.append('--dev')
    return ' '.join(shlex.quote(c) for c in cmd + ['--branch', '<namespace>'])


def run(plan, scores, output):
    P, S = json.load(open(plan)), json.load(open(scores))
    if not S.get('placed'):
        raise SystemExit(f'{scores} has no placed-pose scores: rerun score_fp.py')
    n = normalised(S['candidates'])
    if not n:
        raise SystemExit(f'no scored candidates in {scores}')
    best = max(n, key=n.get)
    mesh = S['candidates'][best]['mesh']
    rec = {'pick': best, 'mesh': mesh, 'score': n[best], 'ranking': sorted(n, key=lambda k: -n[k]),
           'normalised': n, 'mean': {k: v['mean'] for k, v in S['candidates'].items()},
           'method': 'placed pose, per-frame normalised'}
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    json.dump(rec, open(output, 'w'), indent=1)
    print(f'pick {best} (normalised {n[best]:+.3f}): {mesh}')
    cmd = track_command(P, mesh)
    if cmd:
        print('track it with the run\'s own settings (same seed, prompt, segmenter and reference frame):\n  ' + cmd)
