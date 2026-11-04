"""Add score_input/ to scenes built by build_scene.py, so they can be packed for Kaggle.

The team loader (tools/track3/run_load_retarget.py) writes score_input/meshes/episode_XXXXXX/<name>.obj in
write_score_input, its LAST step; our build_scene runs exit at the support step (no USD binding in the CPU venv),
so our scenes lack it. The kit packer needs it (pack_submission.py --meshes DIR: episode_XXXXXX/<object name>.{glb,
obj,...} "in rollout body coordinates"), and Edmond's make_bob_plan.py / scene_qa.py skip scenes without it.

Source = the scene's own processed body meshes processed/<sequence_id>_<name>.obj (+ .mtl + map_Kd texture), copied
under the roster names (tools/track3/episodes.json) exactly as Edmond's tools/track3/import_scene.py does. These ARE
the rollout bodies, and they equal the bundle meshes the loader would have exported (ep5 check: white_pot 5520 and
wooden_spoon 1395 vertices, identical bounds and means). score_input/results/episode_XXXXXX.npy (bundle object poses,
loader format, for tools/track3/score_local.py) is written too when the bundle is reachable. File copies only; no GT.

Usage: export_score_input.py <scene_dir>... [--force]
"""
from pathlib import Path
import sys, os, json, shutil, hashlib, re
import numpy as np
REPO = Path(os.environ.get('V2D_ROOT', '/Volumes/E/Code/V2D')) / 'NVIDIA_V2D_Egocentric'
ROSTER_FILE = Path(os.environ.get('LEAD_ROSTER', str(REPO / 'tools/track3/episodes.json')))  # roster: tools/track3/episodes.json
ROSTER = {e['episode']: [o['name'] for o in e['objects']] for e in json.loads(ROSTER_FILE.read_text())['episodes']}
PROC = 'assets/human_motion_data/ego_recon/processed'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def n_vertices(p): return sum(1 for l in open(p) if l.startswith('v '))

def write_results(bundle: Path, out: Path, episode: int):
    from scipy.spatial.transform import Rotation
    d = dict(np.load(bundle / 'result.npz', allow_pickle=False))
    n = len(d['object_names']) if 'object_names' in d else 1
    keys = ['object_to_world_transform'] + [f'object_{i}_to_world_transform' for i in range(1, n)]
    poses = []
    for k in keys:
        T = d[k].astype(float)
        poses.append(np.concatenate([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()], axis=-1))
    (out / 'score_input/results').mkdir(parents=True, exist_ok=True)
    np.save(out / f'score_input/results/episode_{episode:06d}.npy', np.stack(poses, axis=1))
    return [str(n) for n in d['object_names']] if 'object_names' in d else None

def export(scene: Path, force: bool):
    proc = scene / PROC
    if (scene / 'OUTPUT_CHECK.json').is_file():
        check = json.loads((scene / 'OUTPUT_CHECK.json').read_text())
        ep, seq = int(check['episode']), check['sequence_id']
    else:  # scenes built elsewhere (e.g. lead_v3/stage/scenes/episode_000033/scene): episode from the path, sequence from the partition
        check = {}
        m = re.search(r'episode_(\d+)', str(scene)) or re.search(r'/ep(\d+)_', str(scene))
        ep = int(m.group(1)); parts = list(proc.glob('sequence_id=*')); assert len(parts) == 1, parts
        seq = parts[0].name.split('=', 1)[1]
        try:
            import pyarrow.parquet as pq
            f = next(parts[0].glob('robot_name=*/*.parquet')); raw = pq.read_table(f, columns=['raw_motion_file']).column(0)[0].as_py()
            check['bundle_cleaned'] = str(Path(raw).parent)
        except Exception as e:
            print(f'  ! {scene}: no bundle path ({e}); results npy skipped', file=sys.stderr)
    names = ROSTER[ep]
    sdir = scene / 'score_input/meshes' / f'episode_{ep:06d}'
    if sdir.is_dir() and not force:
        have = sorted(p.stem for p in sdir.glob('*.obj'))
        if have == sorted(names):
            return dict(scene=str(scene), episode=ep, status='already_present', meshes=have)
        if have:  # a different, non-empty set: never overwrite silently
            raise SystemExit(f'{scene}: score_input/meshes exists with {have}, roster {names}; use --force')
        # empty dir left by an aborted run: rewrite
    if sdir.exists(): shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    made = {}
    for name in names:
        src = proc / f'{seq}_{name}.obj'
        if not src.is_file() and len(names) == 1 and (proc / f'{seq}.obj').is_file():
            src = proc / f'{seq}.obj'  # single-object bundle: the loader names the body after the sequence alone
        if not src.is_file():
            cands = [p for p in proc.glob(f'*_{name}.obj')] or ([p for p in proc.glob('*.obj') if not p.name.endswith('_visual.obj')] if len(names) == 1 else [])
            assert len(cands) == 1, (str(scene), name, cands)
            src = cands[0]
        dst = sdir / f'{name}.obj'
        shutil.copy2(src, dst)
        extra = []
        for line in open(src):
            if line.split()[:1] == ['mtllib']:
                mtl = proc / line.split()[1]
                if mtl.is_file():
                    shutil.copy2(mtl, sdir / mtl.name); extra.append(mtl.name)  # the obj's mtllib names it
                    for ml in open(mtl):
                        if ml.split()[:1] == ['map_Kd']:
                            tex = proc / ml.split()[1]
                            if tex.is_file(): shutil.copy2(tex, sdir / tex.name); extra.append(tex.name)
        made[name] = dict(source=str(src), vertices=n_vertices(dst), sha256=sha(dst), with_files=extra)
    bundle_names = None
    bundle = Path(check.get('bundle_cleaned', '')) if check.get('bundle_cleaned') else None
    if bundle and (bundle / 'result.npz').is_file():
        bundle_names = write_results(bundle, scene, ep)
        if bundle_names is not None and bundle_names != names:
            raise SystemExit(f'{scene}: bundle object_names {bundle_names} != roster {names}')
    info = dict(scene=str(scene), episode=ep, sequence_id=seq, status='written', roster=names, meshes=made,
                results_npy=str(scene / f'score_input/results/episode_{ep:06d}.npy') if bundle_names is not None or (bundle and (bundle / 'result.npz').is_file()) else None,
                bundle=str(bundle) if bundle else None)
    (scene / 'SCORE_INPUT_EXPORT.json').write_text(json.dumps(info, indent=1) + '\n')
    return info

if __name__ == '__main__':
    force = '--force' in sys.argv
    failed = []
    for a in sys.argv[1:]:
        if a == '--force': continue
        try:
            o = export(Path(a).resolve(), force)
            print(json.dumps({'scene': a, **{k: o[k] for k in ('episode', 'status')}, 'meshes': {k: v['vertices'] for k, v in o['meshes'].items()} if isinstance(o['meshes'], dict) else o['meshes']}), flush=True)
        except (Exception, SystemExit) as e:
            failed.append(a); print(json.dumps({'scene': a, 'status': 'FAILED', 'error': repr(e)[:300]}), flush=True)
    print(json.dumps({'failed': failed}))
    sys.exit(1 if failed else 0)
