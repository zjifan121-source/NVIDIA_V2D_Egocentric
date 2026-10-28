"""Inventory completed moving-reference policies; validate finalists before selection.

Own-reference scores across reconstructions describe tracking, not hidden task accuracy.
No still references, missing frames, non-finite poses or incompatible evaluator hashes.
"""
import collections
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = Path('/data/user_data/edmondwa/v2d')
NIGHT = DATA / 'ken_night/night_2026-10-24'
HASH = 'b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1'


def run_folders(root):
    for parent in root.iterdir():
        yield parent, parent.name
        if parent.is_dir():
            for early in parent.glob('early_*'):
                if early.is_dir():
                    yield early, parent.name


def board_bottom_lifts(df, meshdir, objects):
    """Actual mesh clearance, so rotation about the table cannot masquerade as pickup."""
    lifts = []
    for slot, obj in enumerate(objects):
        mesh = meshdir / (obj['name'] + '.obj')
        vertices = np.array([[float(x) for x in line.split()[1:4]]
                             for line in mesh.read_text().splitlines() if line.startswith('v ')])
        body = df[df.object_id == slot].sort_values('reference_frame')
        quat = body[[f'achieved_quaternion_{a}' for a in 'xyzw']].to_numpy()
        rotation_z = Rotation.from_quat(quat).as_matrix()[:, 2, :]
        bottom = (rotation_z @ vertices.T).min(axis=1) + body.achieved_position_z.to_numpy()
        lifts.append(float(bottom.max() - bottom[0]))
    return lifts


def main():
    episodes = {e['episode']: e for e in json.loads((REPO / 'tools/track3/episodes.json').read_text())['episodes'] if e['split'] == 'evaluation'}
    candidates = collections.defaultdict(list)
    exclusions = []
    for root in (DATA / 'flash_runs/diag', NIGHT / 'runs'):
        for run, parent_name in run_folders(root):
            match = re.match(r'(?:kv_)?ep(\d+)_', parent_name)
            if not match or int(match[1]) not in episodes:
                continue
            ep = int(match[1])
            result = run / 'results' / f'episode_{ep:06d}.json'
            rollout = result.with_suffix('.parquet')
            if not result.is_file() or not rollout.is_file():
                continue
            if root.name == 'diag' and not (run / 'eval.done').is_file():
                continue
            record = json.loads(result.read_text())
            ref = record.get('parquet', '')
            if '/assets/' not in ref or any(s in (parent_name + ' ' + ref).lower() for s in ('still', 'half', 'oracle', 'scan', '_uw')):
                continue
            if record.get('non_finite_worlds') or record.get('step_count') != episodes[ep]['num_frames']:
                continue
            scene = Path(ref.split('/assets/')[0])
            if str(scene).startswith('/data/user_data/haisenk/'):
                scene = NIGHT / 'scenes' / str(scene).split('/load_retarget/')[-1]
            meshdir = scene / 'score_input/meshes' / f'episode_{ep:06d}'
            if not all((meshdir / (o['name'] + '.obj')).is_file() for o in episodes[ep]['objects']):
                exclusions.append([run.name, 'missing scoring meshes'])
                continue
            label = run.name if run.name == parent_name else parent_name + '/' + run.name
            candidates[ep].append(dict(run=str(run), scene=str(scene), kind='policy', tracking=float(record['metrics']['add_auc']), name=label))
    plan, checks = {}, []
    for ep, roster in sorted(episodes.items()):
        for candidate in sorted(candidates[ep], key=lambda c: c['tracking'], reverse=True):
            table = pq.read_table(Path(candidate['run']) / 'results' / f'episode_{ep:06d}.parquet')
            md = table.schema.metadata or {}
            df = table.to_pandas()
            n, count = roster['num_frames'], len(roster['objects'])
            problems = []
            if md.get(b'flash_chord.eval_code_sha256', b'').decode() != HASH:
                problems.append('evaluator hash')
            if float(md.get(b'flash_chord.control_fps', b'0')) != 20.254119873046875:
                problems.append('fps')
            if len(df) != n * count or sorted(df.reference_frame.unique()) != list(range(n)) or sorted(df.object_id.unique()) != list(range(count)):
                problems.append('coverage')
            if not df.environment_valid.all() or not np.isfinite(df[[c for c in df if c.startswith(('achieved_', 'reference_'))]].to_numpy()).all():
                problems.append('invalid poses')
            movements = []
            for _, body in df.groupby('object_id'):
                xyz = body.sort_values('reference_frame')[[f'reference_position_{a}' for a in 'xyz']].to_numpy()
                movements.append(float(np.linalg.norm(xyz - xyz[0], axis=1).max()))
            if max(movements) < 0.03:
                problems.append('reference lacks substantial motion')
            bottom_lifts = None
            if ep == 25 and not problems:
                meshdir = Path(candidate['scene']) / 'score_input/meshes/episode_000025'
                bottom_lifts = board_bottom_lifts(df, meshdir, roster['objects'])
                if min(bottom_lifts) < 0.05:
                    problems.append('episode25 requires both actual board bottoms to lift >=5cm')
            if problems:
                exclusions.append([candidate['name'], problems])
                continue
            plan[str(ep)] = dict(candidate, reference_max_displacement_m=movements)
            checks.append(dict(episode=ep, rows=len(df), frames=n, issues=[],
                               actual_mesh_bottom_lifts_m=bottom_lifts))
            break
    if len(plan) != len(episodes):
        raise RuntimeError(f'Only {len(plan)} of {len(episodes)} episodes have valid moving policies')
    (HERE / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    summary = dict(episodes=len(plan), mean_own_reference=sum(c['tracking'] for c in plan.values()) / len(plan), all_references_move=True,
                   checks=checks, exclusions=exclusions, candidates={str(ep): sorted(cs, key=lambda c: c['tracking'], reverse=True) for ep, cs in candidates.items()},
                   caveat='Own-reference tracking over multiple reconstruction namespaces; not a hidden-ground-truth estimate')
    (HERE / 'tracking_inventory.json').write_text(json.dumps(summary, indent=2) + '\n')
    for ep, entry in plan.items():
        print(f'{ep:>2} {entry["name"]:30} {entry["tracking"]:.4f}')
    print(f'Mean tracking: {summary["mean_own_reference"]:.6f}')


if __name__ == '__main__':
    main()
