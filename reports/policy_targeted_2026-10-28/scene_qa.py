"""Pre-training check of built Track 3 scenes: will each one train, and is its hand reference usable?

    python scene_qa.py SCENES_ROOT [--out report.json]      # SCENES_ROOT holds ep<N>/ scene dirs

Per episode:
  FAIL  the trainer or the packer will break: no Cylinder/Cube shapes in the support USDA, an object mesh missing or
        unnamed (None.obj), or a frame count that differs from episodes.json.
  WARN  it will train but probably plateau: median fingertip-to-object gap over lifted frames above 4 cm (the hand
        reference does not hold the object; send back to reconstruction), or object orientation tremor above
        3 deg/frame^2 (try the --smooth-object-rotation build). Ken's scenes sit at 0.1-3 cm and about 1 deg/frame^2;
        on Wentao's scored20 bundle every episode above 4 cm stayed below 0.81 tracking until its grasp was repaired.
  PASS  otherwise.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
EPISODES = {e['episode']: e for e in json.loads((REPO / 'tools/track3/episodes.json').read_text())['episodes']}
GAP_CM, TREMOR = 4.0, 3.0

# fingertip-gap measurement shared with gapcmp.py (thumb/index/middle tip to object surface on lifted frames)
_src = (REPO / 'reports/policy_targeted_2026-10-28/gapcmp.py').read_text().split('\nplan = json.loads')[0]
# an unnamed None.obj left beside the named mesh would be taken for the object; skip it
_glob = "meshes = list((scene / 'score_input/meshes' / f'episode_{ep:06d}').glob('*.obj'))"
assert _glob in _src
_src = _src.replace(_glob, _glob[:-1] + " if p.stem != 'None']").replace("meshes = list(", "meshes = [p for p in ", 1)
_argv, sys.argv = sys.argv, sys.argv[:1] + ['/dev/null']
exec(_src)
sys.argv = _argv


def parquet_of(scene):
    p = next((scene / 'assets/human_motion_data/ego_recon/processed').glob('sequence_id=*/robot_name=sharpa_wave'))
    return p if p.is_file() else next(p.glob('*.parquet'))


def check(scene, ep):
    problems, warnings, info = [], [], {}
    support = [p for p in scene.glob('assets/human_motion_data/ego_recon/reconstructed_stage/*_support.usda')]
    shapes = sum(len(re.findall(r'def (?:Cylinder|Cube)\b', p.read_text())) for p in support)
    info['support_shapes'] = shapes
    if not shapes:
        problems.append('support USDA has no Cylinder/Cube shapes (the trainer refuses it)')
    want = [o['name'] for o in EPISODES[ep]['objects']]
    meshes = {p.stem for p in (scene / 'score_input/meshes' / f'episode_{ep:06d}').glob('*.obj')}
    missing = [o for o in want if o not in meshes]
    if missing:
        problems.append(f'object meshes missing: want {want}, have {sorted(meshes)}')
    t = pq.read_table(parquet_of(scene), columns=['object_body_wxyz'])
    quat = np.asarray(t['object_body_wxyz'][0].as_py())
    info['frames'] = len(quat)
    if len(quat) != EPISODES[ep]['num_frames']:
        problems.append(f"{len(quat)} frames, episodes.json says {EPISODES[ep]['num_frames']}")
    tremor = {}
    for slot in range(quat.shape[1]):
        r = Rotation.from_quat(quat[:, slot, [1, 2, 3, 0]])
        tremor[slot] = round(float(np.mean(np.abs(np.diff(np.degrees((r[1:] * r[:-1].inv()).magnitude()))))), 2)
    try:
        gaps = scene_stats(scene, ep)
    except Exception as e:  # a scene the gap code cannot read is itself a finding
        gaps = {}
        warnings.append(f'fingertip gap not measured: {e!r}'[:160])
    info['objects'] = {}
    for slot, (name, g) in enumerate(gaps.items()):
        info['objects'][name] = dict(g, tremor_deg_per_frame2=tremor.get(slot))
        if g.get('median_gap_cm') is not None and g['median_gap_cm'] > GAP_CM:
            warnings.append(f"{name}: fingertips {g['median_gap_cm']} cm from the object while lifted")
        if tremor.get(slot, 0) > TREMOR:
            warnings.append(f'{name}: orientation tremor {tremor[slot]} deg/frame^2')
    status = 'FAIL' if problems else 'WARN' if warnings else 'PASS'
    return dict(episode=ep, status=status, problems=problems, warnings=warnings, **info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    rows = []
    for scene in sorted(args.root.glob('ep*'), key=lambda p: int(re.sub(r'\D', '', p.name) or 0)):
        m = re.fullmatch(r'ep(\d+)', scene.name)
        if not m or int(m[1]) not in EPISODES:
            continue
        try:
            row = check(scene, int(m[1]))
        except Exception as e:
            row = dict(episode=int(m[1]), status='FAIL', problems=[f'scene unreadable: {e!r}'[:200]], warnings=[])
        rows.append(row)
        print(f"ep{row['episode']:>3} {row['status']:4}  " + '; '.join(row['problems'] + row['warnings']), flush=True)
    counts = {s: sum(r['status'] == s for r in rows) for s in ('PASS', 'WARN', 'FAIL')}
    print(counts)
    if args.out:
        args.out.write_text(json.dumps(dict(root=str(args.root), counts=counts, episodes=rows), indent=1))


if __name__ == '__main__':
    main()
