"""Reference grasp check, Ken's selected scenes vs Wentao's alignment mix: for frames where an object is lifted >2 cm,
the nearest thumb/index/middle tip distance to the object surface (either hand). Reference geometry only."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

PLAN = Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/plan.json')
WENTAO = Path('/data/user_data/edmondwa/v2d/wentao_mix/scenes')


def scene_stats(scene, ep):
    parquet = next((scene / 'assets/human_motion_data/ego_recon/processed').glob('sequence_id=*/robot_name=sharpa_wave'))
    parquet = parquet if parquet.is_file() else next(parquet.glob('*.parquet'))
    t = pq.read_table(parquet)
    cell = lambda k: np.asarray(t[k][0].as_py())
    pos, quat = cell('object_body_position'), cell('object_body_wxyz')
    meshes = list((scene / 'score_input/meshes' / f'episode_{ep:06d}').glob('*.obj'))
    out = {}
    for slot, name in enumerate(t['object_body_names'][0].as_py()):
        cand = [p for p in meshes if name.endswith('_' + p.stem)] or (meshes if len(meshes) == 1 else [])
        mesh = trimesh.load(cand[0], force='mesh', process=False)
        rot = Rotation.from_quat(quat[:, slot, [1, 2, 3, 0]]).as_matrix()
        z0 = pos[0, slot, 2]
        frames = [f for f in range(0, len(pos), 5) if pos[f, slot, 2] - z0 > 0.02]
        gaps = []
        for f in frames:
            best = 99.0
            for side in ('left', 'right'):
                names = t[f'{side}_robot_frame_names'][0].as_py()
                ids = [names.index(f'{side}_{d}_tip_site') for d in ('thumb', 'index', 'middle')]
                tips = cell(f'robot_{side}_frames')[f, ids, :3]
                _, d, _ = trimesh.proximity.closest_point(mesh, (tips - pos[f, slot]) @ rot[f])
                best = min(best, 100 * d.min())
            gaps.append(best)
        short = name.split('attach_')[-1]
        out[short] = dict(lifted=len(frames), median_gap_cm=round(float(np.median(gaps)), 2) if gaps else None,
                          frac_over_2cm=round(float(np.mean(np.array(gaps) > 2)), 2) if gaps else None)
    return out


plan = json.loads(PLAN.read_text())
rows = {}
for ep in sorted(int(k) for k in plan):
    r = {}
    for label, scene in (('ken', Path(plan[str(ep)]['scene'])), ('wentao', WENTAO / f'ep{ep}')):
        try:
            r[label] = scene_stats(scene, ep)
        except Exception as e:
            r[label] = f'error: {e!r}'[:200]
    rows[ep] = r
    print(ep, json.dumps(r), flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
