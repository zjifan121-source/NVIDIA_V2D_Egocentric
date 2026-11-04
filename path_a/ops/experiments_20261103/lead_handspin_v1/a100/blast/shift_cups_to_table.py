"""Observation-based cup seating (the team's ep8 T6 idea): at frame 0 a tracked cup floats above the depth-measured table by a gap; move the cup's
mesh in its own body frame by that gap along the body-frame 'down' direction so the cup touches the table at frame 0. The mesh shape is unchanged
(CD-O neutral); the implied cup trajectory is lower by the gap at every frame. Only cups (blue_cup, beige_cup) with gap > --min-gap-cm move.
usage: shift_cups_to_table.py <bundle dir> <nx> <ny> <nz> <cx> <cy> <cz> [--min-gap-cm 0.5]"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
B = Path(sys.argv[1]); n = np.array([float(x) for x in sys.argv[2:5]]); c = np.array([float(x) for x in sys.argv[5:8]])
mg = float(sys.argv[sys.argv.index('--min-gap-cm') + 1]) if '--min-gap-cm' in sys.argv else 0.5
n = n / np.linalg.norm(n)
if n[2] < 0: n = -n
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
d = np.load(B / 'result.npz', allow_pickle=False); names = [str(x) for x in d['object_names']]
rec = []
for i, name in enumerate(names):
    key = 'object_to_world_transform' if i == 0 else f'object_{i}_to_world_transform'; sk = 'object_scale' if i == 0 else f'object_{i}_scale'
    mesh = B / ('mesh.obj' if i == 0 else f'mesh_{i}.obj'); s = float(d[sk]) if sk in d.files else 1.0
    lines = mesh.read_text().splitlines(); vi = [j for j, l in enumerate(lines) if l.startswith('v ')]
    v = np.array([lines[j].split()[1:4] for j in vi], float)
    P = d[key][0].astype(float); W = (s * v) @ P[:3, :3].T + P[:3, 3]; gap = float(((W - c) @ n).min())
    r = dict(object=name, mesh=mesh.name, gap_to_table_cm_f0=round(gap * 100, 2), scale=s, shifted=False)
    if name in ('blue_cup', 'beige_cup') and gap > mg / 100:
        down = P[:3, :3].T @ (-n); down /= np.linalg.norm(down); shift = (gap / s) * down
        r.update(sha_before=sha(mesh)); v2 = v + shift
        for k, j in enumerate(vi):
            parts = lines[j].split(); lines[j] = 'v ' + ' '.join(f'{x:.6f}' for x in v2[k]) + ((' ' + ' '.join(parts[4:])) if len(parts) > 4 else '')
        mesh.write_text('\n'.join(lines) + '\n')
        W2 = (s * v2) @ P[:3, :3].T + P[:3, 3]
        r.update(shifted=True, shift_mesh_units=[float(x) for x in shift], shift_cm=round(gap * 100, 2), gap_after_cm=round(float(((W2 - c) @ n).min()) * 100, 3), sha_after=sha(mesh))
    rec.append(r)
(B / 'SHIFT_CUPS.json').write_text(json.dumps(dict(plane_n=[float(x) for x in n], plane_c=[float(x) for x in c], min_gap_cm=mg, objects=rec), indent=1) + '\n')
print('SHIFT', json.dumps(rec))
