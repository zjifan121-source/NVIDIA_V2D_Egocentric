"""Extend a floating cup mesh downward to the depth-measured table at frame 0 (observation only): the lowest band of the cup is extruded
straight down by the frame-0 gap between the mesh bottom and the depth-fitted table plane, and the bottom is closed. Rim, body frame and
everything above the band are unchanged, so the tracked pose is kept and only the cup's length changes. New vertices and faces are appended
to the OBJ text; texture coordinates are kept for the original faces. Only cups (blue_cup, beige_cup) with gap > --min-gap-cm change.
usage: extend_cups_down.py <bundle dir> <nx> <ny> <nz> <cx> <cy> <cz> [--min-gap-cm 0.5] [--band-mm 8]"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
B = Path(sys.argv[1]); n = np.array([float(x) for x in sys.argv[2:5]]); c = np.array([float(x) for x in sys.argv[5:8]])
mg = float(sys.argv[sys.argv.index('--min-gap-cm') + 1]) if '--min-gap-cm' in sys.argv else 0.5
band = (float(sys.argv[sys.argv.index('--band-mm') + 1]) if '--band-mm' in sys.argv else 8.0) / 1000
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
    faces = []
    for l in lines:
        if l.startswith('f '):
            idx = [int(t.split('/')[0]) for t in l.split()[1:]]
            for k in range(1, len(idx) - 1): faces.append((idx[0], idx[k], idx[k + 1]))
    faces = np.array(faces, int) - 1
    P = d[key][0].astype(float); W = (s * v) @ P[:3, :3].T + P[:3, 3]; gap = float(((W - c) @ n).min())
    r = dict(object=name, mesh=mesh.name, gap_to_table_cm_f0=round(gap * 100, 2), scale=s, extended=False, vertices=int(len(v)), faces=int(len(faces)))
    if name in ('blue_cup', 'beige_cup') and gap > mg / 100:
        down = P[:3, :3].T @ (-n); down /= np.linalg.norm(down); ext = (gap / s) * down  # mesh units, body frame
        h = v @ down; lo = h.max() - band / s; in_band = h >= lo
        fb = faces[in_band[faces].all(1)]  # faces fully inside the lowest band
        if len(fb) == 0: fb = faces[in_band[faces].any(1)]
        edges = {}
        for f in fb:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                e = (min(a, b), max(a, b)); edges[e] = edges.get(e, 0) + 1
        boundary = [e for e, cnt in edges.items() if cnt == 1]
        used = sorted(set(x for e in boundary for x in e)); newidx = {u: len(v) + k for k, u in enumerate(used)}
        v2 = np.vstack([v, v[used] + ext]); new_faces = []
        for a, b in boundary:
            a2, b2 = newidx[a], newidx[b]; new_faces += [(a, b, b2), (a, b2, a2)]
        cap_faces = [(newidx[f[0]], newidx[f[1]], newidx[f[2]]) for f in fb if all(x in newidx for x in f)]  # closed bottom: copy of the band faces that lie on the boundary
        r.update(sha_before=sha(mesh), band_vertices=int(in_band.sum()), band_faces=int(len(fb)), boundary_edges=int(len(boundary)), new_vertices=int(len(used)), new_faces=int(len(new_faces) + len(cap_faces)))
        out = lines + [''] + ['v ' + ' '.join(f'{x:.6f}' for x in v2[k]) for k in range(len(v), len(v2))] + ['f %d %d %d' % (a + 1, b + 1, cc + 1) for a, b, cc in new_faces + cap_faces]
        mesh.write_text('\n'.join(out) + '\n')
        W2 = (s * v2) @ P[:3, :3].T + P[:3, 3]
        r.update(extended=True, extension_cm=round(gap * 100, 2), gap_after_cm=round(float(((W2 - c) @ n).min()) * 100, 3), sha_after=sha(mesh), rim_unchanged=True)
    rec.append(r)
(B / 'EXTEND_CUPS.json').write_text(json.dumps(dict(plane_n=[float(x) for x in n], plane_c=[float(x) for x in c], min_gap_cm=mg, band_mm=band * 1000, objects=rec), indent=1) + '\n')
print('EXTEND', json.dumps(rec))
