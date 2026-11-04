"""Lifted frames per object with no reference contact near it from EITHER hand (what the contact reward penalises).
usage: contact_union.py <scene dir> ..."""
import sys, glob, os
import numpy as np, pyarrow.parquet as pq

def arr(t, name):
    return np.array(t.column(name).combine_chunks().to_pylist()[0], dtype=float)

def runs(idx):
    if not len(idx): return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != p + 1: out.append((s, p)); s = i
        p = i
    return out + [(s, p)]

for scene in sys.argv[1:]:
    fs = sorted(glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet")))
    if not fs: print("=====", scene, "no parquet"); continue
    t = pq.read_table(fs[0]); names = t.column_names
    bodies = t.column("object_body_names").combine_chunks().to_pylist()[0]
    pk = [n for n in names if n in ("object_body_position", "object_positions", "object_body_positions", "object_position")][0]
    P = arr(t, pk); radius = arr(t, "object_mesh_radius").ravel()
    if P.ndim == 3 and P.shape[0] == len(bodies) and P.shape[1] != len(bodies): P = P.transpose(1, 0, 2)
    side = {s: (arr(t, f"mano_{s}_object_contact_part_ids"), arr(t, f"mano_{s}_object_contact_positions")) for s in ("left", "right")}
    print("=====", scene, "frames", P.shape[0])
    for j, b in enumerate(bodies):
        z = P[:, j, 2]; lifted = np.where(z - z[0] > 0.02)[0]
        if not len(lifted): continue
        miss = []
        for i in lifted:
            hit = False
            for pid, cp in side.values():
                m = pid[i].ravel() > 0; pts = cp[i].reshape(-1, 3)[m]
                hit |= bool(len(pts) and np.min(np.linalg.norm(pts - P[i, j], axis=1)) < radius[j] + 0.02)
            if not hit: miss.append(i)
        print(f"  {b.split('_attach_')[-1]:24} lifted {len(lifted):4}  no-contact {len(miss):4} ({len(miss)/len(lifted):.0%})  runs {[(int(a), int(c)) for a, c in runs(miss) if c - a >= 2][:8]}")
