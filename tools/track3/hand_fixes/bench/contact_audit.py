"""hand5: audit the reference contact labels the policy is rewarded with (missed/unintended contact terms).
Per hand side and frame: number of contact points (mano_<side>_object_contact_part_ids > 0) and, per object, how many of
those contact points lie within 2 cm of that object's centre-surface proxy (object pose from the parquet + mesh radius).
Prints, for every object, the frames where it is lifted (> 2 cm above its first-frame height) and the fraction of those
frames that have >= 1 contact point near it from each hand. usage: contact_audit.py <scene dir> [<scene dir> ...]"""
import sys, glob, os
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

def col(t, name):
    a = t.column(name).combine_chunks()
    v = a.to_pylist()[0]
    return np.array(v, dtype=object) if v is not None else None

def arr(t, name):
    v = t.column(name).combine_chunks().to_pylist()[0]
    return np.array(v, dtype=float)

for scene in [a for a in sys.argv[1:] if os.path.isdir(a)]:
    f = sorted(glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet")))[0]
    t = pq.read_table(f)
    names = t.column_names
    print("=====", os.path.basename(scene.rstrip("/")))
    if "obj_debug" in sys.argv: print([n for n in names if "object" in n or "contact" in n])
    bodies = t.column("object_body_names").combine_chunks().to_pylist()[0]
    pos_key = [n for n in names if n in ("object_body_position", "object_positions", "object_body_positions", "object_position")]
    P = arr(t, pos_key[0]) if pos_key else None  # expect (T, n_obj, 3) or (n_obj, T, 3)
    radius = np.array(t.column("object_mesh_radius").combine_chunks().to_pylist()[0], dtype=float).ravel()
    if P is not None and P.ndim == 3 and P.shape[0] == len(bodies) and P.shape[1] != len(bodies): P = P.transpose(1, 0, 2)
    for side in ("left", "right"):
        pid = arr(t, f"mano_{side}_object_contact_part_ids")
        cp = arr(t, f"mano_{side}_object_contact_positions")
        n = (pid > 0).reshape(pid.shape[0], -1).sum(1)
        print(f" {side}: frames {pid.shape[0]}, frames with any contact {int((n>0).sum())}, contact pts/frame max {int(n.max())}")
        if P is None: continue
        for j, b in enumerate(bodies):
            z = P[:, j, 2]; lifted = np.where(z - z[0] > 0.02)[0]
            if not len(lifted): continue
            near = []
            for i in lifted:
                m = pid[i].ravel() > 0
                pts = cp[i].reshape(-1, 3)[m]
                near.append(int(len(pts) and np.min(np.linalg.norm(pts - P[i, j], axis=1)) < radius[j] + 0.02))
            near = np.array(near)
            gaps = lifted[near == 0]
            print(f"   {b}: lifted {len(lifted)} frames [{lifted.min()}..{lifted.max()}], with {side} contact near it {near.sum()} "
                  f"({near.mean():.0%}); no-contact lifted frames e.g. {gaps[:6].tolist()}..{gaps[-3:].tolist() if len(gaps) else ''}")
