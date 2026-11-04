"""Graft one reconstruction's grasps onto another's object trajectories (bundle level, no GT).

While a hand holds an object in the donor bundle, its wrist pose relative to that object (and its finger pose)
is carried over onto the target bundle's object trajectory; away from contact the target's own hand is kept,
with a smooth blend in between. Donor and target must come from the same video (same frame count) and list
the same object names; meshes may differ, in which case the donor mesh is registered onto the target's.

    python tools/track3/graft_grasps.py --target <bundle dir> --donor <bundle dir> --out <bundle dir> \
        [--object-rotation-from-donor blue_cup]

--object-rotation-from-donor replaces those objects' world rotation with the donor's (mapped into the target
world through the average camera-to-camera alignment); their positions stay the target's.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

NEAR_CM, FAR_CM = 6.0, 12.0  # donor wrist-to-surface distance: full graft below NEAR, none beyond FAR


def slots(data):
    n = len(data["object_names"]) if "object_names" in data else 1
    return [("object_to_world_transform", "object_scale", "mesh.obj")] + [
        (f"object_{i}_to_world_transform", f"object_{i}_scale", f"mesh_{i}.obj") for i in range(1, n)]


def mesh(bundle, data, scale_key, mesh_file):
    m = trimesh.load(bundle / mesh_file, force="mesh", process=False)
    m.apply_scale(float(data.get(scale_key, 1.0)))
    return m


def register(donor_mesh, target_mesh):
    """Donor-local -> target-local rigid transform (identity when the meshes match)."""
    if len(donor_mesh.vertices) == len(target_mesh.vertices) and np.allclose(donor_mesh.vertices, target_mesh.vertices):
        return np.eye(4), 0.0
    # principal-axis starts over all 24 proper axis permutations/flips, then rigid ICP without reflection
    pts = donor_mesh.sample(2000, seed=0) if "seed" in donor_mesh.sample.__code__.co_varnames else donor_mesh.sample(2000)
    cd, ct = donor_mesh.centroid, target_mesh.centroid
    Pd = np.linalg.svd(np.cov((donor_mesh.vertices - cd).T))[0]
    Pt = np.linalg.svd(np.cov((target_mesh.vertices - ct).T))[0]
    best = (np.inf, np.eye(4))
    for perm in Rotation.create_group("O").as_matrix():
        R = Pt @ perm @ Pd.T
        if np.linalg.det(R) < 0:
            R = Pt @ (perm @ np.diag([1, 1, -1])) @ Pd.T
        init = np.eye(4)
        init[:3, :3] = R
        init[:3, 3] = ct - R @ cd
        A, _, cost = trimesh.registration.icp(pts, target_mesh, initial=init, threshold=1e-6, max_iterations=60,
                                             reflection=False, scale=False)
        if cost < best[0]:
            best = (cost, A)
    return best[1], float(best[0])


def world_alignment(target, donor):
    """Donor-world -> target-world rotation+translation, averaged over frames where both cameras are valid."""
    ct, cd = target["camera_to_world_transform"].astype(float), donor["camera_to_world_transform"].astype(float)
    ok = target.get("camera_is_valid", np.ones(len(ct))).astype(bool) & donor.get("camera_is_valid", np.ones(len(cd))).astype(bool)
    M = ct[ok] @ np.linalg.inv(cd[ok])
    R = Rotation.from_matrix(M[:, :3, :3])
    out = np.eye(4)
    out[:3, :3] = R.mean().as_matrix()
    out[:3, 3] = np.median(M[:, :3, 3], axis=0)
    spread = np.degrees(np.median((R.mean().inv() * R).magnitude()))
    return out, float(spread)


def surface_cm(m, T_obj, points_world):
    local = (np.linalg.inv(T_obj) @ np.c_[points_world, np.ones(len(points_world))].T).T[:, :3]
    return np.abs(trimesh.proximity.signed_distance(m, local)) * 100


def blend(a, b, w):
    """Per-frame blend of 4x4 poses a->b by weight w (slerp rotations, lerp translations)."""
    out = a.copy()
    for t, wt in enumerate(w):
        if wt <= 0:
            continue
        r = Slerp([0, 1], Rotation.from_matrix([a[t, :3, :3], b[t, :3, :3]]))(wt)
        out[t, :3, :3] = r.as_matrix()
        out[t, :3, 3] = (1 - wt) * a[t, :3, 3] + wt * b[t, :3, 3]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--donor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--object-rotation-from-donor", nargs="*", default=[])
    args = ap.parse_args()

    tgt, don = dict(np.load(args.target / "result.npz")), dict(np.load(args.donor / "result.npz"))
    n = len(tgt["camera_to_world_transform"])
    assert len(don["camera_to_world_transform"]) == n, "donor and target differ in frame count"
    # a single-object bundle may not list its object name; pair the lone objects directly
    tnames = [str(x) for x in tgt["object_names"]] if "object_names" in tgt else ["object"]
    dnames = [str(x) for x in don["object_names"]] if "object_names" in don else ["object"]
    if len(tnames) == 1 and len(dnames) == 1:
        dnames = tnames
    info = {"target": str(args.target), "donor": str(args.donor), "objects": {}, "hands": {}}
    Mw, spread = world_alignment(tgt, don)
    info["world_alignment_spread_deg"] = spread

    objs = []
    for (tk, ts, tm), name in zip(slots(tgt), tnames):
        j = next(i for i, d in enumerate(dnames) if d == name or d.endswith("_" + name))
        dk, ds, dm = slots(don)[j]
        mt, md = mesh(args.target, tgt, ts, tm), mesh(args.donor, don, ds, dm)
        A, cost = register(md, mt)
        if name in args.object_rotation_from_donor:
            Td = don[dk].astype(float)
            R = (Mw[:3, :3] @ Td[:, :3, :3]) @ np.linalg.inv(A[:3, :3])
            T = tgt[tk].astype(float)
            T[:, :3, :3] = R
            tgt[tk] = T.astype(tgt[tk].dtype)
        objs.append(dict(name=name, mt=mt, md=md, A=A, Tt=tgt[tk].astype(float), Td=don[dk].astype(float)))
        info["objects"][name] = {"registration_cost": cost, "rotation_from_donor": name in args.object_rotation_from_donor}

    c2w = tgt["camera_to_world_transform"].astype(float)
    for side in ("left", "right"):
        key = f"hand_{side}_wrist_to_world_transform"
        if key not in tgt or key not in don:
            continue
        Hd, Ht = don[key].astype(float), tgt[key].astype(float)
        dvalid = don[f"hand_{side}_is_valid"].astype(bool)
        dist = np.full((len(objs), n), np.inf)
        for i, o in enumerate(objs):
            dist[i] = [surface_cm(o["md"], o["Td"][t], Hd[t, None, :3, 3])[0] for t in range(n)]
        nearest = dist.argmin(0)
        dmin = dist.min(0)
        w = np.clip((FAR_CM - dmin) / (FAR_CM - NEAR_CM), 0, 1) * dvalid
        w = np.clip(gaussian_filter1d(w, 2.0), 0, 1)
        graft = Ht.copy()
        for t in range(n):
            o = objs[nearest[t]]
            graft[t] = o["Tt"][t] @ o["A"] @ np.linalg.inv(o["Td"][t]) @ Hd[t]
        new = blend(Ht, graft, w)
        w2c = np.linalg.inv(c2w) @ new
        tgt[key] = new.astype(tgt[key].dtype)
        tgt[f"hand_{side}_wrist_to_camera_transform"] = w2c.astype(np.float32)
        tgt[f"hand_{side}_wrist_orient_in_camera"] = Rotation.from_matrix(w2c[:, :3, :3]).as_rotvec().astype(tgt[f"hand_{side}_wrist_orient_in_camera"].dtype)
        tgt[f"hand_{side}_wrist_trans_in_camera"] = w2c[:, :3, 3].astype(tgt[f"hand_{side}_wrist_trans_in_camera"].dtype)
        fk = f"hand_{side}_finger_pose"
        if fk in tgt and fk in don and tgt[fk].shape == don[fk].shape:
            wf = w.reshape((-1,) + (1,) * (tgt[fk].ndim - 1))
            tgt[fk] = ((1 - wf) * tgt[fk] + wf * don[fk]).astype(tgt[fk].dtype)
        valid = tgt[f"hand_{side}_is_valid"].astype(bool) | (w > 0.5)
        tgt[f"hand_{side}_is_valid"] = valid.astype(tgt[f"hand_{side}_is_valid"].dtype)
        info["hands"][side] = {"grafted_frames": int((w > 0.5).sum()), "blended_frames": int(((w > 0) & (w <= 0.5)).sum()),
                               "objects": {objs[i]["name"]: int(((nearest == i) & (w > 0.5)).sum()) for i in range(len(objs))}}

    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.target.iterdir():
        if f.is_file() and f.name != "result.npz":
            shutil.copy2(f, args.out / f.name)
    np.savez(args.out / "result.npz", **tgt)
    (args.out / "graft_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
