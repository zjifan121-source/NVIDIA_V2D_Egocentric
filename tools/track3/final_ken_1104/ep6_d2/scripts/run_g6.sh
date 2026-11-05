set -u; D=/data/user_data/haisenk/v2d/refine1104/ep6_d2; W=$D/work; hostname; date
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
BOB1=/data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep6; PY=/home/haisenk/anaconda3/bin/python
cat > $W/interpen.py << '__I_EOF__'
#!/usr/bin/env python3
"""Object-object interpenetration in a scene parquet (reference) and optionally a rollout parquet.
GT-free. Uses the processed meshes the scene parquet points to (same resolution as render_review).
Inside test = generalized winding number (> 0.5) of one mesh's surface samples w.r.t. the other mesh;
depth = distance to the other mesh's surface (dense samples, ~1 mm)."""
import sys, os, json, glob, argparse
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--rollout", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--nsamp", type=int, default=6000)
ap.add_argument("--cand_cm", type=float, default=1.5)
a = ap.parse_args()

proc = os.path.join(a.scene, "assets/human_motion_data/ego_recon/processed")
pqs = sorted(glob.glob(os.path.join(proc, "sequence_id=*/robot_name=*/*.parquet")), key=os.path.getmtime)
P = pqs[-1]
t = pq.read_table(P, columns=["object_body_names", "object_mesh_paths", "object_body_position", "object_body_wxyz"])
d = {c: t.column(c)[0].as_py() for c in t.column_names}
names = list(d["object_body_names"]); mps = list(d["object_mesh_paths"])
pos = np.array(d["object_body_position"], float); wxyz = np.array(d["object_body_wxyz"], float)
N, K = pos.shape[0], pos.shape[1]
print("parquet", P, "frames", N, "bodies", names, flush=True)

def load_obj(p):
    V, F = [], []
    for ln in open(p, errors="ignore"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            idx = [int(x.split("/")[0]) for x in ln.split()[1:]]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for j in range(1, len(idx) - 1):
                F.append([idx[0], idx[j], idx[j + 1]])
    return np.array(V, float), np.array(F, np.int64)

meshes = []
for k in range(K):
    mp = mps[k]
    cand = [mp, os.path.join(proc, os.path.basename(mp))]
    mpath = next(c for c in cand if c and os.path.isfile(c))
    V, F = load_obj(mpath)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(1234 + k)
    def samp(n):
        fi = rng.choice(len(F), n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        m = u + v > 1; u[m], v[m] = 1 - u[m], 1 - v[m]
        tt = tri[fi]
        return tt[:, 0] + u[:, None] * (tt[:, 1] - tt[:, 0]) + v[:, None] * (tt[:, 2] - tt[:, 0])
    S = samp(a.nsamp)
    D = samp(200000)
    # signed volume (closedness / orientation hint) + boundary edges
    vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    meshes.append(dict(name=names[k], path=mpath, V=V, F=F, tri=tri, S=S, tree=cKDTree(D),
                       vol_cm3=vol * 1e6, boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                       extents_cm=((V.max(0) - V.min(0)) * 100).round(2).tolist()))
    print(f"mesh {names[k]}: {mpath} V{len(V)} F{len(F)} vol {vol*1e6:.1f} cm3 boundary_edges {(cnt==1).sum()} nonmanifold {(cnt>2).sum()} extents_cm {meshes[-1]['extents_cm']}", flush=True)

def winding(Q, tri):
    """generalized winding number of points Q (M,3) w.r.t. triangles tri (F,3,3)."""
    out = np.zeros(len(Q))
    for s in range(0, len(Q), 64):
        q = Q[s:s + 64]
        A = tri[None, :, 0, :] - q[:, None, :]
        B = tri[None, :, 1, :] - q[:, None, :]
        C = tri[None, :, 2, :] - q[:, None, :]
        la, lb, lc = np.linalg.norm(A, axis=2), np.linalg.norm(B, axis=2), np.linalg.norm(C, axis=2)
        det = np.einsum("mfi,mfi->mf", A, np.cross(B, C))
        den = la * lb * lc + np.einsum("mfi,mfi->mf", A, B) * lc + np.einsum("mfi,mfi->mf", B, C) * la + np.einsum("mfi,mfi->mf", C, A) * lb
        out[s:s + 64] = (2 * np.arctan2(det, den)).sum(1) / (4 * np.pi)
    return out

def poses_from(pos_, wxyz_):
    R = Rot.from_quat(wxyz_[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(pos_.shape[:-1] + (3, 3))
    return R

def run(pos_, R_, tag):
    res = {}
    for i in range(K):
        for j in range(K):
            if i == j: continue
            mi, mj = meshes[i], meshes[j]
            n_in = np.zeros(N, int); dmax = np.zeros(N); dmin_surf = np.full(N, np.nan)
            for f in range(N):
                # points of mesh i in body frame of j
                Pw = mi["S"] @ R_[f, i].T + pos_[f, i]
                Pj = (Pw - pos_[f, j]) @ R_[f, j]
                dist, _ = mj["tree"].query(Pj, distance_upper_bound=0.05)
                dmin_surf[f] = np.min(dist) * 100 if np.isfinite(dist).any() else np.nan
                cand = np.where(dist < a.cand_cm / 100)[0]
                # also points far inside: check a coarse subset of points within the other mesh's bbox
                lo, hi = mj["V"].min(0), mj["V"].max(0)
                inb = np.where(np.all((Pj > lo) & (Pj < hi), axis=1))[0]
                inb = np.setdiff1d(inb, cand)[::4]
                cc = np.concatenate([cand, inb])
                if len(cc) == 0: continue
                w = winding(Pj[cc], mj["tri"])
                ins = cc[w > 0.5]
                n_in[f] = len(ins)
                if len(ins):
                    dd, _ = mj["tree"].query(Pj[ins])
                    dmax[f] = dd.max() * 100
            key = f"{mi['name']}_in_{mj['name']}"
            fr = np.where(n_in > 0)[0]
            res[key] = dict(frames_with_penetration=int(len(fr)), first=int(fr[0]) if len(fr) else None,
                            last=int(fr[-1]) if len(fr) else None,
                            max_depth_cm=float(dmax.max()), frame_of_max=int(dmax.argmax()),
                            frames_depth_over_0p5cm=int((dmax > 0.5).sum()), frames_depth_over_1cm=int((dmax > 1.0).sum()),
                            max_points_inside=int(n_in.max()), samples=int(len(mi["S"])),
                            per_frame_depth_cm=[round(float(x), 3) for x in dmax],
                            per_frame_points_inside=[int(x) for x in n_in],
                            per_frame_min_surface_dist_cm=[round(float(x), 3) if np.isfinite(x) else None for x in dmin_surf])
            segs = []
            if len(fr):
                s0 = fr[0]; p = fr[0]
                for x in fr[1:]:
                    if x != p + 1: segs.append([int(s0), int(p)]); s0 = x
                    p = x
                segs.append([int(s0), int(p)])
            res[key]["segments"] = segs
            print(f"[{tag}] {key}: frames {len(fr)} segs {segs[:12]} max depth {dmax.max():.2f} cm @f{dmax.argmax()} (>0.5cm {int((dmax>0.5).sum())} fr, >1cm {int((dmax>1).sum())} fr) max pts {n_in.max()}/{len(mi['S'])}", flush=True)
    return res

out = dict(scene=a.scene, parquet=P, meshes=[{k: m[k] for k in ("name", "path", "vol_cm3", "boundary_edges", "nonmanifold_edges", "extents_cm")} for m in meshes])
R = poses_from(pos, wxyz)
out["reference"] = run(pos, R, "reference")
if a.rollout:
    rt = pq.read_table(a.rollout)
    cols = rt.column_names
    print("rollout columns (first 60):", cols[:60], flush=True)
    out["rollout_columns"] = cols
json.dump(out, open(a.out, "w"))
print("wrote", a.out)
__I_EOF__
cat > $W/g6.py << '__G6_EOF__'
#!/usr/bin/env python3
"""refine1104 ep6 round 2: G6 checks of the built scene rf_ep6_d2 vs bob1 eb1_ep6 (pre-registered in ../gates.json).
Read-only on both scenes; prints JSON and writes WD/out/g6_result.json.  argv: [interpen_new.json interpen_bob1.json]"""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot

D = "/data/user_data/haisenk/v2d/refine1104/ep6_d2"
NEW = os.path.join(D, "scene", "rf_ep6_d2")
BOB = "/data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep6"
TG = np.load(os.path.join(D, "build", "d2_targets.npz"))
out = {}

# ---------------- (a) bundle_cleaned
a = np.load(os.path.join(NEW, "bundle_cleaned", "result.npz"))
b = np.load(os.path.join(BOB, "bundle_cleaned", "result.npz"))
m = TG["object_1_mask"].astype(bool)
diffs = {}
for k in b.files:
    if k not in a.files:
        diffs[k] = "MISSING"
        continue
    x, y = a[k], b[k]
    if x.dtype.kind in "fiub" and x.shape == y.shape:
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        if k == "object_1_to_world_transform":
            diffs[k + "_outside_mask"] = float(d[~m].max())
            diffs[k + "_inside_mask_vs_bob1"] = float(d[m].max())
        else:
            diffs[k] = float(d.max())
    else:
        diffs[k] = "same" if (x.shape == y.shape and np.all(x == y)) else f"nonnumeric/shape {x.shape} {y.shape}"
tgt = np.abs(a["object_1_to_world_transform"].astype(np.float64) - TG["object_1_to_world_transform"])[m].max()
left = {k: diffs[k] for k in diffs if k.startswith("hand_left_")}
right = {k: diffs[k] for k in diffs if k.startswith("hand_right_")}
ga = dict(pot_max=diffs.get("object_to_world_transform"), pot_cam_max=diffs.get("object_to_camera_transform"),
          spoon_outside_mask_max=diffs.get("object_1_to_world_transform_outside_mask"),
          spoon_inside_vs_targets_max=float(tgt), left_hand_max=max(v for v in left.values() if isinstance(v, float)),
          left_keys=left, right_keys=right,
          other_nonzero={k: v for k, v in diffs.items() if not k.startswith(("hand_left_", "hand_right_", "object_")) and v not in (0.0, "same")})
ga["pass"] = bool(ga["pot_max"] <= 1e-6 and ga["spoon_outside_mask_max"] <= 1e-6 and ga["spoon_inside_vs_targets_max"] <= 1e-5
                  and ga["left_hand_max"] <= 1e-6)
out["a_bundle_cleaned"] = ga

# grip: right wrist in the spoon frame during the hold, new vs bob1 (info for (c))
def grip(z):
    T = z["object_1_to_world_transform"].astype(np.float64)
    H = z["hand_right_wrist_to_world_transform"].astype(np.float64)
    return np.einsum("nij,njk->nik", np.linalg.inv(T), H)
Ga, Gb = grip(a), grip(b)
fr = np.arange(38, 180)  # hold interior (outside the 5-frame blends of [32,185])
dt = np.linalg.norm(Ga[fr, :3, 3] - Gb[fr, :3, 3], axis=1) * 100
dr = np.degrees(Rot.from_matrix(np.einsum("nji,njk->nik", Ga[fr, :3, :3], Gb[fr, :3, :3])).magnitude())
out["info_grip_offset_vs_bob1"] = dict(trans_cm_median=round(float(np.median(dt)), 3), trans_cm_max=round(float(dt.max()), 3),
                                       rot_deg_median=round(float(np.median(dr)), 2), rot_deg_max=round(float(dr.max()), 2),
                                       new_grip_const_trans_std_cm=round(float(np.linalg.norm(Ga[fr, :3, 3].std(0)) * 100), 3),
                                       bob1_grip_const_trans_std_cm=round(float(np.linalg.norm(Gb[fr, :3, 3].std(0)) * 100), 3))
# right wrist world displacement vs bob1 (how much the hand reference moves)
hw = np.linalg.norm(a["hand_right_wrist_to_world_transform"][:, :3, 3].astype(np.float64) - b["hand_right_wrist_to_world_transform"][:, :3, 3].astype(np.float64), axis=1) * 100
out["info_right_wrist_world_shift_cm"] = dict(median_hold=round(float(np.median(hw[32:186])), 3), max=round(float(hw.max()), 3),
                                              argmax=int(hw.argmax()), outside_hold_max=round(float(np.concatenate([hw[:32], hw[186:]]).max()), 4))


# ---------------- (b) scene parquet
def parq(scene):
    f = glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet"))
    assert len(f) == 1, f
    t = pq.read_table(f[0])
    g = {c: t.column(c)[0].as_py() for c in ("object_body_names", "object_mesh_paths", "object_body_position", "object_body_wxyz",
                                              "robot_left_wrist_position", "robot_right_wrist_position")}
    return f[0], g


pa, A = parq(NEW)
pb, B = parq(BOB)
posA, posB = np.array(A["object_body_position"], float), np.array(B["object_body_position"], float)
qA, qB = np.array(A["object_body_wxyz"], float), np.array(B["object_body_wxyz"], float)
lwA, lwB = np.array(A["robot_left_wrist_position"], float), np.array(B["robot_left_wrist_position"], float)
rwA, rwB = np.array(A["robot_right_wrist_position"], float), np.array(B["robot_right_wrist_position"], float)
gb = dict(parquet_new=pa, names=A["object_body_names"], shapes=[list(posA.shape), list(qA.shape)],
          pot_pos_max=float(np.abs(posA[:, 0] - posB[:, 0]).max()), pot_wxyz_max=float(np.abs(qA[:, 0] - qB[:, 0]).max()),
          left_wrist_max_m=float(np.abs(lwA - lwB).max()), right_wrist_shift_max_cm=round(float(np.linalg.norm(rwA - rwB, axis=-1).max() * 100), 3))
sp_change = np.flatnonzero((np.abs(posA[:, 1] - posB[:, 1]).max(1) > 1e-7) | (np.abs(qA[:, 1] - qB[:, 1]).max(1) > 1e-7))
gb["spoon_changed_frames"] = [int(sp_change.min()), int(sp_change.max()), int(len(sp_change))] if len(sp_change) else None


def objv(p):
    V = []
    for ln in open(p, errors="ignore"):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
    return np.array(V)


mp = A["object_mesh_paths"][1]
mp = mp if os.path.isfile(mp) else os.path.join(os.path.dirname(pa), "..", "..", os.path.basename(mp))
mu = objv(mp).mean(0)


def steps(pos, q):
    R = Rot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    rs = np.degrees(Rot.from_matrix(np.einsum("nji,njk->nik", R[:-1], R[1:])).magnitude())
    cen = np.einsum("nij,j->ni", R, mu) + pos
    return rs, np.linalg.norm(np.diff(cen, axis=0), axis=1) * 100


rs, ts = steps(posA[:, 1], qA[:, 1])
rsb, tsb = steps(posB[:, 1], qB[:, 1])
x, y = rs[35:172], ts[35:172]
gb["spoon_steps"] = dict(rot_max=round(float(x.max()), 2), rot_p90=round(float(np.percentile(x, 90)), 2), trans_max_cm=round(float(y.max()), 3),
                         trans_p90_cm=round(float(np.percentile(y, 90)), 3), bob1_rot_max=round(float(rsb[35:172].max()), 2),
                         bob1_trans_max_cm=round(float(tsb[35:172].max()), 3), mesh=mp)
gb["pass"] = bool(gb["pot_pos_max"] <= 1e-6 and gb["pot_wxyz_max"] <= 1e-6 and gb["left_wrist_max_m"] <= 1e-3 and x.max() <= 15.0
                  and np.percentile(x, 90) <= 10.6 and y.max() <= 3.0 and np.percentile(y, 90) <= 1.63)
out["b_scene_parquet"] = gb

# ---------------- (c) report.json cleaning
ra = json.load(open(os.path.join(NEW, "report.json")))
rb = json.load(open(os.path.join(BOB, "report.json")))
ca, cb = ra["cleaning"], rb["cleaning"]
holds = ca.get("attached_1", [])
right_holds = [h for h in holds if h[0] == "right"]
gc = dict(attached_1=holds, bob1_attached_1=cb.get("attached_1"), attached_0=ca.get("attached_0"),
          tilt=[ca.get("tilt_deg_0"), ca.get("tilt_deg_1"), ca.get("levelled_0"), ca.get("levelled_1")],
          bob1_tilt=[cb.get("tilt_deg_0"), cb.get("tilt_deg_1"), cb.get("levelled_0"), cb.get("levelled_1")],
          same_keys={k: ca.get(k) == cb.get(k) for k in sorted(set(ca) | set(cb))})
ok_hold = (len(right_holds) == 1 and len(holds) == 1 and abs(right_holds[0][1] - 32) <= 3 and abs(right_holds[0][2] - 185) <= 3)
grip_d = right_holds[0][3] if right_holds else None
bob_grip = cb["attached_1"][0][3]
gc["grip_distance_cm"] = grip_d
gc["bob1_grip_distance_cm"] = bob_grip
gc["grip_distance_diff_cm"] = None if grip_d is None else round(abs(grip_d - bob_grip), 2)
gc["pass"] = bool(ok_hold and grip_d is not None and abs(grip_d - bob_grip) <= 0.5 and not ca.get("attached_0")
                  and gc["tilt"] == gc["bob1_tilt"] and ca.get("tilt_deg_0") == 1.4 and ca.get("tilt_deg_1") == 15.4)
out["c_report_cleaning"] = gc

# ---------------- (e) output_check task errors
oa, ob = ra.get("output_check", {}), rb.get("output_check", {})
te_a, te_b = oa.get("task_error_cm", {}), ob.get("task_error_cm", {})
ge = dict(new=te_a, bob1=te_b, frame0_seating_new=oa.get("frame0_seating_cm"), frame0_seating_bob1=ob.get("frame0_seating_cm"),
          warnings_new=oa.get("warnings", ra.get("warnings")), warnings_bob1=ob.get("warnings", rb.get("warnings")))
try:
    ge["pass"] = bool(te_a["right"]["mean"] <= te_b["right"]["mean"] + 0.5 and te_a["right"]["max"] <= te_b["right"]["max"] + 2.0)
except Exception as e:  # noqa: BLE001
    ge["pass"] = False
    ge["error"] = repr(e)
out["e_task_error"] = ge

# ---------------- (d) interpenetration (interpen.py outputs)
if len(sys.argv) >= 3 and os.path.isfile(sys.argv[1]) and os.path.isfile(sys.argv[2]):
    ia = json.load(open(sys.argv[1]))["reference"]
    ib = json.load(open(sys.argv[2]))["reference"]
    key = [k for k in ia if k.startswith("wooden_spoon_in_")][0]
    da = np.array(ia[key]["per_frame_depth_cm"])
    db = np.array(ib[key]["per_frame_depth_cm"])
    lim = np.maximum(0.5, db + 0.2)
    bad = np.flatnonzero(da > lim)
    gd = dict(key=key, new_max=round(float(da.max()), 3), new_argmax=int(da.argmax()), bob1_max=round(float(db.max()), 3),
              bob1_argmax=int(db.argmax()), frames_over_limit=bad.tolist()[:40], n_over=int(len(bad)),
              new_frames_gt_0p5=int((da > 0.5).sum()), bob1_frames_gt_0p5=int((db > 0.5).sum()),
              pot_in_spoon_new_max=ia.get([k for k in ia if k.startswith("white_pot_in_")][0], {}).get("max_depth_cm"),
              pot_in_spoon_bob1_max=ib.get([k for k in ib if k.startswith("white_pot_in_")][0], {}).get("max_depth_cm"))
    gd["pass"] = bool(len(bad) == 0)
else:
    gd = dict(pass_=None, note="interpen outputs missing")
    gd["pass"] = False
out["d_interpen"] = gd
out["G6_pass"] = bool(all(out[k]["pass"] for k in ("a_bundle_cleaned", "b_scene_parquet", "c_report_cleaning", "d_interpen", "e_task_error")))
json.dump(out, open(os.path.join(D, "work", "out", "g6_result.json"), "w"), indent=1)
print(json.dumps(out, indent=1, default=str))
__G6_EOF__
md5sum $W/interpen.py $W/g6.py | cut -c1-8,33-
(timeout 470 $PY $W/interpen.py --scene $D/scene/rf_ep6_d2 --out $W/out/interpen_new.json > $W/out/interpen_new.log 2>&1; echo EXIT $? >> $W/out/interpen_new.log) &
P1=$!
(timeout 470 $PY $W/interpen.py --scene $BOB1 --out $W/out/interpen_bob1.json > $W/out/interpen_bob1.log 2>&1; echo EXIT $? >> $W/out/interpen_bob1.log) &
P2=$!
while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do sleep 15; echo -n .; done; echo
tail -4 $W/out/interpen_new.log $W/out/interpen_bob1.log
$PY $W/g6.py $W/out/interpen_new.json $W/out/interpen_bob1.json > $W/out/g6.log 2>&1; echo "g6 rc $?"
tail -c 7000 $W/out/g6.log; date
