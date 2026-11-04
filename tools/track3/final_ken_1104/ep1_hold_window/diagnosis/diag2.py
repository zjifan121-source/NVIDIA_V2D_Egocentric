"""ep1 on Wentao's base: object motion/flips/spin, MANO wrist steps, ctl vs raw hand diffs, contact-label windows,
phantom carry, valid gaps. GT-free. usage: diag2.py <ev_ctl.npz> <ev_raw.npz> <scene_ctl> <scene_raw> <raw merged npz> <W bundle npz>"""
import sys, glob, json, numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
ec, er = np.load(sys.argv[1]), np.load(sys.argv[2])
def tab(scene):
    PQ = glob.glob(scene + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave")[0]
    t = pq.read_table(glob.glob(PQ + "/*.parquet")[0]); return {n: t.column(n)[0].as_py() for n in t.column_names}
dc, dr = tab(sys.argv[3]), tab(sys.argv[4])
def runs(m):
    e = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]]))); return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2])]
for tag, d in (("ctl", dc), ("raw", dr)):
    opos = np.asarray(d["object_body_position"], float); ow = np.asarray(d["object_body_wxyz"], float); T = len(opos)
    rot = R.from_quat(ow[..., [1, 2, 3, 0]].reshape(-1, 4)).reshape(T, -1) if False else [R.from_quat(ow[:, b][:, [1, 2, 3, 0]]) for b in range(ow.shape[1])]
    print(f"===== {tag}: T={T} objects {d['object_body_names']}")
    for b in range(opos.shape[1]):
        st = np.linalg.norm(np.diff(opos[:, b], axis=0), axis=1) * 100
        ang = np.degrees((rot[b][1:] * rot[b][:-1].inv()).magnitude())
        rel0 = rot[b] * rot[b][0].inv()
        up0 = rot[b][0].inv().apply([0, 0, 1.0])  # world up in object frame at f0
        # object-frame axial spin: twist of rot[0]^-1 rot[t] about the object axis that is world-up at f0
        rr = rot[b][0].inv() * rot[b]
        q = rr.as_quat(); tw = 2 * np.degrees(np.arctan2(q[:, :3] @ up0, q[:, 3])); tw = (tw + 180) % 360 - 180
        tilt = np.degrees(np.arccos(np.clip(rot[b].apply(up0)[:, 2], -1, 1)))
        dz = (opos[:, b, 2] - opos[0, b, 2]) * 100
        print(f" obj{b}: step cm max {st.max():.2f}@f{st.argmax()+1} p99 {np.percentile(st,99):.2f}; rot step deg max {ang.max():.1f}@f{ang.argmax()+1} p99 {np.percentile(ang,99):.1f}; frames rot step>10deg {np.flatnonzero(ang>10)+1}")
        print(f"   tilt from f0 up: max {tilt.max():.1f}@f{tilt.argmax()} | axial twist range {tw.min():.0f}..{tw.max():.0f} deg | dz range {dz.min():.1f}..{dz.max():.1f} cm; lifted>2cm runs {runs(dz>2)}; >1cm {runs(dz>1)}")
        print("   per 10 f (dz cm / tilt / twist): " + " ".join(f"{t}:{dz[t]:.1f}/{tilt[t]:.0f}/{tw[t]:.0f}" for t in range(0, T, 10)))
    for side in ("left", "right"):
        J = np.asarray(d[f"mano_{side}_joints"], float)
        s = np.linalg.norm(np.diff(J[:, 0], axis=0), axis=1) * 1000
        rw = np.asarray(d[f"robot_{side}_wrist_position"], float); rs = np.linalg.norm(np.diff(rw, axis=0), axis=1) * 1000
        print(f" {side}: MANO wrist step mm max {s.max():.1f}@f{s.argmax()+1} p99 {np.percentile(s,99):.1f} frames>30mm {np.flatnonzero(s>30)+1}; robot wrist step max {rs.max():.1f}@f{rs.argmax()+1} p99 {np.percentile(rs,99):.1f}")
        pid = np.asarray(d[f"mano_{side}_object_contact_part_ids"], float)
        for p in (1, 2):
            print(f"   contact labels part {p}: frame runs {runs((pid == p).sum(1) > 0)}")
# ctl vs raw MANO joint differences
for side in ("left", "right"):
    Jc, Jr = ec[f"{side}_J"], er[f"{side}_J"]
    dw = np.linalg.norm(Jc[:, 0] - Jr[:, 0], axis=1) * 100
    dj = np.linalg.norm(Jc - Jr, axis=2).max(1) * 100
    print(f"ctl vs raw {side}: wrist diff cm median {np.median(dw):.2f} p90 {np.percentile(dw,90):.2f} max {dw.max():.2f}@f{dw.argmax()}; frames any joint >1cm: {runs(dj>1)}")
# phantom carry: object moving (>0.5 cm/frame or lifted >2cm) with no MANO tip/wrist of either hand within 2 cm
for tag, e in (("ctl", ec), ("raw", er)):
    op = e["opos"]; T = len(op)
    for b, nm in enumerate(("pot", "lid")):
        dz = (op[:, b, 2] - op[0, b, 2]); v = np.r_[0, np.linalg.norm(np.diff(op[:, b], axis=0), axis=1)]
        moving = (dz > 0.02) | (v > 0.005)
        near = {s: e[f"{s}_sd"][:, b, :].min(1) for s in ("left", "right")}
        tipnear = {s: e[f"{s}_sd"][:, b, 1:] for s in ("left", "right")}
        ph = moving & (np.minimum(near["left"], near["right"]) > 0.02)
        print(f"{tag} {nm}: moving frames {moving.sum()} runs {runs(moving)}; phantom (no hand point within 2 cm) {ph.sum()} runs {runs(ph)}")
        for s in ("left", "right"):
            m = moving & (near[s] < 0.03)
            if m.sum():
                tn = tipnear[s][m] * 1000
                print(f"   {s} on {nm} while moving ({m.sum()} f): tip sd mm median th {np.median(tn[:,0]):.0f} ix {np.median(tn[:,1]):.0f} mi {np.median(tn[:,2]):.0f} ri {np.median(tn[:,3]):.0f} pi {np.median(tn[:,4]):.0f}; wrist {1000*np.median(e[f'{s}_sd'][m, b, 0]):.0f}")
# valid flags
for p in sys.argv[5:7]:
    z = np.load(p, allow_pickle=True)
    print("==", p)
    for side in ("left", "right"):
        if f"hand_{side}_is_valid" in z:
            v = z[f"hand_{side}_is_valid"].astype(bool); print(f"  {side} valid {v.mean():.3f}; invalid runs {runs(~v)}")
    print("  keys:", [k for k in z.files if "object" in k or "valid" in k][:30])
