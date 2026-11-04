"""Independent review of w_hold_ep1 vs w_ctl_ep1 / w_raw_ep1 (Wentao base). GT-free.
usage: rev.py <scenes root> <wentao episode dir>"""
import sys, glob, json, os
import numpy as np, trimesh, pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

root, wdir = sys.argv[1], sys.argv[2]
NAMES = ("w_ctl_ep1", "w_raw_ep1", "w_hold_ep1")


def runs(m):
    e = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2])]


S = {}
for nm in NAMES:
    sc = f"{root}/{nm}"
    pqd = glob.glob(sc + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave")
    files = glob.glob(pqd[0] + "/*.parquet") if pqd else []
    print(f"== {nm}: parquet dir {pqd}; files {[os.path.basename(f) for f in files]}; "
          f"report {os.path.exists(sc + '/report.json')}; npz {os.path.exists(sc + '/bundle_cleaned/result.npz')}")
    t = pq.read_table(files[0])
    d = {n: t.column(n)[0].as_py() for n in t.column_names}
    zp = sc + "/bundle_cleaned/result.npz"
    if not os.path.exists(zp):
        cands = sorted(glob.glob(f"{root}/../bundles/{nm}/*.npz")) + sorted(glob.glob(sc + "/**/result.npz", recursive=True)) + sorted(glob.glob(sc + "/*.npz"))
        print("   no bundle_cleaned; candidates", cands)
        zp = cands[0]
    print("   npz used:", zp)
    z = dict(np.load(zp, allow_pickle=True))
    rep = json.load(open(sc + "/report.json"))
    S[nm] = dict(d=d, z=z, rep=rep, proc=os.path.dirname(os.path.dirname(pqd[0])))

print("\n== report.json cleaning (hold)")
cl = S["w_hold_ep1"]["rep"].get("cleaning", S["w_hold_ep1"]["rep"])
for k, v in cl.items():
    if "attach" in k or "ep1n" in k or "fill" in k:
        print(" ", k, json.dumps(v)[:400])
print("== report.json cleaning (raw)")
cl = S["w_raw_ep1"]["rep"].get("cleaning", S["w_raw_ep1"]["rep"])
for k, v in cl.items():
    if "attach" in k or "fill" in k:
        print(" ", k, json.dumps(v)[:400])


def npzdiff(a, b, tag):
    za, zb = S[a]["z"], S[b]["z"]
    print(f"\n== npz diff {tag}: keys only in A {sorted(set(za) - set(zb))} only in B {sorted(set(zb) - set(za))}")
    for k in sorted(set(za) & set(zb)):
        x, y = za[k], zb[k]
        if x.shape != y.shape:
            print(f"  {k}: SHAPE {x.shape} vs {y.shape}"); continue
        if x.dtype.kind in "fiub":
            m = float(np.nanmax(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
            nan = int(np.isnan(x.astype(float)).sum()) if x.dtype.kind == "f" else 0
            if m > 1e-9 or nan:
                fr = np.flatnonzero(np.abs(x.astype(float) - y.astype(float)).reshape(len(x), -1).max(1) > 1e-6) if x.ndim and len(x) > 1 else []
                print(f"  {k}: maxabs {m:.4g} nanA {nan} frames {runs(np.isin(np.arange(len(x)), fr)) if len(fr) else []}")
        elif not np.array_equal(x, y):
            print(f"  {k}: non-numeric differs")


npzdiff("w_hold_ep1", "w_raw_ep1", "hold vs raw")
npzdiff("w_hold_ep1", "w_ctl_ep1", "hold vs ctl (hand keys expected to differ)")

# objects vs Wentao delivered
print("\n== objects vs Wentao delivered bundle")
wb = glob.glob(wdir + "/bundle/*.npz")
print("  wentao npz:", wb)
for f in wb:
    zw = np.load(f, allow_pickle=True)
    zh = S["w_hold_ep1"]["z"]
    for k in zw.files:
        if ("object" in k or "camera_to_world" in k) and k in zh and zw[k].dtype.kind in "fiub":
            if zw[k].shape == zh[k].shape:
                print(f"  {k}: maxabs {float(np.abs(zw[k].astype(float) - zh[k].astype(float)).max()):.3g}")
            else:
                print(f"  {k}: SHAPE {zw[k].shape} vs {zh[k].shape}")

# parquet column diffs
print("\n== parquet column diffs")
for a, b in (("w_hold_ep1", "w_raw_ep1"), ("w_hold_ep1", "w_ctl_ep1")):
    da, db = S[a]["d"], S[b]["d"]
    diff = []
    for k in da:
        if k not in db:
            diff.append(k + "(missing)"); continue
        try:
            x, y = np.asarray(da[k], float), np.asarray(db[k], float)
        except Exception:
            continue
        if x.shape != y.shape:
            diff.append(f"{k}:shape"); continue
        nan = int(np.isnan(x).sum())
        m = float(np.nanmax(np.abs(x - y))) if x.size else 0
        if m > 1e-6 or nan:
            diff.append(f"{k}:{m:.3g}{' NAN' + str(nan) if nan else ''}")
    print(f"  {a} vs {b}: {diff}")

# geometry
d0 = S["w_hold_ep1"]["d"]
onames = d0["object_body_names"]
print("\nobjects", onames)
MESH = []
for n in onames:
    m = trimesh.load(f"{S['w_hold_ep1']['proc']}/{n}.obj", force="mesh", process=False)
    np.random.seed(0)
    pts, fi = trimesh.sample.sample_surface(m, 80000)
    pts = np.vstack([pts, m.vertices]); nrm = np.vstack([m.face_normals[fi], m.vertex_normals])
    MESH.append((cKDTree(pts), pts, nrm, m))
    print(f"  mesh {n}: extents {m.extents}, watertight {m.is_watertight}")
TIPS = [4, 8, 12, 16, 20]


def geo(nm):
    d = S[nm]["d"]
    opos = np.asarray(d["object_body_position"], float); ow = np.asarray(d["object_body_wxyz"], float)
    T = len(opos)
    orot = R.from_quat(ow[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(T, -1, 3, 3)
    out = dict(opos=opos, orot=orot, T=T)
    for side in ("left", "right"):
        J = np.asarray(d[f"mano_{side}_joints"], float)
        out[side + "_J"] = J
        L = np.einsum("tbji,tbkj->tbki", orot, J[:, None, :, :] - opos[:, :, None, :])  # (T,B,21,3) local
        out[side + "_L"] = L
        SD = np.zeros(L.shape[:3])
        for b in range(L.shape[1]):
            tree, pts, nrm, _ = MESH[b]
            P = L[:, b].reshape(-1, 3)
            dist, i = tree.query(P)
            s = np.sign(np.einsum("ij,ij->i", P - pts[i], nrm[i])); s[s == 0] = 1
            SD[:, b] = (dist * s).reshape(T, 21)
        out[side + "_SD"] = SD
        out[side + "_pid"] = np.asarray(d[f"mano_{side}_object_contact_part_ids"], float)
    return out


G = {nm: geo(nm) for nm in NAMES}
PH = {"pre[32,127)": (32, 127), "lift[127,221)": (127, 221), "post[221,304)": (221, 304), "all[32,304)": (32, 304)}
for nm in NAMES:
    g = G[nm]; T = g["T"]
    print(f"\n######## {nm}")
    for side in ("left", "right"):
        J = g[side + "_J"]
        nn = int(np.isnan(J).sum())
        st = np.linalg.norm(np.diff(J[:, 0], axis=0), axis=1) * 1000
        sj = np.linalg.norm(np.diff(J, axis=0), axis=2).max(1) * 1000
        print(f" {side}: NaN {nn}; wrist step max {st.max():.1f}@f{st.argmax()+1} p99 {np.percentile(st,99):.1f} >20mm {list(np.flatnonzero(st>20)+1)}; any-joint step max {sj.max():.1f}@f{sj.argmax()+1}")
        for p in (1, 2):
            print(f"   contact part {p} runs {runs((g[side+'_pid'] == p).sum(1) > 0)}")
        SD = g[side + "_SD"]
        for b, on in enumerate(("pot", "lid")):
            pen = SD[:, b].min(1)
            bad = pen < -0.01
            print(f"   pen into {on}: min {1000*pen.min():.1f}mm@f{pen.argmin()}; frames joint < -10mm: {int(bad.sum())} runs {runs(bad)[:8]}; < -5mm {int((pen<-0.005).sum())}")
    # left grip on pot in pot frame
    L = g["left_L"][:, 0]
    SD = g["left_SD"][:, 0]
    for ph, (a, b) in PH.items():
        W = L[a:b, 0]
        dev = np.linalg.norm(W - np.median(L[32:304, 0], axis=0), axis=1) * 1000
        devp = np.linalg.norm(W - np.median(W, axis=0), axis=1) * 1000
        tipd = SD[a:b][:, TIPS] * 1000
        stdtip = tipd.std(0)
        hf = W - np.array([np.convolve(W[:, k], np.ones(7) / 7, mode="same") for k in range(3)]).T
        hf = np.linalg.norm(hf[4:-4], axis=1) * 1000
        notip = int((SD[a:b][:, TIPS].min(1) > 0.01).sum())
        print(f"  L-pot {ph:14s}: wrist dev from all-median med {np.median(dev):5.1f} p90 {np.percentile(dev,90):5.1f} | from phase-median med {np.median(devp):5.1f} | phase-median offset from all {1000*np.linalg.norm(np.median(W,0)-np.median(L[32:304,0],0)):5.1f}mm | wrist HF(7f box) rms {np.sqrt((hf**2).mean()):4.1f} | tip sd med {np.round(np.median(tipd,0)).astype(int)} std {np.round(stdtip,1)} | frames no tip<1cm {notip}")
    # right on lid
    L = g["right_L"][:, 1]; SD = g["right_SD"][:, 1]
    for a, b in ((45, 68), (250, 274)):
        W = L[a:b, 0]
        devp = np.linalg.norm(W - np.median(W, axis=0), axis=1) * 1000
        tipd = SD[a:b][:, TIPS] * 1000
        print(f"  R-lid [{a},{b}): wrist dev from phase median med {np.median(devp):5.1f} p90 {np.percentile(devp,90):5.1f} | wrist sd med {1000*np.median(SD[a:b,0]):.0f} | tip sd med {np.round(np.median(tipd,0)).astype(int)} std {np.round(tipd.std(0),1)} | frames no tip<1cm {int((SD[a:b][:,TIPS].min(1)>0.01).sum())} | min joint sd {1000*SD[a:b].min():.1f}")
    # the right hand vs pot during right lid holds (attach to wrong object?)
    SDp = g["right_SD"][:, 0]
    print(f"  R-pot min sd in lid windows: {1000*SDp[45:68].min():.1f} / {1000*SDp[250:274].min():.1f} mm")

# hand-shape identity hold vs raw (fingers unchanged): pairwise joint distances
for side in ("left", "right"):
    Ja, Jb = G["w_hold_ep1"][side + "_J"], G["w_raw_ep1"][side + "_J"]
    pa = np.linalg.norm(Ja[:, :, None] - Ja[:, None], axis=-1); pb = np.linalg.norm(Jb[:, :, None] - Jb[:, None], axis=-1)
    dw = np.linalg.norm(Ja[:, 0] - Jb[:, 0], axis=1) * 1000
    print(f"\nhold vs raw {side}: max pairwise-joint-distance change {1000*np.abs(pa-pb).max():.3f} mm; wrist moved >1mm runs {runs(dw>1)}; max {dw.max():.1f}@f{dw.argmax()}")
    Jc = G["w_ctl_ep1"][side + "_J"]
    dc = np.linalg.norm(Ja[:, 0] - Jc[:, 0], axis=1) * 1000
    print(f"hold vs ctl {side}: wrist diff med {np.median(dc):.1f} p90 {np.percentile(dc,90):.1f} max {dc.max():.1f}@f{dc.argmax()}")
# blend-boundary steps for hold
for side in ("left", "right"):
    J = G["w_hold_ep1"][side + "_J"]; st = np.linalg.norm(np.diff(J[:, 0], axis=0), axis=1) * 1000
    Jr = G["w_raw_ep1"][side + "_J"]; sr = np.linalg.norm(np.diff(Jr[:, 0], axis=0), axis=1) * 1000
    for f in ((26, 40), (298, 312)) if side == "left" else ((40, 50), (63, 73), (245, 255), (268, 279)):
        print(f" {side} wrist steps hold/raw f{f[0]}-{f[1]}: " + " ".join(f"{st[t]:.0f}/{sr[t]:.0f}" for t in range(f[0], f[1])))
# object poses identical across builds
for nm in NAMES[:2]:
    print(f"obj pos maxabs hold vs {nm}: {np.abs(G['w_hold_ep1']['opos']-G[nm]['opos']).max():.2e}; rot {np.abs(G['w_hold_ep1']['orot']-G[nm]['orot']).max():.2e}")
