"""ep8 depth-table check (GT-free, read-only): real table from our FoundationStereo depth vs the sim
support plane vs each resting object's mesh bottom; plus object-surface depth vs our mesh.
usage: python -I dt8.py <scene_dir> <kenc_episode_dir> <out_dir>
Depth PNG: metres = 65535 / raw - 1 (raw 65535 / 0 = hole). Work at half resolution (stride 2)."""
import sys, os, glob, json, time, importlib.util
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

t0 = time.time()


def log(*a):
    print(f"[dt8 {time.time() - t0:6.1f}s]", *a, flush=True)


spec = importlib.util.spec_from_file_location("rr", "/home/haisenk/v2d_ops/kenc/review/tool/render_review.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

scene, E, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
P = rr.load_scene_parquet(rr.find_parquet(scene))
N, K = P["N"], P["pos"].shape[1]
proc = os.path.dirname(os.path.dirname(os.path.dirname(P["path"])))
b = rr.load_bundle(os.path.join(scene, "bundle_cleaned", "result.npz"))
short = [n.split("_attach_")[-1] for n in P["body_names"]]
assert list(b["names"]) == short, (b["names"], short)
objs = []
for k in range(K):
    mp = P["mesh_paths"][k]
    mp = mp if os.path.isfile(mp) else os.path.join(proc, os.path.basename(mp))
    V, F = rr.load_obj(mp)
    R = rr.wxyz_to_R(P["wxyz"][:, k]).reshape(N, 3, 3)
    T = rr.to_T(R, P["pos"][:, k])
    lo, hi = V.min(0), V.max(0)
    C = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    objs.append(dict(name=short[k], V=V, F=F, T=T, mp=mp, C=C))
A, res = rr.estimate_A([o["T"] for o in objs], [b["objT"][k] for k in range(K)], [o["C"] for o in objs])
rotA = float(np.degrees(np.arccos(np.clip((np.trace(A[:3, :3]) - 1) / 2, -1, 1))))
log("loader A t", A[:3, 3].round(4), "rot deg %.3f" % rotA, "residual max mm %.4f" % float(np.nanmax(res) * 1000))
c2w = A[None] @ b["c2w"]
Kf = np.asarray(b["K"][0], float)
H, W, S = 1520, 2028, 2
H2, W2 = (H + S - 1) // S, (W + S - 1) // S
K2 = Kf / S  # stride sampling: half-res pixel i == full-res pixel S*i
vv, uu = np.mgrid[0:H2, 0:W2]
xn = (uu - K2[2]) / K2[0]
yn = (vv - K2[3]) / K2[1]
raylen = np.sqrt(1 + xn ** 2 + yn ** 2)

usd = sorted(glob.glob(os.path.join(scene, "assets/human_motion_data/ego_recon/reconstructed_stage/*.usda")))[0]
prims = rr.parse_usda(usd)
sup = max(prims, key=lambda p: p["area"])
z_sup = float(sup["top_z"])
sup_c = sup["pts"][:, :2].mean(0)
sup_r = float(np.linalg.norm(sup["pts"][:, :2] - sup_c, axis=1).max())
log("support", sup["name"], "top z %.5f centre" % z_sup, sup_c.round(4), "radius %.3f" % sup_r, "n prims", len(prims))

rng = np.random.default_rng(0)


def sample_surface(V, F, dens=6e6, nmin=300000, nmax=1500000):
    a, bb, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ar = 0.5 * np.linalg.norm(np.cross(bb - a, c - a), axis=1)
    n = int(np.clip(ar.sum() * dens, nmin, nmax))
    idx = rng.choice(len(F), n, p=ar / ar.sum())
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    Pm = (1 - r1)[:, None] * a[idx] + (r1 * (1 - r2))[:, None] * bb[idx] + (r1 * r2)[:, None] * c[idx]
    return np.concatenate([Pm, V]), float(ar.sum())


for o in objs:
    o["Ps"], o["area"] = sample_surface(o["V"], o["F"])
    log(o["name"], "mesh", o["mp"], "verts", len(o["V"]), "area m2 %.4f" % o["area"], "samples", len(o["Ps"]))


def zbuf(Pm, T, f):
    w2c = np.linalg.inv(c2w[f])
    M = w2c @ T
    Pc = Pm @ M[:3, :3].T + M[:3, 3]
    z = Pc[:, 2]
    ok = z > 0.05
    Pc, z = Pc[ok], z[ok]
    u = np.round(K2[0] * Pc[:, 0] / z + K2[2]).astype(np.int64)
    v = np.round(K2[1] * Pc[:, 1] / z + K2[3]).astype(np.int64)
    inb = (u >= 0) & (u < W2) & (v >= 0) & (v < H2)
    zb = np.full(H2 * W2, np.inf)
    np.minimum.at(zb, v[inb] * W2 + u[inb], z[inb])
    return zb.reshape(H2, W2)


def bottom(o, f):
    T = o["T"][f]
    Vw = o["V"] @ T[:3, :3].T + T[:3, 3]
    zb = float(Vw[:, 2].min())
    low = Vw[Vw[:, 2] < zb + 0.003]
    bc = low[:, :2].mean(0)
    lowbody = Vw[Vw[:, 2] < zb + 0.03]
    r_fp = float(np.percentile(np.linalg.norm(lowbody[:, :2] - bc, axis=1), 95))
    return zb, bc, r_fp


def load_depth(f):
    raw = np.asarray(Image.open(f"{E}/white_pot/depth/{f:06d}.png")).astype(np.float64)[::S, ::S]
    ok = (raw > 0) & (raw < 65535)
    Z = np.where(ok, 65535.0 / np.maximum(raw, 1) - 1.0, np.nan)
    Z[(Z < 0.1) | (Z > 2.5)] = np.nan
    return Z


def load_mask(p):
    if not os.path.isfile(p):
        return np.zeros((H2, W2), bool)
    return np.asarray(Image.open(p).convert("L"))[::S, ::S] > 127


def backproject(Z, sel, f):
    Pc = np.stack([xn[sel] * Z[sel], yn[sel] * Z[sel], Z[sel]], 1)
    return Pc @ c2w[f, :3, :3].T + c2w[f, :3, 3]


def edt_out(m):
    """distance (half-res px) of each pixel to mask m (0 inside)."""
    if not m.any():
        return np.full(m.shape, 1e9)
    return ndi.distance_transform_edt(~m)


def fit_plane(Pts, thr=0.006, iters=400, seed=0):
    n = len(Pts)
    if n < 100:
        return None
    r_ = np.random.default_rng(seed)
    Ps = Pts if n <= 80000 else Pts[r_.choice(n, 80000, replace=False)]
    best_cnt, best = -1, None
    for _ in range(iters):
        s = Ps[r_.choice(len(Ps), 3, replace=False)]
        nv = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(nv)
        if nn < 1e-12:
            continue
        nv /= nn
        if abs(nv[2]) < 0.85:
            continue
        d = -nv @ s[0]
        cnt = int((np.abs(Ps @ nv + d) < thr).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nv, d)
    if best is None:
        return None
    nv, d = best
    inl = np.abs(Pts @ nv + d) < thr
    coef = None
    for _ in range(4):
        Q = Pts[inl]
        M = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]
        coef, *_ = np.linalg.lstsq(M, Q[:, 2], rcond=None)
        r = Pts[:, 2] - (Pts[:, 0] * coef[0] + Pts[:, 1] * coef[1] + coef[2])
        inl = np.abs(r) < thr
    a, bq, c = [float(x) for x in coef]
    ri = r[inl]
    above = float((r > thr).mean())
    below = float((r < -thr).mean())
    return dict(a=a, b=bq, c=c, tilt_deg=float(np.degrees(np.arctan(np.hypot(a, bq)))),
                ascent_az_deg=float(np.degrees(np.arctan2(bq, a))), n=int(n), inlier_frac=float(inl.mean()),
                resid_std_mm=float(ri.std() * 1000), frac_above=above, frac_below=below)


def zat(pl, xy):
    return None if pl is None else float(pl["a"] * xy[0] + pl["b"] * xy[1] + pl["c"])


cup = [i for i, o in enumerate(objs) if o["name"] == "beige_cup"][0]
pot = [i for i, o in enumerate(objs) if o["name"] == "white_pot"][0]
phases = {
    "start_0_30": list(range(0, 31, 2)),
    "pot_contact_16_94": list(range(16, 95, 6)),
    "pot_late_166_196": list(range(166, 197, 6)),
    "end_200_218": list(range(200, 219, 2)),
}
obj_phase = {  # which phases each object rests on the table (from the review numbers: cup lifted 45-185, pot lifted 95-164)
    "beige_cup": ["start_0_30", "end_200_218"],
    "white_pot": ["start_0_30", "pot_contact_16_94", "pot_late_166_196", "end_200_218"],
}
frames = sorted(set(sum(phases.values(), [])))
per = {}
tab_pts = {ph: [] for ph in phases}
ring_pts = {ph: {o["name"]: [] for o in objs} for ph in phases}
viz = {}
for f in frames:
    Z = load_depth(f)
    fin = np.isfinite(Z)
    mx = ndi.maximum_filter(np.where(fin, Z, -1.0), size=5)
    mn = ndi.minimum_filter(np.where(fin, Z, 99.0), size=5)
    edge = (mx - mn) > 0.015
    mk = {o["name"]: load_mask(f"{E}/{o['name']}/masks/1/{f:06d}.png") for o in objs}
    hand = np.zeros((H2, W2), bool)
    for o in objs:
        for h in (2, 3):
            hand |= load_mask(f"{E}/{o['name']}/masks/{h}/{f:06d}.png")
    zbs = [zbuf(o["Ps"], o["T"][f], f) for o in objs]
    zall = np.minimum.reduce(zbs)
    rend = np.isfinite(zall)
    d_obj = edt_out(mk["beige_cup"] | mk["white_pot"] | rend)
    d_hand = edt_out(hand)
    cand = fin & ~edge & (d_obj > 8) & (d_hand > 25)
    Pw = backproject(Z, cand, f)
    keep = (np.abs(Pw[:, 2] - z_sup) < 0.10) & (np.linalg.norm(Pw[:, :2] - sup_c, axis=1) < 0.45)
    Pt = Pw[keep]
    rec = dict(frame=f, n_table_px=int(len(Pt)), objects={})
    plf = fit_plane(Pt, seed=f)
    rec["frame_plane"] = plf
    for ph, fl in phases.items():
        if f in fl:
            sub = Pt if len(Pt) <= 30000 else Pt[np.random.default_rng(f).choice(len(Pt), 30000, replace=False)]
            tab_pts[ph].append(sub)
    for k, o in enumerate(objs):
        nm = o["name"]
        zb_, bc, r_fp = bottom(o, f)
        own = np.isfinite(zbs[k]) & (zbs[k] <= zall + 1e-9)
        other = [j for j in range(K) if j != k][0]
        d_in = ndi.distance_transform_edt(mk[nm]) if mk[nm].any() else np.zeros((H2, W2))
        d_oth = edt_out(mk[objs[other]["name"]] | np.isfinite(zbs[other]))
        sel = (d_in > 4) & own & (d_hand > 10) & ~edge & fin & (d_oth > 4)
        sil_iou = float((mk[nm] & np.isfinite(zbs[k])).sum() / max((mk[nm] | np.isfinite(zbs[k])).sum(), 1))
        r = dict(bottom_z=zb_, bottom_xy=bc.tolist(), r_fp=r_fp, float_vs_sim_cm=(zb_ - z_sup) * 100,
                 sil_iou_halfres=sil_iou, n_obj_px=int(sel.sum()))
        if sel.sum() > 50:
            dz = Z[sel] - zbs[k][sel]
            r["dz_med_cm"] = float(np.median(dz) * 100)
            r["dz_p25_cm"] = float(np.percentile(dz, 25) * 100)
            r["dz_p75_cm"] = float(np.percentile(dz, 75) * 100)
            r["dray_med_cm"] = float(np.median(dz * raylen[sel]) * 100)
            r["ratio_med"] = float(np.median(Z[sel] / zbs[k][sel]))
            Pr = backproject(Z, sel, f)
            Pm = backproject(zbs[k], sel, f)
            dl = Pr - Pm
            r["dworld_med_cm"] = (np.median(dl, 0) * 100).tolist()
            r["real_low_p2_z"] = float(np.percentile(Pr[:, 2], 2))
            r["mesh_low_p2_z"] = float(np.percentile(Pm[:, 2], 2))
            cam = c2w[f, :3, 3]
            ray = np.array([bc[0], bc[1], zb_]) - cam
            r["ray_elev_deg"] = float(np.degrees(np.arcsin(-ray[2] / np.linalg.norm(ray))))
            r["cam_dist_m"] = float(np.linalg.norm(ray))
            if f in (10, 210):
                viz.setdefault(f, {})[nm] = (sel, Z - zbs[k])
        # image ring around the SAM mask (pose-independent table selection)
        if mk[nm].any():
            dm = edt_out(mk[nm])
            ring = cand & (dm > 6) & (dm <= 60)
            Pr2 = backproject(Z, ring, f)
            kk = (np.abs(Pr2[:, 2] - z_sup) < 0.10)
            for ph, fl in phases.items():
                if f in fl:
                    ring_pts[ph][nm].append(Pr2[kk])
        rec["objects"][nm] = r
    if f in (10, 210):
        Ztab = np.full((H2, W2), np.nan)
        idx = np.nonzero(cand)
        zw = Pw[:, 2].copy()
        zw[~keep] = np.nan
        Ztab[idx] = zw
        viz.setdefault(f, {})["table"] = Ztab
        viz[f]["masks"] = {nm: mk[nm] for nm in mk}
        viz[f]["rend"] = {o["name"]: np.isfinite(zbs[k]) for k, o in enumerate(objs)}
        viz[f]["hand"] = hand
    per[f] = rec
    fp = rec["frame_plane"]
    log(f"f{f:3d} table px {len(Pt):6d} plane tilt {fp['tilt_deg'] if fp else float('nan'):.2f} deg z@cup "
        f"{zat(fp, rec['objects']['beige_cup']['bottom_xy']) if fp else float('nan'):.4f} z@pot "
        f"{zat(fp, rec['objects']['white_pot']['bottom_xy']) if fp else float('nan'):.4f} | cup bot "
        f"{rec['objects']['beige_cup']['bottom_z']:.4f} dz {rec['objects']['beige_cup'].get('dz_med_cm', float('nan')):.2f} cm"
        f" | pot bot {rec['objects']['white_pot']['bottom_z']:.4f} dz {rec['objects']['white_pot'].get('dz_med_cm', float('nan')):.2f} cm")

# ---------------------------------------------------------------- phase summaries
summ = dict(scene=scene, depth="FoundationStereo (ken_c seed fs_depth) " + f"{E}/white_pot/depth (== beige_cup/depth, md5 equal)",
            loader_A=dict(t=A[:3, 3].tolist(), rot_deg=rotA, resid_mm=float(np.nanmax(res) * 1000)),
            support=dict(name=sup["name"], top_z=z_sup, centre=sup_c.tolist(), radius=sup_r), phases={})
for ph, fl in phases.items():
    Pt = np.concatenate(tab_pts[ph]) if tab_pts[ph] else np.zeros((0, 3))
    pl = fit_plane(Pt, seed=1)
    pls = [per[f]["frame_plane"] for f in fl if per[f]["frame_plane"]]
    out = dict(frames=[fl[0], fl[-1]], n_frames=len(fl), global_plane=pl, objects={})
    if pl:
        # also the plane height at the support centre
        out["global_plane_z_at_support_centre_minus_sim_cm"] = (zat(pl, sup_c) - z_sup) * 100
    for o in objs:
        nm = o["name"]
        if ph not in obj_phase[nm]:
            continue
        rs = [per[f]["objects"][nm] for f in fl]
        bz = float(np.median([r["bottom_z"] for r in rs]))
        bxy = np.median(np.array([r["bottom_xy"] for r in rs]), 0)
        r_fp = float(np.median([r["r_fp"] for r in rs]))
        zg = zat(pl, bxy)
        zf = [zat(p, bxy) for p in pls]
        # 3D annulus around the mesh footprint
        dh = np.linalg.norm(Pt[:, :2] - bxy, axis=1) if len(Pt) else np.zeros(0)
        ann = Pt[(dh > r_fp + 0.02) & (dh < r_fp + 0.12)] if len(Pt) else Pt
        pla = fit_plane(ann, seed=2)
        rp = np.concatenate(ring_pts[ph][nm]) if ring_pts[ph][nm] else np.zeros((0, 3))
        plr = fit_plane(rp, seed=3)
        okr = [r for r in rs if "dz_med_cm" in r]
        oo = dict(mesh_bottom_z=bz, bottom_xy=bxy.tolist(), footprint_r_cm=r_fp * 100,
                  float_vs_sim_cm=(bz - z_sup) * 100,
                  depth_table_global_z=zg, float_vs_depth_table_global_cm=None if zg is None else (bz - zg) * 100,
                  depth_table_minus_sim_cm=None if zg is None else (zg - z_sup) * 100,
                  per_frame_plane_z_med=float(np.median(zf)) if zf else None,
                  per_frame_plane_z_iqr_mm=float((np.percentile(zf, 75) - np.percentile(zf, 25)) * 1000) if zf else None,
                  annulus_plane=pla, annulus_z=zat(pla, bxy), annulus_n=int(len(ann)),
                  float_vs_annulus_cm=None if pla is None else (bz - zat(pla, bxy)) * 100,
                  ring2d_plane=plr, ring2d_z=zat(plr, bxy),
                  float_vs_ring2d_cm=None if plr is None else (bz - zat(plr, bxy)) * 100)
        if okr:
            def med(key):
                return float(np.median([r[key] for r in okr]))
            oo.update(dz_med_cm=med("dz_med_cm"), dz_p25_cm=med("dz_p25_cm"), dz_p75_cm=med("dz_p75_cm"),
                      dray_med_cm=med("dray_med_cm"), ratio_med=med("ratio_med"),
                      dworld_med_cm=np.median(np.array([r["dworld_med_cm"] for r in okr]), 0).tolist(),
                      real_low_p2_z=med("real_low_p2_z"), mesh_low_p2_z=med("mesh_low_p2_z"),
                      real_low_minus_table_cm=None if zg is None else (med("real_low_p2_z") - zg) * 100,
                      mesh_low_minus_table_cm=None if zg is None else (med("mesh_low_p2_z") - zg) * 100,
                      ray_elev_deg=med("ray_elev_deg"), cam_dist_m=med("cam_dist_m"),
                      n_obj_px_med=float(np.median([r["n_obj_px"] for r in okr])),
                      sil_iou_halfres=med("sil_iou_halfres"))
            if zg is not None:
                oo["along_ray_shift_to_seat_on_depth_table_cm"] = (bz - zg) * 100 / np.sin(np.radians(oo["ray_elev_deg"]))
        out["objects"][nm] = oo
    summ["phases"][ph] = out

# depth-to-mask alignment check (gradient of depth on the SAM mask boundary vs shift)
al = {}
for f in (10, 210):
    Z = load_depth(f)
    Zf = np.where(np.isfinite(Z), Z, np.nanmedian(Z))
    gy, gx = np.gradient(Zf)
    G = np.minimum(np.hypot(gx, gy), 0.05)
    for o in objs:
        m = load_mask(f"{E}/{o['name']}/masks/1/{f:06d}.png")
        B = m & ~ndi.binary_erosion(m, iterations=1)
        ys, xs = np.nonzero(B)
        sc = {}
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                sc[(dx, dy)] = float(np.mean(G[np.clip(ys + dy, 0, H2 - 1), np.clip(xs + dx, 0, W2 - 1)]))
        kb = max(sc, key=sc.get)
        al[f"f{f}_{o['name']}"] = dict(best_shift_halfpx=list(kb), best=sc[kb], at0=sc[(0, 0)])
summ["depth_mask_alignment"] = al
summ["per_frame"] = {str(f): per[f] for f in frames}
with open(os.path.join(OUT, "ep8_depthtab.json"), "w") as fh:
    json.dump(summ, fh, indent=1, default=float)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
for row, f in enumerate((10, 210)):
    if f not in viz:
        continue
    v = viz[f]
    ax = axs[row, 0]
    im = ax.imshow((v["table"] - z_sup) * 100, cmap="RdBu_r", vmin=-6, vmax=6)
    for nm, col in (("beige_cup", "b"), ("white_pot", "orange")):
        ax.contour(v["masks"][nm], levels=[0.5], colors=[col], linewidths=1)
        ax.contour(v["rend"][nm], levels=[0.5], colors=[col], linewidths=1, linestyles="--")
    ax.contour(v["hand"], levels=[0.5], colors=["g"], linewidths=1)
    ax.set_title(f"f{f}: depth-table height minus sim support top (cm); solid = SAM mask, dashed = our mesh")
    plt.colorbar(im, ax=ax, fraction=0.03)
    ax = axs[row, 1]
    D = np.full((H2, W2), np.nan)
    for nm in ("beige_cup", "white_pot"):
        if nm in v:
            sel, dd = v[nm]
            D[sel] = dd[sel] * 100
    im = ax.imshow(D, cmap="PuOr_r", vmin=-10, vmax=10)
    for nm, col in (("beige_cup", "b"), ("white_pot", "orange")):
        ax.contour(v["masks"][nm], levels=[0.5], colors=[col], linewidths=1)
    ax.set_title(f"f{f}: real depth minus our mesh depth on object pixels (cm, + = real farther)")
    plt.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "ep8_depthtab.png"), dpi=70)

# ---------------------------------------------------------------- print summary
print("==== SUMMARY ====")
print("support top z %.5f centre %s radius %.3f" % (z_sup, np.round(sup_c, 4), sup_r))
for ph, out in summ["phases"].items():
    pl = out["global_plane"]
    if pl:
        print(f"[{ph}] global depth-table plane: tilt {pl['tilt_deg']:.2f} deg (ascent az {pl['ascent_az_deg']:.0f}), n {pl['n']}, "
              f"inliers {pl['inlier_frac']:.2f} (above {pl['frac_above']:.2f} below {pl['frac_below']:.2f}), resid std {pl['resid_std_mm']:.1f} mm, "
              f"z@support-centre minus sim {out.get('global_plane_z_at_support_centre_minus_sim_cm', float('nan')):+.2f} cm")
    for nm, oo in out["objects"].items():
        print(f"   {nm}: mesh bottom {oo['mesh_bottom_z']:.4f} xy {np.round(oo['bottom_xy'], 3)} | float vs sim {oo['float_vs_sim_cm']:+.2f} cm"
              f" | depth table z {oo['depth_table_global_z']} (minus sim {oo['depth_table_minus_sim_cm']}) | float vs depth table {oo['float_vs_depth_table_global_cm']}"
              f" | annulus {oo['float_vs_annulus_cm']} (n {oo['annulus_n']}) | ring2d {oo['float_vs_ring2d_cm']}")
        if "dz_med_cm" in oo:
            print(f"      object depth: real - mesh dz med {oo['dz_med_cm']:+.2f} cm [p25 {oo['dz_p25_cm']:+.2f}, p75 {oo['dz_p75_cm']:+.2f}], along ray {oo['dray_med_cm']:+.2f} cm, ratio {oo['ratio_med']:.3f},"
                  f" dworld {np.round(oo['dworld_med_cm'], 2)} cm; real lowest visible minus table {oo['real_low_minus_table_cm']}, mesh lowest visible minus table {oo['mesh_low_minus_table_cm']};"
                  f" ray elev {oo['ray_elev_deg']:.1f} deg, cam dist {oo['cam_dist_m']:.3f} m; along-ray shift to seat {oo.get('along_ray_shift_to_seat_on_depth_table_cm')}; px {oo['n_obj_px_med']:.0f}, sil IoU {oo['sil_iou_halfres']:.3f}")
print("alignment", json.dumps(al))
log("done")
