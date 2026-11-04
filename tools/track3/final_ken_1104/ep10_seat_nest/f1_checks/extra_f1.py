"""k10 F1 extra GT-free numbers (anaconda python, CPU step):
 1) attach shift: cleaned minus merged wrist in the camera frame (levelling-free), per hand, ctl vs seat
 2) blue silhouette vs SAM 3 visible mask, f63-122, ctl vs seat: render_review IoU (same raster), coverage
    |sil & mask| / |mask|, occlusion ceiling |mask| / |sil| and occlusion-aware IoU (hand + beige masks removed
    from both). Same silhouette code as render_review.mask_check (1/4 resolution).
usage: extra_f1.py <ctl scene> <seat scene> <out.json>"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/haisenk/v2d_ops/kenc/review/tool")
import render_review as rr  # noqa: E402

CTL, SEAT, OUTJ = sys.argv[1:4]
KENC = rr.KENC_ROOT + "/episode_000010"
RES = {}

# ---- 1) attach shift
ash = {}
for lab, sc in (("ctl", CTL), ("seat", SEAT)):
    zm = np.load(os.path.join(sc, "bundle_merged/result.npz"), allow_pickle=True)
    zc = np.load(os.path.join(sc, "bundle_cleaned/result.npz"), allow_pickle=True)
    for side in ("left", "right"):
        d = np.linalg.norm(zc[f"hand_{side}_wrist_trans_in_camera"].astype(float) -
                           zm[f"hand_{side}_wrist_trans_in_camera"].astype(float), axis=1) * 100
        ash[f"{lab}_{side}"] = {"max_cm": round(float(d.max()), 2), "argmax": int(d.argmax()),
                                "f45_81_max_cm": round(float(d[45:82].max()), 2),
                                "f82_124_max_cm": round(float(d[82:125].max()), 2),
                                "outside_f45_124_max_cm": round(float(np.concatenate([d[:45], d[125:]]).max()), 2),
                                "every4_f40_128": {int(f): round(float(d[f]), 2) for f in range(40, 129, 4)}}
RES["attach_shift_cleaned_minus_merged_camera_cm"] = ash
print(json.dumps(ash, indent=0)[:3000], flush=True)


# ---- 2) silhouettes
def setup(SC):
    pqp = rr.find_parquet(SC)
    P = rr.load_scene_parquet(pqp)
    N = P["N"]
    proc_dir = os.path.dirname(os.path.dirname(os.path.dirname(pqp)))
    Bc = rr.load_bundle(os.path.join(SC, "bundle_cleaned", "result.npz"))
    zm = np.load(os.path.join(SC, "bundle_merged", "result.npz"), allow_pickle=True)
    objs = []
    for k, bn in enumerate(P["body_names"]):
        nm = "blue_cup" if bn.endswith("blue_cup") else ("beige_cup" if bn.endswith("beige_cup") else bn)
        mp = P["mesh_paths"][k]
        mpath = next(c for c in [mp, os.path.join(proc_dir, os.path.basename(mp))] if c and os.path.isfile(c))
        V, F = rr.load_obj(mpath)
        R = rr.wxyz_to_R(P["wxyz"][:, k]).reshape(N, 3, 3)
        T = rr.to_T(R, P["pos"][:, k])
        lo, hi = V.min(0), V.max(0)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        objs.append(dict(name=nm, V=V, F=np.asarray(F, np.int64), T=T, corners=corners))
    bnames = [str(x) for x in zm["object_names"]]
    A, resA = rr.estimate_A([o["T"] for o in objs], [Bc["objT"][bnames.index(o["name"])] for o in objs],
                            [o["corners"] for o in objs])
    c2w = A[None] @ Bc["c2w"]
    return objs, np.linalg.inv(c2w), Bc["K"], N, float(np.nanmax(resA))


W, H, s = 2028, 1520, 4
h4, w4 = H // s, W // s


def load_small(p):
    try:
        im = Image.open(p).convert("L")
    except Exception:
        return None
    if im.size != (W, H):
        return None
    if im.getextrema()[1] == 1:
        im = im.point(lambda v: 255 if v else 0)
    return np.asarray(im.crop((0, 0, w4 * s, h4 * s)).reduce(s)) > 127


FR = list(range(55, 161))
mb = {f: load_small(os.path.join(KENC, "blue_cup", "masks", "1", f"{f:06d}.png")) for f in FR}
mg = {f: load_small(os.path.join(KENC, "beige_cup", "masks", "1", f"{f:06d}.png")) for f in FR}
occ = {}
for f in FR:
    o_ = np.zeros((h4, w4), bool)
    for nm in ("blue_cup", "beige_cup"):
        for hid in ("2", "3"):
            x = load_small(os.path.join(KENC, nm, "masks", hid, f"{f:06d}.png"))
            if x is not None:
                o_ |= x
    occ[f] = o_
sil_res = {}
for lab, sc in (("ctl", CTL), ("seat", SEAT)):
    objs, w2c, Kc, N, resA = setup(sc)
    ob = next(o for o in objs if o["name"] == "blue_cup")
    rows = {}
    for f in FR:
        uv, ok, _ = rr.project(rr.xf(ob["T"][f], ob["V"]), w2c[f], Kc[f])
        sil = rr.raster_mask((uv + 0.5) / s - 0.5, ob["F"], h4, w4, ok)
        m = mb[f]
        if m is None:
            continue
        inter, uni = (sil & m).sum(), (sil | m).sum()
        oc = occ[f] | (mg[f] if mg[f] is not None else False)
        sv, mv = sil & ~oc, m & ~oc
        rows[f] = dict(iou=float(inter / uni) if uni else np.nan, cov=float(inter / m.sum()) if m.sum() else np.nan,
                       ceil=float(m.sum() / sil.sum()) if sil.sum() else np.nan,
                       iou_occl=float((sv & mv).sum() / (sv | mv).sum()) if (sv | mv).sum() else np.nan,
                       mask_px=int(m.sum()), sil_px=int(sil.sum()))
    sil_res[lab] = rows
    RES[f"loader_A_residual_{lab}"] = resA


def summ(rows, a, b, key):
    v = np.array([rows[f][key] for f in range(a, b + 1) if f in rows])
    return {"mean": round(float(np.nanmean(v)), 3), "median": round(float(np.nanmedian(v)), 3),
            "min": round(float(np.nanmin(v)), 3), "max": round(float(np.nanmax(v)), 3)}


S = {}
for lab, rows in sil_res.items():
    for (a, b) in ((63, 122), (124, 160), (55, 62)):
        for key in ("iou", "cov", "ceil", "iou_occl"):
            S[f"{lab}_{key}_f{a}_{b}"] = summ(rows, a, b, key)
    S[f"{lab}_iou_over_ceiling_f63_122_median"] = round(float(np.nanmedian(
        [rows[f]["iou"] / rows[f]["ceil"] for f in range(63, 123) if f in rows and rows[f]["ceil"] > 0])), 3)
RES["blue_silhouette_summary"] = S
RES["blue_silhouette_every4"] = {lab: {f: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[f].items()}
                                       for f in range(55, 161, 4) if f in rows} for lab, rows in sil_res.items()}
json.dump(RES, open(OUTJ, "w"), indent=1)
for k, v in S.items():
    print(k, v)
print("EXTRA_DONE")
