"""ep3 hand identity (handoff "Four hand failures", p2 and p7): Path B's shipped bundle (HF path_b/runs/v4_eval_p4s_bundles)
against the doc's scan-free acceptance checks, with the release (scored20_alignment_mix_v1_20261030 @ 2b9c4a1) under the
same measures. Both hands re-posed by the baseline loader's own MANO path (pathb/drivers/mano_world.py). Observations, none
from HaMeR: Path B's P1b SAM 3.1 hand masks, WiLoR's detections (side + box, P1d wilor_raw), P1d's stereo depth.
- identity: WiLoR's side per hand; the mesh on its own hand's mask, not the other's
- export: the shipped bundle re-posed = Path B's P4 export mesh (camera frame, each with its own camera)
- image fit: silhouette share inside its mask, IoU with the mask, IoU with the mask inside WiLoR's box (the hand without
  the forearm), the mesh's 2-D box against WiLoR's box
- metric placement: |mesh depth - stereo depth| median / p90 on the same pixels (silhouette, mask, stereo), with coverage
- temporal: native MANO wrist rotation steps and wrist translation steps
Usage: python ep3_verify.py <Path B episode dir> <shipped bundle dir> <shipped mano npz> <release bundle dir> <release mano npz> <out dir>"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

E, SB, SM, RB, RM, OUT = (Path(a) for a in sys.argv[1:7])
H, W = 1520, 2028
SIDES = ("left", "right")
HELD_OUT = (0, 50, 100, 150, 200, 250, 300)  # approach, hold, release, and the doc's samples 50 / 100 / 150


def project(Vw, T_cw, K):
    T = np.linalg.inv(T_cw)
    Vc = Vw @ T[:3, :3].T + T[:3, 3]
    fx, fy, cx, cy = K
    return np.stack([fx * Vc[:, 0] / Vc[:, 2] + cx, fy * Vc[:, 1] / Vc[:, 2] + cy], 1), Vc[:, 2]


def render(uv, z, faces, depth=False):
    sil = np.zeros((H, W), np.uint8)
    tri = np.round(uv[faces]).astype(np.int32)
    cv2.fillPoly(sil, list(tri), 1)
    if not depth:
        return sil.astype(bool), None
    zb = np.full((H, W), np.inf, np.float32)
    for i in np.argsort(-z[faces].mean(1)):
        cv2.fillPoly(zb, [tri[i]], float(z[faces[i]].mean()))
    return sil.astype(bool), zb


def read_depth(path):  # P1d: inverse-depth PNG, pixel = 65535 / (depth_m + 1); 0 = none
    px = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    with np.errstate(divide="ignore"):
        d = 65535.0 / px - 1.0
    d[px == 0] = 0.0
    return d


def wilor_box(f, side):
    dets = [d for d in json.loads((E / f"p1d_hands/wilor_raw/{f:06d}.json").read_text()) if bool(d["is_right"]) == (side == "right")]
    if not dets:
        return None
    b = max(dets, key=lambda d: d["score"])["bbox"]
    return int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"])


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1)


def load(bundle, mano):
    m, b = np.load(mano), np.load(bundle / "result.npz")
    return {"verts": {s: m[f"{s}_vertices"] for s in SIDES}, "faces": {s: m[f"{s}_faces"] for s in SIDES},
            "orient": {s: np.nan_to_num(m[f"{s}_global_orient"]) for s in SIDES}, "joints": {s: m[f"{s}_joints"] for s in SIDES},
            "valid": {s: m[f"{s}_valid"].astype(bool) for s in SIDES},
            "T_cw": b["camera_to_world_transform"].astype(np.float64), "K": b["camera_intrinsics"]}


bundles = {"pathb": load(SB, SM), "release": load(RB, RM)}
n = len(bundles["pathb"]["T_cw"])

# export agreement: the shipped bundle through the loader = Path B's P4 export mesh (camera frame)
hv = np.load(E / "p4_export/hands/hand_verts_world.npz")
p4cam = np.load(E / "p4_export/episode_000003/white_pot/result_slam_gravity_aligned/result.npz")["camera_to_world_transform"].astype(np.float64)
export = {}
for i, s in enumerate(SIDES):
    ok = hv["valid"][:, i].astype(bool) & bundles["pathb"]["valid"][s]
    d = []
    for f in np.flatnonzero(ok):
        a = hv["verts"][f, i] @ np.linalg.inv(p4cam[f])[:3, :3].T + np.linalg.inv(p4cam[f])[:3, 3]
        T = np.linalg.inv(bundles["pathb"]["T_cw"][f])
        b = bundles["pathb"]["verts"][s][f] @ T[:3, :3].T + T[:3, 3]
        d.append(np.linalg.norm(a - b, axis=1).max())
    export[s] = {"frames": int(ok.sum()), "max_vertex_mm": round(float(np.max(d)) * 1e3, 3), "median_of_frame_max_mm": round(float(np.median(d)) * 1e3, 3)}

masks = {s: E / f"p1b_masks/hand_{s[0].upper()}" for s in SIDES}
res = {k: {s: [] for s in SIDES} for k in bundles}
for f in range(n):
    obs = {s: cv2.imread(str(masks[s] / f"{f:06d}.png"), 0) > 0 for s in SIDES}
    boxes = {s: wilor_box(f, s) for s in SIDES}
    dimg = read_depth(E / f"p1d_hands/depth/{f:06d}.png") if (f % 10 == 0 or f in HELD_OUT) else None
    for k, b in bundles.items():
        for s in SIDES:
            if not b["valid"][s][f]:
                res[k][s].append({"f": f, "valid": False}); continue
            uv, z = project(b["verts"][s][f], b["T_cw"][f], b["K"])
            sil, zb = render(uv, z, b["faces"][s], depth=dimg is not None)
            own, other = obs[s], obs["right" if s == "left" else "left"]
            r = {"f": f, "valid": True, "sil_px": int(sil.sum()), "mask_px": int(own.sum()),
                 "in_same": float((sil & own).sum() / max(sil.sum(), 1)), "in_other": float((sil & other).sum() / max(sil.sum(), 1)),
                 "iou_mask": float((sil & own).sum() / max((sil | own).sum(), 1))}
            box = boxes[s]
            if box is not None:
                x0, y0, x1, y1 = box
                hand = np.zeros_like(own); hand[max(y0, 0):y1, max(x0, 0):x1] = own[max(y0, 0):y1, max(x0, 0):x1]
                r["iou_hand"] = float((sil & hand).sum() / max((sil | hand).sum(), 1))
                ys, xs = np.nonzero(sil)
                r["box_iou"] = float(box_iou((xs.min(), ys.min(), xs.max(), ys.max()), box)) if len(xs) else 0.0
            if dimg is not None:
                elig = sil & own & np.isfinite(zb)
                px = elig & (dimg > 0)
                err = np.abs(zb[px] - dimg[px])
                r.update({"depth_px": int(px.sum()), "depth_coverage": float(px.sum() / max(elig.sum(), 1)),
                          "depth_med_mm": float(np.median(err) * 1e3) if px.any() else None,
                          "depth_p90_mm": float(np.percentile(err, 90) * 1e3) if px.any() else None})
            res[k][s].append(r)


def steps(b, s):
    ok = b["valid"][s]
    R = Rotation.from_rotvec(b["orient"][s])
    pair = ok[1:] & ok[:-1]
    deg = np.degrees((R[:-1].inv() * R[1:]).magnitude())[pair]
    wrist = np.asarray(b["joints"][s][:, 0], np.float64)  # MANO joint 0: the wrist
    mm = np.linalg.norm(np.diff(wrist, axis=0), axis=1)[pair] * 1e3
    return {"rot_step_max_deg": round(float(deg.max()), 1), "rot_steps_over_30deg": int((deg > 30).sum()),
            "rot_steps_over_90deg": int((deg > 90).sum()), "trans_step_max_mm": round(float(mm.max()), 1),
            "trans_step_median_mm": round(float(np.median(mm)), 1)}


def med(rows, key, q=50):
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.percentile(v, q)), 3) if v else None


handed = json.loads((E / "p1d_hands/handedness.json").read_text())
summary = {"wilor_handedness": {"2 (left track)": handed["2"], "3 (right track)": handed["3"]}, "export_agreement": export}
for k, b in bundles.items():
    for s in SIDES:
        rows = [r for r in res[k][s] if r["valid"]]
        boxed = [r for r in rows if "box_iou" in r]
        summary[f"{k}/{s}"] = {
            "frames": len(rows), "in_same_median": med(rows, "in_same"), "in_same_p10": med(rows, "in_same", 10),
            "frames_in_same_lt_0.5": sum(r["in_same"] < 0.5 for r in rows), "frames_mostly_on_other_hand": sum(r["in_other"] > 0.5 for r in rows),
            "iou_mask_median": med(rows, "iou_mask"), "iou_hand_median": med(boxed, "iou_hand"), "iou_hand_p10": med(boxed, "iou_hand", 10),
            "box_iou_median": med(boxed, "box_iou"), "box_iou_p10": med(boxed, "box_iou", 10),
            "frames_box_iou_lt_0.3": sum(r["box_iou"] < 0.3 for r in boxed), "frames_with_wilor_box": len(boxed),
            "depth_med_mm": med(rows, "depth_med_mm"), "depth_p90_mm": med(rows, "depth_p90_mm"),
            "depth_coverage": med(rows, "depth_coverage"), "depth_frames": sum(r.get("depth_med_mm") is not None for r in rows),
            **steps(b, s),
            "held_out": {f"f{r['f']}": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in r.items()
                                        if kk in ("in_same", "in_other", "iou_mask", "iou_hand", "box_iou", "depth_med_mm", "depth_p90_mm")}
                         for r in rows if r["f"] in HELD_OUT}}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "ep3_verify.json").write_text(json.dumps({"summary": summary, "per_frame": res}, indent=1))
print(json.dumps(summary, indent=1))

# figure: the held-out frames (but 0), Path B filled, the release outlined, SAM masks thin, WiLoR boxes dashed-like thin
cap = cv2.VideoCapture(str(E / "p1d_hands/cam_a.mp4"))
tiles = []
col = {("pathb", "left"): (255, 255, 0), ("pathb", "right"): (255, 0, 255), ("release", "left"): (0, 255, 255), ("release", "right"): (0, 140, 255)}
for f in HELD_OUT[1:]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = cap.read()
    assert ok, f
    over = img.copy()
    for k, b in bundles.items():
        for s in SIDES:
            if not b["valid"][s][f]:
                continue
            uv, z = project(b["verts"][s][f], b["T_cw"][f], b["K"])
            sil, _ = render(uv, z, b["faces"][s])
            if k == "pathb":
                over[sil] = (0.45 * over[sil] + 0.55 * np.array(col[(k, s)])).astype(np.uint8)
            cnts, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cnts, -1, col[(k, s)], 8 if k == "release" else 3)
    for s, c in (("left", (0, 255, 0)), ("right", (255, 128, 0))):
        m = (cv2.imread(str(masks[s] / f"{f:06d}.png"), 0) > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, c, 3)
        bx = wilor_box(f, s)
        if bx:
            cv2.rectangle(over, bx[:2], bx[2:], c, 2)
    cv2.putText(over, f"frame {f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (255, 255, 255), 6)
    tiles.append(cv2.resize(over, (W // 3, H // 3)))
grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
legend = np.full((70, grid.shape[1], 3), 255, np.uint8)
cv2.putText(legend, "ep3. Filled: Path B (cyan = left, magenta = right). Thick outline: release (yellow = left, orange = right). "
            "Thin: SAM 3.1 hand masks and WiLoR boxes (green = left, blue = right).", (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2)
cv2.imwrite(str(OUT / "ep3_heldout_frames.jpg"), np.vstack([legend, grid]), [cv2.IMWRITE_JPEG_QUALITY, 82])
print("figure:", OUT / "ep3_heldout_frames.jpg")
