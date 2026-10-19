"""One self-contained HTML page per episode: where Path B's scale bridge gains and loses points.

Sections: score waterfall, what stereo sees (frame 0), the two rulers frame by frame, mesh size vs
world scale, motion against ground truth, camera checks, every measurement. Charts are inline SVG
drawn by a small script in visualize_page.html; frames are embedded as JPEG, so the file opens
offline. Reads a finished run (run.py run, then run.py waterfall) and writes
<episode>/ep<N>_path_b_scale.html.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .geometry import read_depth
from .paths import TRACK3, episode_info
from .stages import p3_scale_bridge as P3
from .team import retarget_tools, scorer

TEMPLATE = Path(__file__).with_name("visualize_page.html")
WIDTH = 640  # displayed image width, px
# page series colours (s1 blue, s2 orange, s3 green, s7 purple) as BGR, for drawing on frames
BGR = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (214, 80, 98)]
VARIANT_LABELS = {"raw": "Team bundle as uploaded", "stage2": "+ stage 2 repairs",
                  "stage2_stereo": "+ Path B stereo scale", "stage2_scan": "Scan-oracle scale (comparison)"}


def data_url(image: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def shrink(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA)


def depth_colour(depth: np.ndarray, near: float = 0.2, far: float = 1.6) -> np.ndarray:
    """Turbo colours, near = red; no depth = dark grey."""
    t = np.clip((far - depth) / (far - near), 0, 1)
    image = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[depth <= 0] = (40, 40, 40)
    return image


def colour_scale(near: float = 0.2, far: float = 1.6) -> str:
    strip = np.linspace(near, far, 256)[None, :].repeat(12, axis=0)
    return data_url(depth_colour(strip), 90)


def stereo_view(root: Path, data: dict, bundle: Path, names: list[str], scale: dict) -> dict:
    """Frame-0 images: ego_cam_a with the ray sample points, stereo depth, and the planes found."""
    L = retarget_tools()
    p1, p2 = root / "p1_calibrate", root / "p2_depth"
    cal = json.loads((p1 / "calib.json").read_text())
    alignment = json.loads((p2 / "alignment.json").read_text())
    rect = cal["rectification"]
    Kr = np.array([[rect["fx"], 0, rect["cx"]], [0, rect["fy"], rect["cy"]], [0, 0, 1.0]])
    K_a = np.array(cal["cam_a"]["K"])
    files = sorted((p2 / "depth_rect").glob("*.png"))
    stereo_frame = min(max(int(alignment.get("offset", 0)), 0), len(files) - 1)  # ego_cam_a frame 0

    # ego_cam_a frame 0 with each object's ray (where P3 samples stereo depth)
    cam_a = cv2.imread(str(p1 / "cam_a/000000.jpg"))
    s = WIDTH / cam_a.shape[1]
    cam_a = shrink(cam_a)
    w2c = np.linalg.inv(data["camera_to_world_transform"][0].astype(np.float64))
    for i, (name, (pose, scale_key, mesh)) in enumerate(zip(names, L.object_slots(data))):
        T = w2c @ data[pose][0].astype(np.float64)
        centre = (L.object_vertices(bundle, data, scale_key, mesh) @ T[:3, :3].T + T[:3, 3]).mean(axis=0)
        u, v = (K_a @ centre)[:2] / (K_a @ centre)[2] * s
        colour = BGR[(i + 2) % len(BGR)]
        cv2.circle(cam_a, (int(u), int(v)), 9, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(cam_a, (int(u), int(v)), 9, colour, 2, cv2.LINE_AA)
        cv2.putText(cam_a, name, (int(u) + 13, int(v) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(cam_a, name, (int(u) + 13, int(v) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

    depth = read_depth(files[stereo_frame])
    left = cv2.imread(str(p1 / f"rect_left/{stereo_frame:06d}.png"), cv2.IMREAD_GRAYSCALE)

    # planes: colour each plane's inliers (1 cm) on the rectified left image, outline the support plane
    planes = json.loads((p2 / "planes.json").read_text())["planes"][stereo_frame]["planes"]
    in_a = P3.planes_in_cam_a(planes, alignment)
    support = next((r["table_stereo"] for r in scale["frames"] if r["frame"] == 0), None)
    chosen = int(np.argmin([abs(d - support) for _, d in in_a])) if support is not None and in_a else None
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    X = np.stack([(u - Kr[0, 2]) / Kr[0, 0] * depth, (v - Kr[1, 2]) / Kr[1, 1] * depth, depth], axis=-1)
    overlay = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    shaded = overlay.copy()
    masks, legend = [], []
    for k, p in enumerate(planes):
        mask = (depth > 0) & (np.abs(X @ np.array(p["normal"]) + p["d"]) < 0.01)
        shaded[mask] = BGR[k % len(BGR)]
        masks.append(mask)
        legend.append({"colour": ["s1", "s2", "s3", "s7"][k % 4], "distance_m": in_a[k][1],
                       "share": p["inlier_fraction"], "chosen": k == chosen})
    overlay = cv2.addWeighted(shaded, 0.55, overlay, 0.45, 0)
    if chosen is not None:
        mask = cv2.morphologyEx(masks[chosen].astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 400]
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
    return {"cam_a": data_url(cam_a), "depth": data_url(shrink(depth_colour(depth))),
            "planes": data_url(shrink(overlay)), "scale_strip": colour_scale(), "stereo_frame": stereo_frame,
            "legend": legend}


def extents(vertices: np.ndarray) -> list[float]:
    return [round(100 * e, 1) for e in retarget_tools().oriented_extents(vertices)]


def mesh_sizes(data: dict, bundle: Path, names: list[str], scale: dict) -> list[dict]:
    L = retarget_tools()
    out = []
    for name, (_, scale_key, mesh) in zip(names, L.object_slots(data)):
        bundle_ext = extents(L.object_vertices(bundle, data, scale_key, mesh))
        scan = next((p for p in (L.SCANS / f"{name}/{name}_visual.glb", L.SCANS / f"{name}/{name}.glb") if p.is_file()), None)
        out.append({
            "name": name, "bundle": bundle_ext,
            "scan": extents(np.asarray(trimesh.load(scan, force="mesh", process=False).vertices)) if scan else None,
            "stereo": [round(e * scale["factor"], 1) for e in bundle_ext],
            "scan_rescaled": [round(e * scale["oracle_scan_factor"], 1) for e in bundle_ext]
            if scale.get("oracle_scan_factor") else None,
        })
    return out


def motion(root: Path, number: int, names: list[str], raw: Path) -> dict:
    """Distance of each object from its start position, per frame, in cm (rotation-free)."""
    L, S = retarget_tools(), scorer()
    gt = S.ground_truth(number)
    sources = {"raw": raw, **{v: root / "waterfall" / v for v in ("stage2", "stage2_stereo", "stage2_scan")}}
    objects = []
    for i, name in enumerate(names):
        series = {"gt": np.linalg.norm(gt[:, i, :3] - gt[0, i, :3], axis=1)}
        for variant, bundle in sources.items():
            if not (bundle / "result.npz").is_file():
                continue
            data = dict(np.load(bundle / "result.npz"))
            pose = L.object_slots(data)[i][0]
            p = data[pose][: len(gt), :3, 3].astype(np.float64)
            series[variant] = np.linalg.norm(p - p[0], axis=1)
        peak = series["gt"].max()
        objects.append({
            "name": name, "series": {k: np.round(100 * v, 2).tolist() for k, v in series.items()},
            "gt_peak_cm": round(100 * float(peak), 1),
            "peak_cm": {k: round(100 * float(v.max()), 1) for k, v in series.items()},
            "peak_ratio": {k: round(float(v.max() / peak), 3) for k, v in series.items() if k != "gt" and peak > 0},
        })
    moving = [o for o in objects if o["gt_peak_cm"] >= 10 and "stage2" in o["peak_ratio"]]
    # the factor that would make the bundle's motion as large as the real motion
    motion_factor = float(np.median([1 / o["peak_ratio"]["stage2"] for o in moving])) if moving else None
    return {"frames": len(gt), "objects": objects, "motion_factor": motion_factor,
            "moving_objects": [o["name"] for o in moving]}


def camera(root: Path) -> dict:
    cal = json.loads((root / "p1_calibrate/calib.json").read_text())
    alignment = json.loads((root / "p2_depth/alignment.json").read_text())
    offsets = sorted({t["offset"] for t in alignment["trials"].values()})
    best = {o: max(t["within_5px"] for t in alignment["trials"].values() if t["offset"] == o) for o in offsets}
    return {"offsets": offsets, "within_5px": [best[o] for o in offsets], "chosen_offset": alignment["offset"],
            "pose": alignment["pose"], "calibration_vs_pnp": alignment["calibration_vs_pnp"],
            "rows_before": cal["readings_checked"][cal["reading"]]["median_dy_px"],
            "rows_after": cal["after_row_correction"]["median_dy_px"], "row_fit": cal["row_correction"],
            "reading": cal["reading"]}


def motion_verdict(factor: float, motion: float | None, oracle: float | None, closer: str | None) -> tuple:
    if not motion:
        return ("neutral", "too little motion", "No object moves 10 cm or more, so motion cannot check the scale here.")
    off = factor / motion - 1
    text = (f"Real motion asks for a factor of {motion:.3f} (objects moving 10 cm or more); stereo gives {factor:.3f} "
            f"({off:+.0%})" + (f", the scan oracle {oracle:.3f}." if oracle else ".") +
            " Peak distance is a rough referee: a tracking jump (a flip, a lost frame) inflates it, so check the curves.")
    if abs(off) <= 0.10:
        return ("good", "stereo matches motion", text)
    return ("warn", {"stereo": "stereo closer than scan", "scan": "scan closer to motion"}.get(closer, "stereo off motion"),
            text)


def verdicts(page: dict) -> dict:
    """Section verdicts and one-line takeaways, from the numbers."""
    s = page["waterfall"]["scores"]
    gain = s["stage2_stereo"]["AUC"] - s["stage2"]["AUC"]
    p3 = {m["name"]: m for m in page["measurements"].get("p3_scale_bridge", [])}
    ruler_flags = [n for n in ("scale spread over rest frames (IQR / median)", "camera-to-support spread, stereo",
                               "object-ray factor vs support-plane factor") if p3.get(n, {}).get("status") in ("WARN", "FAIL")]
    off_meshes = [m["name"] for m in page["mesh"] if m["scan"] and not 0.8 <= np.mean(np.array(m["stereo"]) / np.array(m["scan"])) <= 1.25]
    f = page["factors"]
    factor = page["rulers"]["factor"]
    closer = None
    if f.get("motion") and f.get("scan oracle"):
        closer = "stereo" if abs(factor - f["motion"]) < abs(f["scan oracle"] - f["motion"]) else "scan"
    flagged = [m for ms in page["measurements"].values() for m in ms if m["status"] in ("WARN", "FAIL")]
    fails = sum(m["status"] == "FAIL" for m in flagged)
    # P3 takes the nearest horizontal plane below the object's surface point; a larger plane a few
    # cm below that one is usually the table, and the chosen one an object's base or the hands.
    legend = page["stereo"]["legend"]
    chosen = next((p for p in legend if p["chosen"]), None)
    larger = [p for p in legend if chosen and not p["chosen"] and p["share"] > chosen["share"]
              and 0 < p["distance_m"] - chosen["distance_m"] <= 0.10]
    if larger:
        b = max(larger, key=lambda p: p["share"])
        stereo = ("warn", "support plane suspect",
                  f"At frame 0 the plane used for scale ({chosen['distance_m']:.3f} m from the camera, "
                  f"{chosen['share']:.0%} of points) lies {100 * (b['distance_m'] - chosen['distance_m']):.1f} cm "
                  f"above a larger plane ({b['distance_m']:.3f} m, {b['share']:.0%}): most likely an object's base "
                  "or the hands rather than the table.")
    else:
        stereo = ("neutral", f"{len(legend)} planes",
                  "The support plane is the largest horizontal plane at most 35 cm below the object: not the largest "
                  "plane overall (often the floor), nor the nearest (an object's base, flat palms). Circles: where "
                  "each object's ray samples stereo depth.")
    return {
        "waterfall": ("good" if gain > 0.005 else "bad" if gain < -0.005 else "neutral",
                      f"stereo scale {gain:+.3f} AUC",
                      f"Path B's stereo scale moves AUC from {s['stage2']['AUC']:.3f} to {s['stage2_stereo']['AUC']:.3f}; "
                      f"the scan-oracle scale reaches {s['stage2_scan']['AUC']:.3f}."),
        "stereo": stereo,
        "rulers": ("bad" if any(p3[n]["status"] == "FAIL" for n in ruler_flags) else "warn" if ruler_flags else "good",
                   "unstable" if ruler_flags else "stable",
                   ("Flagged: " + "; ".join(ruler_flags) + ". The factor is the median, so it survives some "
                    "switching, but these frames are where it can go wrong.") if ruler_flags else
                   "Both rulers agree frame by frame."),
        "mesh": ("bad" if off_meshes else "good", "mesh off the scan" if off_meshes else "mesh fits",
                 (f"After the stereo rescale {', '.join(off_meshes)} no longer match{'es' if len(off_meshes) == 1 else ''} "
                  "the scan: the bundle's mesh size and its world scale disagree, so one factor cannot fix both.")
                 if off_meshes else "After the stereo rescale every mesh is within 0.8-1.25x of its scan."),
        "motion": motion_verdict(factor, f.get("motion"), f.get("scan oracle"), closer),
        "camera": ("good", f"offset {page['camera']['chosen_offset']:+d} corrected",
                   f"Stereo frame = ego_cam_a frame {page['camera']['chosen_offset']:+d}. Rows were off by "
                   f"{page['camera']['rows_before']:.2f} px with the factory calibration, "
                   f"{page['camera']['rows_after']:.2f} px after the correction."),
        "table": ("bad" if fails else "warn" if flagged else "good", f"{fails} fail · {len(flagged) - fails} warn",
                  "Every measurement Path B took on this episode, with its bar."),
    }


def clean(x):
    """JSON-safe: numpy to Python, non-finite numbers to null."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def page(root: Path) -> Path:
    number = int(root.name.removeprefix("episode_"))
    episode = episode_info(number)
    names = [o["name"] for o in episode["objects"]]
    loss = json.loads((root / "loss_report.json").read_text())
    scale = json.loads((root / "p3_scale_bridge/scale.json").read_text())
    waterfall = json.loads((root / "waterfall.json").read_text())
    run = json.loads((root / "run.json").read_text())
    bundle = Path(scale["bundle"])
    data = dict(np.load(bundle / "result.npz"))

    # the rulers again, with each object's ray ratio kept apart
    p2 = root / "p2_depth"
    planes = {p["frame"]: p["planes"] for p in json.loads((p2 / "planes.json").read_text())["planes"]}
    alignment = json.loads((p2 / "alignment.json").read_text())
    K_half = np.array(json.loads((p2 / "depth_a_half/intrinsics.json").read_text())["K"])
    rows = P3.measure(data, bundle, [r["frame"] for r in scale["frames"]], planes, alignment, p2 / "depth_a_half", K_half)

    content = {
        "episode": number, "task": episode.get("task"), "split": episode["split"], "objects": names,
        "commit": run["commit"]["sha"][:8], "finished": run["finished_utc"], "gpu": run.get("gpu"),
        "first_failing": loss["first_failing_stage"],
        "waterfall": {"variants": [v for v in waterfall["variants"] if v in waterfall["scores"]],
                      "labels": VARIANT_LABELS, "scores": waterfall["scores"],
                      "attribution": waterfall["auc_attribution"]},
        "stereo": stereo_view(root, data, bundle, names, scale),
        "rulers": {"frames": [r["frame"] for r in rows], "rest_end": scale["rest_end_frame"],
                   "stereo_m": [r["table_stereo"] for r in rows], "bundle": [r["table_bundle"] for r in rows],
                   "plane_ratio": [r["table_stereo"] / r["table_bundle"] if r["table_stereo"] else None for r in rows],
                   "ray_ratio": {n: [r["ray_by_object"][i] for r in rows] for i, n in enumerate(names)},
                   "factor": scale["factor"], "oracle": scale.get("oracle_scan_factor"), "method": scale["method"]},
        "mesh": mesh_sizes(data, bundle, names, scale),
        "motion": motion(root, number, names, bundle),
        "camera": camera(root),
        "measurements": loss["measurements"],
        "stages": [{"stage": s["stage"], "status": s["status"]} for s in loss["stages"]],
    }
    content["factors"] = {"support plane": scale["plane_factor"], "object ray": scale["ray_factor"],
                          "scan oracle": scale.get("oracle_scan_factor"), "motion": content["motion"]["motion_factor"]}
    content = clean(content)
    content["verdicts"] = verdicts(content)
    html = TEMPLATE.read_text().replace("__PAGE_DATA__", json.dumps(content).replace("</", "<\\/"))
    out = root / f"ep{number}_path_b_scale.html"
    out.write_text(html)
    return out
