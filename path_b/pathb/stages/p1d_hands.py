"""P1d hands: MANO hands per frame, aligned to the stereo depth (V3_BUILD_SPEC.md P1d).

The baseline's hand path (run_ego_wilor.py steps 3 and 14a-14e, hand_pose_source hamer), each step
through its own wrapper and container: WiLoR over the video (detections and handedness, used for the
coverage and handedness checks), HaMeR on P1b's hand masks (2 = left, 3 = right, from
hand_tracks.json), depth alignment against depth/ = P2's depth_a_half upsampled to full resolution
(hand masks inside, object masks outside), gap filling (tracks_interpolate, the pipeline's 15-frame
gaps, fixed betas), the filled overlay video. A hand that enters late has its own mask track from
P1b already (SAM 3.1 propagates both directions), which is what --late-hands does in the baseline.

Out: hamer_aligned_filled/{2,3}/%06d.json, hand_scale.json, handedness.json, overlay.mp4.
Bars: per hand, frames with a pose >= 90 % of frames with a mask; median hand scale 0.7-1.2 (PASS)
/ 0.6-1.5 (WARN); wrist speed < 2 m/s on >= 99 % of frames (PASS) / >= 95 % (WARN)
(run_load_retarget.bad_hand_frames's limit; world via P1c; V4 §13.5 E3). P1b's hand bar: a mask on >= 90 % of the frames WiLoR finds that hand.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from .. import baseline as B
from ..geometry import read_depth, read_transforms, write_depth
from ..paths import TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json

HANDS = {"2": "hand_L", "3": "hand_R"}  # P1b hand_tracks.json ids
BBOX_EXPANSION = 1.7  # the pipeline's hamer_bbox_expansion
MASK_MIN_PIXELS = 1  # the pipeline's value in hamer-primary mode
MAX_GAP = 15  # the pipeline's interp_max_gap_frames
MAX_WRIST_SPEED = 2.0  # m/s, run_load_retarget.bad_hand_frames
SPIKE_WINDOW = 2  # frames each side
BOX_GROW = 1.1  # default: WiLoR's box, grown, bounds the P1b hand pixels kept for that side
BOX_SHRINK = 0.9  # default: WiLoR's box, shrunk, is the mask where P1b has no hand pixel in it
SPIKE_M = MAX_WRIST_SPEED * SPIKE_WINDOW / TRACK3_FPS  # 0.2 m: farther than the speed bar allows over the window


def prepare(ctx, d, n: int) -> None:
    """masks/{2,3} (P1b hands), object_masks/ (union of P1b objects), depth/ (full resolution),
    intrinsics.json, hand_tracks.json, cam_a.mp4."""
    tracks = B.mask_tracks(ctx.root)
    masks = d / "masks"
    masks.mkdir(exist_ok=True)
    for oid, name in HANDS.items():  # files, not a folder link: the containers mount p1d_hands only
        link = masks / oid
        if link.is_symlink():
            link.unlink()
        link.mkdir(exist_ok=True)
        for f in tracks[name].glob("*.png"):
            target = link / f.name
            target.unlink(missing_ok=True)
            try:
                os.link(f, target)
            except OSError:
                shutil.copy2(f, target)
    calib = json.loads((ctx.root / "p1_calibrate/calib.json").read_text())
    W, H = calib["cam_a"]["size"]
    (d / "object_masks").mkdir(exist_ok=True)
    (d / "depth").mkdir(exist_ok=True)
    half = ctx.root / "p2_depth/depth_a_half"
    for k in range(n):
        union = np.zeros((H, W), bool)
        for name, folder in tracks.items():
            if name.startswith("obj_"):
                m = B.read_mask(folder, k)
                if m is not None:
                    union |= m
        cv2.imwrite(str(d / f"object_masks/{k:06d}.png"), union.astype(np.uint8) * 255)
        src = half / f"{k:06d}.png"
        if src.is_file() and not (d / f"depth/{k:06d}.png").is_file():
            write_depth(d / f"depth/{k:06d}.png", cv2.resize(read_depth(src), (W, H), interpolation=cv2.INTER_NEAREST))
    B.intrinsics_json(calib["cam_a"]["K"], (W, H), d / "intrinsics.json")
    shutil.copy(ctx.root / "p1b_masks/hand_tracks.json", d / "hand_tracks.json")
    B.frames_to_mp4(ctx.root / "p1_calibrate/cam_a", d / "cam_a.mp4")


def cam_t(r: dict) -> list:
    """A hand record's camera-frame wrist position, in either layout (see host_layout)."""
    return r["camera"]["cam_t"] if "camera" in r else r.get("cam_t", [])


def host_layout(folder) -> int:
    """The GHCR HaMeR / WiLoR images (afe7dad) write cam_t and intrinsics at the record's top level; the
    baseline host code that reads P1d's records (v2d_common.result_bundle._load_hand_track) takes
    camera.cam_t and falls back to [0, 0, 0] without a word. Each record gains camera: {cam_t,
    intrinsics}; the top-level fields stay. Returns the number of records changed."""
    changed = 0
    for f in sorted(Path(folder).glob("*/*.json")):
        r = json.loads(f.read_text())
        if "camera" not in r and "cam_t" in r:
            r["camera"] = {"cam_t": r["cam_t"], "intrinsics": r.get("intrinsics")}
            f.write_text(json.dumps(r, indent=2))
            changed += 1
    return changed


def hand_depth(d, tracks: dict, n: int) -> Path:
    """depth_hands/: P1d's depth with the pixels inside each hand mask that have no stereo depth set
    to the median of the mask's valid depth. run_align_hands takes the median depth over the hand mask
    with no validity test (our "no depth" decodes as infinity, which biases it far: ep40, ~30 % of the
    mask); values equal to the median leave the median unchanged, so it gets the valid pixels' median.
    Only align_hands reads this copy; depth/ (S3a, S3b) is unchanged."""
    out = d / "depth_hands"
    out.mkdir(exist_ok=True)
    for k in range(n):
        src = d / f"depth/{k:06d}.png"
        if not src.is_file():
            continue
        depth = read_depth(src)
        for name in HANDS.values():
            m = B.read_mask(tracks[name], k)
            if m is None:
                continue
            valid = m & (depth > 0)
            if valid.any():
                depth[m & (depth <= 0)] = float(np.median(depth[valid]))
        write_depth(out / f"{k:06d}.png", depth)
    return out


def reject_nonfinite(d) -> dict[str, list[int]]:
    """Aligned records whose depth alignment is not finite move to hamer_aligned_rejected/: run_align_hands
    takes the median depth over the hand mask with no validity test, and our "no depth" pixels decode
    as infinity, so a hand mostly over missing depth gets an infinite scale and position. Rejected
    frames are left to tracks_interpolate (gaps up to MAX_GAP), like frames HaMeR did not see."""
    out = {}
    for oid in HANDS:
        frames = []
        for f in sorted((d / f"hamer_aligned/{oid}").glob("*.json")):
            r = json.loads(f.read_text())
            values = list(cam_t(r)) + [r.get("diagnostics", {}).get("scale", 0.0)]
            if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                dest = d / f"hamer_aligned_rejected/{oid}"
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), dest / f.name)
                frames.append(int(f.stem))
        out[oid] = frames
    return out


def reject_spikes(d) -> dict[str, list[int]]:
    """Aligned records farther than SPIKE_M from the median of their real neighbours (SPIKE_WINDOW
    frames each side, camera frame) move to hamer_aligned_rejected/. ep40 (2026-10-23): the unchecked
    median over a hand mask with ~30 % missing depth puts single frames 0.5-0.8 m too far (0.46 ->
    0.99 -> 0.47 m in one frame); like non-finite fits, they are left to tracks_interpolate."""
    out = {}
    for oid in HANDS:
        files = sorted((d / f"hamer_aligned/{oid}").glob("*.json"))
        frames = np.array([int(f.stem) for f in files])
        pos = np.array([cam_t(json.loads(f.read_text())) for f in files], dtype=np.float64).reshape(-1, 3)
        bad = []
        for i, k in enumerate(frames):
            near = (np.abs(frames - k) <= SPIKE_WINDOW) & (frames != k)
            if near.sum() >= 2 and np.linalg.norm(pos[i] - np.median(pos[near], axis=0)) > SPIKE_M:
                bad.append(i)
        dest = d / f"hamer_aligned_rejected/{oid}"
        for i in bad:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(files[i]), dest / files[i].name)
        out[oid] = [int(frames[i]) for i in bad]
    return out


def side_masks(d, tracks: dict, n: int, votes: dict) -> dict[str, dict[str, int]]:
    """masks/2 = the left hand, masks/3 = the right, by WiLoR's handedness frame by frame: P1b's hand
    pixels (either track) inside WiLoR's best box of that side (expanded BOX_GROW), else that box
    (shrunk BOX_SHRINK); with no WiLoR box of that side, the P1b track WiLoR votes that side on most.
    P1b's prompts can drift to the other hand (ep0: both tracks voted left, 263:20 and 61:21, so the
    bundle had no right hand) or lose a hand for good (ep13: the left on 14 of the 171 frames WiLoR saw
    it). The files in masks/ are P1b's hard links: each is unlinked before it is written."""
    side_track = {}
    for right in (False, True):
        key = "right" if right else "left"
        best = max(votes, key=lambda name: votes[name][key] - votes[name]["left" if right else "right"])
        if votes[best][key] > votes[best]["left" if right else "right"]:
            side_track[right] = best
    counts = {"2": {"wilor_and_p1b": 0, "wilor_box": 0, "p1b_only": 0, "none": 0},
              "3": {"wilor_and_p1b": 0, "wilor_box": 0, "p1b_only": 0, "none": 0}}
    for k in range(n):
        p = d / f"wilor_raw/{k:06d}.json"
        dets = json.loads(p.read_text()) if p.is_file() else []
        hands_px = None
        for name in HANDS.values():
            m = B.read_mask(tracks[name], k)
            if m is not None:
                hands_px = m if hands_px is None else hands_px | m
        shape = hands_px.shape if hands_px is not None else None
        for oid, right in (("2", False), ("3", True)):
            target = d / f"masks/{oid}/{k:06d}.png"
            if shape is None:
                continue
            mine = [x for x in dets if bool(x["is_right"]) == right]
            out = np.zeros(shape, bool)
            if mine:
                b = max(mine, key=lambda x: x.get("score", 0))["bbox"]
                cx, cy, w, h = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, b["x1"] - b["x0"], b["y1"] - b["y0"]
                def box(f):
                    x0, x1 = int(max(0, cx - f * w / 2)), int(min(shape[1], cx + f * w / 2))
                    y0, y1 = int(max(0, cy - f * h / 2)), int(min(shape[0], cy + f * h / 2))
                    r = np.zeros(shape, bool)
                    r[y0:y1, x0:x1] = True
                    return r
                inside = hands_px & box(BOX_GROW)
                if inside.any():
                    out, kind = inside, "wilor_and_p1b"
                else:
                    out, kind = box(BOX_SHRINK), "wilor_box"
            elif right in side_track:
                m = B.read_mask(tracks[side_track[right]], k)
                out, kind = (m, "p1b_only") if m is not None and m.any() else (out, "none")
            else:
                kind = "none"
            counts[oid][kind] += 1
            target.unlink(missing_ok=True)
            cv2.imwrite(str(target), out.astype(np.uint8) * 255)
    return counts


def records(folder) -> dict[int, dict]:
    return {int(p.stem): json.loads(p.read_text()) for p in sorted(folder.glob("*.json")) if p.stem.isdigit()}


def wilor_hands(d, n: int) -> dict[bool, set[int]]:
    """Frames on which WiLoR finds a right (True) / left (False) hand."""
    out = {True: set(), False: set()}
    for k in range(n):
        p = d / f"wilor_raw/{k:06d}.json"
        if p.is_file():
            for det in json.loads(p.read_text()):
                out[bool(det["is_right"])].add(k)
    return out


def handedness_votes(d, tracks: dict, n: int, min_iou: float = 0.3) -> dict[str, dict]:
    """Per P1b hand track, WiLoR's left/right votes on the frames where one of its boxes overlaps the
    track's mask box (IoU >= min_iou): the baseline takes handedness from WiLoR, P1b from a text
    prompt, so WiLoR decides."""
    votes = {name: {"right": 0, "left": 0} for name in HANDS.values()}
    for k in range(n):
        p = d / f"wilor_raw/{k:06d}.json"
        if not p.is_file():
            continue
        dets = json.loads(p.read_text())
        for name in HANDS.values():
            m = B.read_mask(tracks[name], k)
            if m is None or not m.any():
                continue
            ys, xs = np.nonzero(m)
            mb = (xs.min(), ys.min(), xs.max(), ys.max())
            best, side = 0.0, None
            for det in dets:
                b = det["bbox"]
                ix = max(0, min(mb[2], b["x1"]) - max(mb[0], b["x0"]))
                iy = max(0, min(mb[3], b["y1"]) - max(mb[1], b["y0"]))
                inter = ix * iy
                union = (mb[2] - mb[0]) * (mb[3] - mb[1]) + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best:
                    best, side = iou, bool(det["is_right"])
            if side is not None and best >= min_iou:
                votes[name]["right" if side else "left"] += 1
    return votes


def run(ctx) -> StageResult:
    d = ctx.dir("p1d_hands")
    n = len(list((ctx.root / "p1_calibrate/cam_a").glob("*.jpg")))
    prepare(ctx, d, n)
    log = d / "hands.log"
    log.write_text("")
    W = B.WEIGHTS
    mano = W / "hamer/_DATA/data"
    if not any((d / "wilor_raw").glob("*.json")):
        B.call("v2d.wilor.docker.run_video_to_hands:run_video_to_hands",
               {"video_path": d / "cam_a.mp4", "output_dir": d / "wilor_raw", "weights_dir": W / "wilor"}, log)
    tracks = B.mask_tracks(ctx.root)
    # masks_to_hands and align_hands skip frames whose output exists: a rerun (new P1b masks, new
    # depth) must start from empty folders. WiLoR's raw detections depend on the video only.
    for stale in ("hamer", "hamer_aligned", "hamer_aligned_rejected", "hamer_aligned_filled"):
        shutil.rmtree(d / stale, ignore_errors=True)
    votes = handedness_votes(d, tracks, n)
    sources = side_masks(d, tracks, n, votes)  # masks/2 left, masks/3 right, by WiLoR's handedness
    meta = json.loads((d / "hand_tracks.json").read_text())
    for t in meta["tracks"]:
        v = votes[HANDS[str(t["object_id"])]]
        t["prompt_is_right"] = t["is_right"]
        t["is_right"] = str(t["object_id"]) == "3"  # side_masks: 2 is the left hand, 3 the right
        t["wilor_votes"] = v
        t["mask_sources"] = sources[str(t["object_id"])]
    write_json(d / "hand_tracks.json", meta)
    B.call("v2d.hamer.docker.run_masks_to_hands:run_masks_to_hands",
           {"frames_dir": ctx.root / "p1_calibrate/cam_a", "masks_dir": d / "masks", "tracks_path": d / "hand_tracks.json",
            "output_dir": d / "hamer", "weights_dir": W / "hamer", "bbox_expansion": BBOX_EXPANSION,
            "mask_min_pixels": MASK_MIN_PIXELS}, log)
    B.call("v2d.hamer.docker.run_align_hands:run_align_hands",
           {"hamer_dir": d / "hamer", "depth_dir": hand_depth(d, tracks, n), "intrinsics_path": d / "intrinsics.json",
            "mano_assets_root": mano, "output_dir": d / "hamer_aligned", "hand_masks_dir": d / "masks",
            "object_masks_dir": d / "object_masks", "mask_min_pixels": MASK_MIN_PIXELS}, log)
    shutil.rmtree(d / "depth_hands", ignore_errors=True)  # ~2 MB a frame; only align_hands reads it
    shutil.rmtree(d / "hamer_aligned_rejected", ignore_errors=True)
    rejected = reject_nonfinite(d)
    spikes = reject_spikes(d)
    shutil.rmtree(d / "hamer_aligned_filled", ignore_errors=True)
    B.call("v2d.wilor.docker.run_tracks_interpolate:run_tracks_interpolate",
           {"aligned_dir": d / "hamer_aligned", "masks_dir": d / "masks", "output_dir": d / "hamer_aligned_filled",
            "betas": "fixed", "max_gap_frames": MAX_GAP}, log)
    B.call("v2d.hamer.docker.run_render_hands_aligned_video:run_render_hands_aligned_video",
           {"frames_dir": ctx.root / "p1_calibrate/cam_a", "aligned_dir": d / "hamer_aligned_filled",
            "mano_assets_root": mano, "output_path": d / "overlay.mp4", "fps": TRACK3_FPS}, log)
    host_layout(d / "hamer_aligned_filled")
    handed = d / "hamer_aligned_filled/handedness.json"
    if handed.is_file():
        shutil.copy(handed, d / "handedness.json")

    c2w, have_cam = read_transforms(ctx.root / "p1c_camera/camera_to_world", n)
    wilor = wilor_hands(d, n)
    metrics, scales = [], {}
    for t in meta["tracks"]:
        name = HANDS[str(t["object_id"])]
        src = t["mask_sources"]
        metrics.append(Metric(f"{name}: hand mask by WiLoR's handedness", "right" if t["is_right"] else "left", "",
                              note=f"frames: WiLoR box with P1b pixels {src['wilor_and_p1b']}, WiLoR box only "
                                   f"{src['wilor_box']}, P1b only {src['p1b_only']}, none {src['none']}; P1b track "
                                   f"{name} votes {t['wilor_votes']}"))
    for oid, name in HANDS.items():
        filled = records(d / f"hamer_aligned_filled/{oid}")
        real = records(d / f"hamer_aligned/{oid}")
        masked = {k for k in range(n) if (m := B.read_mask(d / f"masks/{oid}", k)) is not None and m.any()}
        posed = set(filled) & masked
        s = [r["diagnostics"]["scale"] for r in real.values() if r.get("diagnostics", {}).get("n_pixels", 0) > 0]
        scales[name] = {"median": float(np.median(s)) if s else None, "frames": len(s)}
        cam_t = {k: np.array(r["camera"]["cam_t"], dtype=np.float64) for k, r in filled.items()}
        frames = sorted(k for k in cam_t if have_cam[k])
        world = np.array([c2w[k, :3, :3] @ cam_t[k] + c2w[k, :3, 3] for k in frames])
        speeds = (np.linalg.norm(np.diff(world, axis=0), axis=1) * TRACK3_FPS / np.diff(frames)
                  if len(frames) > 1 else np.array([]))
        right = next(t["is_right"] for t in meta["tracks"] if HANDS[str(t["object_id"])] == name)
        seen = wilor[right]
        holes = [float((read_depth(d / f"depth/{k:06d}.png")[m] <= 0).mean())
                 for k in sorted(masked)[:: max(1, len(masked) // 20)]
                 if (m := B.read_mask(d / f"masks/{oid}", k)) is not None]
        metrics += [
            Metric(f"{name}: hand-mask pixels without stereo depth (median)", float(np.median(holes)) if holes else None,
                   "share", note="run_align_hands takes the median over the mask with no validity test"),
            Metric(f"{name}: aligned frames rejected (non-finite depth fit)", len(rejected[oid]), "frames",
                   note="left to tracks_interpolate; frames " + (", ".join(map(str, rejected[oid][:12])) or "none")),
            Metric(f"{name}: aligned frames rejected (> {SPIKE_M:.2f} m from their neighbours)", len(spikes[oid]), "frames",
                   note="left to tracks_interpolate; frames " + (", ".join(map(str, spikes[oid][:12])) or "none")),
            Metric(f"{name}: frames with a pose / frames with a mask", len(posed) / len(masked) if masked else None,
                   "share", ok=(0.9, 1), note=f"{len(posed)} of {len(masked)}"),
            Metric(f"{name}: hand scale (median, depth fit)", scales[name]["median"], "x", ok=(0.7, 1.2),
                   warn=(0.6, 1.5), note=f"{len(s)} aligned frames"),
            Metric(f"{name}: frames with wrist speed < {MAX_WRIST_SPEED} m/s", float(np.mean(speeds < MAX_WRIST_SPEED))
                   if len(speeds) else None, "share", ok=(0.99, 1), warn=(0.95, 1),
                   note="E3: 0.95-0.99 is WARN (a few frames of the hand tracker's spikes, left to tracks_interpolate)"),
            Metric(f"{name}: hand mask on frames WiLoR finds this hand", len(seen & masked) / len(seen) if seen else None,
                   "share", ok=(0.9, 1), note=f"WiLoR finds the {'right' if right else 'left'} hand on {len(seen)} frames"),
        ]
    write_json(d / "hand_scale.json", scales)
    return StageResult(metrics, {"hands": d / "hamer_aligned_filled", "hand_scale": d / "hand_scale.json",
                                 "handedness": d / "handedness.json", "overlay": d / "overlay.mp4"})
