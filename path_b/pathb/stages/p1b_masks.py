"""P1b masks: one mask track per roster object and per hand over ego_cam_a's undistorted frames.

SAM 3.1 (facebook/sam3.1, gated: the machine's HF login) in its own venv ($SAM3_PY, default
/root/v2d/venvs/sam3/bin/python) through pathb/drivers/sam3_masks.py, with the roster's text prompt
(tools/track3/episodes.json) and "left hand" / "right hand". If SAM 3.1 cannot run, the fallback is
the baseline's Grounding DINO box + SAM2 (pathb/drivers/sam2_masks.py in the baseline host venv),
seeded on the same rule. Seed frame: the first with an instance of probability >= 0.35 covering
>= 0.5 % of the image; propagation over the whole clip, both directions (V3_BUILD_SPEC.md P1b).

Distinct hands (V4_BUILD_SPEC.md §14.5 G-A, HAND_RESEED): the two hand tracks must be two hands, their
median mask IoU (frames with both) below RESEED_IOU and neither voting the other side on more frames than
its own (WiLoR's handedness, P1d's handedness_votes on WiLoR's raw detections, run here into wilor_raw/,
which P1d reuses). Otherwise the track whose votes favour the other side most is rebuilt by the driver's
re-seed from WiLoR's boxes of its side (the late-hand path from the first frame WiLoR sees that hand,
seeded again after each switch to the other hand); the first pass's track stays in initial_<track>/.
ep21/22/43/44: both prompts on one hand (IoU 0.67-0.99); ep0: the right track on the left hand from frame 230.
On an episode whose S1 exists, rerun P1b, P1d, P3 and P4, not P1c: DROID-SLAM is not repeatable (ep45 on identical
inputs: 0.5 mm median, 6.7 mm max), and G-A's P1c rerun moved ep0's camera 22.7 mm inside its rest (0.654 -> 0.354);
with the S1 camera kept (G-A′, v4_ga2) ep0 rises to 0.693. P3 does not read the hands (identical with old and new).

Out: obj_<i>/ (roster order), hand_L/, hand_R/ (%06d.png, 255 = inside), prompts.json,
hand_tracks.json (the baseline's track schema: object 1, hands 2 = left and 3 = right, as P1d's
hamer_aligned_filled/{2,3}), overlay/ (8 frames), masks_overlay.mp4, hand_check.json.
Bars: object mask on >= 95 % of frames (PASS), >= 80 % (WARN). The hand bar (masks wherever WiLoR
finds a hand) needs P1d's detections, so P1d reports it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .. import baseline as B
from ..paths import HOST_PY, PATH_B, RECON, TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json

MIN_PROB = 0.35  # default
MIN_AREA = 0.005  # default: share of the image
HANDS = (("hand_L", "left hand"), ("hand_R", "right hand"))
OVERLAY_FRAMES = 8
HAND_REACH = 0.04  # share of the image width: a hand's pieces lie this close to its main blob (fingers cut by an object)
COLORS = [(40, 200, 40), (40, 120, 255), (255, 80, 40), (220, 40, 220), (40, 220, 220)]
HAND_RESEED = True  # default (§14.5 G-A, adopted as G-A′: run v4_ga2): hand tracks that are not two distinct hands are re-seeded
RESEED_IOU = 0.1  # spec: two hand tracks are distinct below this median mask IoU (and per frame in the re-seed)
RESEED_BOX = 0.3  # default: a re-seed instance's mask box against WiLoR's box of its side (P1d's handedness bar)
SWITCH_RUN = 3  # default: frames in a row repeating the other hand that make a switch to it
MAX_SEGMENTS = 8  # default: re-seeds per track
SIDE = {"hand_L": "left", "hand_R": "right"}
OTHER = {"hand_L": "hand_R", "hand_R": "hand_L"}


def tracks(ctx) -> list[dict]:
    return ([{"track": f"obj_{i}", "text": o["prompt"], "object": o["name"]} for i, o in enumerate(ctx.episode["objects"])]
            + [{"track": t, "text": text} for t, text in HANDS])


def frames_video(frames: list, path) -> None:
    first = cv2.imread(str(frames[0]))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), TRACK3_FPS, first.shape[1::-1])
    for f in frames:
        writer.write(cv2.imread(str(f)))
    writer.release()


def overlay(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    out = image.astype(np.float32)
    for m, color in zip(masks, COLORS):
        out[m] = 0.5 * out[m] + 0.5 * np.array(color, np.float32)
    return out.astype(np.uint8)


def hand_blob(mask: np.ndarray, previous: np.ndarray | None, reach: int) -> np.ndarray:
    """One hand's mask without far pieces: the anchor is the component overlapping the previous
    frame's kept mask most (else the largest); components within `reach` px of it stay."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    anchor = 1 + int(np.argmax(areas))
    if previous is not None and previous.any():
        overlap = np.bincount(labels[previous], minlength=n)[1:]
        if overlap.max() > 0:
            anchor = 1 + int(np.argmax(overlap))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    near = cv2.dilate((labels == anchor).astype(np.uint8), kernel) > 0
    keep = np.unique(labels[near & (labels > 0)])
    return np.isin(labels, keep)


def clean_hands(d: Path, n: int, width: int, names: tuple[str, ...] = tuple(t for t, _ in HANDS)) -> dict[str, dict]:
    """Hand masks in place without far pieces (hand_blob, frame by frame in order). SAM 3.1's hand masks
    leak onto a second region on some frames (ep40: the box widens 375 -> 909 px for one frame),
    and HaMeR crops around the mask's box, so the wrist jumps ~15 cm. Objects are left as they are:
    a hand legitimately cuts an object in pieces."""
    out = {}
    reach = max(1, round(HAND_REACH * width))
    for track in names:
        previous, frames, removed = None, 0, []
        for k in range(n):
            f = d / track / f"{k:06d}.png"
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = m > 0
            kept = hand_blob(m, previous, reach)
            if kept.sum() < m.sum():
                frames += 1
                removed.append(1 - kept.sum() / m.sum())
                cv2.imwrite(str(f), kept.astype(np.uint8) * 255)
            previous = kept if kept.any() else previous
        out[track] = {"frames": frames, "removed_share_median": float(np.median(removed)) if removed else 0.0}
    return out


def hand_overlap(d: Path, n: int) -> tuple[float | None, int]:
    """Median IoU of the two hand tracks over the frames where both have a mask, and that count."""
    overlap = []
    for k in range(n):
        L, R = (cv2.imread(str(d / t / f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE) for t in ("hand_L", "hand_R"))
        if L is None or R is None:
            continue
        L, R = L > 0, R > 0
        if L.any() and R.any():
            overlap.append((L & R).sum() / max((L | R).sum(), 1))
    return (float(np.median(overlap)) if overlap else None), len(overlap)


def hand_check(d: Path, n: int, votes: dict, seeds: dict) -> dict:
    """Whether the two hand tracks are two hands (module docstring): median IoU < RESEED_IOU and each track's
    WiLoR votes for its own side at least those for the other. If not, the track to re-seed: the one whose
    votes favour the other side most (ties: the later seed)."""
    iou, both = hand_overlap(d, n)
    margin = {t: votes[t][SIDE[t]] - votes[t][SIDE[OTHER[t]]] for t in SIDE}
    distinct = (iou is None or iou < RESEED_IOU) and all(m >= 0 for m in margin.values())
    worst = min(SIDE, key=lambda t: (margin[t], -(seeds.get(t) if seeds.get(t) is not None else -1)))
    return {"median_iou": iou, "frames_with_both": both, "votes": votes, "margin": margin, "distinct": distinct,
            "reseed": None if distinct else worst}


def side_boxes(d: Path, n: int, right: bool) -> dict[int, list[float]]:
    """WiLoR's best box (by score) of one side per frame, pixels (x0, y0, x1, y1)."""
    out = {}
    for k in range(n):
        p = d / f"wilor_raw/{k:06d}.json"
        dets = [x for x in (json.loads(p.read_text()) if p.is_file() else []) if bool(x["is_right"]) == right]
        if dets:
            b = max(dets, key=lambda x: x.get("score", 0))["bbox"]
            out[k] = [b["x0"], b["y0"], b["x1"], b["y1"]]
    return out


def wilor(ctx, d: Path) -> None:
    """WiLoR's raw detections over the video into wilor_raw/ (they depend on the video only: P1d reuses them)."""
    if any((d / "wilor_raw").glob("*.json")):
        return
    video = B.frames_to_mp4(ctx.root / "p1_calibrate/cam_a", d / "wilor_video.mp4")  # as P1d's cam_a.mp4
    B.call("v2d.wilor.docker.run_video_to_hands:run_video_to_hands",
           {"video_path": video, "output_dir": d / "wilor_raw", "weights_dir": B.WEIGHTS / "wilor"}, d / "wilor.log")
    video.unlink(missing_ok=True)


def reseed(ctx, d: Path, n: int, req: dict, sam3: str, track: str) -> dict:
    """The driver's re-seed of one hand track; the first pass's track moves to initial_<track>/."""
    text = dict(HANDS)[track]
    request = {k: v for k, v in req.items() if k != "tracks"} | {"reseed": {
        "track": track, "other": OTHER[track], "texts": [text, "hand"],
        "boxes": {str(k): b for k, b in side_boxes(d, n, SIDE[track] == "right").items()},
        "max_iou": RESEED_IOU, "box_match": RESEED_BOX, "switch_run": SWITCH_RUN, "max_segments": MAX_SEGMENTS}}
    write_json(d / "reseed_request.json", request)
    with open(d / "sam3_reseed.log", "w") as log:
        rc = subprocess.run([sam3, str(PATH_B / "pathb/drivers/sam3_masks.py"), str(d / "reseed_request.json")],
                            stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        tail = "\n".join((d / "sam3_reseed.log").read_text().splitlines()[-15:])
        raise RuntimeError(f"SAM 3.1's re-seed of {track} failed (exit {rc}):\n{tail}")
    initial = d / f"initial_{track}"  # not hand_*: B.mask_tracks lists every hand_* folder as a track
    if initial.exists():
        shutil.rmtree(initial)
    (d / track).rename(initial)
    (d / f"{track}_reseed").rename(d / track)
    return json.loads((d / "sam3_reseed_meta.json").read_text())["reseed"]


def run(ctx) -> StageResult:
    d = ctx.dir("p1b_masks")
    frames = sorted((ctx.root / "p1_calibrate/cam_a").glob("*.jpg"))
    n = len(frames)
    req = {"frames": str(ctx.root / "p1_calibrate/cam_a"), "out": str(d), "tracks": tracks(ctx),
           "min_prob": MIN_PROB, "min_area": MIN_AREA}
    write_json(d / "request.json", req)
    sam3 = os.environ.get("SAM3_PY", "/root/v2d/venvs/sam3/bin/python")
    fallback = None
    with open(d / "sam3.log", "w") as log:
        rc = subprocess.run([sam3, str(PATH_B / "pathb/drivers/sam3_masks.py"), str(d / "request.json")],
                            stdout=log, stderr=subprocess.STDOUT).returncode if os.path.exists(sam3) else -1
    if rc != 0:
        fallback = (f"SAM 3.1 failed (exit {rc}; {d / 'sam3.log'})" if rc != -1 else f"no SAM 3.1 venv at {sam3}")
        frames_video(frames, d / "cam_a.mp4")
        req.update(video=str(d / "cam_a.mp4"), weights=str(RECON / "data/weights"))
        write_json(d / "request.json", req)
        with open(d / "sam2.log", "w") as log:
            rc2 = subprocess.run([str(HOST_PY), str(PATH_B / "pathb/drivers/sam2_masks.py"), str(d / "request.json")],
                                 cwd=RECON, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc2:
            tail = "\n".join((d / "sam2.log").read_text().splitlines()[-15:])
            raise RuntimeError(f"{fallback}; the SAM2 fallback failed too (exit {rc2}):\n{tail}")
    meta = json.loads((d / ("sam2_meta.json" if fallback else "sam3_meta.json")).read_text())
    width = cv2.imread(str(frames[0])).shape[1]
    cleaned = clean_hands(d, n, width)
    check = None
    if HAND_RESEED and fallback:
        check = {"skipped": "the re-seed runs on SAM 3.1 only"}
    elif HAND_RESEED:
        from .p1d_hands import handedness_votes
        wilor(ctx, d)
        hands = {t: d / t for t in SIDE}
        seeds = {t: meta["tracks"][t]["seed_frame"] for t in SIDE}
        check = hand_check(d, n, handedness_votes(d, hands, n), seeds)
        if not check["distinct"]:
            t = check["reseed"]
            got = reseed(ctx, d, n, req, sam3, t)
            first = got["segments"][0]["seed_frame"] if got["segments"] else None
            meta["tracks"][t] |= {"seed_frame": first, "present_frames": got["present_frames"], "reseed": got["segments"],
                                  "initial_seed_frame": meta["tracks"][t]["seed_frame"]}
            cleaned |= clean_hands(d, n, width, names=(t,))
            after = hand_check(d, n, handedness_votes(d, hands, n), seeds)
            check["after"] = {k: after[k] for k in ("median_iou", "frames_with_both", "votes", "margin", "distinct")}
            check["segments"] = got["segments"]
        write_json(d / "hand_check.json", check)
    write_json(d / "prompts.json", {"model": meta["model"], "fallback_reason": fallback,
                                    "tracks": {t: {k: v for k, v in m.items() if k != "present_frames"}
                                               for t, m in meta["tracks"].items()}})
    # hands only, with the baseline's ids (2, 3: P1d's hamer_aligned_filled/{2,3}); objects apart
    write_json(d / "hand_tracks.json", {
        "tracks": [{"object_id": 2, "role": "hand", "is_right": False, "mask_dir": "hand_L"},
                   {"object_id": 3, "role": "hand", "is_right": True, "mask_dir": "hand_R"}],
        "objects": [{"name": t["object"], "mask_dir": t["track"]} for t in req["tracks"] if "object" in t]})

    names = [t["track"] for t in req["tracks"]]
    present = {t: set(meta["tracks"][t]["present_frames"]) for t in names}
    picks = [int(i) for i in np.linspace(0, n - 1, OVERLAY_FRAMES)]
    (d / "overlay").mkdir(exist_ok=True)
    first = cv2.imread(str(frames[0]))
    size = (first.shape[1] // 2, first.shape[0] // 2)
    writer = cv2.VideoWriter(str(d / "masks_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), TRACK3_FPS, size)
    both, overlap = 0, []
    objects = [i for i, t in enumerate(req["tracks"]) if "object" in t]
    pairs = {(i, j): [] for a, i in enumerate(objects) for j in objects[a + 1:]}  # ep23: two tracks, one board
    for k, f in enumerate(frames):
        masks = [cv2.imread(str(d / t / f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0 for t in names]
        image = overlay(cv2.imread(str(f)), masks)
        writer.write(cv2.resize(image, size))
        if k in picks:
            cv2.imwrite(str(d / f"overlay/{k:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        for (i, j), seen in pairs.items():
            if masks[i].any() and masks[j].any():
                seen.append((masks[i] & masks[j]).sum() / max((masks[i] | masks[j]).sum(), 1))
        L, R = masks[names.index("hand_L")], masks[names.index("hand_R")]
        if L.any() and R.any():
            both += 1
            overlap.append((L & R).sum() / max((L | R).sum(), 1))
    writer.release()

    metrics = [Metric("segmenter", meta["model"], "", note=fallback or ""),
               Metric("SAM 3.1 ran (1) or the SAM2 fallback (0)", 0 if fallback else 1, "", ok=(1, 1),
                      note="V4 §0 rule 6: a stage whose primary model does not load FAILs (the masks stay, flagged)")]
    for t in req["tracks"]:
        share = len(present[t["track"]]) / n
        is_object = "object" in t
        metrics.append(Metric(f"{t['track']} ({t.get('object', t['text'])}): frames with a mask", share, "share",
                              ok=(0.95, 1) if is_object else None, warn=(0.8, 1) if is_object else None,
                              note=f"prompt {t['text']!r}, seed frame {meta['tracks'][t['track']]['seed_frame']}"))
    metrics += [Metric(f"{t}: frames with far pieces removed", c["frames"], "frames",
                       note=f"median share of the mask removed {c['removed_share_median']:.2f}; pieces farther than "
                            f"{HAND_REACH:.0%} of the width from the hand's blob")
                for t, c in cleaned.items()]
    metrics.append(Metric("left/right hand mask overlap (median IoU, frames with both)",
                          float(np.median(overlap)) if overlap else None, "", ok=(0, 0.5), warn=(0, 0.8),
                          note=f"{both} frames with both; near 1 means both prompts found the same hand"))
    if check and "distinct" in check:
        def said(c):
            iou = "none" if c["median_iou"] is None else f"{c['median_iou']:.2f}"
            return f"median IoU {iou} on {c['frames_with_both']} frames; WiLoR votes {c['votes']}"
        metrics.append(Metric("hand tracks distinct on the first pass (median IoU < 0.1, own-side WiLoR votes)",
                              int(check["distinct"]), "", note=said(check)))
        if not check["distinct"]:
            segs = "; ".join(f"seed {s['seed_frame']} ({s['text']!r}, box IoU {s['box_iou']}) frames {s['start']}-{s['end']}"
                             for s in check["segments"]) or "no seed found"
            metrics.append(Metric(f"re-seeded {check['reseed']}: distinct after", int(check["after"]["distinct"]), "",
                                  ok=(1, 1), note=f"{said(check['after'])}; {segs}"))
    worst = max(((float(np.median(v)), names[i], names[j]) for (i, j), v in pairs.items() if v), default=None)
    metrics.append(Metric("object mask overlap (median IoU, worst pair, frames with both)",
                          worst[0] if worst else None, "", ok=(0, 0.5), warn=(0, 0.8),
                          note=f"{worst[1]} / {worst[2]}; near 1 means two prompts found the same object" if worst else
                               "fewer than two objects"))
    return StageResult(metrics, {"masks": d, "prompts": d / "prompts.json", "hand_tracks": d / "hand_tracks.json",
                                 "overlay": d / "overlay", "video": d / "masks_overlay.mp4"})
