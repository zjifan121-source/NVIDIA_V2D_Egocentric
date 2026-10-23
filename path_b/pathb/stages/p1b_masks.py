"""P1b masks: one mask track per roster object and per hand over ego_cam_a's undistorted frames.

SAM 3.1 (facebook/sam3.1, gated: the machine's HF login) in its own venv ($SAM3_PY, default
/root/v2d/venvs/sam3/bin/python) through pathb/drivers/sam3_masks.py, with the roster's text prompt
(tools/track3/episodes.json) and "left hand" / "right hand". If SAM 3.1 cannot run, the fallback is
the baseline's Grounding DINO box + SAM2 (pathb/drivers/sam2_masks.py in the baseline host venv),
seeded on the same rule. Seed frame: the first with an instance of probability >= 0.35 covering
>= 0.5 % of the image; propagation over the whole clip, both directions (V3_BUILD_SPEC.md P1b).

Out: obj_<i>/ (roster order), hand_L/, hand_R/ (%06d.png, 255 = inside), prompts.json,
hand_tracks.json (the baseline's track schema: object 1, hands 2 = left and 3 = right, as P1d's
hamer_aligned_filled/{2,3}), overlay/ (8 frames), masks_overlay.mp4.
Bars: object mask on >= 95 % of frames (PASS), >= 80 % (WARN). The hand bar (masks wherever WiLoR
finds a hand) needs P1d's detections, so P1d reports it.
"""
from __future__ import annotations

import json
import os
import subprocess

import cv2
import numpy as np

from ..paths import HOST_PY, PATH_B, RECON, TRACK3_FPS
from ..report import Metric, StageResult
from ..runner import write_json

MIN_PROB = 0.35  # default
MIN_AREA = 0.005  # default: share of the image
HANDS = (("hand_L", "left hand"), ("hand_R", "right hand"))
OVERLAY_FRAMES = 8
COLORS = [(40, 200, 40), (40, 120, 255), (255, 80, 40), (220, 40, 220), (40, 220, 220)]


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
    for k, f in enumerate(frames):
        masks = [cv2.imread(str(d / t / f"{k:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0 for t in names]
        image = overlay(cv2.imread(str(f)), masks)
        writer.write(cv2.resize(image, size))
        if k in picks:
            cv2.imwrite(str(d / f"overlay/{k:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        L, R = masks[names.index("hand_L")], masks[names.index("hand_R")]
        if L.any() and R.any():
            both += 1
            overlap.append((L & R).sum() / max((L | R).sum(), 1))
    writer.release()

    metrics = [Metric("segmenter", meta["model"], "", note=fallback or ""),
               Metric("SAM 3.1 ran (1) or the SAM2 fallback (0)", 0 if fallback else 1, "", ok=(1, 1), warn=(0, 1),
                      note="spec P1b: SAM 3.1 primary, SAM2 fallback")]
    for t in req["tracks"]:
        share = len(present[t["track"]]) / n
        is_object = "object" in t
        metrics.append(Metric(f"{t['track']} ({t.get('object', t['text'])}): frames with a mask", share, "share",
                              ok=(0.95, 1) if is_object else None, warn=(0.8, 1) if is_object else None,
                              note=f"prompt {t['text']!r}, seed frame {meta['tracks'][t['track']]['seed_frame']}"))
    metrics.append(Metric("left/right hand mask overlap (median IoU, frames with both)",
                          float(np.median(overlap)) if overlap else None, "", ok=(0, 0.5), warn=(0, 0.8),
                          note=f"{both} frames with both; near 1 means both prompts found the same hand"))
    return StageResult(metrics, {"masks": d, "prompts": d / "prompts.json", "hand_tracks": d / "hand_tracks.json",
                                 "overlay": d / "overlay", "video": d / "masks_overlay.mp4"})
