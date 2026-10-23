"""Object masks for one video with SAM 3, in place of Grounding DINO + SAM2's object track (V2D Track 3).

Called by run_ego_wilor.py step 6b when --object_segmenter sam3 is set, after SAM2 has tracked the hands only.
SAM 3 takes the text prompt at the reference frame, finds the object itself (no detector box) and propagates it
through the video in both directions. Outputs, in the formats the pipeline already uses:
  <masks_out>/NNNNNN.png   object mask per frame, uint8 0/255, video resolution, hand pixels removed
  <detections_out>         [{label, confidence, box: {x0, y0, x1, y1}}]: the reference-frame mask's box (what
                           Grounding DINO's dino_detections.json holds), so later steps read it unchanged
  <meta_out>               all reference-frame candidates, the chosen one, per-frame probability and area
Instance choice: the most probable candidate at the reference frame (Grounding DINO's top box did the same).
If SAM 3 finds nothing at its default cut-offs (score_threshold_detection 0.5, new_det_thresh 0.7) it retries at 0.3
and 0.15: on public episodes 39 (dust pan) and 22 (watering pitcher) the object is found only at 0.42 and 0.19. At a
lowered cut-off it takes only a clear winner (twice the next candidate's probability) and fails otherwise.
A frame where SAM 3 loses the object takes the nearest frame's mask, with a warning (never an empty mask).
Run with a Python environment that has SAM 3 installed (README.md in this folder):
  $V2D_SAM3_PYTHON sam3_object_masks.py --video undistorted.mp4 --prompt "white pot" --ref 0 \
      --checkpoint data/weights/sam3/sam3.pt --masks_out out/masks/1 --hand_masks out/masks/2 out/masks/3 \
      --detections_out out/dino_detections.json --meta_out out/sam3_meta.json
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

RETRY_THRESHOLDS = (0.3, 0.15)


def read_mask(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


def candidates(pred, session: str, frame: int, prompt: str) -> list[dict]:
    o = pred.handle_request(dict(type="add_prompt", session_id=session, frame_index=frame, text=prompt))["outputs"]
    ids = list(o["out_obj_ids"])
    probs = list(o.get("out_probs", [None] * len(ids)))
    out = []
    for k, oid in enumerate(ids):
        m = np.asarray(o["out_binary_masks"][k]).astype(bool)
        out.append({"obj_id": int(oid), "prob": None if probs[k] is None else float(probs[k]), "area": int(m.sum())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ref", type=int, default=0, help="reference frame (the pipeline's --reference_frame)")
    ap.add_argument("--checkpoint", required=True, help="sam3.pt from the facebook/sam3 Hugging Face repo")
    ap.add_argument("--masks_out", type=Path, required=True)
    ap.add_argument("--hand_masks", type=Path, nargs="*", default=[], help="SAM2 hand mask folders to subtract")
    ap.add_argument("--detections_out", type=Path, required=True)
    ap.add_argument("--meta_out", type=Path, required=True)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    t0 = time.time()
    pred = Sam3VideoPredictor(checkpoint_path=a.checkpoint)
    session = pred.handle_request(dict(type="start_session", resource_path=a.video,
                                       offload_video_to_cpu=True))["session_id"]
    cut_off = None
    cands = candidates(pred, session, a.ref, a.prompt)
    for thresh in RETRY_THRESHOLDS:
        if cands:
            break
        # the lowered cut-offs stay for propagation: SAM 3's hot-start check would otherwise drop the object
        pred.model.score_threshold_detection = pred.model.new_det_thresh = cut_off = thresh
        pred.handle_request(dict(type="reset_session", session_id=session))
        cands = candidates(pred, session, a.ref, a.prompt)
    print(f"SAM 3 {a.prompt!r} at frame {a.ref} (cut-off {cut_off or 'default'}): {cands}", flush=True)
    if not cands:
        raise SystemExit(f"SAM 3 found no {a.prompt!r} at frame {a.ref}, even at cut-off {RETRY_THRESHOLDS[-1]}. "
                         "Try another prompt or --reference_frame.")
    ranked = sorted(cands, key=lambda c: c["prob"] or 0.0, reverse=True)
    if cut_off is not None and len(ranked) > 1 and (ranked[0]["prob"] or 0.0) < 2 * (ranked[1]["prob"] or 0.0):
        # at a lowered cut-off a look-alike can outscore the target: take only a clear winner
        raise SystemExit(f"SAM 3 found {a.prompt!r} only at cut-off {cut_off}, with no clear winner among {ranked}. "
                         "Try a more specific prompt or another --reference_frame.")
    chosen = ranked[0]

    per = {}
    for item in pred.handle_stream_request(dict(type="propagate_in_video", session_id=session,
                                                propagation_direction="both")):
        t, o = int(item["frame_index"]), item["outputs"]
        oids = list(o["out_obj_ids"])
        if chosen["obj_id"] in oids:
            k = oids.index(chosen["obj_id"])
            p = o.get("out_probs")
            m = np.asarray(o["out_binary_masks"][k]).astype(bool)
            if m.any():
                per[t] = (m, None if p is None else float(p[k]))
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"propagated {len(per)} of {n_video} frames in {time.time() - t0:.0f} s, peak GPU memory {peak:.1f} GiB",
          flush=True)

    if not per:
        raise SystemExit(f"SAM 3 tracked no {a.prompt!r} on any frame")
    # A frame where SAM 3 lost the object takes the nearest frame's mask (earlier first): an empty mask would blank
    # FoundationPose's whole depth image for that frame. 0 such frames on 18 public objects (Oct 2026).
    have = sorted(per)
    filled = []
    for t in range(n_video):
        if t not in per:
            earlier = [s for s in have if s < t]
            per[t] = (per[earlier[-1] if earlier else have[0]][0], None)
            filled.append(t)
    if filled:
        print(f"WARNING: SAM 3 lost the object on {len(filled)} of {n_video} frames ({filled[:10]}...): "
              "nearest frame's mask used", flush=True)

    if a.masks_out.exists():
        shutil.rmtree(a.masks_out)
    a.masks_out.mkdir(parents=True)
    meta = {"prompt": a.prompt, "ref": a.ref, "video": a.video, "frames": n_video, "cut_off": cut_off,
            "candidates": cands, "chosen": chosen["obj_id"], "frames_filled": filled, "per_frame": {}}
    for t in range(n_video):
        m, p = per[t]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        whole = m.copy()
        for d in a.hand_masks:
            hm = read_mask(d / f"{t:06d}.png")
            if hm is not None and hm.shape == m.shape:
                m &= ~hm
        if not m.any():  # the hand covers all of what SAM 3 sees: keep it rather than nothing
            m = whole
        cv2.imwrite(str(a.masks_out / f"{t:06d}.png"), m.astype(np.uint8) * 255)
        meta["per_frame"][t] = {"prob": p, "area": int(m.sum())}

    ref = read_mask(a.masks_out / f"{a.ref:06d}.png")
    ys, xs = np.nonzero(ref)
    box = {"x0": float(xs.min()), "y0": float(ys.min()), "x1": float(xs.max()), "y1": float(ys.max())}
    write_json(a.detections_out, [{"label": a.prompt, "confidence": chosen["prob"], "box": box}])
    write_json(a.meta_out, meta)
    print(f"wrote {n_video} object masks to {a.masks_out} ({len(filled)} frames filled from a neighbour); box {box}",
          flush=True)


def write_json(path: Path, data) -> None:
    """Replace, never write through: run folders hard-link these files from other runs."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
