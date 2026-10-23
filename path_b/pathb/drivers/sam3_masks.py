"""P1b driver, run in the sam3 venv: SAM 3.1 text-prompted masks over a folder of JPEG frames.

    $SAM3_PY path_b/pathb/drivers/sam3_masks.py <request.json>

request.json: {"frames": <dir of %06d.jpg>, "out": <dir>, "tracks": [{"track": "obj_0", "text": "white pot"},
...], "min_prob": 0.35, "min_area": 0.005}. For each track: the seed frame is the first frame where
some instance of the text has probability >= min_prob and a mask covering >= min_area of the image
(V3_BUILD_SPEC.md P1b); that instance alone is kept and propagated over the whole clip in both
directions. Writes <out>/<track>/%06d.png (255 = the object, every frame) and <out>/sam3_meta.json.
Imports only the sam3 package, numpy and PIL (no pathb).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def best(outputs: dict, min_prob: float, min_area: float, pixels: int):
    """(object id, prob, mask, box) of the most probable instance that passes both bars, or None."""
    ids = np.asarray(outputs["out_obj_ids"]).tolist()
    if not ids:
        return None
    probs = np.asarray(outputs["out_probs"]).tolist() if outputs.get("out_probs") is not None else [1.0] * len(ids)
    masks = np.asarray(outputs["out_binary_masks"]).astype(bool)
    boxes = np.asarray(outputs["out_boxes_xywh"]).tolist() if outputs.get("out_boxes_xywh") is not None else [None] * len(ids)
    keep = [i for i in range(len(ids)) if probs[i] >= min_prob and masks[i].sum() >= min_area * pixels]
    if not keep:
        return None
    i = max(keep, key=lambda k: probs[k])
    return ids[i], float(probs[i]), masks[i], boxes[i]


def main(request_path: str) -> int:
    import torch
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    req = json.loads(Path(request_path).read_text())
    frames, out = Path(req["frames"]), Path(req["out"])
    files = sorted(frames.glob("*.jpg"))
    n = len(files)
    w, h = Image.open(files[0]).size
    pixels = w * h
    t0 = time.time()
    predictor = build_sam3_multiplex_video_predictor()
    meta = {"model": "sam3.1 (build_sam3_multiplex_video_predictor)", "frames": n, "size": [w, h],
            "load_seconds": round(time.time() - t0, 1), "tracks": {}}
    session = predictor.handle_request(dict(type="start_session", resource_path=str(frames),
                                            offload_video_to_cpu=True))["session_id"]
    for track in req["tracks"]:
        t1 = time.time()
        seed = None
        for f in range(n):  # the first frame where the text finds a clear instance
            predictor.handle_request(dict(type="reset_session", session_id=session))
            got = predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=f,
                                                text=track["text"]))["outputs"]
            pick = best(got, req["min_prob"], req["min_area"], pixels)
            if pick is not None:
                seed = (f, *pick)
                break
        d = out / track["track"]
        d.mkdir(parents=True, exist_ok=True)
        present = []
        if seed is not None:
            f, obj, prob, _, box = seed
            for other in np.asarray(got["out_obj_ids"]).tolist():  # keep the chosen instance only
                if other != obj:
                    predictor.handle_request(dict(type="remove_object", session_id=session, obj_id=other,
                                                  frame_index=f))
            masks = {}
            for response in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session,
                                                                  propagation_direction="both", start_frame_index=f)):
                o = response["outputs"]
                ids = np.asarray(o["out_obj_ids"]).tolist()
                if obj in ids:
                    m = np.asarray(o["out_binary_masks"])[ids.index(obj)].astype(bool)
                    if m.any():
                        masks[response["frame_index"]] = m
            for k in range(n):
                m = masks.get(k)
                Image.fromarray(((m if m is not None else np.zeros((h, w), bool)) * 255).astype(np.uint8)).save(
                    d / f"{k:06d}.png")
            present = sorted(masks)
            meta["tracks"][track["track"]] = {"text": track["text"], "seed_frame": f, "seed_prob": prob,
                                              "seed_box_xywh_rel": box, "obj_id": obj, "present_frames": present,
                                              "seconds": round(time.time() - t1, 1)}
        else:
            for k in range(n):
                Image.fromarray(np.zeros((h, w), np.uint8)).save(d / f"{k:06d}.png")
            meta["tracks"][track["track"]] = {"text": track["text"], "seed_frame": None, "present_frames": [],
                                              "seconds": round(time.time() - t1, 1)}
        print(f"{track['track']}: seed {meta['tracks'][track['track']]['seed_frame']}, "
              f"present on {len(present)}/{n} frames", flush=True)
    predictor.handle_request(dict(type="close_session", session_id=session))
    meta["gpu_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
    (out / "sam3_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
