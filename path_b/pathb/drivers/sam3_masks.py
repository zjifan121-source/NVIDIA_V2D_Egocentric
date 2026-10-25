"""P1b driver, run in the sam3 venv: SAM 3.1 text-prompted masks over a folder of JPEG frames.

    $SAM3_PY path_b/pathb/drivers/sam3_masks.py <request.json>

request.json: {"frames": <dir of %06d.jpg>, "out": <dir>, "tracks": [{"track": "obj_0", "text": "white pot"},
...], "min_prob": 0.35, "min_area": 0.005}. For each track: the seed frame is the first frame where
some instance of the text has probability >= min_prob and a mask covering >= min_area of the image
(V3_BUILD_SPEC.md P1b); every instance is propagated over the whole clip in both directions and
that one alone is kept; a hand's instance must differ from the other hand's (mask IoU <= 0.5 on the seed
frame), and an object's must differ from every earlier track's with the same text (ep23: two "wooden
board" tracks both seeded on the left board). Look-alikes take instances in probability order, so which
is which is arbitrary. Writes <out>/<track>/%06d.png (255 = the object, every frame) and
<out>/sam3_meta.json.
Imports only the sam3 package, numpy and PIL (no pathb).
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


OTHER_HAND = {"hand_L": "hand_R", "hand_R": "hand_L"}


def best(outputs: dict, min_prob: float, min_area: float, pixels: int, avoid: list | None = None):
    """(object id, prob, mask, box) of the most probable instance that passes both bars, or None;
    avoid: masks the instance must not repeat (IoU > 0.5), e.g. the other hand's."""
    ids = np.asarray(outputs["out_obj_ids"]).tolist()
    if not ids:
        return None
    probs = np.asarray(outputs["out_probs"]).tolist() if outputs.get("out_probs") is not None else [1.0] * len(ids)
    masks = np.asarray(outputs["out_binary_masks"]).astype(bool)
    boxes = np.asarray(outputs["out_boxes_xywh"]).tolist() if outputs.get("out_boxes_xywh") is not None else [None] * len(ids)
    keep = [i for i in range(len(ids)) if probs[i] >= min_prob and masks[i].sum() >= min_area * pixels]
    for a in avoid or []:
        if a.any():
            keep = [i for i in keep if (masks[i] & a).sum() <= 0.5 * (masks[i] | a).sum()]
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
    try:  # FlashAttention 3 (the builder's default) exists for Hopper GPUs only; the L40S is Ada
        import flash_attn_interface  # noqa: F401
        fa3 = torch.cuda.get_device_capability()[0] >= 9
    except ImportError:
        fa3 = False
    predictor = build_sam3_multiplex_video_predictor(use_fa3=fa3)
    # the pinned SAM 3 base predictor passes offload_state_to_cpu to init_state, which the multiplex
    # model's init_state does not take: pass on only the arguments it declares
    init, takes = predictor.model.init_state, set(inspect.signature(predictor.model.init_state).parameters)
    predictor.model.init_state = lambda **kw: init(**{k: v for k, v in kw.items() if k in takes})
    meta = {"model": "sam3.1 (build_sam3_multiplex_video_predictor)", "use_fa3": fa3, "frames": n, "size": [w, h],
            "load_seconds": round(time.time() - t0, 1), "tracks": {}}
    def fresh(old=None):
        """A new session for every pass: each starts with an empty action history, which SAM 3.1 reads
        to choose between a full propagation and a fetch of cached frames."""
        if old is not None:
            predictor.handle_request(dict(type="close_session", session_id=old))
        return predictor.handle_request(dict(type="start_session", resource_path=str(frames),
                                             offload_video_to_cpu=True))["session_id"]
    session, done = None, []
    for track in req["tracks"]:
        t1 = time.time()
        session = fresh(session)
        seed = None
        for f in range(n):  # the first frame where the text finds a clear instance
            predictor.handle_request(dict(type="reset_session", session_id=session))
            got = predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=f,
                                                text=track["text"]))["outputs"]
            others = [t["track"] for t in done if t["text"] == track["text"]]  # written before this one
            others += [OTHER_HAND[track["track"]]] if track["track"] in OTHER_HAND else []
            avoid = [np.asarray(Image.open(out / o / f"{f:06d}.png")) > 0 for o in others
                     if (out / o / f"{f:06d}.png").is_file()]
            pick = best(got, req["min_prob"], req["min_area"], pixels, avoid)
            if pick is not None:
                seed = (f, *pick)
                break
        d = out / track["track"]
        d.mkdir(parents=True, exist_ok=True)
        present = []
        if seed is not None:
            f, obj, prob, _, box = seed
            masks, backward_error = {}, None

            def collect(direction, chosen):
                for response in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session,
                                                                      propagation_direction=direction,
                                                                      start_frame_index=f)):
                    o = response["outputs"]
                    ids = np.asarray(o["out_obj_ids"]).tolist()
                    if chosen in ids:
                        m = np.asarray(o["out_binary_masks"])[ids.index(chosen)].astype(bool)
                        if m.any():
                            masks.setdefault(response["frame_index"], m)
            # every instance is propagated and only the chosen one kept: in the pinned SAM 3.1 a remove_object
            # before the first propagation makes it a "fetch" of the cached frames, of which only the prompt
            # frame exists (ep18's second track: KeyError 1)
            collect("forward", obj)
            if f > 0:  # backward from a fresh prompt: after a forward pass the pinned SAM 3's multiplex
                # tracker raised "No points are provided" going back (ep0, a hand seeded at frame 25)
                try:
                    session = fresh(session)
                    again = predictor.handle_request(dict(type="add_prompt", session_id=session, frame_index=f,
                                                          text=track["text"]))["outputs"]
                    ids = np.asarray(again["out_obj_ids"]).tolist()
                    ref = masks.get(f)
                    if ids:
                        same = [np.asarray(again["out_binary_masks"])[i].astype(bool) for i in range(len(ids))]
                        overlap = [((x & ref).sum() / max((x | ref).sum(), 1)) if ref is not None else 0 for x in same]
                        back_obj = ids[int(np.argmax(overlap))]
                        collect("backward", back_obj)
                except RuntimeError as err:
                    backward_error = str(err)[:200]
                    print(f"{track['track']}: backward propagation failed ({backward_error}); frames before {f} "
                          "stay empty", flush=True)
            for k in range(n):
                m = masks.get(k)
                Image.fromarray(((m if m is not None else np.zeros((h, w), bool)) * 255).astype(np.uint8)).save(
                    d / f"{k:06d}.png")
            present = sorted(masks)
            meta["tracks"][track["track"]] = {"text": track["text"], "seed_frame": f, "seed_prob": prob,
                                              "seed_box_xywh_rel": box, "obj_id": obj, "present_frames": present,
                                              "backward_error": backward_error, "seconds": round(time.time() - t1, 1)}
        else:
            for k in range(n):
                Image.fromarray(np.zeros((h, w), np.uint8)).save(d / f"{k:06d}.png")
            meta["tracks"][track["track"]] = {"text": track["text"], "seed_frame": None, "present_frames": [],
                                              "seconds": round(time.time() - t1, 1)}
        print(f"{track['track']}: seed {meta['tracks'][track['track']]['seed_frame']}, "
              f"present on {len(present)}/{n} frames", flush=True)
        done.append(track)
    if session is not None:
        predictor.handle_request(dict(type="close_session", session_id=session))
    meta["gpu_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None
    (out / "sam3_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
