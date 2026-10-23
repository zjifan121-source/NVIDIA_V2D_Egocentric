"""P1b fallback driver, run in the baseline host venv: Grounding DINO boxes seed SAM2 (the baseline's
run_ego_wilor.py steps 4-6, through its own container wrappers), when SAM 3.1 cannot run.

    baseline/reconstruction/.venv/bin/python path_b/pathb/drivers/sam2_masks.py <request.json>

request.json as for sam3_masks.py, plus "video" (the frames as mp4) and "weights" (baseline
data/weights). Per track: Grounding DINO on every frame; the seed is the first frame whose best box
has confidence >= min_prob and covers >= min_area of the image (the box stands in for the mask);
SAM2 propagates all tracks from their seed boxes. Writes <out>/<track>/%06d.png and sam2_meta.json.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def main(request_path: str) -> int:
    from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
    from v2d.grounding_dino.docker.run_image_list_to_object_bboxes import run_image_list_to_object_bboxes
    from v2d.sam2.docker.run_video_to_masks import run_video_to_masks

    req = json.loads(Path(request_path).read_text())
    frames, out, weights = Path(req["frames"]), Path(req["out"]), Path(req["weights"])
    files = sorted(frames.glob("*.jpg"))
    n = len(files)
    w, h = Image.open(files[0]).size
    work = out / "sam2_work"
    work.mkdir(parents=True, exist_ok=True)
    meta = {"model": "sam2 + grounding_dino (baseline images)", "frames": n, "size": [w, h], "tracks": {}}
    prompts = []
    for i, track in enumerate(req["tracks"]):
        t0 = time.time()
        dets_path = work / f"dino_{track['track']}.json"
        if not dets_path.is_file():
            run_image_list_to_object_bboxes(image_dir=str(frames), output_path=str(dets_path), prompt=track["text"],
                                            model_dir=str(weights / "grounding_dino"))
        dets = json.loads(dets_path.read_text())
        seed = None
        for stem in sorted(dets, key=int):
            good = [d for d in dets[stem] if d["confidence"] >= req["min_prob"]
                    and (d["box"]["x1"] - d["box"]["x0"]) * (d["box"]["y1"] - d["box"]["y0"]) >= req["min_area"] * w * h]
            if good:
                seed = (int(stem), good[0])
                break
        meta["tracks"][track["track"]] = {"text": track["text"], "object_id": i + 1,
                                          "seed_frame": seed[0] if seed else None,
                                          "seed_prob": seed[1]["confidence"] if seed else None,
                                          "seed_box": seed[1]["box"] if seed else None,
                                          "dino_seconds": round(time.time() - t0, 1)}
        if seed:
            prompts.append(Sam2Prompt(frame_index=seed[0], object_id=i + 1, box=BoundingBox.from_dict(seed[1]["box"])))
    masks_dir = work / "masks"
    if prompts:
        Sam2Prompts(prompts=prompts).save(str(work / "sam2_prompts.json"))
        t0 = time.time()
        run_video_to_masks(video_path=req["video"], prompts_path=str(work / "sam2_prompts.json"),
                           masks_dir=str(masks_dir), weights_dir=str(weights / "sam2"))
        meta["sam2_seconds"] = round(time.time() - t0, 1)
    for i, track in enumerate(req["tracks"]):
        d = out / track["track"]
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        present = []
        for k in range(n):
            src = masks_dir / str(i + 1) / f"{k:06d}.png"
            m = np.asarray(Image.open(src)) > 0 if src.is_file() else np.zeros((h, w), bool)
            if m.ndim == 3:
                m = m.any(axis=2)
            if m.any():
                present.append(k)
            Image.fromarray((m * 255).astype(np.uint8)).save(d / f"{k:06d}.png")
        meta["tracks"][track["track"]]["present_frames"] = present
        print(f"{track['track']}: seed {meta['tracks'][track['track']]['seed_frame']}, present on {len(present)}/{n}",
              flush=True)
    (out / "sam2_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
