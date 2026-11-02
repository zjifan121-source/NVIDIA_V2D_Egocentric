"""Recon P9 (James, 2026-11-02: "Use the gpu to focus on reconstruction score increases"): re-seed an object track
that collapsed onto another object's (B.lost_frames: public ep0's pot, P1b obj_0, on the lid from frame 74) with
SAM 3.1, through the P1b driver's re-seed (pathb/drivers/sam3_masks.py, written for the hands): seeded on the first
frame from the collapse where an instance of the track's text fills the track's last good mask box (frame f0 - 1)
without repeating the other track's mask (IoU < 0.5), propagated both ways, cut where it repeats the other's for 3
frames. The first pass's track is kept before f0. Run on the GPU machine with SAM 3.1's venv (its HF login):

    /root/v2d/venvs/sam3/bin/python path_b/vm/p1b_object_reseed.py <p1b dir copy> <cam_a dir> <f0> \
        path_b/pathb/drivers/sam3_masks.py [--track obj_0 --other obj_1 --texts "white pot" pot]

Writes <p1b dir>/<track> (merged; the first pass's in initial_<track>/, which B.mask_tracks does not list),
p9_reseed/ (the driver's request and output) and p9_report.json, with the GT-free check: the new mask against the
other track's on f0..end (IoU median, frames >= 0.9). Public ep0, 2026-11-02 (run v4_p9_s1): seeded at 120, pot
masks on 198 of the 299 frames from 74, IoU with the lid 0.0 (median), none >= 0.9.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("p1b", type=Path)
ap.add_argument("frames")
ap.add_argument("f0", type=int)
ap.add_argument("driver")
ap.add_argument("--track", default="obj_0")
ap.add_argument("--other", default="obj_1")
ap.add_argument("--texts", nargs="+", default=["white pot", "pot"])
a = ap.parse_args()
d, frames, f0, driver, track, other, texts = a.p1b, a.frames, a.f0, a.driver, a.track, a.other, a.texts
n = len(sorted(Path(frames).glob("*.jpg")))
mask = lambda p: (np.asarray(Image.open(p)) > 0) if p.is_file() else None
last = mask(d / f"{track}/{f0 - 1:06d}.png")
ys, xs = np.nonzero(last)
box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
work = d / "p9_reseed"
work.mkdir(exist_ok=True)
if not (work / other).exists():
    (work / other).symlink_to((d / other).resolve())  # the driver reads the other track's masks from its out
req = {k: v for k, v in json.loads((d / "request.json").read_text()).items() if k != "tracks"}
req |= {"frames": frames, "out": str(work), "reseed": {
    "track": track, "other": other, "texts": texts, "boxes": {str(t): box for t in range(f0, n)},
    "max_iou": 0.5, "box_match": 0.5, "switch_run": 3, "max_segments": 8}}
(work / "request.json").write_text(json.dumps(req, indent=1))
rc = subprocess.run([sys.executable, driver, str(work / "request.json")]).returncode
if rc:
    raise SystemExit(f"driver exit {rc}")
initial = d / f"initial_{track}"
if not initial.exists():
    (d / track).rename(initial)
out = d / track
out.mkdir(exist_ok=True)
ious, present = [], 0
for t in range(n):
    m = mask(initial / f"{t:06d}.png") if t < f0 else mask(work / f"{track}_reseed/{t:06d}.png")
    if m is None:
        m = np.zeros_like(last)
    Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{t:06d}.png")
    if t >= f0 and m.any():
        present += 1
        o = mask(d / f"{other}/{t:06d}.png")
        if o is not None and o.any():
            ious.append(float((m & o).sum() / (m | o).sum()))
meta = json.loads((work / "sam3_reseed_meta.json").read_text())
report = {"track": track, "other": other, "first_lost_frame": f0, "seed_box_xyxy": box, "texts": texts, "frames": n,
          "pot_mask_frames_from_f0": present, "of": n - f0,
          "iou_with_lid_median": float(np.median(ious)) if ious else None,
          "frames_iou_with_lid_ge_0.9": int(sum(i >= 0.9 for i in ious)),
          "segments": meta["reseed"]["segments"], "seconds": meta["reseed"].get("seconds"),
          "gpu_peak_mib": meta.get("gpu_peak_mib")}
(d / "p9_report.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "segments"}))
