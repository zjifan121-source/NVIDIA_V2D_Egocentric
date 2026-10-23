"""p1b_masks host side (the SAM 3.1 driver runs on the GPU machine): with a stand-in for the driver,
the stage writes prompts.json, hand_tracks.json, overlays, and bars object coverage."""
import json
import stat
import sys
from pathlib import Path

import cv2
import numpy as np

from pathb import runner

STUB = '''
import json, sys
from pathlib import Path
import cv2, numpy as np
req = json.loads(Path(sys.argv[2]).read_text())
out = Path(req["out"])
frames = sorted(Path(req["frames"]).glob("*.jpg"))
h, w = cv2.imread(str(frames[0])).shape[:2]
meta = {"model": "stub", "frames": len(frames), "tracks": {}}
for j, t in enumerate(req["tracks"]):
    d = out / t["track"]; d.mkdir(parents=True, exist_ok=True)
    present = []
    for k in range(len(frames)):
        m = np.zeros((h, w), np.uint8)
        if not (t["track"] == "obj_0" and k == 3):  # the object is missed on one frame
            cv2.circle(m, (10 + 12 * j, 20), 6, 255, -1); present.append(k)
        cv2.imwrite(str(d / f"{k:06d}.png"), m)
    meta["tracks"][t["track"]] = {"text": t["text"], "seed_frame": 0, "present_frames": present}
(out / "sam3_meta.json").write_text(json.dumps(meta))
'''


def test_p1b_host_side(tmp_path, monkeypatch):
    root = tmp_path / "episode_000040"
    cam = root / "p1_calibrate/cam_a"
    cam.mkdir(parents=True)
    for k in range(20):
        cv2.imwrite(str(cam / f"{k:06d}.jpg"), np.full((40, 64, 3), 90, np.uint8))
    (tmp_path / "stub.py").write_text(STUB)
    fake = tmp_path / "fakepy"
    fake.write_text(f"#!/bin/sh\nexec {sys.executable} {tmp_path / 'stub.py'} \"$@\"\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("SAM3_PY", str(fake))
    episode = {"episode": 40, "split": "public", "num_frames": 20,
               "objects": [{"name": "mini_dust_pan", "prompt": "yellow dust pan"}]}
    report, _ = runner.run_stage("p1b_masks", runner.Context(episode, root, {}), force=True)
    assert report["status"] == "PASS", report.get("error")
    d = root / "p1b_masks"
    cover = {m["name"].split(" ")[0]: m["value"] for m in report["metrics"] if "frames with a mask" in m["name"]}
    assert cover == {"obj_0": 0.95, "hand_L": 1.0, "hand_R": 1.0}
    hands = json.loads((d / "hand_tracks.json").read_text())
    assert [(t["object_id"], t["is_right"]) for t in hands["tracks"]] == [(2, False), (3, True)]
    assert hands["objects"] == [{"name": "mini_dust_pan", "mask_dir": "obj_0"}]
    assert len(list((d / "overlay").glob("*.jpg"))) == 8 and (d / "masks_overlay.mp4").stat().st_size > 0
    assert json.loads((d / "prompts.json").read_text())["tracks"]["obj_0"]["text"] == "yellow dust pan"
