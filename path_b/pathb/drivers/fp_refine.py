"""S3e driver (V4_BUILD_SPEC.md §14.4 R1), run inside the v2d_foundation_pose container as fp_track.py is (path_b
mounted read-only on PYTHONPATH): FoundationPose's refiner started from our own track.

Per requested frame, the request's pose (object_to_camera, S3d's settled track) is refined by FoundationPose's
refiner (`iterations` passes, the frame's RGB and its stereo depth masked to the object, eroded and bilateral-
filtered as the tracker does) and both poses are measured as fp_track measures a pose: the rendered silhouette's IoU
with the object's P1b mask, and the median |rendered - stereo depth| inside it. The refined pose is kept only where
its IoU is higher and its residual lower than the start's, and it moved at most `max_step_m` / `max_step_deg` from
the start (the refiner is a local polish, not a re-registration); otherwise the start is kept.
Writes <output_dir>/poses/%06d.json (object_to_camera Transform3d, every requested frame) and refine_meta.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from v2d.common.datatypes import Transform3d
from pathb.drivers.fp_track import Track
from Utils import bilateral_filter_depth, depth2xyzmap, erode_depth  # FoundationPose, on sys.path via the tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fp_refine")


def keep(start: dict, refined: dict, step_m: float, step_deg: float, bounds: dict) -> bool:
    """The driver's rule: better on both measures, and within the step bound."""
    return (refined["iou"] > start["iou"] and refined["residual_m"] < start["residual_m"]
            and step_m <= bounds["max_step_m"] and step_deg <= bounds["max_step_deg"])


class Refine(Track):
    def refine(self, rgb, depth, P: np.ndarray, iterations: int) -> np.ndarray:
        d = torch.as_tensor(depth.depth, device="cuda", dtype=torch.float)
        d = bilateral_filter_depth(erode_depth(d, radius=2, device="cuda"), radius=2, device="cuda")
        xyz = depth2xyzmap(d.cpu().numpy(), self.K)
        with torch.no_grad():
            refined, _ = self.est.refiner.predict(mesh=self.est.mesh, mesh_tensors=self.est.mesh_tensors, rgb=rgb.data,
                                                  depth=d, K=self.K, ob_in_cams=self.centred(P)[None], normal_map=None,
                                                  xyz_map=xyz, glctx=self.est.glctx, mesh_diameter=self.est.diameter,
                                                  iteration=iterations, get_vis=False)
        return refined.data.cpu().numpy()[0] @ self.tf

    def run(self) -> None:
        req, out = self.req, Path(self.a.output_dir)
        (out / "poses").mkdir(parents=True, exist_ok=True)
        rows = []
        for k in req["frames"]:
            P0 = np.asarray(req["poses"][str(k)], dtype=np.float64)
            rgb, depth, mask, stereo = self.load(k)
            iou0, res0 = self.measure(P0, mask, stereo)
            P1 = self.refine(rgb, depth, P0, int(req["iterations"]))
            iou1, res1 = self.measure(P1, mask, stereo)
            step_m = float(np.linalg.norm(P1[:3, 3] - P0[:3, 3]))
            step_deg = float(np.degrees(Rotation.from_matrix(P0[:3, :3].T @ P1[:3, :3]).magnitude()))
            start, refined = {"iou": iou0, "residual_m": res0}, {"iou": iou1, "residual_m": res1}
            kept = keep(start, refined, step_m, step_deg, req["bounds"])
            Transform3d.from_matrix(P1 if kept else P0).save(str(out / f"poses/{k:06d}.json"))
            rows.append({"frame": k, "start": start, "refined": refined, "step_m": step_m, "step_deg": step_deg,
                         "kept": kept})
        (out / "refine_meta.json").write_text(json.dumps({"iterations": req["iterations"], "bounds": req["bounds"],
                                                          "frames": rows}, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("request", "frames_dir", "depth_dir", "masks_dir", "mesh_path", "intrinsics_path", "weights_dir",
                 "output_dir"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args()
    t0 = time.time()
    Refine(a, json.loads(Path(a.request).read_text())).run()
    log.info(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
