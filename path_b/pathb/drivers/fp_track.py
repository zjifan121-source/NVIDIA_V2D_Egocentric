"""S3b driver, run inside the v2d_foundation_pose container (path_b mounted read-only on PYTHONPATH):
FoundationPose tracking with the V3_BUILD_SPEC.md S3b recovery rule.

Registration on the request's reference frame, then tracking forward and backward as the baseline
does (track_one, 5 iterations, depth masked to the object as the pipeline runs it). Frames where the
hands cover more than 40 % of the object mask use track_one_particles (20 particles, mask IoU weight
1.0). When a tracked pose's mask IoU drops below 0.3, the candidates are the last good pose x the
object's symmetry group plus register()'s top-5 hypotheses, each refined by 2 iterations (one
batched refiner call), and the best by
    FoundationPose scorer logit + 1.0 log(mask IoU) - lambda * (rotation to the last good pose, rad)
    - mu * (median |rendered - stereo depth| in the mask, m)
is kept and the frame marked recovered. Renders use FoundationPose's centred mesh with the matching
centred pose. Writes <output_dir>/poses/%06d.json (object_to_camera Transform3d) and track_meta.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Transform3d
from v2d.common.datatypes import Image as V2dImage
from v2d.foundation_pose.lib.backends import DEFAULT_BACKEND
from v2d.foundation_pose.lib.foundation_pose_tracker import FoundationPoseTracker
from v2d.mesh.lib.mesh import Mesh
from Utils import bilateral_filter_depth, depth2xyzmap, erode_depth, nvdiffrast_render  # FoundationPose, on sys.path via the tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("fp_track")


class Track:
    def __init__(self, a, req):
        self.a, self.req = a, req
        self.mesh = Mesh.load(a.mesh_path, force_mesh=Path(a.mesh_path).suffix.lower() == ".glb")
        self.tracker = FoundationPoseTracker(self.mesh, a.weights_dir, backend=DEFAULT_BACKEND)
        self.est = self.tracker._est
        self.intr = CameraIntrinsics.load(a.intrinsics_path)
        self.K = self.intr.to_matrix()
        self.tf = self.est.get_tf_to_centered_mesh().cpu().numpy()
        self.G = np.array(req["symmetry"], dtype=np.float64)

    def load(self, k: int):
        bgr = cv2.imread(str(Path(self.a.frames_dir) / f"{k:06d}.jpg"))
        stereo = DepthImage.load(str(Path(self.a.depth_dir) / f"{k:06d}.png")).depth.astype(np.float32)
        stereo[~np.isfinite(stereo)] = 0.0  # pathb writes "no depth" as pixel 0, which decodes to infinity
        mask = Mask.load(str(Path(self.a.masks_dir) / f"{k:06d}.png"))
        masked = stereo.copy()
        masked[~mask.mask.astype(bool)] = 0.0  # the pipeline tracks with mask_depth=True
        return V2dImage(data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), DepthImage(depth=masked), mask, stereo

    def centred(self, P: np.ndarray) -> np.ndarray:
        return P @ np.linalg.inv(self.tf)

    def render(self, P: np.ndarray) -> np.ndarray:
        """Rendered depth of the mesh at caller-frame pose P."""
        pose = torch.as_tensor(self.centred(P)[None], device="cuda", dtype=torch.float)
        with torch.no_grad():
            _, depth, _ = nvdiffrast_render(self.K, self.intr.height, self.intr.width, pose, glctx=self.tracker._glctx,
                                            mesh_tensors=self.est.mesh_tensors, get_normal=False)
        return depth[0].cpu().numpy()

    def measure(self, P: np.ndarray, mask: Mask, stereo: np.ndarray) -> tuple[float, float]:
        """(mask IoU, median |rendered - stereo| inside the mask in m)."""
        r = self.render(P)
        sil, m = r > 0.001, mask.mask.astype(bool)
        union = (sil | m).sum()
        sel = sil & m & (stereo > 0)
        res = float(np.median(np.abs(r[sel] - stereo[sel]))) if sel.sum() >= 20 else 1.0
        return float((sil & m).sum() / union) if union else 0.0, res

    def recover(self, rgb, depth, mask, stereo, last_good: np.ndarray):
        """The S3b candidate set, refined and scored; (pose, details)."""
        p = self.req["recovery"]
        self.tracker.register(rgb, depth, mask, self.intr, iteration=10)
        top = [P.cpu().numpy() @ self.tf for P in self.est.poses[:5]]
        cands = [last_good @ S for S in self.G] + top
        d = torch.as_tensor(depth.depth, device="cuda", dtype=torch.float)
        d = bilateral_filter_depth(erode_depth(d, radius=2, device="cuda"), radius=2, device="cuda")
        xyz = depth2xyzmap(d.cpu().numpy(), self.K)
        with torch.no_grad():
            refined, _ = self.est.refiner.predict(mesh=self.est.mesh, mesh_tensors=self.est.mesh_tensors, rgb=rgb.data,
                                                  depth=d, K=self.K, ob_in_cams=np.stack([self.centred(P) for P in cands]),
                                                  normal_map=None, xyz_map=xyz, glctx=self.est.glctx,
                                                  mesh_diameter=self.est.diameter, iteration=2, get_vis=False)
            logits, _ = self.est.scorer.predict(mesh=self.est.mesh, rgb=rgb.data, depth=d, K=self.K,
                                                ob_in_cams=refined.data.cpu().numpy(), normal_map=None,
                                                mesh_tensors=self.est.mesh_tensors, glctx=self.est.glctx,
                                                mesh_diameter=self.est.diameter, get_vis=False)
        rows = []
        for i, Pc in enumerate(refined.data.cpu().numpy()):
            P = Pc @ self.tf
            iou, res = self.measure(P, mask, stereo)
            turn = float(Rotation.from_matrix(last_good[:3, :3].T @ P[:3, :3]).magnitude())
            score = float(logits[i]) + p["iou_weight"] * np.log(max(iou, 1e-3)) - p["lambda"] * turn - p["mu"] * res
            rows.append({"source": "symmetry" if i < len(self.G) else "register", "logit": float(logits[i]), "iou": iou,
                         "turn_rad": turn, "residual_m": res, "score": score, "pose": P})
        best = max(rows, key=lambda r: r["score"])
        return best["pose"], {k: v for k, v in best.items() if k != "pose"} | {"candidates": len(rows)}

    def run(self) -> None:
        req, out = self.req, Path(self.a.output_dir)
        (out / "poses").mkdir(parents=True, exist_ok=True)
        ref, frames, overlap = req["reference_frame"], req["frames"], req["hand_overlap"]
        meta = {}
        rgb, depth, mask, stereo = self.load(ref)
        P0 = self.tracker.register(rgb, depth, mask, self.intr, iteration=10).to_matrix()
        iou, res = self.measure(P0, mask, stereo)
        meta[ref] = {"frame": ref, "iou": iou, "recovered": False, "particles": False, "depth_residual_m": res,
                     "method": "register"}
        Transform3d.from_matrix(P0).save(str(out / f"poses/{ref:06d}.json"))
        p = req["recovery"]
        for order in ([f for f in frames if f > ref], [f for f in reversed(frames) if f < ref]):
            self.tracker.reset_to_pose(Transform3d.from_matrix(P0))
            last_good = P0
            for k in order:
                rgb, depth, mask, stereo = self.load(k)
                particles = overlap.get(str(k), 0.0) > p["particle_overlap"]
                if particles:
                    P = self.tracker.track_one_particles(rgb, depth, self.intr, mask=mask, n_particles=p["particles"],
                                                         mask_iou_weight=1.0).to_matrix()
                else:
                    P = self.tracker.track_one(rgb, depth, self.intr, iteration=5).to_matrix()
                seen = bool(mask.mask.any())
                iou, res = self.measure(P, mask, stereo) if seen else (None, None)
                row = {"frame": k, "iou": iou, "recovered": False, "particles": particles, "depth_residual_m": res,
                       "method": ("particles" if particles else "track") + ("" if seen else " (no mask)")}
                if seen and iou < p["iou_thresh"]:  # without a mask there is nothing to re-register on
                    P, info = self.recover(rgb, depth, mask, stereo, last_good)
                    self.tracker.reset_to_pose(Transform3d.from_matrix(P))
                    iou, res = info["iou"], info["residual_m"]
                    row.update(iou=iou, depth_residual_m=res, recovered=True, method="recovered", recovery=info)
                if seen and not row["recovered"] and iou >= p["trusted_iou"]:
                    last_good = P
                meta[k] = row
                Transform3d.from_matrix(P).save(str(out / f"poses/{k:06d}.json"))
        rows = [meta[k] for k in sorted(meta)]
        (out / "track_meta.json").write_text(json.dumps({"reference_frame": ref, "recovery": p, "frames": rows},
                                                        indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("request", "frames_dir", "depth_dir", "masks_dir", "mesh_path", "intrinsics_path", "weights_dir",
                 "output_dir"):
        ap.add_argument(f"--{name}", required=True)
    a = ap.parse_args()
    t0 = time.time()
    Track(a, json.loads(Path(a.request).read_text())).run()
    log.info(f"done in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
