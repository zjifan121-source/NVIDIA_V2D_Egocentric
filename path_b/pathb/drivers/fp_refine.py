"""S3e driver (V4_BUILD_SPEC.md §14.4 R1), run inside the v2d_foundation_pose container as fp_track.py is (path_b
mounted read-only on PYTHONPATH): FoundationPose's refiner started from our own track.

Per requested frame, the request's pose (object_to_camera, S3d's settled track) is refined by FoundationPose's
refiner (`iterations` passes, the frame's RGB and its stereo depth masked to the object, eroded and bilateral-
filtered as the tracker does) and both poses are measured as fp_track measures a pose: the rendered silhouette's IoU
with the object's P1b mask, and the median |rendered - stereo depth| inside it. The refined pose is kept only where
its IoU is higher and its residual lower than the start's, and it moved at most `max_step_m` / `max_step_deg` from
the start (the refiner is a local polish, not a re-registration); otherwise the start is kept.
Writes <output_dir>/poses/%06d.json (object_to_camera Transform3d, every requested frame) and refine_meta.json,
whose rows also hold the refined pose (object_to_camera) and FoundationPose's scorer logits of [start, refined]
(one scorer call, so the two are comparable; null if the scorer failed), so S3e can apply another keep rule
without the GPU (Recon round P4). With "checkpoints" in the request (refiner passes, e.g. [5, 10, 20]), the passes
go on from one checkpoint's pose to the next and each row also holds "hypotheses" (pose, IoU, residual, step and
logit per checkpoint) and "start_logit", all scored in one call with the start.
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
    scorer_error = None

    def filtered(self, depth):
        """The frame's masked depth, eroded and bilateral-filtered as the tracker does (GPU tensor)."""
        d = torch.as_tensor(depth.depth, device="cuda", dtype=torch.float)
        return bilateral_filter_depth(erode_depth(d, radius=2, device="cuda"), radius=2, device="cuda")

    def refine(self, rgb, d, P: np.ndarray, iterations: int) -> np.ndarray:
        xyz = depth2xyzmap(d.cpu().numpy(), self.K)
        with torch.no_grad():
            refined, _ = self.est.refiner.predict(mesh=self.est.mesh, mesh_tensors=self.est.mesh_tensors, rgb=rgb.data,
                                                  depth=d, K=self.K, ob_in_cams=self.centred(P)[None], normal_map=None,
                                                  xyz_map=xyz, glctx=self.est.glctx, mesh_diameter=self.est.diameter,
                                                  iteration=iterations, get_vis=False)
        return refined.data.cpu().numpy()[0] @ self.tf

    def logits(self, rgb, d, poses: list) -> list | None:
        """FoundationPose's scorer logits of caller-frame poses, scored together (as S3b's recovery scores its
        candidates); None, logged once, if the scorer fails."""
        try:
            with torch.no_grad():
                out, _ = self.est.scorer.predict(mesh=self.est.mesh, rgb=rgb.data, depth=d, K=self.K,
                                                 ob_in_cams=np.stack([self.centred(P) for P in poses]), normal_map=None,
                                                 mesh_tensors=self.est.mesh_tensors, glctx=self.est.glctx,
                                                 mesh_diameter=self.est.diameter, get_vis=False)
            return [float(x) for x in out]
        except Exception as err:  # the refined poses are kept without scores
            if self.scorer_error is None:
                self.scorer_error = repr(err)[:300]
                log.warning(f"scorer failed, logits left null: {self.scorer_error}")
            return None

    def run(self) -> None:
        req, out = self.req, Path(self.a.output_dir)
        (out / "poses").mkdir(parents=True, exist_ok=True)
        n_iter = int(req["iterations"])
        stops = sorted({n_iter, *[int(c) for c in req.get("checkpoints") or []]})  # refiner passes to record poses at
        rows = []
        for k in req["frames"]:
            P0 = np.asarray(req["poses"][str(k)], dtype=np.float64)
            rgb, depth, mask, stereo = self.load(k)
            d = self.filtered(depth)
            iou0, res0 = self.measure(P0, mask, stereo)
            poses, P, done = {}, P0, 0
            for it in stops:  # the refiner's passes go on from the last checkpoint's pose
                P = self.refine(rgb, d, P, it - done)
                poses[it], done = P, it
            hyps = []
            for it in stops:
                iou, res = self.measure(poses[it], mask, stereo)
                hyps.append({"iterations": it, "pose": poses[it].tolist(), "iou": iou, "residual_m": res,
                             "step_m": float(np.linalg.norm(poses[it][:3, 3] - P0[:3, 3])),
                             "step_deg": float(np.degrees(Rotation.from_matrix(P0[:3, :3].T @ poses[it][:3, :3]).magnitude()))})
            logits = self.logits(rgb, d, [P0] + [poses[it] for it in stops])  # one call: comparable logits
            first = hyps[stops.index(n_iter)]
            P1 = poses[n_iter]
            start, refined = {"iou": iou0, "residual_m": res0}, {"iou": first["iou"], "residual_m": first["residual_m"]}
            kept = keep(start, refined, first["step_m"], first["step_deg"], req["bounds"])
            Transform3d.from_matrix(P1 if kept else P0).save(str(out / f"poses/{k:06d}.json"))
            row = {"frame": k, "start": start, "refined": refined, "step_m": first["step_m"], "step_deg": first["step_deg"],
                   "kept": kept, "refined_pose": P1.tolist(),
                   "logits": None if logits is None else [logits[0], logits[1 + stops.index(n_iter)]]}
            if len(stops) > 1:
                row["start_logit"] = None if logits is None else logits[0]
                row["hypotheses"] = [h | {"logit": None if logits is None else logits[1 + i]} for i, h in enumerate(hyps)]
            rows.append(row)
        (out / "refine_meta.json").write_text(json.dumps({"iterations": req["iterations"], "bounds": req["bounds"],
                                                          "scorer_error": self.scorer_error, "frames": rows},
                                                         indent=1) + "\n")


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
