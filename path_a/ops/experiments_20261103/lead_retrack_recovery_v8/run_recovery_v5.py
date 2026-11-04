"""lead_retrack_recovery_v5 runner (adds plain and recovery_cont_t05) (recovery with depth guard, recovery_cont, particles): provided-pose start, then FP tracking with one frozen variant.

Variant "recovery": track_one + mask-IoU gated re-registration (iou < thresh -> register from the current
frame mask). Variant "particles": FP refiner/scorer particle filter with mask-IoU weighting.
Derived from the scored11-fulltrack-20261103 runner; only FIXED/VARIANTS and the track loop differ.

Fixed-metric asset tracking extracted from the eight-sequence experiment.
No production modules are changed. --validate-only never imports torch or FP.
"""
from pathlib import Path
import argparse
import datetime
import gc
import hashlib
import importlib
import inspect
import json
import random
import shlex
import sys
import time

import numpy as np


FIXED = dict(seed=0, register_iteration=10, track_iteration=5,
             mask_depth=True, scale_search=False, EKF=False, referee=False)
VARIANTS = {
    "recovery": dict(recovery=True, n_particles=1, recovery_iou_thresh=0.3, recovery_iteration=10),
    "particles": dict(recovery=False, n_particles=20, particle_noise_t=0.005, particle_noise_r=0.02,
                      particle_iteration=3, mask_iou_weight=1.0),
    # continuity-aware recovery: among registration candidates within cont_score_margin of the best score take the
    # one closest in rotation to the tracked pose; decline if the centre jumps more than cont_max_trans_mm.
    "recovery_cont": dict(recovery=True, n_particles=1, recovery_iou_thresh=0.3, recovery_iteration=10,
                          cont_score_margin=1.0, cont_max_trans_mm=60.0, min_register_depth_px=200),
}
VARIANTS["recovery"]["min_register_depth_px"] = 200  # never re-register from a frame without measured object depth
VARIANTS["plain"] = dict(recovery=False, n_particles=1)  # track_one only, no re-registration
VARIANTS["recovery_cont_t05"] = dict(VARIANTS["recovery_cont"], recovery_iou_thresh=0.5)  # earlier re-registration


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def as_np(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def schedule(anchor, count):
    if not isinstance(anchor, int) or not 0 <= anchor < count:
        raise ValueError("anchor must name one existing frame")
    return {"forward": list(range(anchor + 1, count)),
            "backward": list(range(anchor - 1, -1, -1))}


def checked_path(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Non-portable payload path: {relative}")
    out = (root / path).resolve()
    if not out.is_relative_to(root):
        raise ValueError(f"Payload escape: {relative}")
    return out


def validate_transform(T):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    if not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Non-finite or non-homogeneous pose")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or abs(np.linalg.det(R)-1) > 1e-3:
        raise ValueError("Public pose is not a unit-scale proper SE(3)")
    return T


def public_to_internal(public, original_to_centered):
    return validate_transform(public) @ np.linalg.inv(original_to_centered)


def assert_pair(internal, original_to_centered, public):
    err = float(np.max(np.abs(np.asarray(internal).reshape(4, 4) @ original_to_centered - public)))
    if err >= 1e-5:
        raise ValueError(f"Centered mesh/public pose mismatch: {err}")
    return err


def validate_runtime(actual, expected):
    # Paths can be remounted; all model, precision-engine and hardware pins cannot.
    for key in ("backend", "model_registry", "model_version", "tensorrt_version",
                "gpu_compute_capability", "models"):
        if json.loads(json.dumps(actual.get(key))) != json.loads(json.dumps(expected.get(key))):
            raise ValueError(f"FP runtime/model mismatch: {key}")


def decode_frame(payload, row, K):
    import cv2
    rgb = cv2.imread(str(checked_path(payload, row["rgb"])), cv2.IMREAD_COLOR)
    raw = cv2.imread(str(checked_path(payload, row["depth"])), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(checked_path(payload, row["mask"])), cv2.IMREAD_UNCHANGED)
    shape = (int(K["height"]), int(K["width"]))
    if rgb is None or raw is None or mask is None:
        raise ValueError(f"Unreadable input frame {row['frame_id']}")
    if rgb.shape != (*shape, 3) or raw.shape != shape or mask.shape != shape or raw.dtype != np.uint16:
        raise ValueError(f"Unmatched grid/depth encoding at frame {row['frame_id']}")
    if not np.isin(np.unique(mask), [0, 1, 255]).all():
        raise ValueError("Expected frozen binary object mask")
    mask = mask > 0
    valid = (raw > 0) & (raw < 65535)
    depth = np.zeros(shape, np.float32)
    depth[valid] = 65535. / raw[valid].astype(np.float32) - 1.
    depth[~mask] = 0
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError("Invalid decoded metric depth")
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth, mask


def load_package(manifest_path, protocol_path, payload_override=None):
    manifest_path, protocol_path = Path(manifest_path), Path(protocol_path)
    m, cfg = json.loads(manifest_path.read_text()), json.loads(protocol_path.read_text())
    if cfg.get("runner_sha256") != sha(__file__):
        raise ValueError("Runner is not pinned by frozen protocol")
    count = cfg["frame_count"]
    if not isinstance(count, int) or count <= 0 or m["frame_count"] != count:
        raise ValueError("Manifest/protocol frame count mismatch")
    if not isinstance(cfg["episode"], int) or not cfg["object"]:
        raise ValueError("Expected episode identifier and object name")
    if m.get("metric_mesh_reviewed") is not True or m.get("anchor_selection_reviewed") is not True:
        raise ValueError("Metric mesh and input-only anchor selection require explicit review")
    if any(m.get(k) != cfg.get(k) for k in ["episode","object","frame_count","anchor_frame"]):
        raise ValueError("Manifest/protocol episode/object/frame/anchor mismatch")
    for key, value in FIXED.items():
        if cfg.get(key) != value:
            raise ValueError(f"Non-frozen setting {key}")
    if cfg.get("variant") not in VARIANTS:
        raise ValueError("Unknown tracking variant")
    for key, value in VARIANTS[cfg["variant"]].items():
        if cfg.get(key) != value:
            raise ValueError(f"Non-frozen variant setting {key}")
    if cfg["GPU_window_seconds"] != 900:
        raise ValueError("Expected one total 15-minute window")
    if cfg.get("input_manifest_sha256") != sha(manifest_path):
        raise ValueError("Manifest is not pinned by frozen protocol")
    provided = cfg.get('provided_initial_pose')
    if provided is not None:
        path = checked_path(protocol_path.parent, provided['file'])
        if sha(path) != provided['sha256'] or provided['mesh_sha256'] != m['files'][m['mesh_file']] or provided['source_frame'] != m['anchor_frame']:
            raise ValueError('Provided mesh/pose pair mismatch')
        validate_transform(np.load(path, allow_pickle=False))
    payload = Path(payload_override).resolve() if payload_override else checked_path(manifest_path.parent, m["payload_root"])
    required = {m["mesh_file"], m["K_file"]}
    rows = m["frames"]
    if [x["frame_id"] for x in rows] != list(range(count)):
        raise ValueError("Frames must be complete, unique and ordered complete frame IDs")
    for row in rows:
        required.update(row[k] for k in ("rgb", "depth", "mask"))
    for key in ("rgb", "depth", "mask"):
        if len({row[key] for row in rows}) != count:
            raise ValueError(f"Frame aliases are not a complete explicit {key} time map")
    if not required.issubset(m["files"]):
        raise ValueError("Required file has no frozen digest")
    for name, digest in m["files"].items():
        if sha(checked_path(payload, name)) != digest:
            raise ValueError(f"SHA mismatch: {name}")
    K = json.loads(checked_path(payload, m["K_file"]).read_text())
    if any(k not in K for k in ("fx", "fy", "cx", "cy", "width", "height")):
        raise ValueError("K must use native CameraIntrinsics JSON schema")
    if not isinstance(K["width"], int) or not isinstance(K["height"], int) or min(K["width"], K["height"]) <= 0 or min(K["fx"], K["fy"]) <= 0:
        raise ValueError("Unexpected frozen FP image grid/intrinsics")
    if not np.isfinite([K[k] for k in ("fx", "fy", "cx", "cy")]).all():
        raise ValueError("Nonfinite K")
    directions = schedule(m["anchor_frame"], count)
    perframe = []
    for row in rows:
        _, d, mask = decode_frame(payload, row, K)
        perframe.append(dict(frame=row["frame_id"], object_pixels=int(mask.sum()), valid_object_depth_pixels=int((d > 0).sum())))
    if perframe[m["anchor_frame"]]["valid_object_depth_pixels"] == 0:
        raise ValueError("Anchor has no usable measured object depth")
    # Explicit OBJ references are local and hashed; loading must not find an unpinned texture.
    mesh_path = checked_path(payload, m["mesh_file"])
    if mesh_path.suffix.lower() != ".obj":
        raise ValueError("Frozen provided mesh must be OBJ")
    for line in mesh_path.read_text().splitlines():
        if line.startswith("mtllib "):
            ref = str((Path(m["mesh_file"]).parent / line.split(maxsplit=1)[1]).as_posix())
            if ref not in m["files"]:
                raise ValueError(f"Unpinned material dependency: {ref}")
            for material_line in checked_path(payload, ref).read_text().splitlines():
                tokens = shlex.split(material_line, comments=True)
                if tokens and (tokens[0].lower().startswith("map_") or tokens[0].lower() in ("bump", "disp", "decal")):
                    texture = (Path(ref).parent / tokens[-1]).as_posix()
                    if texture not in m["files"]:
                        raise ValueError(f"Unpinned texture dependency: {texture}")
    from contact import load_contact
    if cfg.get('contact_constraint') is not None and cfg.get('contact_runner_sha256') != sha(Path(__file__).with_name('contact.py')):
        raise ValueError('Contact implementation must be pinned in protocol')
    load_contact(m, cfg, payload, checked_path, validate_transform)
    return m, cfg, payload, K, directions, perframe


def validate_review(review, protocol_digest, manifest_digest, mesh_digest, anchor_path, anchor, provided=False):
    required = ['proceed_to_fixed_tracking', 'geometry_and_pose_numerically_valid',
                'matching_mesh_pose_provenance_checked', 'quality_veto_disabled_by_user']
    if any(review.get(k) is not True for k in required):
        raise ValueError('Missing execution review; quality is evaluated only after full tracking')
    expected = dict(protocol_sha256=protocol_digest, input_manifest_sha256=manifest_digest,
                    mesh_sha256=mesh_digest, anchor_public_sha256=sha(anchor_path), anchor_frame=anchor)
    for key, value in expected.items():
        if review.get(key) != value:
            raise ValueError(f"Anchor review provenance mismatch: {key}")
    if not review.get("reviewer") or not review.get("reviewed_evidence"):
        raise ValueError("Anchor review must identify reviewer and reviewed output artifacts")


def execute(args, package):
    m, cfg, payload, K_dict, directions, perframe = package
    count = m["frame_count"]
    from contact import load_contact, apply_to_tracker
    contact = load_contact(m, cfg, payload, checked_path, validate_transform)
    provided_spec = cfg.get('provided_initial_pose')
    provided = np.load(checked_path(Path(args.protocol).parent, provided_spec['file']), allow_pickle=False) if provided_spec else None
    has_initial = contact is not None or provided is not None
    O = Path(args.output).resolve()
    O.mkdir(parents=True, exist_ok=True)
    stage = O / args.stage
    stage.mkdir(exist_ok=False)  # An attempted stage can never be resumed/overwritten.
    record = dict(state="PRECHECK", stage=args.stage, register_calls=0, track_calls=0, rows=[],
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  protocol_sha256=sha(args.protocol), input_manifest_sha256=sha(args.manifest),
                  mesh_sha256=m["files"][m["mesh_file"]], geometry_units="metres_baked_once",
                  pose_convention="T_camera_from_original_provided_mesh; column vectors",
                  no_GT=True, initialization="single_registration_rank0")
    record['initialization'] = 'provided_matching_pose_then_track' if has_initial else 'single_registration_rank0'
    record['variant'] = cfg['variant']
    record['constraint'] = 'observed_contact_point' if contact else None
    if contact:
        record['contact_implementation_sha256'] = sha(Path(__file__).with_name('contact.py'))
    tick = time.monotonic()
    def save():
        record["seconds"] = time.monotonic() - tick
        write_json(stage / "RECEIPT.json", record)
    save()
    try:
        window_path = O / "WINDOW.json"
        if args.stage == "anchor":
            if window_path.exists():
                raise RuntimeError("Existing GPU window; no new attempt permitted")
            write_json(window_path, dict(start_unix=time.time(), deadline_unix=time.time()+900,
                                        protocol_sha256=sha(args.protocol), input_manifest_sha256=sha(args.manifest)))
        window = json.loads(window_path.read_text())
        if window["protocol_sha256"] != sha(args.protocol) or window["input_manifest_sha256"] != sha(args.manifest):
            raise ValueError("Window is bound to different inputs")
        def deadline():
            if time.time() >= window["deadline_unix"]:
                raise TimeoutError("Shared 15-minute window reached; no retry")
        deadline()
        anchor_path = O / "anchor" / "anchor_public.npy"
        if args.stage == "track":
            ar = json.loads((O / "anchor" / "RECEIPT.json").read_text())
            if ar["state"] != "COMPLETE_AWAITING_HUMAN_REVIEW" or ar["register_calls"] != (0 if has_initial else 1):
                raise ValueError("No complete frozen initialization")
            validate_review(json.loads((O / "ANCHOR_REVIEW.json").read_text()), sha(args.protocol),
                            sha(args.manifest), record["mesh_sha256"], anchor_path, m["anchor_frame"], provided=has_initial)
        import cv2
        import torch
        from v2d.common.datatypes import CameraIntrinsics, DepthImage, Mask, Image, Transform3d
        from v2d.mesh.lib.mesh import Mesh
        from v2d.foundation_pose.lib import trt_engine
        def no_build(*a, **k):
            raise RuntimeError("Matching existing TensorRT engines required; no engine build allowed")
        trt_engine.build_engine = no_build
        import v2d.foundation_pose.lib.foundation_pose_tracker as ft
        estimator_module = importlib.import_module(ft.FoundationPose.__module__)
        loaded = {"v2d_foundation_pose/lib/foundation_pose_tracker.py": Path(inspect.getfile(ft)),
                  "v2d_foundation_pose/lib/FoundationPose/estimater.py": Path(inspect.getfile(estimator_module))}
        for name, file in loaded.items():
            if sha(file) != cfg["expected_source_sha256"][name]:
                raise ValueError(f"Runtime source mismatch: {name}")
        record["loaded_source_sha256"] = {key: sha(file) for key, file in loaded.items()}

        torch.set_grad_enabled(False)
        def seed():
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        def new_tracker():
            deadline(); seed()
            tracker = ft.FoundationPoseTracker(Mesh.load(str(checked_path(payload, m["mesh_file"]))),
                                               args.weights_dir, backend="nvidia_tensorrt")
            validate_runtime(tracker.runtime_info, cfg["expected_backend_runtime"])
            return tracker
        K = CameraIntrinsics.from_dict(K_dict)
        def inputs(t):
            rgb, depth, mask = decode_frame(payload, m["frames"][t], K_dict)
            return Image(data=rgb), DepthImage(depth=depth), Mask(mask=mask)
        def constrain(tracker, pose, frame):
            raw = validate_transform(pose.to_matrix())
            if contact is None:
                return pose, raw
            corrected_pose, raw, info = apply_to_tracker(tracker, pose, contact, frame, Transform3d.from_matrix)
            record.setdefault('contact_rows', []).append(info)
            return corrected_pose, raw
        (stage / "poses").mkdir()
        record["state"] = "RUNNING"; save()
        if args.stage == "anchor":
            tracker = new_tracker()
            image, depth, mask = inputs(m["anchor_frame"])
            deadline()
            if has_initial:
                tracker.reset_to_pose(Transform3d.from_matrix(contact['initial_pose_file'] if contact else provided))
                pose = tracker.track_one(image, depth, K, iteration=5)
                record['track_calls'] = 1
            else:
                record['register_calls'] = 1
                pose = tracker.register(image, depth, mask, K, iteration=10)
            pose, raw = constrain(tracker, pose, m['anchor_frame'])
            np.save(stage / 'raw_anchor_public.npy', raw)
            save()
            T = validate_transform(pose.to_matrix())
            pose.save(str(stage / "poses" / f"{m['anchor_frame']:06d}.json"))
            np.save(stage / "anchor_public.npy", T)
            est = tracker._est
            A = as_np(est.get_tf_to_centered_mesh())
            if not has_initial:
                candidates, scores = as_np(est.poses), as_np(est.scores)
                assert_pair(candidates[0], A, T)
                np.savez_compressed(stage / "registration_candidates.npz", internal_centered_poses=candidates,
                                    scores=scores, original_to_centered=A, selected_public=T)
            else:
                np.savez_compressed(stage / 'provided_anchor.npz', initial=contact['initial_pose_file'] if contact else provided,
                                    before_contact=raw, selected_public=T)
            tracker.reset_to_pose(pose)
            error = assert_pair(as_np(est.pose_last), A, T)
            # Rendering uses centered geometry and its matching internal pose exactly once.
            internal = public_to_internal(T, A)
            _, rendered, _ = ft.nvdiffrast_render(K.to_matrix(), K.height, K.width,
                torch.as_tensor(internal[None], dtype=torch.float32, device="cuda"),
                glctx=tracker._glctx, mesh_tensors=est.mesh_tensors, get_normal=False)
            rz = as_np(rendered)[0]
            physical = rz > 0
            observed = mask.mask
            overlay = image.data.copy()
            overlay[physical] = (0.55*overlay[physical]+0.45*np.array([0,255,255])).astype(np.uint8)
            contours, _ = cv2.findContours(observed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
            cv2.imwrite(str(stage / "anchor_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            np.save(stage / "anchor_rendered_depth.npy", rz)
            support = (depth.depth > 0); overlap = support & physical
            residual = np.abs(rz[overlap]-depth.depth[overlap])
            record.update(state="COMPLETE_AWAITING_HUMAN_REVIEW", selected_rank=None if has_initial else 0,
                anchor_frame=m["anchor_frame"], anchor_public_sha256=sha(stage / "anchor_public.npy"),
                coordinate_pair_error=error, runtime_info=tracker.runtime_info,
                anchor_input_metrics=dict(mask_iou=float((physical&observed).sum()/max(1,(physical|observed).sum())),
                    valid_object_depth_pixels=int(support.sum()), rendered_support_pixels=int(overlap.sum()),
                    depth_residual_median_m=float(np.median(residual)) if residual.size else None),
                gate="Numerical validity and mesh/pose provenance only; all quality outcomes continue to full tracking")
        else:
            T = validate_transform(np.load(anchor_path))
            anchor = Transform3d.from_matrix(T)
            anchor.save(str(stage / "poses" / f"{m['anchor_frame']:06d}.json"))
            allposes = {m["anchor_frame"]: T}
            rawposes = {m['anchor_frame']: np.load(O / 'anchor' / 'raw_anchor_public.npy')}
            record["directions"] = directions
            for direction, ids in directions.items():
                tracker = new_tracker()  # Separate estimator, particles and history for each direction.
                tracker.reset_to_pose(anchor)
                A = as_np(tracker._est.get_tf_to_centered_mesh())
                pair_error = assert_pair(as_np(tracker._est.pose_last), A, T)
                record.setdefault("direction_initialization", []).append(dict(direction=direction,
                    fresh_tracker=True, same_anchor_public_sha256=sha(anchor_path), coordinate_pair_error=pair_error))
                for t in ids:
                    deadline(); ftick=time.monotonic(); image, depth, mask = inputs(t)
                    recovered = False
                    if cfg["variant"] == "plain":
                        pose = tracker.track_one(image, depth, K, iteration=5)
                    elif cfg["variant"] in ("recovery", "recovery_cont", "recovery_cont_t05"):
                        pose = tracker.track_one(image, depth, K, iteration=5)
                        iou_tracked = float(tracker._mask_iou(mask, K, pose))
                        valid_px = int((depth.depth > 0).sum())
                        if iou_tracked < cfg["recovery_iou_thresh"] and valid_px >= cfg["min_register_depth_px"]:
                            tracked_pub = validate_transform(pose.to_matrix())
                            tracker.register(image, depth, mask, K, iteration=cfg["recovery_iteration"])
                            record["register_calls"] += 1
                            if cfg["variant"] == "recovery":
                                pose = Transform3d.from_matrix(validate_transform(as_np(tracker._est.pose_last) @ as_np(tracker._est.get_tf_to_centered_mesh())))
                                recovered = True; record.setdefault("recovery_frames", []).append(t)
                            else:
                                est = tracker._est; cands = as_np(est.poses); scs = as_np(est.scores).reshape(-1)
                                Ac = as_np(est.get_tf_to_centered_mesh()); pubs = np.einsum("nij,jk->nik", cands, Ac)
                                order = list(np.argsort(-scs)); keep = [i for i in order if scs[i] >= scs[order[0]] - cfg["cont_score_margin"]]
                                def rotd(x, y):
                                    return float(np.degrees(np.arccos(np.clip((np.trace(x[:3, :3].T @ y[:3, :3]) - 1) / 2, -1, 1))))
                                best = min(keep, key=lambda i: rotd(pubs[i], tracked_pub))
                                chosen = validate_transform(pubs[best]); jump = float(np.linalg.norm(chosen[:3, 3] - tracked_pub[:3, 3]) * 1000)
                                if jump <= cfg["cont_max_trans_mm"]:
                                    pose = Transform3d.from_matrix(chosen); tracker.reset_to_pose(pose); recovered = True
                                    record.setdefault("recovery_frames", []).append(t)
                                    record.setdefault("recovery_choices", []).append(dict(frame=t, rank=int(order.index(best)), n_keep=len(keep),
                                        rot_to_tracked_deg=rotd(chosen, tracked_pub), jump_mm=jump, iou_tracked=iou_tracked, best_score=float(scs[order[0]]), chosen_score=float(scs[best])))
                                else:
                                    tracker.reset_to_pose(Transform3d.from_matrix(tracked_pub))
                                    record.setdefault("recovery_declined", []).append(dict(frame=t, jump_mm=jump, iou_tracked=iou_tracked))
                        elif iou_tracked < cfg["recovery_iou_thresh"]:
                            record.setdefault("recovery_skipped_no_depth", []).append(t)
                    else:
                        pose = tracker.track_one_particles(image, depth, K, mask=mask, n_particles=cfg["n_particles"],
                            process_noise_t=cfg["particle_noise_t"], process_noise_r=cfg["particle_noise_r"],
                            iteration=cfg["particle_iteration"], mask_iou_weight=cfg["mask_iou_weight"])
                    mask_iou = float(tracker._mask_iou(mask, K, pose))
                    pose, raw = constrain(tracker, pose, t)
                    rawposes[t] = raw
                    Tt = validate_transform(pose.to_matrix())
                    if cfg["variant"] in ("plain", "recovery", "recovery_cont", "recovery_cont_t05"):
                        pair_err = assert_pair(as_np(tracker._est.pose_last), A, Tt)
                    else:
                        # Particle filter returns the weighted mean while pose_last keeps the best particle.
                        pair_err = float(np.max(np.abs(as_np(tracker._est.pose_last).reshape(4, 4) @ A - Tt)))
                    pose.save(str(stage / "poses" / f"{t:06d}.json")); allposes[t] = Tt
                    record["track_calls"] += 1
                    record["rows"].append(dict(frame=t, direction=direction, seconds=time.monotonic()-ftick,
                                               recovered=recovered, mask_iou=mask_iou, pose_last_pair_error=pair_err))
                    save(); print("FRAME", t, direction, flush=True)
                record["runtime_info"] = tracker.runtime_info
                del tracker; gc.collect(); torch.cuda.empty_cache()
            if sorted(allposes) != list(range(count)) or record["track_calls"] != count-1:
                raise ValueError("Incomplete tracking is never a reusable completed cache")
            np.savez_compressed(stage / "matched_mesh_poses.npz", frame_ids=np.arange(count),
                                public_poses=np.stack([allposes[t] for t in range(count)]),
                                raw_before_contact=np.stack([rawposes[t] for t in range(count)]))
            write_json(stage / "MESH_POSE_PAIR.json", dict(mesh_file=m["mesh_file"],
                mesh_sha256=record["mesh_sha256"], input_manifest_sha256=sha(args.manifest),
                poses_sha256=sha(stage / "matched_mesh_poses.npz"), frame_count=m["frame_count"],
                convention=record["pose_convention"], mesh_scale_already_baked=True,
                additional_scale=1.0, exported_center_conversion_count=0,
                note="Native FP outputs already use original provided-mesh coordinates; do not recenter or attach old poses."))
            record["state"] = "COMPLETE_FULL_TRACK"
        deadline()
        record.update(exit_code=0, outputs_sha256={str(p.relative_to(stage)):sha(p) for p in stage.rglob("*")
                      if p.is_file() and p.name!="RECEIPT.json"})
        save(); return 0
    except BaseException as e:
        record.update(state="TIMEOUT" if isinstance(e, TimeoutError) else "FAILED", exit_code=1, error=repr(e))
        save(); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--stage", choices=["anchor", "track"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights-dir", help="Native FoundationPose weights directory (required for GPU stages)")
    args=parser.parse_args()
    package=load_package(args.manifest, args.protocol, args.payload)
    m,cfg,payload,K,directions,perframe=package
    if args.validate_only:
        result=dict(state="CPU_INPUT_CONTRACT_PASS", frame_count=m["frame_count"], anchor_frame=m["anchor_frame"],
                    planned_track_updates=m["frame_count"]-1, directions=directions, perframe=perframe,
                    input_manifest_sha256=sha(args.manifest), protocol_sha256=sha(args.protocol),
                    torch_imported="torch" in sys.modules, GPU_run=False,
                    validation_scope="Input contract only; no tracking executed")
        if args.validation_output:
            if args.validation_output.exists(): raise FileExistsError(args.validation_output)
            write_json(args.validation_output,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("directions","perframe")},indent=2));return 0
    if args.stage is None or args.output is None or args.weights_dir is None:
        parser.error("--stage, --output and --weights-dir are required for GPU execution")
    return execute(args, package)


if __name__ == "__main__":
    sys.exit(main())
