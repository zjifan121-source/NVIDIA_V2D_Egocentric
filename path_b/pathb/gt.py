"""The only way into ground truth and the public scans (V3_BUILD_SPEC.md §7, the GT firewall).

Public ground truth (assets/data/track_3/public/data/*.parquet) and the scans (public/mesh/) are
read only here, only for public episodes, and every read is appended to the caller's `read` list,
which the runner writes to report.json: files_read. An evaluation episode raises
GroundTruthFirewall before anything is opened. The team's score_local and run_load_retarget do
the reading; this module decides whether they may. tests/test_gt_firewall.py checks that no other
pathb module calls them for ground truth or scans.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import TRACK3
from .team import retarget_tools, scorer

GT_DATA = TRACK3 / "public/data"
SCANS = TRACK3 / "public/mesh"


class GroundTruthFirewall(PermissionError):
    pass


def allowed(episode: dict) -> None:
    """Raise unless `episode` (its episodes.json entry) is public."""
    if episode.get("split") != "public":
        raise GroundTruthFirewall(f"episode {episode.get('episode')} is {episode.get('split')!r}: "
                                  "ground truth and scans are read for public episodes only")


def parquet(number: int) -> Path:
    return GT_DATA / f"data/chunk-000/episode_{number:06d}.parquet"


def scan(name: str) -> Path | None:
    return next((p for p in (SCANS / f"{name}/{name}_visual.glb", SCANS / f"{name}/{name}.glb") if p.is_file()),
                None)


def ground_truth(episode: dict, read: list) -> np.ndarray:
    """(frames, objects, 7) world_T_object, xyzw, in episodes.json object order."""
    allowed(episode)
    read.append(parquet(episode["episode"]))
    return scorer().ground_truth(episode["episode"])


def score(episode: dict, poses: np.ndarray, meshes: dict[str, Path], read: list) -> dict:
    """The six official metrics of one episode's (frames, objects, 7) poses with one mesh per object."""
    allowed(episode)
    read.append(parquet(episode["episode"]))
    read.extend(p for p in (scan(o["name"]) for o in episode["objects"]) if p)
    n = episode["episode"]
    return scorer().score([episode], {n: poses}, {(n, name): path for name, path in meshes.items()})


def scan_size_ratio(episode: dict, vertices: np.ndarray, name: str, read: list) -> float | None:
    """Reconstructed mesh size / public scan size (run_load_retarget.scan_size_ratio)."""
    allowed(episode)
    path = scan(name)
    if path is None:
        return None
    read.append(path)
    return retarget_tools().scan_size_ratio(vertices, name)


def scan_vertices(episode: dict, name: str, read: list) -> np.ndarray | None:
    import trimesh
    allowed(episode)
    path = scan(name)
    if path is None:
        return None
    read.append(path)
    return np.asarray(trimesh.load(path, force="mesh", process=False).vertices)


def scan_mesh(episode: dict, name: str, read: list) -> tuple[np.ndarray, np.ndarray] | None:
    """The scan as the official metric uses it (score_local.reference_object): (vertices m, faces)."""
    allowed(episode)
    path = scan(name)
    if path is None:
        return None
    read.append(path)
    ref = scorer().reference_object(name)
    return np.asarray(ref["vertices"], dtype=np.float64), np.asarray(ref["faces"], dtype=np.int64)


def scan_in_frame(episode: dict, name: str, vertices: np.ndarray, faces: np.ndarray, read: list
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """The scan at its true size placed in a mesh's own frame, as the official Track 3 metric registers
    that mesh to the scan (surface centroids, then _align_rigid on size-normalised samples): the scan's
    points p go to cc + R^T (p - rc - offset), without the metric's division by the size ratio. Scored
    with the mesh's poses it is the pose-only ceiling: a perfect mesh where ours is (not submittable)."""
    ref = scan_mesh(episode, name, read)
    if ref is None:
        return None
    M = scorer().metric_module("AUC")
    cv, cf = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    rv, rf = ref
    cc, cr = M._surface_moments(cv, cf)
    rc, rr = M._surface_moments(rv, rf)
    seed = 1000 * int(episode["episode"]) + 11
    cp = (M._sample_mesh_surface(cv, cf, 3000, seed) - cc) * (rr / cr)
    rp = M._sample_mesh_surface(rv, rf, 3000, seed) - rc
    R, offset = M._align_rigid(rp, cp)
    return cc + (rv - rc - offset) @ R, rf


def rescale_by_scan(episode: dict, name: str, read: list) -> str:
    """Permission for run_load_retarget's --rescale-by-scan (cleaned_bundle(rescale_by_scan=name))."""
    allowed(episode)
    path = scan(name)
    if path:
        read.append(path)
    return name


def scan_obj(episode: dict, name: str, out: Path, read: list) -> Path:
    """The public scan as a textured OBJ in metres (run_baseline_reconstruction.scan_obj), for oracle
    tests only (S0.1): the FAQ asks for our own meshes on scored episodes."""
    import sys
    from .paths import TOOLS_TRACK3
    allowed(episode)
    path = scan(name)
    if path is None:
        raise FileNotFoundError(f"no public scan for {name}")
    read.append(path)
    if str(TOOLS_TRACK3) not in sys.path:
        sys.path.insert(0, str(TOOLS_TRACK3))
    import run_baseline_reconstruction
    return run_baseline_reconstruction.scan_obj(name, out)


def cdo_cm(episode: dict, name: str, vertices: np.ndarray, faces: np.ndarray, read: list) -> float:
    """The official CD-O (cm) of a mesh against the object's scan, as tools/track3/eval_mesh.py
    computes it (score_local's reference object and the kit's CD-O code)."""
    allowed(episode)
    path = scan(name)
    if path is None:
        raise FileNotFoundError(f"no public scan for {name}")
    read.append(path)
    S = scorer()
    ref = S.reference_object(name)
    return float(S.metric_module("CD-O")._object_chamfer_cm(
        (ref["vertices"], ref["faces"]), (np.asarray(vertices), np.asarray(faces, dtype=np.int64)),
        1000 * int(episode["episode"]) + 7))


def official_frame_errors(episode: dict, poses: np.ndarray, meshes: dict[str, Path], read: list) -> dict:
    """Per frame and object, the errors the official Track 3 pose metrics see: each mesh registered to
    its scan and the poses moved to the scan's anchor (the kit's _register_object_body /
    _convert_object_body), the scene aligned by its frame-0 pose (_align_registered_scene); then the
    anchor's position error (m) and the symmetry-aware rotation error (rad, _symmetry_orientation_errors)
    and the symmetry-aware mean point error (m, _symmetry_point_errors, AUC's ADD). poses (n, k, 7)
    xyzw in episodes.json object order; meshes budgeted as the packer does (score_local.mesh_payload)."""
    allowed(episode)
    S = scorer()
    M = S.metric_module("AUC")
    number = int(episode["episode"])
    names = [o["name"] for o in episode["objects"]]
    truth = ground_truth(episode, read)
    achieved = np.asarray(poses, dtype=np.float64)[:, None].copy()
    reference = truth[:, None].astype(np.float64).copy()
    points, axes, orders = [], [], []
    for slot, name in enumerate(names):
        read.append(scan(name))
        ref = S.reference_object(name)
        _, v, f = S.mesh_payload(Path(meshes[name]))
        anchor_c, rotation = M._register_object_body((v, f), (ref["vertices"], ref["faces"]), ref["anchor"],
                                                     1_000_003 * number + 101 * slot)
        achieved[:, :, slot] = M._convert_object_body(achieved[:, :, slot], anchor_c, rotation)
        reference[:, :, slot, :3] += np.einsum("...ij,j->...i", M._rotation_matrix(reference[:, :, slot, 3:]),
                                               ref["anchor"])
        points.append(ref["points"] - ref["anchor"])
        axes.append(ref["axis"])
        orders.append(int(ref["order"]))
    achieved = M._align_registered_scene(achieved, reference, axes)
    position = np.linalg.norm(achieved[:, 0, :, :3] - reference[:, 0, :, :3], axis=-1)
    rotation = M._symmetry_orientation_errors(achieved, reference, axes, orders)[:, 0]
    add = np.stack([M._symmetry_point_errors(achieved[:, :, i], reference[:, :, i], points[i], axes[i], orders[i])[:, 0]
                    for i in range(len(names))], axis=1)
    return {"names": names, "position_m": position, "rotation_rad": rotation, "add_m": add,
            "truth_displacement_m": np.linalg.norm(reference[:, 0, :, :3] - reference[:1, 0, :, :3], axis=-1),
            "ours_displacement_m": np.linalg.norm(achieved[:, 0, :, :3] - achieved[:1, 0, :, :3], axis=-1)}
