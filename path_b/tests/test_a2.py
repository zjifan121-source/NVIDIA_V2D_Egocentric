"""Phase A2 (spec §11) on a synthetic episode: a box rests, is moved, rests again, seen by a sweeping
camera; stereo depth = the box rendered (pathb/scan.py), masks = where it renders."""
import json

import cv2
import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation

from pathb import runner
from pathb import scan as S
from pathb.geometry import write_depth, write_transform
from pathb.stages import s3a2_fuse_scan as A21

K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
SHAPE = (240, 320)  # depth_a_half; masks at twice that, as P1b's full resolution
N, REST_B, REST_A = 40, (0, 14), (26, 39)


def box():
    m = o3d.geometry.TriangleMesh.create_box(0.12, 0.08, 0.05)
    m.translate([-0.06, -0.04, 0.0])
    m.compute_vertex_normals()
    return m


def pose(t):
    """Object pose: at rest, then 25 cm over and turned 40 degrees, at rest again."""
    s = np.clip((t - REST_B[1]) / (REST_A[0] - REST_B[1]), 0, 1)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0, 0, 0.7 * s]).as_matrix()
    T[:3, 3] = [0.25 * s, 0.05 * s, 0.0]
    return T


def camera(t):
    yaw = -0.8 + 1.6 * t / (N - 1)
    c = np.array([0.55 * np.cos(yaw), 0.55 * np.sin(yaw), 0.5]) + [0.12, 0.0, 0.0]
    z = np.array([0.12, 0.0, 0.0]) - c
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.stack([x, np.cross(z, x), z], axis=1), c
    return T


@pytest.fixture
def episode(tmp_path):
    root = tmp_path / "episode_000999"
    half, cams = root / "p2_depth/depth_a_half", root / "p4_export/camera_to_world"
    for d in (half, cams, root / "p1b_masks/obj_0", root / "p1b_masks/hand_L", root / "p1b_masks/hand_R",
              root / "s4a_phases/box", root / "s3c_fuse/box"):
        d.mkdir(parents=True)
    (half / "intrinsics.json").write_text(json.dumps({"K": K.tolist(), "size": [SHAPE[1], SHAPE[0]]}))
    for t in range(N):
        mesh = o3d.geometry.TriangleMesh(box()).transform(pose(t))
        depth = S.rendered_depth(mesh, K, camera(t), SHAPE)
        write_depth(half / f"{t:06d}.png", depth)
        write_transform(cams / f"{t:06d}.json", camera(t))
        big = cv2.resize((depth > 0).astype(np.uint8) * 255, (SHAPE[1] * 2, SHAPE[0] * 2),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(root / f"p1b_masks/obj_0/{t:06d}.png"), big)
        hand = np.zeros_like(big)
        if t == 5:  # a hand over the object on one rest frame
            hand = cv2.dilate(big, np.ones((9, 9), np.uint8))
        cv2.imwrite(str(root / f"p1b_masks/hand_L/{t:06d}.png"), hand)
        cv2.imwrite(str(root / f"p1b_masks/hand_R/{t:06d}.png"), np.zeros_like(big))
    (root / "s4a_phases/box/phases.json").write_text(json.dumps({"rest_before": list(REST_B), "rest_after": list(REST_A)}))
    off = np.eye(4)
    off[:3, 3] = [0.006, 0.0, 0.0]  # S3c's rest-after anchor 6 mm off: ICP takes it out
    (root / "s3c_fuse/box/anchors.json").write_text(json.dumps({
        "rest_before": {"window": list(REST_B), "pose": pose(0).tolist()},
        "rest_after": {"window": list(REST_A), "pose": (off @ pose(N - 1)).tolist()}}))
    e = {"episode": 999, "split": "evaluation", "num_frames": N, "objects": [{"name": "box"}]}
    return runner.Context(e, root, {})


def test_fuse_scan_on_a_moved_box(episode):
    result = A21.run(episode)
    report = json.loads((episode.root / "s3a2_fuse_scan/box/fuse_report.json").read_text())
    rows = {r["frame"]: r for r in report["frames"]}
    assert rows[5]["used"] is False and rows[5]["why"] == "hand touches the object"
    assert len(report["fused"]) == (15 - 1) + 14  # both windows, less the touched frame
    move = report["rest_after_move"]
    assert move["icp"]["fitness"] > 0.5 and abs(move["icp"]["correction_mm"] - 6) < 2
    assert report["held_out_rms"]["rms_m"] < 0.003
    partial = o3d.io.read_triangle_mesh(str(episode.root / "s3a2_fuse_scan/box/partial_scan.ply"))
    truth = o3d.geometry.TriangleMesh(box()).transform(pose(0))  # the frame-0 placement
    pts = np.asarray(S.mesh_points(partial, 5000).points)
    assert S.one_sided_cm(pts, np.asarray(truth.vertices), np.asarray(truth.triangles)) < 0.2
    status = {m.name: m.status for m in result.metrics}
    assert status["box: frames fused"] == "WARN"  # 28 < 30
    assert status["box: surface vs held-out stereo depth (RMS)"] == "PASS"


def test_scan_registered_onto_the_partial(episode):
    A21.run(episode)
    partial = o3d.io.read_triangle_mesh(str(episode.root / "s3a2_fuse_scan/box/partial_scan.ply"))
    elsewhere = np.eye(4)
    elsewhere[:3, :3] = Rotation.from_rotvec([0.4, -1.1, 2.0]).as_matrix()
    elsewhere[:3, 3] = [1.0, -0.5, 0.3]  # the scan's own frame, unrelated to the world
    scan = o3d.geometry.TriangleMesh(box()).transform(elsewhere)
    best = A21.chamfer_to_scan(partial, np.asarray(scan.vertices), np.asarray(scan.triangles))
    assert best["cm"] < 0.2
