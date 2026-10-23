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


# ---------------------------------------------------------------- A2.2 canonical mesh
from pathb import canonical as C  # noqa: E402
import trimesh  # noqa: E402


def seen(mesh, eye, keep=None):
    """The triangles a camera at eye sees (facing it), less the table side."""
    dense = mesh.subdivide_midpoint(3)
    dense.compute_triangle_normals()
    c = np.asarray(dense.vertices)[np.asarray(dense.triangles)].mean(axis=1)
    facing = ((np.asarray(eye) - c) * np.asarray(dense.triangle_normals)).sum(1) > 0
    part = o3d.geometry.TriangleMesh(dense)
    part.remove_triangles_by_mask(~facing | (c[:, 2] < 0.0005))
    part.remove_unreferenced_vertices()
    part.compute_vertex_normals()
    return part


def distance_cm(mesh, origin, truth):
    pts = np.asarray(S.mesh_points(mesh, 20000).points) + origin
    d = trimesh.Trimesh(np.asarray(truth.vertices), np.asarray(truth.triangles)).nearest.on_surface(pts)[1]
    return np.percentile(d, [50, 95, 100]) * 100


def dust_pan():
    base = o3d.geometry.TriangleMesh.create_box(0.20, 0.16, 0.01).translate([-0.10, -0.08, 0])
    wall = o3d.geometry.TriangleMesh.create_box(0.20, 0.01, 0.06).translate([-0.10, 0.07, 0.01])
    handle = o3d.geometry.TriangleMesh.create_box(0.03, 0.12, 0.02).translate([-0.015, 0.08, 0.03])
    m = (base + wall + handle).translate([0.2, 0.1, 0])
    m.compute_vertex_normals()
    return m


def test_canonical_revolution_completes_a_half_seen_pot():
    pot = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=0.12, resolution=120, split=24)
    pot.translate([0.3, -0.1, 0.06])
    pot.compute_vertex_normals()
    mesh, rep = C.build("white_pot", [seen(pot, [0.8, -0.1, 0.5])])
    assert rep["watertight"] and rep["extents_ok"] and rep["faces"] <= C.MAX_FACES
    assert np.degrees(np.arccos(rep["symmetry"]["axis"][2])) < 1.0
    p50, p95, worst = distance_cm(mesh, -rep["world_to_canonical"], pot)
    assert p95 < 0.3 and worst < 1.0
    assert rep["coverage_before_completion"] < 0.6 and "coverage < 0.6" in rep["flags"]


def test_canonical_mirror_finds_the_dust_pan_plane():
    pan = dust_pan()
    mesh, rep = C.build("mini_dust_pan", [seen(pan, [0.45, -0.25, 0.5])])
    sym = rep["symmetry"]
    assert sym["applied"] and abs(abs(sym["normal"][0]) - 1) < 1e-3 and abs(abs(sym["d"]) - 0.2) < 0.002
    assert rep["watertight"] and rep["extents_ok"]
    p50, p95, worst = distance_cm(mesh, -rep["world_to_canonical"], pan)
    assert p95 < 0.6 and worst < 1.5


def test_canonical_registers_clips_from_two_placements():
    pan = dust_pan()
    first = seen(pan, [0.45, -0.25, 0.5])
    moved = np.eye(4)
    moved[:3, :3] = Rotation.from_rotvec([0, 0, 1.2]).as_matrix()
    moved[:3, 3] = [-0.3, 0.4, 0.0]  # another episode: elsewhere on its table, turned
    second = seen(o3d.geometry.TriangleMesh(pan).transform(moved), moved[:3, :3] @ [-0.2, 0.5, 0.5] + moved[:3, 3])
    ref, moves = C.register([first, second])
    assert ref == 0 and all(m["kept"] for m in moves)
    T = moves[1]["T"] @ moved  # second's placement brought onto the first's: the identity
    assert np.abs(T[:3, 3]).max() < 0.004 and Rotation.from_matrix(T[:3, :3]).magnitude() < np.radians(2)


# ---------------------------------------------------------------- A2.3 register + re-expressed track
from pathb.stages import s3a2_register as A23  # noqa: E402


def test_register_carries_the_track_to_the_canonical_mesh(episode):
    A21.run(episode)
    np.savez(episode.root / "s3c_fuse/box/object_to_world.npz",
             object_to_world=np.stack([pose(t) for t in range(N)]))  # phase A's track (its mesh frame = the box's)
    C.main("box", episode.root.parent)
    result = A23.run(episode)
    reg = json.loads((episode.root / "s3a2_register/box/registration.json").read_text())
    assert reg["start"] == "canonical.json" and reg["partial_explained"] > 0.8
    canon = trimesh.load(episode.root.parent / "canonical/box/mesh.obj", force="mesh", process=False)
    track = np.load(episode.root / "s3a2_register/box/object_to_world.npz")["object_to_world"]
    truth = box()
    for t in (0, N - 1):  # at rest before, and after the move
        world = canon.vertices @ track[t, :3, :3].T + track[t, :3, 3]
        placed = o3d.geometry.TriangleMesh(truth).transform(pose(t))
        assert S.one_sided_cm(world, np.asarray(placed.vertices), np.asarray(placed.triangles)) < 0.5
    assert {m.name: m.status for m in result.metrics}["box: share of the partial the canonical mesh explains"] == "PASS"


# ---------------------------------------------------------------- A2.4 rigid motion from point tracks
from pathb import symmetry  # noqa: E402
from pathb.stages import s3b2_track_points as A24  # noqa: E402


def test_point_tracks_give_the_rigid_motion():
    rng = np.random.default_rng(3)
    n, q = 30, 5
    P = rng.normal(0, 0.05, (200, 3)) + [0.3, 0.1, 0.05]
    truth = np.stack([pose(t + 10) for t in range(n)])  # moving between rest windows
    move = truth @ np.linalg.inv(truth[q])  # world motion since frame q
    coords = np.einsum("tij,nj->tni", move[:, :3, :3], P) + move[:, None, :3, 3]
    coords += rng.normal(0, 0.001, coords.shape)
    bad = rng.random(200) < 0.2  # a fifth of the tracks drift off the object
    coords[:, bad] += rng.normal(0, 0.05, (n, bad.sum(), 3))
    visible = rng.random((n, 200)) > 0.3
    visible[q] = True
    M, rows = A24.motions(coords, visible, q, symmetry.group("mini_dust_pan"))
    def on_object(T):  # the object's own points (the motion's origin is 0.3 m away: a lever arm)
        return np.einsum("tij,nj->tni", T[:, :3, :3], P) + T[:, None, :3, 3]
    err = np.linalg.norm(on_object(M) - on_object(move), axis=-1).mean(axis=1)
    assert np.isfinite(M[:, 0, 0]).all() and err.max() < 0.002
    assert max(r["rmse_m"] for r in rows) < 0.003 and min(r["iou"] for r in rows if r["frame"] != q) > 0.6


# ---------------------------------------------------------------- A2.5 decision rule
from pathb import a2 as A25  # noqa: E402


def test_a2_decision_rule():
    row = lambda auc, cdo: {"AUC": auc, "mean CD-O cm": cdo}  # noqa: E731
    # better CD-O on both episodes, AUC split: A2 has the higher mean rank
    t = {40: {"phase_a": row(0.54, 0.6), "a2_mesh_fp": row(0.50, 0.3)},
         26: {"phase_a": row(0.30, 1.0), "a2_mesh_fp": row(0.35, 0.4)}}
    assert A25.decide(t)["choice"] == "a2_mesh_fp"
    # equal ranks: the deterministic (A2) variant
    t = {40: {"phase_a": row(0.54, 0.3), "a2_mesh_fp": row(0.50, 0.4)},
         26: {"phase_a": row(0.30, 1.0), "a2_mesh_fp": row(0.35, 0.4)}}
    d = A25.decide(t)
    assert d["tied"] == ["a2_mesh_fp", "phase_a"] and d["choice"] == "a2_mesh_fp"
    # A2 worse on both AUC and CD-O everywhere: phase A stays
    t = {40: {"phase_a": row(0.54, 0.3), "a2_mesh_fp": row(0.50, 0.4)},
         26: {"phase_a": row(0.36, 0.4), "a2_mesh_fp": row(0.35, 0.5)}}
    d = A25.decide(t)
    assert d["a2_lost_on_both"] and d["choice"] == "phase_a"
