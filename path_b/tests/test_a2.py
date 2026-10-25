"""Phase A2 (spec §11) on a synthetic episode: a box rests, is moved, rests again, seen by a sweeping
camera; stereo depth = the box rendered (pathb/scan.py), masks = where it renders."""
import json
from pathlib import Path

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
        if REST_B[1] < t < REST_A[0]:  # the hand on the object while it carries it
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


def test_touch_windows():
    touched = [False] * 3 + [True] * 4 + [False] * 13
    assert A21.touch_windows(touched) == {"rest_before": [0, 2], "rest_after": [7 + A21.SETTLE, 19]}
    assert A21.touch_windows([False] * 8) == {"rest_before": [0, 7], "rest_after": None}  # never touched
    assert A21.touch_windows([True] + [False] * 3) == {"rest_before": None, "rest_after": None}  # no frame left after settling
    moving = [False] + [True] * 3 + [False] + [True] + [False] * 14  # ep43: scattered flags before the touch
    assert A21.touch_windows([False] * 7 + [True] * 5 + [False] * 8, moving)["rest_before"] == [0, 6]
    carried = [False] * 5 + [True] * 8 + [False] * 7  # ep13: a carry the hand masks missed
    assert A21.touch_windows([False] * 20, carried) == {"rest_before": [0, 4], "rest_after": [13 + A21.SETTLE, 19]}


def test_area_bar_is_relative_for_thin_objects():
    from pathb.observe import area_bar
    assert area_bar([0.05] * 50 + [0.0] * 10) == 0.01  # the spec's 1 % for an object that shows large
    assert area_bar([0.009] * 40 + [0.002] * 10) == pytest.approx(0.0045)  # ep18's sweeper: half its own share
    assert area_bar([0.0] * 5) == 0.01  # never seen


def test_fuse_scan_on_a_moved_box(episode):
    result = A21.run(episode)
    report = json.loads((episode.root / "s3a2_fuse_scan/box/fuse_report.json").read_text())
    # the hand touches 15-25 and the box moves 15-25; the motion test (+-5 frames) ends rest-before a few
    # frames before the touch and starts rest-after a few frames after the settle
    w = report["windows"]
    assert 9 <= w["rest_before"][1] <= REST_B[1] and w["rest_before"][0] == 0
    assert REST_A[0] + A21.SETTLE <= w["rest_after"][0] <= REST_A[0] + A21.SETTLE + 5 and w["rest_after"][1] == N - 1
    parts = report["partials"]
    assert len(parts["rest_before"]["fused"]) == w["rest_before"][1] + 1
    assert len(parts["rest_after"]["fused"]) == N - w["rest_after"][0]
    assert parts["rest_before"]["held_out_rms"]["rms_m"] < 0.003
    for key, file, t in (("rest_before", "partial_scan.ply", 0), ("rest_after", "partial_scan_after.ply", N - 1)):
        partial = o3d.io.read_triangle_mesh(str(episode.root / f"s3a2_fuse_scan/box/{file}"))
        truth = o3d.geometry.TriangleMesh(box()).transform(pose(t))  # each window's own placement
        pts = np.asarray(S.mesh_points(partial, 5000).points)
        assert S.one_sided_cm(pts, np.asarray(truth.vertices), np.asarray(truth.triangles)) < 0.2, key
    status = {m.name: m.status for m in result.metrics}
    assert status["box: frames fused"] == "WARN"  # ~11 + ~5 < 30
    assert status["box: rest_before surface vs held-out stereo depth (RMS)"] == "PASS"


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


def test_canonical_revolution_keeps_a_handle_as_seen():
    """ep41's case: a half-seen pot with a 17 cm handle out of one side, both seen by the camera. The
    body is revolved (radius 0.1 m), the handle stays as observed: not a disc of the handle's reach."""
    pot = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=0.12, resolution=120, split=24)
    handle = o3d.geometry.TriangleMesh.create_box(0.17, 0.03, 0.02).translate([0.1, -0.015, 0.09])
    both = pot.translate([0.0, 0.0, 0.06]) + handle
    both.translate([0.3, -0.1, 0.0])
    both.compute_vertex_normals()
    mesh, rep = C.build("white_pot", [seen(both, [0.8, -0.1, 0.5])])
    ext = np.asarray(rep["extents_cm"])
    assert rep["watertight"] and abs(rep["symmetry"]["wall_radius_m"] - 0.1) < 0.01
    assert 35 < ext[0] < 39 and 19 < ext[1] < 23 and 11 < ext[2] < 14, ext  # 20 cm body + the 17 cm handle; a disc: 54 x 54


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


def test_voxel_solid_fills_a_sparse_closed_shell_and_not_an_open_plate():
    """The closing grows until a sparse shell encloses an interior; an open plate keeps the first one."""
    rng = np.random.default_rng(4)
    pts = rng.uniform([-0.06, -0.04, 0.0], [0.06, 0.04, 0.05], (20000, 3))
    axis = rng.integers(0, 3, len(pts))
    lo, hi = np.array([-0.06, -0.04, 0.0]), np.array([0.06, 0.04, 0.05])
    pts[np.arange(len(pts)), axis] = np.where(rng.random(len(pts)) < 0.5, lo[axis], hi[axis])  # on a face
    keep = rng.random(len(pts)) < 0.06  # ~1 200 points: 6-8 mm apart, gaps a 2-step closing leaves
    box = C.voxel_solid(pts[keep])
    vol = abs(trimesh.Trimesh(np.asarray(box.vertices), np.asarray(box.triangles), process=False).volume)
    assert 0.6 * 0.12 * 0.08 * 0.05 < vol < 1.4 * 0.12 * 0.08 * 0.05  # was 0.13 x before the fill (marching cubes loses ~half a voxel a side)
    plate = rng.uniform([-0.1, -0.08, 0.0], [0.1, 0.08, 0.0], (4000, 3))  # a dust pan's flat sheet
    sheet = C.voxel_solid(plate)
    assert np.ptp(np.asarray(sheet.vertices)[:, 2]) < 0.03  # not grown into a slab


# ---------------------------------------------------------------- A2.3 register + re-expressed track
from pathb.stages import s3a2_register as A23  # noqa: E402


SIM = Path.home() / "V2D/venvs/sim/bin/python"  # a venv with coacd (FlashCHORD's 1.0.11) on the Mac


def test_register_places_the_canonical_mesh_and_writes_its_files(episode, monkeypatch):
    if SIM.exists():
        monkeypatch.setenv("COACD_PY", str(SIM))
    A21.run(episode)
    C.main("box", episode.root.parent)
    result = A23.run(episode)
    reg = json.loads((episode.root / "s3a2_register/box/registration.json").read_text())
    assert reg["start"] == "canonical.json" and reg["partial_explained"] > 0.8 and reg["window"] == "rest_before"
    canon = trimesh.load(episode.root.parent / "canonical/box/mesh.obj", force="mesh", process=False)
    T0 = np.asarray(reg["canonical_to_world"])
    world = canon.vertices @ T0[:3, :3].T + T0[:3, 3]
    placed = o3d.geometry.TriangleMesh(box()).transform(pose(0))  # the frame-0 placement
    assert S.one_sided_cm(world, np.asarray(placed.vertices), np.asarray(placed.triangles)) < 0.5
    files = episode.root / "s3a_mesh/box"
    assert (files / "mesh.glb").is_file() and (files / "mesh.obj").is_file() and (files / "sha256.txt").is_file()
    mass = json.loads((files / "mass.json").read_text())  # one clip never sees the +y side: not the box's volume
    assert mass["volume_m3"] > 0 and mass["density_kg_m3"] > 0 and "volume_from" in mass
    status = {m.name: m.status for m in result.metrics}
    assert status["box: share of the partial the canonical mesh explains"] == "PASS"
    if SIM.exists():
        assert status["box: convex pieces (CoACD 0.05)"] == "PASS" and (files / "coacd/pieces.json").is_file()
