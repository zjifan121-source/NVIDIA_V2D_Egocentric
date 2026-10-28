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


def test_neighbour_pixels_leave_the_scan():
    """V4 §13.1 A2: the cup's mask also covers rack wires (in the rack's mask) and cup pixels 1 cm from
    the rack; both go, the cup pixels 5 cm away stay."""
    K = np.array([[500.0, 0, 50], [0, 500.0, 50], [0, 0, 1]])
    depth = np.full((100, 100), 0.5)
    cup = np.zeros((100, 100), bool)
    cup[20:80, 20:60] = True
    rack = np.zeros((100, 100), bool)
    rack[20:80, 58:90] = True  # overlaps the cup's mask in columns 58-59
    out, dropped = A21.without_neighbours(depth, cup, rack, K)
    # at 0.5 m and f = 500 px a pixel is 1 mm: within 2 cm of column 58 = columns 39-59 (38 on the edge)
    assert (out[:, 58:60] == 0).all() and (out[20:80, 40:58] == 0).all()
    assert (out[20:80, 20:38] == 0.5).all() and dropped in (60 * 21, 60 * 22)
    assert (out[~cup] == 0).all()


def test_nested_masks_leave_the_overlap_to_the_smaller_object(tmp_path):
    """ep0: the pot's mask holds the lid's; the lid keeps its pixels, the pot loses them."""
    pot, lid = np.zeros((40, 40), np.uint8), np.zeros((40, 40), np.uint8)
    pot[5:35, 5:35], lid[10:30, 10:30] = 255, 255
    for name, m in (("obj_0", pot), ("obj_1", lid)):
        (tmp_path / "p1b_masks" / name).mkdir(parents=True)
        cv2.imwrite(str(tmp_path / "p1b_masks" / name / "000000.png"), m)

    class Ctx:
        root, objects = tmp_path, ["white_pot", "white_pot_lid"]
    for_lid = A21.neighbour_masks(Ctx, 1, 0, (40, 40))
    for_pot = A21.neighbour_masks(Ctx, 0, 0, (40, 40))
    assert not for_lid[10:30, 10:30].any() and for_lid[5:10, 5:35].all()
    assert for_pot[10:30, 10:30].all() and not for_pot[5:10, 5:35].any()


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


def pot_with_handle(turn_deg=0.0):
    pot = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=0.12, resolution=120, split=24)
    handle = o3d.geometry.TriangleMesh.create_box(0.17, 0.03, 0.02).translate([0.1, -0.015, 0.09])
    both = pot.translate([0.0, 0.0, 0.06]) + handle
    both.rotate(Rotation.from_rotvec([0, 0, np.radians(turn_deg)]).as_matrix(), center=[0, 0, 0])
    both.compute_vertex_normals()
    return both


def test_canonical_rejects_an_axis_far_from_the_table_normal():
    """V4 §13.1 A1: a cup lying 30 or 60 degrees over (axis fitted beyond 20) or 45 (no side wall about
    the vertical from this view) is not revolved about the vertical (the old cap); one leaning 10
    degrees is, about the table normal."""
    for lean, applied in ((30, False), (45, False), (60, False), (10, True)):
        cup = o3d.geometry.TriangleMesh.create_cylinder(radius=0.05, height=0.15, resolution=120, split=30)
        cup.rotate(Rotation.from_rotvec([np.radians(lean), 0, 0]).as_matrix(), center=[0, 0, 0])
        cup.translate([0.3, -0.1, 0.12])
        cup.compute_vertex_normals()
        mesh, rep = C.build("blue_cup", [seen(cup, [0.3, -0.6, 0.6])])
        sym = rep["symmetry"]
        assert sym["applied"] is applied, (lean, sym)
        if applied:
            assert np.degrees(np.arccos(sym["axis"][2])) < 1.0
        else:
            assert "rejected" in sym and (sym["fitted_tilt_deg"] is None or sym["fitted_tilt_deg"] > C.AXIS_REJECT_DEG)


def test_canonical_turns_clips_onto_the_handle_and_drops_a_clip_120_degrees_off():
    """V4 §13.1 A1 (ep27/ep42's pot): clips registered with the handle 0, 0, +12 and +120 degrees about
    the axis; the +12 one is turned onto the others, the +120 one is left out."""
    turns = [0.0, 0.0, 12.0, 120.0]
    clouds = [S.mesh_points(pot_with_handle(t), 20000) for t in turns]
    moves = [{"T": np.eye(4), "kept": True} for _ in turns]
    fit = {"axis": np.array([0.0, 0.0, 1.0]), "point": np.zeros(3), "wall_radius_m": 0.1}
    out = C.align_features(clouds, moves, 0, fit)
    assert out["clips_with_feature"] == 4
    assert [m["kept"] for m in moves] == [True, True, True, False]
    assert abs(moves[3]["feature_off_deg"]) > 100
    after = [C.feature_azimuth(np.asarray(c.points), fit) for c, m in zip(clouds, moves) if m["kept"]]
    assert max(abs(C.wrap(a - after[0])) for a in after) < 1.0
    two = [S.mesh_points(pot_with_handle(t), 20000) for t in (0.0, 120.0)]  # no majority: neither goes
    moves = [{"T": np.eye(4), "kept": True} for _ in two]
    assert C.align_features(two, moves, 0, fit)["consensus_azimuth_deg"] is None and all(m["kept"] for m in moves)


def test_canonical_drops_a_clip_away_from_the_rest_of_the_pool():
    """V4 §13.1 A1: three dust-pan clips agree, a fourth registered 1.5 cm above them is left out (a
    slide along the pan's base would not show: its points stay on the plane)."""
    pan = dust_pan()
    eyes = ([0.45, -0.25, 0.5], [0.40, -0.30, 0.55], [0.50, -0.20, 0.45], [0.45, -0.25, 0.5])
    clouds = [S.mesh_points(seen(pan, e), 20000) for e in eyes]
    clouds[3].translate([0.0, 0.0, 0.015])
    moves = [{"T": np.eye(4), "kept": True} for _ in eyes]
    C.pool_distances("mini_dust_pan", clouds, moves, 0)
    assert [m["kept"] for m in moves] == [True, True, True, False], [m.get("pool_distance_mm") for m in moves]
    assert max(m["pool_distance_mm"] for m in moves[:3]) < C.POOL_DIST * 1000


def test_canonical_leaves_out_a_partial_failing_the_held_out_check(tmp_path):
    """V4 §13.1 A1: a table patch fused as the pot (held-out RMS 900 mm, the largest surface) stays out
    of the pool; the pot clip makes the mesh."""
    slab = o3d.geometry.TriangleMesh.create_box(0.30, 0.40, 0.03).translate([0.0, 0.0, -0.03])
    for n, (mesh, rms) in enumerate(((seen(pot_with_handle().translate([0.3, -0.1, 0.0]), [0.8, -0.1, 0.5]), 0.003),
                                     (slab, 0.9)), start=1):
        d = tmp_path / f"episode_{n:06d}/s3a2_fuse_scan/white_pot"
        d.mkdir(parents=True)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(d / "partial_scan.ply"), mesh)
        (d / "fuse_report.json").write_text(json.dumps({"partials": {"rest_before": {
            "file": "partial_scan.ply", "fused": list(range(40)), "held_out_rms": {"trimmed_rms_m": rms}}}}))
    dest = C.main("white_pot", tmp_path)
    rep = json.loads((dest / "canonical.json").read_text())
    assert rep["clips"] == ["episode_000001/s3a2_fuse_scan/white_pot/partial_scan.ply"]
    assert rep["clips_left_out"][0]["reason"] == "held-out depth RMS 900 mm > 10"
    assert max(rep["extents_cm"]) < 40 and rep["explained_min_kept"] > 0.8


def test_voxel_solid_fills_a_sparse_closed_shell_and_not_an_open_plate():
    """The closing grows until a sparse shell encloses an interior; an open plate keeps the first one."""
    rng = np.random.default_rng(4)
    pts = rng.uniform([-0.06, -0.04, 0.0], [0.06, 0.04, 0.05], (20000, 3))
    axis = rng.integers(0, 3, len(pts))
    lo, hi = np.array([-0.06, -0.04, 0.0]), np.array([0.06, 0.04, 0.05])
    pts[np.arange(len(pts)), axis] = np.where(rng.random(len(pts)) < 0.5, lo[axis], hi[axis])  # on a face
    keep = rng.random(len(pts)) < 0.06  # ~1 200 points: 6-8 mm apart, gaps a 2-step closing leaves
    box, _ = C.voxel_solid(pts[keep])
    vol = abs(trimesh.Trimesh(np.asarray(box.vertices), np.asarray(box.triangles), process=False).volume)
    assert 0.6 * 0.12 * 0.08 * 0.05 < vol < 1.4 * 0.12 * 0.08 * 0.05  # was 0.13 x before the fill (marching cubes loses ~half a voxel a side)
    plate = rng.uniform([-0.1, -0.08, 0.0], [0.1, 0.08, 0.0], (4000, 3))  # a dust pan's flat sheet
    sheet, _ = C.voxel_solid(plate)
    assert np.ptp(np.asarray(sheet.vertices)[:, 2]) < 0.03  # not grown into a slab


def test_voxel_solid_of_a_sheet_on_the_fused_grid_is_one_piece_no_ray_passes():
    """A thin sheet as fused_cloud leaves it (4 mm voxel-downsampled, 1 mm noise) plus a few stray points:
    one closed piece that every ray through the sheet hits (closing 1 on the 2 mm grid left ep40's dust pan
    a net that 27 % of its mask's rays passed through)."""
    import open3d as o3d
    rng = np.random.default_rng(5)
    dense = rng.uniform([-0.1, -0.08, 0.0], [0.1, 0.08, 0.0], (60000, 3))
    dense[:, 2] = 0.3 * dense[:, 0] ** 2 + rng.normal(0, 0.001, len(dense))  # a shallow scoop
    pts = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense)).voxel_down_sample(0.004).points)
    stray = rng.uniform([-0.1, -0.08, 0.03], [0.1, 0.08, 0.05], (5, 3))
    mesh, how = C.voxel_solid(np.concatenate([pts, stray]))
    assert C.pieces(mesh) == 1 and C.watertight(mesh)
    assert how["steps"][-1]["see_through"] <= C.SEE_THROUGH_BAR
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    xy = rng.uniform([-0.09, -0.07], [0.09, 0.07], (2000, 2))
    rays = np.c_[xy, np.full(len(xy), 0.2), np.tile([0, 0, -1.0], (len(xy), 1))].astype(np.float32)
    assert np.mean(~np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())) <= 0.01  # a pinhole at most


# ---------------------------------------------------------------- A2.3 register + re-expressed track
from pathb.stages import s3a2_register as A23  # noqa: E402


SIM = Path.home() / "V2D/venvs/sim/bin/python"  # a venv with coacd (FlashCHORD's 1.0.11) on the Mac


def test_b1_margin_small_for_a_symmetric_mesh_large_with_a_handle():
    """V4 §13.2 B1: a plain cylinder registers as well upside down (margin ~0); the handle pins the pot."""
    from pathb.ambiguity import margin
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=0.08, height=0.11, resolution=120, split=20)
    cyl.translate([0, 0, 0.055])
    cyl.compute_vertex_normals()
    out = {}
    for label, mesh in (("cylinder", cyl), ("pot", pot_with_handle())):
        part = seen(mesh, [0.6, -0.3, 0.5])
        out[label] = margin(np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(part.vertices),
                            np.asarray(part.triangles), 7, revolution=True)
    assert out["pot"]["margin_m"] is not None and out["pot"]["margin_m"] > 3 * max(out["cylinder"]["margin_m"] or 0, 1e-4), out


def test_mesh_candidate_choice():
    """V4 §13.1 A4: the pooled mesh unless it fails the bar and another explains clearly more."""
    r = lambda s: {"partial_explained": s}  # noqa: E731
    assert A23.choose({"pooled": r(0.62), "own": r(0.93)})[0] == "own"
    assert A23.choose({"pooled": r(0.90), "own": r(0.93)})[0] == "pooled"
    assert A23.choose({"pooled": r(0.86), "own": r(0.91)})[0] == "pooled"  # ep45: passes the bar
    assert A23.choose({"pooled": r(0.5), "own": {"skipped": "x"}})[0] == "pooled"
    assert A23.choose({"pooled": r(0.5), "own": r(0.6)})[0] == "own"
    assert A23.ORDER == ("pooled", "own")  # no team mesh (James, 2026-10-28)
    b = lambda s, mm: {"partial_explained": s, "b1": {"margin_m": mm / 1000}}  # noqa: E731
    assert A23.insure({"pooled": b(0.89, 0.24), "own": b(0.95, 1.62)}, "pooled", "x")[0] == "own"  # ep21's cup
    assert A23.insure({"pooled": b(0.99, 0.005), "own": b(1.0, 0.009)}, "pooled", "x")[0] == "pooled"  # boards
    assert A23.insure({"pooled": b(0.89, 0.24), "own": b(0.70, 1.62)}, "pooled", "x")[0] == "pooled"  # own fails the bar
    assert A23.insure({"pooled": b(0.89, 2.0), "own": b(0.95, 5.0)}, "pooled", "x")[0] == "pooled"  # not flagged


def test_register_places_the_canonical_mesh_and_writes_its_files(episode, monkeypatch):
    if SIM.exists():
        monkeypatch.setenv("COACD_PY", str(SIM))
    A21.run(episode)
    C.main("box", episode.root.parent)
    result = A23.run(episode)
    reg = json.loads((episode.root / "s3a2_register/box/registration.json").read_text())
    assert reg["start"] == "canonical.json" and reg["partial_explained"] > 0.8 and reg["window"] == "rest_before"
    assert reg["mesh"] == "pooled" and "team" not in reg["candidates"]  # own ties it: the more complete
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


from types import SimpleNamespace  # noqa: E402

from pathb.stages import s4b_reference as S4B  # noqa: E402


def test_b3_is_the_settle_of_the_static_segment_from_frame_0(tmp_path):
    def statics(*rows):
        (tmp_path / "s3d_settle/cup").mkdir(parents=True, exist_ok=True)
        (tmp_path / "s3d_settle/cup/settle_report.json").write_text(json.dumps({"object": "cup", "statics": list(rows)}))
    ctx = SimpleNamespace(root=tmp_path)
    assert S4B.frame0_check(ctx, "cup") is None
    statics({"segment": [0, 14], "support": "rack", "settled": False, "moved_cm": 0.3, "turned_deg": 0.4,
             "contacts": 12}, {"segment": [60, 99], "settled": True, "moved_cm": 5.0, "turned_deg": 9.0, "contacts": 3})
    b3 = S4B.frame0_check(ctx, "cup")
    assert b3["dt_mm"] == pytest.approx(3.0) and b3["dtheta_deg"] == 0.4 and not b3["settled"]
    statics({"segment": [0, 14], "settled": False, "why": "settle.py failed"})
    assert S4B.frame0_check(ctx, "cup") is None
    statics({"segment": [5, 14], "settled": True, "moved_cm": 0.1, "turned_deg": 0.1, "contacts": 4})
    assert S4B.frame0_check(ctx, "cup") is None
