"""pathb/invert.py (V4_BUILD_SPEC.md S3c, task 6) on synthetic clips: a box at rest (0-19), carried 30 cm
and turned 40 degrees by the right hand (20-59), at rest again (60-79), seen only through rendered
stereo depth (1 mm noise) inside its mask; the hand's wrist and two fingertips with noise."""
import numpy as np
import open3d as o3d
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from pathb import invert as I
from pathb import scan as S

N, GRASP = 80, (20, 59)
K = np.array([[300.0, 0, 160], [0, 300, 120], [0, 0, 1]])
SHAPE = (240, 320)
FPS = 20.25


def box():
    m = trimesh.creation.box((0.12, 0.08, 0.06))
    m.apply_translation([0, 0, 0.03])  # canonical frame: footprint centre on the table, z up
    return m


def truth(t):
    s = np.clip((t - GRASP[0]) / (GRASP[1] - GRASP[0]), 0, 1)
    s = 3 * s ** 2 - 2 * s ** 3
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec([0, 0, np.radians(40) * s]).as_matrix()
    T[:3, 3] = [0.3 * s, 0.05 * s, 0.08 * np.sin(np.pi * s)]  # lifted 8 cm on the way
    return T


def camera():
    c = np.array([0.15, -0.45, 0.45])
    z = np.array([0.15, 0.03, 0.03]) - c
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = np.stack([x, np.cross(z, x), z], axis=1), c
    return T


HAND_TO_OBJECT = np.eye(4)
HAND_TO_OBJECT[:3, 3] = [0.0, 0.12, -0.02]  # the object 12 cm in front of the wrist


def clip(slip=None, seed=0):
    """slip: (frame, metres along x the object shifts in the hand from that frame, frames to do it)."""
    rng = np.random.default_rng(seed)
    m = box()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    c2w = np.repeat(camera()[None], N, axis=0)
    G = np.stack([truth(t) for t in range(N)])
    shift = np.zeros(N)
    if slip:
        f, dx, over = slip
        shift[f:] = dx * np.clip((np.arange(N - f) + 1) / over, 0, 1)
    obj = G.copy()  # the object's true pose, with the slip
    for t in range(N):
        D = np.eye(4)
        D[0, 3] = shift[t]
        obj[t] = G[t] @ D if GRASP[0] <= t <= GRASP[1] + 1 else (obj[t - 1] if t > GRASP[1] else G[t])
    obs, sil = [], []
    for t in range(N):
        depth = S.rendered_depth(o3d.geometry.TriangleMesh(mesh).transform(obj[t]), K, c2w[t], SHAPE)
        ys, xs = np.nonzero(depth > 0)
        obs.append((np.stack([xs, ys], 1).astype(float), depth[ys, xs] + rng.normal(0, 0.001, len(ys)),
                    len(ys) / (4 * SHAPE[0] * SHAPE[1])))
        sil.append(I.silhouette_map(depth > 0))
    tips = np.full((N, 2, 5, 3), np.nan)
    wrist = np.tile(np.eye(4), (N, 2, 1, 1))
    valid = np.zeros((N, 2), bool)
    contacts = np.array([[-0.03, -0.04, 0.03], [0.03, -0.04, 0.03]])  # on the box's -y face
    for t in range(15, 66):
        W = G[t] @ np.linalg.inv(HAND_TO_OBJECT)  # the hand moves with the un-slipped plan
        noise = np.eye(4)
        noise[:3, :3] = Rotation.from_rotvec(rng.normal(0, np.radians(0.5), 3)).as_matrix()
        noise[:3, 3] = rng.normal(0, 0.002, 3)
        wrist[t, 1], valid[t, 1] = W @ noise, True
        tips[t, 1, :2] = (contacts @ obj[t][:3, :3].T + obj[t][:3, 3]) + rng.normal(0, 0.001, (2, 3))
        tips[t, 1, 2:] = tips[t, 1, 0] + [0, -0.05, 0.02]
    surface = m.sample(4000, return_index=True)

    def target(a, b):  # the fused surface of a static segment: the posed mesh's sampled surface
        P = obj[a]
        pts, faces = surface
        pc = S.cloud(pts @ P[:3, :3].T + P[:3, 3], m.face_normals[faces] @ P[:3, :3].T)
        return pc
    c = I.Clip(K=K, c2w=c2w, obs=obs, tips=tips, wrist=wrist, valid=valid, vertices=np.asarray(m.vertices),
               faces=np.asarray(m.faces), group=np.eye(4)[None], target=target, silhouette=sil)
    segs = [{"kind": "static", "start": 0, "end": GRASP[0] - 1, "hand": None, "fingertips": []},
            {"kind": "grasp", "start": GRASP[0], "end": GRASP[1], "hand": "R", "fingertips": [["R", 0], ["R", 1]]},
            {"kind": "static", "start": GRASP[1] + 1, "end": N - 1, "hand": None, "fingertips": []}]
    return c, segs, obj, m


def add_mm(P, Q, vertices):
    """Mean vertex distance (mm) per frame between two tracks."""
    a = np.einsum("nij,vj->nvi", P[:, :3, :3], vertices) + P[:, None, :3, 3]
    b = np.einsum("nij,vj->nvi", Q[:, :3, :3], vertices) + Q[:, None, :3, 3]
    return np.linalg.norm(a - b, axis=-1).mean(axis=1) * 1000


def start_pose():
    off = np.eye(4)
    off[:3, :3] = Rotation.from_rotvec([0, 0, np.radians(3)]).as_matrix()
    off[:3, 3] = [0.01, -0.005, 0.0]
    return off @ truth(0)  # the registration 1 cm and 3 degrees off


def test_depth_residual_is_zero_at_the_true_pose():
    c, _, obj, _ = clip()
    r = c.depth_residual(obj[30], 30)
    assert np.median(np.abs(r)) < 0.002 and (np.abs(r) < 0.01).mean() > 0.95
    moved = obj[30].copy()
    moved[:3, 3] += [0, 0, 0.02]
    assert np.median(np.abs(c.depth_residual(moved, 30))) > 0.005


def test_rest_carry_rest_within_3mm():
    c, segs, obj, m = clip()
    P, report = I.solve(c, segs, start_pose(), FPS)
    err = add_mm(P, obj, np.asarray(m.vertices))
    kinds = [r["kind"] for r in report]
    assert kinds == ["static", "grasp", "static"], report
    assert np.median(err) < 3.0 and np.percentile(err, 95) < 6.0, (np.median(err), err.max())
    g = report[1]
    assert not g["failed"] and g["share_within_1cm"] > 0.9
    assert all(r["residual_m"] < I.STATIC_BAR_M for r in report if r["kind"] == "static")


def test_a_small_slip_is_absorbed_and_a_large_one_splits():
    c, segs, obj, m = clip(slip=(35, 0.012, 10))  # 12 mm along the hand over 10 frames
    P, report = I.solve(c, segs, start_pose(), FPS)
    err = add_mm(P, obj, np.asarray(m.vertices))
    assert [r["kind"] for r in report] == ["static", "grasp", "static"]
    assert np.median(err[GRASP[0]:GRASP[1] + 1]) < 4.0
    c, segs, obj, m = clip(slip=(35, 0.09, 1))  # 9 cm at once: more than 10 frames of the 5 mm limit
    P, report = I.solve(c, segs, start_pose(), FPS)
    assert any(r["kind"] == "unstable" and r.get("why") == "slip guard" for r in report), [(r["kind"], r["start"], r["end"]) for r in report]
    err = add_mm(P, obj, np.asarray(m.vertices))
    assert np.median(err[:GRASP[0]]) < 1 and np.median(err[GRASP[1] + 1:]) < 3  # the rests stay right
    assert err.max() < 150  # the bridge interpolates across the jump; bounded, not lost


def test_stage_s3b_then_s3c_on_a_rendered_episode(tmp_path, monkeypatch):
    """s3b_segments then s3c_invert from files: the box's true mesh as the canonical mesh, registered
    3 mm / 2 degrees off at frame 0; the solved track within a few mm of the truth."""
    import json
    import test_a2 as A
    from test_segments import synthetic_episode
    from pathb import runner
    from pathb.stages import s3b_segments, s3c_invert
    monkeypatch.delenv("RETARGET_VENV", raising=False)
    root, e, truth = synthetic_episode(tmp_path)
    ctx = runner.Context(e, root, {})
    s3b_segments.run(ctx)
    m = A.box()
    (root / "s3a_mesh/box").mkdir(parents=True)
    tm = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False)
    tm.export(root / "s3a_mesh/box/mesh.obj")
    off = np.eye(4)
    off[:3, :3] = Rotation.from_rotvec([0, 0, np.radians(2)]).as_matrix()
    off[:3, 3] = [0.003, 0.0, 0.0]
    (root / "s3a2_register/box").mkdir(parents=True)
    (root / "s3a2_register/box/registration.json").write_text(json.dumps(
        {"canonical_to_world": (off @ truth[0]).tolist(), "window": "rest_before"}))
    result = s3c_invert.run(ctx)
    P = np.load(root / "s3c_invert/box/object_to_world.npz")["object_to_world"]
    err = add_mm(P, truth, np.asarray(tm.vertices))
    report = json.loads((root / "s3c_invert/box/invert_report.json").read_text())
    # the p90 is the unstable bridges around this fixture's 2.3 cm/frame carry: 14.x mm on the Mac, 17.9 on the L40S
    assert np.median(err) < 3 and np.percentile(err, 90) < 25, (np.round(err, 1), [(r["kind"], r["start"], r["end"]) for r in report["segments"]])
    status = {x.name: x.status for x in result.metrics}
    assert status["box: static residual (worst segment median)"] == "PASS"
    b = np.load(root / "s3c_invert/box/bundle/result.npz")
    assert np.allclose(b["object_to_world_transform"], P, atol=1e-6) and (root / "s3c_invert/box/bundle/mesh.obj").is_file()


SIM = __import__("pathlib").Path.home() / "V2D/venvs/sim/bin/python"


@pytest.mark.skipif(not SIM.exists(), reason="needs a venv with mujoco (~/V2D/venvs/sim)")
def test_stage_s3d_settles_the_rests_and_resolves_the_carry(tmp_path, monkeypatch):
    """s3b_segments, s3c_invert, then s3d_settle on the rendered episode (the box as one convex piece):
    both rests end on the table and the carry is solved again against them."""
    import json
    import test_a2 as A
    from test_segments import synthetic_episode
    from pathb import runner
    from pathb.stages import s3b_segments, s3c_invert, s3d_settle
    monkeypatch.delenv("RETARGET_VENV", raising=False)
    monkeypatch.setenv("SETTLE_PY", str(SIM))
    root, e, truth = synthetic_episode(tmp_path)
    ctx = runner.Context(e, root, {})
    s3b_segments.run(ctx)
    m = A.box()
    tm = trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), process=False)
    (root / "s3a_mesh/box/coacd").mkdir(parents=True)
    tm.export(root / "s3a_mesh/box/mesh.obj")
    tm.export(root / "s3a_mesh/box/coacd/piece_000.obj")
    (root / "s3a_mesh/box/coacd/pieces.json").write_text(json.dumps({"pieces": [{"file": "piece_000.obj"}]}))
    (root / "s3a_mesh/box/mass.json").write_text(json.dumps({"mass_kg": 0.2}))
    (root / "s3a2_register/box").mkdir(parents=True)
    (root / "s3a2_register/box/registration.json").write_text(json.dumps({"canonical_to_world": truth[0].tolist(),
                                                                         "window": "rest_before"}))
    s3c_invert.run(ctx)
    r = s3d_settle.run(ctx)
    P = np.load(root / "s3d_settle/box/object_to_world.npz")["object_to_world"]
    rep = json.loads((root / "s3d_settle/box/settle_report.json").read_text())
    assert all(s["settled"] for s in rep["statics"]) and rep["resolved"], rep["statics"]
    assert abs(P[0, 2, 3]) < 0.003 and abs(P[-1, 2, 3]) < 0.003  # both rests on the table (bottom at z = 0)
    err = add_mm(P, truth, np.asarray(tm.vertices))
    # the p90 is the unstable bridges around this fixture's 2.3 cm/frame carry (S3c alone: just under 15 mm)
    assert np.median(err) < 3 and np.percentile(err, 90) < 25
    status = {x.name: x.status for x in r.metrics}
    assert status["box: penetration into the support after settling (worst)"] == "PASS"


def test_unsolved_runs_from_frame_0_bridge_from_the_registered_pose():
    """ep18's dust pan: no rest before the carry (0-6 unstable) and the carry lost to the slip guard; the
    two bridge as one from s3a2_register's pose to the final rest, not held at the final rest."""
    c, segs, obj, m = clip()
    segs = [{"kind": "unstable", "start": 0, "end": 6, "hand": None, "fingertips": []},
            {"kind": "unstable", "start": 7, "end": GRASP[1], "hand": None, "fingertips": []}, segs[2]]
    P, report = I.solve(c, segs, start_pose(), FPS)
    err = add_mm(P, obj, np.asarray(m.vertices))
    assert [r["kind"] for r in report] == ["unstable", "unstable", "static"]
    assert report[0]["why"] != "no neighbouring pose"
    assert err[0] < 20 and np.median(err[GRASP[1] + 1:]) < 3  # frame 0 near the registration (1 cm off)
    held = add_mm(np.repeat(P[-1:], GRASP[1] + 1, axis=0), obj[:GRASP[1] + 1], np.asarray(m.vertices))
    assert np.median(err[:GRASP[1] + 1]) < 0.6 * np.median(held)  # the bridge follows the carry's ends


def test_static_rays_missed_flags_a_mesh_the_mask_sees_past():
    """A static pose's mask rays all hit the true box; a box half its size (a see-through or shrunken mesh:
    ep40's voxel net let 27 % through) misses most of them, though ICP onto the fused surface may fit it."""
    c, segs, obj, m = clip()
    assert I.rays_missed(c, 0, GRASP[0] - 1, obj[0]) < I.MISSED_BAR  # edge pixels: 2 %
    small = (np.asarray(m.vertices) - [0, 0, 0.03]) * 0.5 + [0, 0, 0.03]
    half = I.Clip(K=c.K, c2w=c.c2w, obs=c.obs, tips=c.tips, wrist=c.wrist, valid=c.valid, vertices=small,
                  faces=np.asarray(m.faces), group=c.group, target=c.target, silhouette=c.silhouette)
    assert I.rays_missed(half, 0, GRASP[0] - 1, obj[0]) > 0.5
    _, report = I.solve(c, segs, start_pose(), FPS)
    assert all(r["rays_missed"] < I.MISSED_BAR for r in report if r["kind"] == "static")


def test_pass2_moves_the_offset_at_most_one_step_per_frame(monkeypatch):
    """An ICP start a quarter turn off (ep41's near-round pot: ICP turned it 165 degrees in the hand at 2 mm)
    moves the in-hand offset D by at most the spec's 5 mm / 3 degrees per component per frame."""
    c, _, _, _ = clip()
    quarter = np.eye(4)
    quarter[:3, :3] = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    monkeypatch.setattr(I, "icp_start", lambda clip, t, guess: (guess @ quarter, 1.0))
    a, b = GRASP
    poses, _ = I.pass2(c, a, b, 1, [0, 1], HAND_TO_OBJECT)
    D = np.array([I.x_of(np.linalg.inv(c.wrist[t, 1] @ HAND_TO_OBJECT) @ poses[t - a]) for t in range(a, b + 1)])
    step = np.diff(np.vstack([np.zeros(6), D]), axis=0)
    assert np.abs(step[:, :3]).max() <= I.STEP_RAD + 1e-9 and np.abs(step[:, 3:]).max() <= I.STEP_M + 1e-9


def test_static_retry_turns_about_gravity_to_the_turn_the_mask_sees():
    """ep45's pot: the fused surface (here a square top) fits a quarter turn as well; the mask still sees the
    handle where it is. From a start a quarter turn off, the retry's turns about gravity find it, and among
    fits tied on the fused surface the one the fewest mask rays miss wins (the least-turned one kept 81 deg)."""
    body = trimesh.creation.box((0.10, 0.10, 0.06))
    body.apply_translation([0, 0, 0.03])
    handle = trimesh.creation.box((0.08, 0.03, 0.03))
    handle.apply_translation([0.09, 0, 0.035])  # sticks out along +x, beside the camera's view of the body
    m = trimesh.util.concatenate([body, handle])
    true = np.eye(4)
    frames = 6
    c2w = np.repeat(camera()[None], frames, axis=0)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    obs, sil = [], []
    for t in range(frames):
        depth = S.rendered_depth(o3d.geometry.TriangleMesh(mesh).transform(true), K, c2w[t], SHAPE)
        ys, xs = np.nonzero(depth > 0)
        obs.append((np.stack([xs, ys], 1).astype(float), depth[ys, xs], len(ys) / (4 * SHAPE[0] * SHAPE[1])))
        sil.append(I.silhouette_map(depth > 0))
    g = np.stack(np.meshgrid(np.linspace(-0.045, 0.045, 40), np.linspace(-0.045, 0.045, 40)), -1).reshape(-1, 2)
    top = S.cloud(np.column_stack([g, np.full(len(g), 0.06)]), np.tile([0, 0, 1.0], (len(g), 1)))
    c = I.Clip(K=K, c2w=c2w, obs=obs, tips=np.full((frames, 2, 5, 3), np.nan), wrist=np.tile(np.eye(4), (frames, 2, 1, 1)),
               valid=np.zeros((frames, 2), bool), vertices=np.asarray(m.vertices), faces=np.asarray(m.faces),
               group=np.eye(4)[None], target=lambda a, b: top, silhouette=sil)
    off = np.eye(4)
    off[:3, :3] = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    assert I.rays_missed(c, 0, frames - 1, off) > I.MISSED_BAR  # the handle's pixels see past the turned mesh
    r = I.solve_static(c, 0, frames - 1, off)
    assert r["residual_m"] < I.STATIC_BAR_M and r["rays_missed"] < I.MISSED_BAR, r
    assert np.degrees(Rotation.from_matrix(r["pose"][:3, :3]).magnitude()) < 5, r["start_from"]


def test_a_failed_grasp_is_bridged_not_placed(monkeypatch):
    """A grasp whose fit fails (ep26's carry: 56 cm off, none of its frames within 1 cm) anchors nothing: its
    frames are bridged between the rests like an unstable run, and the row still says failed (S4b's fallback)."""
    c, segs, obj, m = clip()

    def far(clip, a, b, *args, **kwargs):
        P = obj[a:b + 1].copy()
        P[:, :3, 3] += [0.0, 0.0, 0.5]
        return [{"kind": "grasp", "start": a, "end": b, "poses": P, "failed": True, "share_within_1cm": 0.0}]
    monkeypatch.setattr(I, "solve_grasp", far)
    P, report = I.solve(c, segs, start_pose(), FPS)
    g = next(r for r in report if r["kind"] == "grasp")
    assert g["failed"] and "poses" not in g
    a, b = GRASP
    assert np.isfinite(P).all()
    assert np.abs(P[a:b + 1, 2, 3] - obj[a:b + 1, 2, 3]).max() < 0.15  # bridged near the path, not 50 cm up


def test_overhang_counts_the_mesh_hanging_out_of_the_mask():
    """rays_missed's other direction: a handle turned out of the mask's outline hangs out of it (ep13's cup upside
    down: 2 % of rays missed, as upright, but 16 % hanging out against 3 %)."""
    body = trimesh.creation.box((0.10, 0.10, 0.06))
    body.apply_translation([0, 0, 0.03])
    handle = trimesh.creation.box((0.04, 0.06, 0.02))
    handle.apply_translation([0, -0.08, 0.05])  # towards the camera, inside the body's outline
    m = trimesh.util.concatenate([body, handle])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    depth = S.rendered_depth(mesh, K, camera(), SHAPE)
    ys, xs = np.nonzero(depth > 0)
    c = I.Clip(K=K, c2w=camera()[None], obs=[(np.stack([xs, ys], 1).astype(float), depth[ys, xs], 0.0)],
               tips=np.full((1, 2, 5, 3), np.nan), wrist=np.tile(np.eye(4), (1, 2, 1, 1)), valid=np.zeros((1, 2), bool),
               vertices=np.asarray(m.vertices), faces=np.asarray(m.faces), group=np.eye(4)[None],
               target=lambda a, b: None, silhouette=[I.silhouette_map(depth > 0)])
    turned = np.eye(4)
    turned[:3, :3] = Rotation.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    assert I.overhang(c, 0, 0, np.eye(4)) < 0.01
    assert I.overhang(c, 0, 0, turned) > I.MISSED_BAR
    f = {"rays_missed": I.rays_missed(c, 0, 0, turned), "overhang": I.overhang(c, 0, 0, turned)}
    assert I.disagreement(f) == pytest.approx(f["rays_missed"] + f["overhang"])
    assert I.disagreement({"rays_missed": None}) is None


def test_a_grasp_anchored_only_at_its_end_is_tracked_from_that_end():
    """The part after a slip knows only the rest after it: pass 1's offset fits that end, the wrist drifts
    away from it towards the start (here 6 cm), and pass 2 tracks from the end backwards (ep41's 91-frame
    part, tracked from its start, never locked on)."""
    c, _, obj, m = clip()
    a, b = GRASP
    wrist = c.wrist.copy()
    for t in range(a, b + 1):
        drift = np.eye(4)
        drift[0, 3] = 0.06 * (b - t) / (b - a)
        wrist[t, 1] = wrist[t, 1] @ drift
    c = I.Clip(K=c.K, c2w=c.c2w, obs=c.obs, tips=np.full_like(c.tips, np.nan), wrist=wrist, valid=c.valid,
               vertices=c.vertices, faces=c.faces, group=c.group, target=c.target, silhouette=c.silhouette)
    [g] = I.solve_grasp(c, a, b, 1, [], None, obj[b + 1], split=False)
    err = add_mm(g["poses"], obj[a:b + 1], np.asarray(m.vertices))
    assert np.median(err) < 5 and np.median(err[:10]) < 8, (np.median(err), np.median(err[:10]))


def test_the_hand_letting_go_is_bridged_not_tracked(monkeypatch):
    """The wrist leaves the object over the grasp's last 4 frames (3 cm a frame, as ep41's left hand at
    the release): those frames are bridged and the carry stays on the object."""
    c, segs, obj, m = clip()
    wrist = c.wrist.copy()
    for t in range(GRASP[1] - 3, GRASP[1] + 1):
        away = np.eye(4)
        away[:3, :3] = Rotation.from_rotvec([0, 0, np.radians(25) * (t - GRASP[1] + 4)]).as_matrix()
        away[1, 3] = -0.03 * (t - GRASP[1] + 4)
        wrist[t, 1] = wrist[t, 1] @ away
    c = I.Clip(K=c.K, c2w=c.c2w, obs=c.obs, tips=c.tips, wrist=wrist, valid=c.valid, vertices=c.vertices,
               faces=c.faces, group=c.group, target=c.target, silhouette=c.silhouette)
    P, report = I.solve(c, segs, start_pose(), FPS)
    assert any(r["kind"] == "unstable" and r["why"] == "the hand letting go" for r in report), report
    err = add_mm(P, obj, np.asarray(m.vertices))
    assert np.median(err[GRASP[0]:GRASP[1] + 1]) < 4 and err[GRASP[1] - 3:GRASP[1] + 1].max() < 10, err[GRASP[1] - 5:]
    monkeypatch.setattr(I, "RELEASE_M", np.inf)  # tracked to the end: the last frames follow the hand
    P, _ = I.solve(c, segs, start_pose(), FPS)
    assert add_mm(P, obj, np.asarray(m.vertices))[GRASP[1] - 3:GRASP[1] + 1].max() > 10


def test_a_carry_lost_to_the_slip_guard_is_bridged_with_the_wrist(monkeypatch):
    """ep43: every grasp of the carry tripped the slip guard and the bridge between the rests was interpolated,
    missing what the hand did with the object (there: turned it over; here: the 8 cm lift). With the hand S3b saw
    holding it, the bridge moves with that wrist, and keeps that guess because it explains the frames better."""
    monkeypatch.setattr(I, "CARRIED_BRIDGES", True)
    c, segs, obj, m = clip()

    def slipped(clip, a, b, *args, **kwargs):
        return [{"kind": "unstable", "start": a, "end": b, "why": "slip guard"}]
    monkeypatch.setattr(I, "solve_grasp", slipped)
    a, b = GRASP
    P, report = I.solve(c, segs, start_pose(), FPS)
    row = next(r for r in report if r["kind"] == "unstable" and r["start"] == a)
    assert row["guess"] == "carried" and row["explains_carried_m"] < row["explains_interpolated_m"], row
    err = add_mm(P, obj, np.asarray(m.vertices))[a:b + 1]
    monkeypatch.setattr(I, "CARRIED_BRIDGES", False)
    P0, report0 = I.solve(c, segs, start_pose(), FPS)
    err0 = add_mm(P0, obj, np.asarray(m.vertices))[a:b + 1]
    assert next(r for r in report0 if r["start"] == a)["guess"] == "interpolated"
    assert np.median(err) < 10 and np.median(err) < 0.5 * np.median(err0), (np.median(err), np.median(err0))


def test_carried_holds_both_ends_and_follows_the_wrist_between():
    """carried: forward from the pose before, backward from the pose after, blended; a frame without a carrier
    keeps the object where it was."""
    c, segs, obj, m = clip()
    a, b = GRASP
    carrier = np.full(N, -1)
    carrier[a:b + 1] = 1
    G = I.carried(c, a, b, obj[a - 1], obj[b + 1], carrier)
    # the ends: one wrist step from the known pose (two noisy wrist samples, 2 mm / 0.5 degrees each at a 12 cm lever)
    assert add_mm(G[[0, -1]], obj[[a, b]], np.asarray(m.vertices)).max() < 15
    assert np.median(add_mm(G, obj[a:b + 1], np.asarray(m.vertices))) < 8
    still = I.carried(c, a, b, obj[a - 1], None, np.full(N, -1))
    assert still is None


def test_a_grasp_takes_its_end_terms_across_a_short_unstable_gap(monkeypatch):
    """ep43: S3b put 0-5 and 272-274 between the carry and its rests, and the carry had neither end term (pass 1
    drifted, the slip guard took 6-271). A gap of at most ANCHOR_GAP frames with a rest (or the clip's start)
    beyond it passes that rest's pose on as the end term, as across the grasp's own trimmed frames."""
    c, segs, obj, m = clip()
    a, b = GRASP
    gapped = [{"kind": "static", "start": 0, "end": a - 5, "hand": None, "fingertips": []},
              {"kind": "unstable", "start": a - 4, "end": a - 1, "hand": None, "fingertips": []},
              segs[1],
              {"kind": "unstable", "start": b + 1, "end": b + 3, "hand": None, "fingertips": []},
              {"kind": "static", "start": b + 4, "end": N - 1, "hand": None, "fingertips": []}]
    monkeypatch.setattr(I, "ANCHOR_GAP", I.RELEASE_FRAMES)
    seen = []
    real = I.pass1

    def spy(clip, a_, b_, hand, fingers, start, end, T0):
        seen.append((start is not None, end is not None))
        return real(clip, a_, b_, hand, fingers, start, end, T0)
    monkeypatch.setattr(I, "pass1", spy)
    P, report = I.solve(c, gapped, start_pose(), FPS)
    assert seen[0] == (True, False) and seen[-1] == (True, True), seen  # sweep 1: the rest before; sweep 2: both
    assert np.median(add_mm(P, obj, np.asarray(m.vertices))[a:b + 1]) < 4
    seen.clear()
    monkeypatch.setattr(I, "ANCHOR_GAP", 3)  # the 4-frame gap is longer: no end term across it
    I.solve(c, gapped, start_pose(), FPS)
    assert seen[0] == (False, False), seen
    seen.clear()
    monkeypatch.setattr(I, "ANCHOR_GAP", I.RELEASE_FRAMES)
    start = [{"kind": "unstable", "start": 0, "end": 3, "hand": None, "fingertips": []},
             {"kind": "grasp", "start": 4, "end": b, "hand": "R", "fingertips": [["R", 0], ["R", 1]]}, segs[2]]
    I.solve(c, start, start_pose(), FPS)
    assert seen[0][0], seen  # the clip's start: s3a2_register's pose


def test_a_rest_beyond_the_hands_reach_keeps_the_rest_before():
    """ep0 / ep11 / ep22: a final rest fitted ~1 m from the rest before, while the hand that carried the object in
    between reached ~25 cm, is not where the object lies: it keeps the rest before. A normal carry is not flagged."""
    c, segs, obj, m = clip()
    P, report = I.solve(c, segs, start_pose(), FPS)
    assert not any("beyond_reach" in r for r in report)
    far = c.target

    def target(a, b):  # the last rest's fused surface 1 m away (a wrong mask or a far background)
        pc = far(a, b)
        return pc.translate([1.0, 0.0, 0.0]) if a > GRASP[1] else pc
    moved = I.Clip(K=c.K, c2w=c.c2w, obs=c.obs, tips=c.tips, wrist=c.wrist, valid=c.valid, vertices=c.vertices,
                   faces=c.faces, group=c.group, target=target, silhouette=c.silhouette)
    P, report = I.solve(moved, segs, start_pose(), FPS)
    last = next(r for r in report if r["kind"] == "static" and r["start"] > GRASP[1])
    assert last["beyond_reach"]["moved_m"] > 0.9 and last["beyond_reach"]["reach_m"] < 0.5, last
    assert np.allclose(P[-1], P[GRASP[0] - 1], atol=1e-9)  # the rest before, held


def test_slip_retry_solves_from_the_end_pose_unless_it_jumps(monkeypatch):
    """SLIP_RETRY: a grasp that slips throughout from its start pose is solved again from its end pose (start
    None); the retry is kept only if no grasp it yields jumps more than JUMP_M between frames."""
    from types import SimpleNamespace
    n, a, b = 40, 5, 30
    clip = SimpleNamespace(wrist=np.tile(np.eye(4), (n, 2, 1, 1)), valid=np.ones((n, 2), bool))
    calls = []

    def fake_pass2(clip, a, b, hand, fingers, T, backward=False):
        calls.append(backward)
        poses = np.tile(np.eye(4), (b - a + 1, 1, 1))
        poses[:, 0, 3] = np.linspace(0, jump_per_frame * (b - a), b - a + 1)
        q75 = np.full(b - a + 1, 0.05 if not backward else 0.001)  # slips from the start pose only
        return poses, {"residual_a_median_m": np.full(b - a + 1, 0.002), "residual_a_q75_m": q75,
                       "drift_max_m": 0.0, "drift_max_deg": 0.0, "icp_paired_median": 1.0}

    monkeypatch.setattr(I, "trimmed", lambda clip, a, b, hand, s, e: (a, b))
    monkeypatch.setattr(I, "pass1", lambda *args: {"T": np.eye(4)})
    monkeypatch.setattr(I, "pass2", fake_pass2)
    monkeypatch.setattr(I, "SLIP_RETRY", True)
    start, end = np.eye(4), np.eye(4)
    jump_per_frame = 0.005
    rows = I.solve_grasp(clip, a, b, 0, [1], start, end)
    assert calls == [False, True] and [r["kind"] for r in rows] == ["grasp"] and rows[0]["retried"]
    calls.clear()
    jump_per_frame = 0.05  # 5 cm a frame: the retry is dropped, the slip split stands
    rows = I.solve_grasp(clip, a, b, 0, [1], start, end)
    assert calls == [False, True] and not any(r.get("retried") for r in rows) and rows[0]["why"] == "slip guard"
    calls.clear()
    monkeypatch.setattr(I, "SLIP_RETRY", False)
    assert I.solve_grasp(clip, a, b, 0, [1], start, end)[0]["why"] == "slip guard" and calls == [False]


def test_upright_prior_on_a_revolution_rest():
    # a pot-like body (a cylinder with a handle: its turn is scored), canonical z up, resting upright; the seen side only
    body = trimesh.creation.cylinder(radius=0.08, height=0.12, sections=48)
    body.apply_translation([0, 0, 0.06])
    handle = trimesh.creation.box((0.06, 0.02, 0.02))
    handle.apply_translation([0.11, 0, 0.09])
    m = trimesh.util.concatenate([body, handle])
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    truth[:3, 3] = [0.3, 0.1, 0.0]
    pts, faces = m.sample(6000, return_index=True)
    seen = (m.face_normals[faces] @ truth[:3, :3].T)[:, 0] < 0.2  # the camera's side
    world = pts[seen] @ truth[:3, :3].T + truth[:3, 3]
    c = I.Clip(K=np.eye(3), c2w=np.eye(4)[None], obs=[None], tips=np.full((1, 2, 5, 3), np.nan),
               wrist=np.tile(np.eye(4), (1, 2, 1, 1)), valid=np.zeros((1, 2), bool), vertices=np.asarray(m.vertices),
               faces=np.asarray(m.faces), group=np.eye(4)[None], target=None, revolution=True)
    tilted = truth.copy()
    tilted[:3, :3] = Rotation.from_euler("x", 25, degrees=True).as_matrix() @ truth[:3, :3]
    local = (world - tilted[:3, 3]) @ tilted[:3, :3]
    res = float(np.median(np.abs(c.scene.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy())))
    out = I.upright(c, 0, 0, {"pose": tilted, "residual_m": res, "start_from": "global"}, world)
    assert out["upright"]["kept"] == "upright" and I.tilt_deg(out["pose"]) < 0.5
    assert np.linalg.norm(out["pose"][:3, 3] - truth[:3, 3]) < 0.005 and out["residual_m"] < 0.001
    assert Rotation.from_matrix(out["pose"][:3, :3].T @ truth[:3, :3]).magnitude() < np.radians(2)  # the handle's turn found again
    # a rest already vertical is left alone; one whose evidence beats the vertical keeps its tilt
    assert "upright" not in I.upright(c, 0, 0, {"pose": truth, "residual_m": 0.0005}, world)
    lean = truth.copy()
    lean[:3, :3] = Rotation.from_euler("x", 18, degrees=True).as_matrix() @ truth[:3, :3]
    leaning = world @ (lean[:3, :3] @ truth[:3, :3].T).T + (lean[:3, 3] - lean[:3, :3] @ truth[:3, :3].T @ truth[:3, 3])
    kept = I.upright(c, 0, 0, {"pose": lean, "residual_m": 0.0002}, leaning)
    assert kept["upright"]["kept"].startswith("the free fit") and kept["pose"] is lean


def test_a_cup_nested_in_a_cup_is_fitted_on_its_axis():
    # ep11: a cup stands in another one of about its size; only its part above the outer cup's top shows, and the
    # free fit lay on its side off the outer cup's axis
    outer = trimesh.creation.cylinder(radius=0.05, height=0.175, sections=48)
    outer.apply_translation([0, 0, 0.0875])
    inner = trimesh.creation.cylinder(radius=0.048, height=0.169, sections=48)
    inner.apply_translation([0, 0, 0.0845])
    host = np.eye(4)
    host[:3, 3] = [0.3, 0.1, 0.0]
    truth = host.copy()
    truth[:3, 3] += [0, 0, 0.025]  # 2.5 cm up inside it
    pts, faces = inner.sample(8000, return_index=True)
    world = pts @ truth[:3, :3].T + truth[:3, 3]
    seen = (world[:, 2] > 0.176) & ((inner.face_normals[faces] @ truth[:3, :3].T)[:, 0] < 0.2)
    world = world[seen]
    c = I.Clip(K=np.eye(3), c2w=np.eye(4)[None], obs=[None], tips=np.full((1, 2, 5, 3), np.nan),
               wrist=np.tile(np.eye(4), (1, 2, 1, 1)), valid=np.zeros((1, 2), bool), vertices=np.asarray(inner.vertices),
               faces=np.asarray(inner.faces), group=np.eye(4)[None], target=None, revolution=True,
               hosts=[{"name": "blue_cup", "P": host[None], "vertices": np.asarray(outer.vertices), "rests": [(0, 0)],
                       "silhouette": None}])
    side = np.eye(4)
    side[:3, :3] = Rotation.from_euler("y", 80, degrees=True).as_matrix()
    side[:3, 3] = [0.3, 0.2, 0.14]  # on its side, 10 cm off the axis
    local = (world - side[:3, 3]) @ side[:3, :3]
    res = float(np.median(np.abs(c.scene.compute_distance(o3d.core.Tensor(local.astype(np.float32))).numpy())))
    out = I.upright(c, 0, 0, {"pose": side, "residual_m": res, "start_from": "global"}, world)
    assert out["stacked"]["kept"] == "stacked" and out["stacked"]["host"] == "blue_cup"
    assert I.tilt_deg(out["pose"]) < 0.5 and np.linalg.norm(out["pose"][:3, 3] - truth[:3, 3]) < 0.005
    # beside the outer cup, or not above its top, there is no host
    assert I.host_for(c, 0, 0, world + [0.2, 0, 0]) is None and I.host_for(c, 0, 0, world - [0, 0, 0.05]) is None


def test_a_grasp_that_leaves_the_object_where_it_was_is_held_still():
    # ep11's blue cup: a hand steadies it, it never moves, and the grasp's track drifted 5 cm off
    c, _, obj, m = clip()
    still = np.repeat(obj[:1], N, axis=0)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
    depth = S.rendered_depth(o3d.geometry.TriangleMesh(mesh).transform(still[0]), K, c.c2w[0], SHAPE)
    ys, xs = np.nonzero(depth > 0)
    c.obs = [(np.stack([xs, ys], 1).astype(float), depth[ys, xs], len(ys) / (4 * SHAPE[0] * SHAPE[1]))] * N
    c.silhouette = [I.silhouette_map(depth > 0)] * N
    a, b = GRASP
    P = still.copy()
    P[a:b + 1, :3, 3] += np.outer(np.sin(np.linspace(0, np.pi, b - a + 1)), [0.05, 0.0, 0.02])
    report = [{"kind": "grasp", "start": a, "end": b}]
    I.held_still(c, a, b, P, report)
    assert np.abs(P[a:b + 1, :3, 3] - still[0, :3, 3]).max() < 1e-9 and "held" in report[0]
    # a real carry that ends where it began explains its frames better than standing still: kept
    c, _, obj, m = clip()
    P = obj.copy()
    P[b + 1:] = P[a - 1]
    track = P[a:b + 1].copy()
    report = [{"kind": "grasp", "start": a, "end": b}]
    I.held_still(c, a, b, P, report)
    assert np.allclose(P[a:b + 1], track) and "held" not in report[0]


def test_a_tracked_bridge_follows_the_frames_where_interpolation_cuts_the_corner():
    # the carry's middle third bridged from its true ends while the object slips 4 cm in the hand: interpolation cuts
    # the 8 cm lift's arc; the tracked guess moves with the wrist and steps with the depth, from both ends
    c, _, obj, m = clip(slip=(30, 0.04, 15))
    a, b = GRASP[0] + 12, GRASP[1] - 12
    carrier = np.full(N, -1)
    carrier[GRASP[0]:GRASP[1] + 1] = 1
    T = I.tracked(c, a, b, obj[a - 1], obj[b + 1], carrier)
    err_t = add_mm(T, obj[a:b + 1], m.vertices)
    err_i = add_mm(I.interpolate(obj[a - 1], obj[b + 1], b - a + 1), obj[a:b + 1], m.vertices)
    assert err_t.mean() < 0.4 * err_i.mean() and err_t.max() < 20
    assert np.allclose(I.tracked(c, a, b, obj[a - 1], None, carrier)[0][:3, 3], T[0][:3, 3], atol=0.02)  # one end only
    # switched on, the bridge keeps whichever guess explains its frames best
    I.TRACKED_BRIDGES = True
    try:
        u = I.solve_unstable(c, a, b, obj[a - 1], obj[b + 1], carrier)
    finally:
        I.TRACKED_BRIDGES = False
    scores = {k[9:-2]: v for k, v in u.items() if k.startswith("explains_") and v is not None}
    assert set(scores) == {"interpolated", "carried", "tracked"} and u["guess"] == min(scores, key=scores.get)
