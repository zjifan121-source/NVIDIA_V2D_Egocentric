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
