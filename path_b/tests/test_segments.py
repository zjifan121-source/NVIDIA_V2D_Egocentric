"""pathb/segments.py (V4_BUILD_SPEC.md S3b, task 5): contact segmentation on synthetic clips.

A box rests on the table (0-29), the right hand grasps it (30-69: thumb and index on its sides,
carried 40 cm over 35-64), releases it (70) and it rests again. The ep41/ep13 checks run on the dev
cache (BUILD_LOG)."""
import numpy as np
import pytest

from pathb import segments as G

N, GRASP, MOVE, RELEASE = 120, 30, (35, 64), 70
SIZE = np.array([0.12, 0.08, 0.06])


def box_points(centre, rng, n=1500):
    """The box's top and its four sides (the camera's view), 1 mm noise."""
    lo, hi = centre - SIZE / 2 * [1, 1, 0], centre + SIZE / 2 * [1, 1, 2]
    p = rng.uniform(lo, hi, (n, 3))
    face = rng.integers(0, 5, n)
    p[face == 0, 2] = hi[2]
    p[face == 1, 0], p[face == 2, 0] = lo[0], hi[0]
    p[face == 3, 1], p[face == 4, 1] = lo[1], hi[1]
    return p + rng.normal(0, 0.001, p.shape)


def path(t):
    s = np.clip((t - MOVE[0]) / (MOVE[1] - MOVE[0]), 0, 1)
    return np.array([0.0, 0.0, 0.0]) + s * np.array([0.4, 0.05, 0.0])


def clip(gap=None):
    rng = np.random.default_rng(0)
    clouds = [box_points(path(t), rng) for t in range(N)]
    tips = np.full((N, 2, 5, 3), np.nan)
    wrist = np.tile(np.eye(4), (N, 2, 1, 1))
    valid = np.zeros((N, 2), bool)
    for t in range(N):
        c = path(t)
        # the right hand: far before 20, approaching 20-29, on the box 30-69, leaving 70-79, far after
        if t < GRASP:
            away = 0.15 if t < 20 else 0.15 - 0.012 * (t - 20)
        elif t < RELEASE:
            away = 0.004
        else:
            away = min(0.15, 0.004 + 0.015 * (t - RELEASE + 1))
        side = c + [0, -SIZE[1] / 2 - away, SIZE[2] / 2]
        tips[t, 1, 0] = side + [-0.03, 0, 0]  # thumb
        tips[t, 1, 1] = side + [0.03, 0, 0]  # index
        tips[t, 1, 2:] = side + [0, -0.05, 0.02]  # the other three, curled away
        wrist[t, 1, :3, 3] = side + [0, -0.08, 0.02]
        valid[t, 1] = True
    if gap:
        tips[gap[0]:gap[1] + 1, 1], valid[gap[0]:gap[1] + 1, 1] = np.nan, False
    return clouds, tips, wrist, valid


def test_rest_grasp_rest():
    clouds, tips, wrist, valid = clip()
    segs, report, _ = G.segment(clouds, tips, wrist, valid, {})
    kinds = [s["kind"] for s in segs]
    grasps = [s for s in segs if s["kind"] == "grasp"]
    assert kinds[0] == "static" and kinds[-1] == "static" and len(grasps) == 1, report["segments"]
    g = grasps[0]
    assert g["hand"] == "R" and abs(g["start"] - GRASP) <= 3 and abs(g["end"] - (RELEASE - 1)) <= 3
    assert g["fingertips"] == [["R", 0], ["R", 1]]  # thumb and index, not the curled fingers
    th = report["thresholds"]
    assert th["certain_frames"] >= G.MIN_CERTAIN and 0.006 < th["near_m"] < 0.012  # 2 x the 4 mm contact
    assert [s["start"] for s in segs] == [0] + [s["end"] + 1 for s in segs[:-1]] and segs[-1]["end"] == N - 1
    ph = G.phases(segs, N, G.centroids(clouds))
    assert ph["carry"] == [g["start"], g["end"]] and ph["rest_before"] == [0, g["start"] - 1]
    assert abs(ph["displacement_cm"] - 40.3) < 1.5 and not ph["static"]


def test_long_hand_gap_splits_the_grasp():
    clouds, tips, wrist, valid = clip(gap=(45, 62))
    segs, _, _ = G.segment(clouds, tips, wrist, valid, {"R": [(45, 62)]})
    grasps = [(s["start"], s["end"]) for s in segs if s["kind"] == "grasp"]
    assert len(grasps) == 2 and grasps[0][1] < 45 and grasps[1][0] > 62
    assert any(s["kind"] == "unstable" and s["start"] <= 45 and s["end"] >= 62 for s in segs)


def test_never_grasped_is_one_static_segment():
    rng = np.random.default_rng(1)
    clouds = [box_points(np.zeros(3), rng) for _ in range(40)]
    tips = np.full((40, 2, 5, 3), np.nan)
    segs, report, _ = G.segment(clouds, tips, np.tile(np.eye(4), (40, 2, 1, 1)), np.zeros((40, 2), bool), {})
    assert segs == [{"kind": "static", "start": 0, "end": 39, "hand": None, "fingertips": [], "support": None}]
    assert report["static"] and report["thresholds"]["near_m"] == G.NEAR_DEFAULT


def test_hysteresis_ignores_blips():
    v = ["static"] * 5 + ["unstable"] + ["static"] * 4 + ["grasp:R"] * 6 + [None, None] + ["grasp:R"] * 3
    out = G.hysteresis(v)
    assert out[:10] == ["static"] * 10 and out[10:] == ["grasp:R"] * 11


def test_support_table_rack_and_floating():
    rng = np.random.default_rng(2)
    rack = rng.uniform([-0.2, -0.15, 0.0], [0.2, 0.15, 0.12], (4000, 3))
    cup = rng.uniform([-0.04, -0.04, 0.04], [0.04, 0.04, 0.14], (2000, 3))  # standing in the rack, 4 cm up
    on_table = rng.uniform([0.5, 0.0, 0.0], [0.58, 0.08, 0.1], (2000, 3))
    assert G.support(rack, {"blue_cup": cup})[0] == "table"
    assert G.support(cup, {"plastic_dish_rack": rack})[0] == "plastic_dish_rack"
    assert G.support(on_table, {"plastic_dish_rack": rack})[0] == "table"
    lifted = on_table + [0, 0, 0.1]  # 10 cm up, nothing under it
    name, ev = G.support(lifted, {"plastic_dish_rack": rack})
    assert name == "table" and ev["floating"]


@pytest.mark.parametrize("gap", [None])
def test_motion_is_small_at_rest_and_large_in_the_carry(gap):
    clouds, *_ = clip(gap)
    m = G.motion(clouds)
    assert np.nanmax(m[:25]) < G.STATIC_M and np.nanmin(m[40:60]) > 0.02


def synthetic_episode(tmp_path):
    """test_a2's box (rest 0-14, moved 15-25, rest 26-39) seen through masks, stereo depth and cameras,
    with a right hand whose thumb and index hold its sides over 15-25; P1d-style hand exports and a
    P4-style bundle. Returns (root, episode entry, true poses (N, 4, 4))."""
    import json
    import cv2
    import test_a2 as A
    from pathb import scan as S
    from pathb.geometry import write_depth, write_transform
    root = tmp_path / "episode_000999"
    half, cams, hd = root / "p2_depth/depth_a_half", root / "p4_export/camera_to_world", root / "p4_export/hands"
    bundle = root / "p4_export/episode_000999/box/result_slam_gravity_aligned"
    for d in (half, cams, hd, bundle, root / "p1b_masks/obj_0", root / "p1b_masks/hand_L", root / "p1b_masks/hand_R"):
        d.mkdir(parents=True)
    (half / "intrinsics.json").write_text(json.dumps({"K": A.K.tolist(), "size": [A.SHAPE[1], A.SHAPE[0]]}))
    n = A.N
    tips = np.full((n, 2, 5, 3), np.nan)
    wrist = np.tile(np.eye(4), (n, 2, 1, 1))
    valid = np.zeros((n, 2), bool)
    for t in range(n):
        mesh = A.o3d.geometry.TriangleMesh(A.box()).transform(A.pose(t))
        depth = S.rendered_depth(mesh, A.K, A.camera(t), A.SHAPE)
        write_depth(half / f"{t:06d}.png", depth)
        write_transform(cams / f"{t:06d}.json", A.camera(t))
        big = cv2.resize((depth > 0).astype(np.uint8) * 255, (A.SHAPE[1] * 2, A.SHAPE[0] * 2), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(root / f"p1b_masks/obj_0/{t:06d}.png"), big)
        for side in ("hand_L", "hand_R"):
            cv2.imwrite(str(root / f"p1b_masks/{side}/{t:06d}.png"), np.zeros_like(big))
        held = A.REST_B[1] < t < A.REST_A[0]
        away = 0.004 if held else 0.2
        local = np.array([[-0.03, -0.04 - away, 0.025], [0.03, -0.04 - away, 0.025]])  # on its -y side, in the box frame
        T = A.pose(t)
        tips[t, 1, :2] = local @ T[:3, :3].T + T[:3, 3]
        tips[t, 1, 2:] = tips[t, 1, 0] + [0, -0.05, 0.03]
        wrist[t, 1] = T @ np.array([[1, 0, 0, 0], [0, 1, 0, -0.12 - away], [0, 0, 1, 0.03], [0, 0, 0, 1.0]])
        valid[t, 1] = True
    np.savez(hd / "fingertips_world.npz", tips=tips, valid=valid, filled=np.zeros_like(valid))
    np.savez(hd / "wrist_world.npz", wrist=wrist, valid=valid, filled=np.zeros_like(valid))
    (hd / "hand_gaps.json").write_text(json.dumps({"gaps": {"L": [{"start": 0, "end": n - 1, "frames": n, "kind": "long"}],
                                                            "R": []}}))
    c2w = np.stack([A.camera(t) for t in range(n)])
    np.savez(bundle / "result.npz", camera_to_world_transform=c2w.astype(np.float32),
             object_to_world_transform=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
             object_to_camera_transform=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)), object_scale=np.float32(1))
    e = {"episode": 999, "split": "evaluation", "num_frames": n, "objects": [{"name": "box"}]}
    return root, e, np.stack([A.pose(t) for t in range(n)])


def test_stage_on_a_synthetic_episode(tmp_path, monkeypatch):
    """s3b_segments end to end on the synthetic episode."""
    import json
    import test_a2 as A
    from pathb import runner
    from pathb.stages import s3b_segments
    monkeypatch.delenv("RETARGET_VENV", raising=False)
    root, e, _ = synthetic_episode(tmp_path)
    result = s3b_segments.run(runner.Context(e, root, {}))
    segs = json.loads((root / "s3b_segments/box/segments.json").read_text())
    kinds = [s["kind"] for s in segs]
    grasps = [s for s in segs if s["kind"] == "grasp"]
    # contact and motion start together here, so the 10-frame motion window leaves unstable bridges
    assert kinds[0] == kinds[-1] == "static" and len(grasps) == 1 and set(kinds) <= {"static", "grasp", "unstable"}, segs
    g = grasps[0]
    assert g["hand"] == "R" and abs(g["start"] - (A.REST_B[1] + 1)) <= 3 and abs(g["end"] - (A.REST_A[0] - 1)) <= 3
    assert segs[0]["support"] == "table" and segs[-1]["support"] == "table"
    ph = json.loads((root / "s3b_segments/box/phases.json").read_text())
    assert ph["carry"] == [g["start"], g["end"]] and ph["source"] == "s3b_segments"
    cov = json.loads((root / "s3b_segments/box/coverage.json").read_text())
    assert cov["coverage_carry"] == 1.0
    status = {m.name: m.status for m in result.metrics}
    assert status["box: every frame in one segment"] == "PASS" and status["box: grasp found"] == "PASS"


def test_wristcheck_measures_the_wrist_against_a_still_object(tmp_path):
    """S0.3: a hand rests on a still box for 30 frames with 1 cm of wrist noise per axis; the pooled
    distance sigma reads ~1 cm (its radial part) and the gate builds S3c; with 4 cm it does not."""
    import json
    from pathb import wristcheck as W
    rng = np.random.default_rng(0)
    n = 60
    for noise, decision in ((0.01, "build S3c as specified"), (0.04, "do not build S3c")):
        root = tmp_path / f"n{noise}" / "episode_000041"
        seg, hd = root / "s3b_segments/white_pot", root / "p4_export/hands"
        seg.mkdir(parents=True)
        hd.mkdir(parents=True)
        motion = np.full(n, 0.001)
        motion[40:] = 0.02  # carried after 40
        d_tip = np.full((n, 2, 5), np.nan)
        d_tip[10:40, 1] = 0.006  # the right hand resting on it 10-39
        cent = np.zeros((n, 3))
        wrist = np.tile(np.eye(4), (n, 2, 1, 1))
        wrist[:, 1, :3, 3] = [0.1, 0.0, 0.05] + rng.normal(0, noise, (n, 3))
        np.savez(seg / "per_frame.npz", motion=motion, d_tip=d_tip, centroid=cent)
        (seg / "segments_report.json").write_text(json.dumps({"thresholds": {"near_m": 0.012}}))
        (seg / "coverage.json").write_text(json.dumps({"coverage_carry": 0.95, "grasps": [{"start": 40, "end": 59}]}))
        np.savez(hd / "wrist_world.npz", wrist=wrist, valid=np.ones((n, 2), bool))
        report = json.loads(W.main(root.parent, [41]).read_text())
        e = report["episodes"][0]
        assert e["window_frames"] == 30 and abs(e["sigma_cm"] - noise * 100) < noise * 100 * 0.35
        assert abs(e["sigma_vec_cm"] - noise * 100 * np.sqrt(3)) < noise * 100 * 0.5
        assert report["decision"].startswith(decision) and report["hand_gap_rule"] == "optional"
