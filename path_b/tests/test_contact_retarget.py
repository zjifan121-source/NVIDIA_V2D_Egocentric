"""Path B's contact-preserving retarget pass (pathb/contact_retarget.py) on a toy planar finger over a box (the parts of
SharpaHand's interface the pass uses), and the driver's step 2b. The real Sharpa model needs pinocchio ($RETARGET_VENV)."""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from pathb import contact_retarget as C

TOP = 0.0  # the box's top face (world z)


class ToyFinger:
    """A planar two-link 'index' finger in the x-z plane (both joints rotate about +y), its pad just under the tip."""
    L1, L2 = 0.05, 0.03

    def __init__(self, base=(0.0, 0.0, 0.05)):
        self.base = np.array(base, float)
        self.lower, self.upper = np.array([-0.2, 0.0]), np.array([1.6, 1.6])
        self.finger_dofs = {"index": [0, 1]}
        self.geoms = [C.Geometry("link1", 1, None, "index", False), C.Geometry("link2", 2, None, "index", False),
                      C.Geometry("pad", 2, None, "index", True, axis=np.array([0.0, 0.0, -1.0]))]
        self.pads = {"index": self.geoms[2]}
        self.x = np.zeros(2)

    def q(self, wrist_position, wrist_wxyz, fingers):
        return np.asarray(fingers, float)

    def update(self, q):
        self.x = np.asarray(q, float)

    def _dirs(self):
        a, b = self.x[0], self.x[0] + self.x[1]
        d1, d2 = np.array([np.cos(a), 0, -np.sin(a)]), np.array([np.cos(b), 0, -np.sin(b)])
        n2 = np.array([np.sin(b), 0, np.cos(b)])  # d2 turned up: the pad is on the other side
        return d1, d2, n2, self.base + self.L1 * d1

    def points(self, g):
        d1, d2, n2, j2 = self._dirs()
        if g.name == "link1":
            return self.base + np.linspace(0, 1, 6)[:, None] * self.L1 * d1
        if g.name == "link2":
            return j2 + np.linspace(0, 1, 6)[:, None] * self.L2 * d2
        return j2 + (self.L2 - np.linspace(0, 0.008, 5)[:, None]) * d2 - 0.004 * n2

    def point_jacobians(self, g, world):
        _, _, _, j2 = self._dirs()
        y = np.array([0.0, 1.0, 0.0])
        J = np.zeros((len(world), 3, 2))
        J[:, :, 0] = np.cross(y, world - self.base)
        if g.joint == 2:
            J[:, :, 1] = np.cross(y, world - j2)
        return J

    def pad_normal(self, g):
        return -self._dirs()[2]  # the pad's palm side: away from link 2's 'up'

    def normal_jacobian(self, g, n):
        return np.stack([np.cross([0.0, 1.0, 0.0], n)] * 2, axis=1)  # both joints turn about +y

    def frames(self):
        _, d2, _, j2 = self._dirs()
        return np.array([[*p, 1.0, 0.0, 0.0, 0.0] for p in (self.base, j2, j2 + self.L2 * d2)])


def box_surface(watertight=True):
    mesh = trimesh.creation.box(extents=(0.4, 0.4, 0.1))
    if not watertight:
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:-2], process=False)
    return C.ObjectSurface(mesh, samples=20000)


BOX_P, BOX_Q = np.array([0.0, 0.0, TOP - 0.05]), np.array([1.0, 0.0, 0.0, 0.0])


def toy_row(n=40, contact=range(15, 25), x0=(0.3, 0.3)):
    """One hand ('right') over the box, NVIDIA's joints constant, the index distal link labelled on `contact` with the
    surface point under the fingertip and the box's outward normal."""
    hand = ToyFinger()
    hand.update(np.array(x0))
    tip = hand.points(hand.pads["index"])[0]
    pid = np.zeros((n, 16), int)
    pos, nrm = np.zeros((n, 16, 3)), np.zeros((n, 16, 3))
    for t in contact:
        pid[t, C.DISTAL_LINK["index"]] = 1
        pos[t, C.DISTAL_LINK["index"]] = [tip[0], 0.0, TOP]
        nrm[t, C.DISTAL_LINK["index"]] = [0.0, 0.0, 1.0]
    return hand, {"robot_right_finger_joints": np.tile(x0, (n, 1)), "robot_right_wrist_position": np.zeros((n, 3)),
                  "robot_right_wrist_wxyz": np.tile([1.0, 0, 0, 0], (n, 1)), "mano_right_object_contact_part_ids": pid,
                  "mano_right_object_contact_positions": pos, "mano_right_object_contact_normals": nrm,
                  "object_body_position": np.tile(BOX_P, (n, 1, 1)), "object_body_wxyz": np.tile(BOX_Q, (n, 1, 1))}


def test_point_jacobians_are_the_velocity_of_points_on_a_body():
    rng = np.random.default_rng(0)
    J, r = rng.normal(size=(6, 3)), rng.normal(size=(4, 3))
    out = C.point_jacobians(J, r)
    for k in range(3):
        assert np.allclose(out[:, :, k], J[:3, k] + np.cross(J[3:, k], r))


def test_contact_runs_close_short_gaps_and_drop_flicker():
    pid = np.zeros((20, 16), int)
    pid[[2, 3, 4, 5, 6, 8, 9, 10, 15], C.DISTAL_LINK["index"]] = 1  # one-frame gap at 7, a lone frame at 15
    pid[[12, 13], C.DISTAL_LINK["thumb"]] = 2  # a two-frame run: shorter than MIN_RUN
    runs = C.contact_runs(pid)
    assert np.flatnonzero(runs["index"]).tolist() == list(range(2, 11))
    assert not runs["thumb"].any()


def test_blend_weights_ramp_in_and_out_of_a_run():
    active = np.zeros(12, bool)
    active[5:8] = True
    w, src = C.blend_weights(active)
    assert np.allclose(w[2:11], [0.25, 0.5, 0.75, 1, 1, 1, 0.75, 0.5, 0.25])
    assert src[4] == 5 and src[8] == 7 and src[1] == -1 and w[1] == 0


def test_solve_frame_puts_the_pad_on_the_surface_without_going_in():
    hand, row = toy_row()
    surf = box_surface()
    x_ik = np.array([0.3, 0.3])
    t = 20
    y, n = row["mano_right_object_contact_positions"][t, 6], row["mano_right_object_contact_normals"][t, 6]
    x, info = C.solve_frame(hand, (np.zeros(3), BOX_Q), x_ik, None, {"index": (y, n, 0)}, [C.Placed(surf, BOX_P, BOX_Q)])
    hand.update(x)
    pad_z = hand.points(hand.pads["index"])[:, 2]
    assert abs(pad_z.min() - (TOP + C.MARGIN)) < 5e-4
    assert min(hand.points(g)[:, 2].min() for g in hand.geoms) > TOP - 1e-3
    assert hand.pad_normal(hand.pads["index"]) @ np.array([0.0, 0.0, -1.0]) > C.FACING_COS  # the palm side faces the surface
    assert np.all(np.abs(x - x_ik) <= C.MAX_DELTA + 1e-9) and np.all(x >= hand.lower) and np.all(x <= hand.upper)


def test_correct_hand_changes_the_contact_run_and_its_ramps_only():
    hand, row = toy_row()
    X, summary = C.correct_hand(hand, row, "right", [box_surface()])
    X_ik = row["robot_right_finger_joints"]
    changed = np.abs(X - X_ik).max(1) > 1e-7
    assert changed[15:25].all() and not changed[:8].any() and not changed[32:].any()
    for t in range(15, 25):  # touching on the labelled frames (smoothing leaves a little at the run's ends)
        hand.update(X[t])
        assert hand.points(hand.pads["index"])[:, 2].min() - TOP < 0.004
    steps = np.abs(np.diff(X, axis=0)).max()
    assert steps <= 0.6 * np.abs(X - X_ik).max()  # ramped in and out: no one-frame jump to the correction
    assert summary["frames_solved"] == 10 and summary["contact_frames"]["index"] == 10


def test_metrics_report_unknown_penetration_on_open_meshes():
    hand, row = toy_row()
    closed = C.metrics(row, {"right": hand}, [box_surface()])["right"]
    open_ = C.metrics(row, {"right": hand}, [box_surface(watertight=False)])["right"]
    assert isinstance(closed["penetration_all_points_exact"], int)
    assert str(open_["penetration_all_points_exact"]).startswith("UNKNOWN")
    assert closed["index"]["frames"] == 10 and closed["index"]["pad_gap_median_mm"] > 5  # NVIDIA's pose: not touching


def test_driver_step_2b_is_off_by_default_and_runs_once(tmp_path, monkeypatch):
    from pathb.drivers import nvidia_stage23 as D
    calls = []
    monkeypatch.setattr(C, "apply", lambda part, out, xmls: calls.append((part, xmls)) or
                        (Path(out) / "contact_retarget.json").write_text("{}"))
    req = {"robotic": str(tmp_path / "rg")}
    assert C.ENABLED is False
    assert D.contact(req, tmp_path, tmp_path / "processed", "s") is False and not calls
    req["contact_retarget"] = True
    assert D.contact(req, tmp_path, tmp_path / "processed", "s") is True
    assert D.contact(req, tmp_path, tmp_path / "processed", "s") is True  # done: the report marks it
    assert len(calls) == 1 and calls[0][0] == D.partition(tmp_path / "processed", "s")
    assert str(calls[0][1]).endswith("xmls/sharpawave")


def test_sharpa_pad_jacobians_match_finite_differences():
    pytest.importorskip("pinocchio")
    xml = Path(__file__).resolve().parents[1] / ("data/nvidia_d9264a4e/robotic_grounding/" + "source/robotic_grounding/"
                                                  "robotic_grounding/assets/xmls/sharpawave/right_sharpawave.xml")
    if not xml.is_file():
        pytest.skip("NVIDIA mirror not built here")
    hand = C.SharpaHand("right", xml)
    x = (hand.lower + hand.upper) / 2
    wp, wq = np.array([0.1, -0.2, 1.1]), np.array([1.0, 0.0, 0.0, 0.0])
    g = hand.pads["index"]
    hand.update(hand.q(wp, wq, x))
    P = hand.points(g)
    Ja = hand.point_jacobians(g, P)
    for k in hand.finger_dofs["index"]:
        dx = x.copy()
        dx[k] += 1e-6
        hand.update(hand.q(wp, wq, dx))
        assert np.allclose(Ja[:, :, k], (hand.points(g) - P) / 1e-6, atol=1e-5)
