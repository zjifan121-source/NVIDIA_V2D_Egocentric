"""pathb/phases.py: windows from synthetic wrist/object tracks, and ep41's first reach.

fixtures/ep41_white_pot_tracks.npz: the team's ep41 white_pot bundle (hub, result.npz sha256
0ee136fdf860c476..., the one P3 measured in run 4c2eac31): object track, metric mesh vertices, both
wrists in the world and their validity.
"""
from pathlib import Path

import numpy as np

from pathb import phases as P
from pathb.team import retarget_tools

FIXTURE = Path(__file__).with_name("fixtures") / "ep41_white_pot_tracks.npz"


def still_object(n: int) -> np.ndarray:
    return np.tile(np.eye(4), (n, 1, 1))


def cube(half: float = 0.05) -> np.ndarray:
    g = np.linspace(-half, half, 5)
    return np.array([[x, y, z] for x in g for y in g for z in g])


def wrist_track(n: int, close: range) -> np.ndarray:
    w = np.tile([0.0, 0.0, 0.5], (n, 1))  # 45 cm above the cube's top
    w[list(close)] = [0.0, 0.0, 0.08]  # 3 cm above it
    return w


def test_reach_carry_release_windows():
    n = 100
    T = still_object(n)
    T[40:61, 0, 3] = np.linspace(0, 0.3, 21)  # carried 30 cm
    T[61:, 0, 3] = 0.3
    w = wrist_track(n, range(30, 66))
    w[40:66, 0] += T[40:66, 0, 3]  # the hand moves with the object
    r = P.phases(P.near_frames(T, cube(), [(w, np.ones(n, bool)), (np.full((n, 3), 9.0), np.ones(n, bool))]), T)
    assert (r["first_reach"], r["last_release"]) == (30, 65)
    assert (r["rest_before"], r["carry"], r["rest_after"]) == ([0, 24], [25, 70], [71, 99])
    assert not r["static"] and P.length(r["carry"]) == 46


def test_untouched_static_object_is_rest_all_clip():
    n = 50
    T = still_object(n)
    T[:, 2, 3] += np.random.default_rng(0).normal(0, 0.002, n)  # 2 mm jitter
    r = P.phases(P.near_frames(T, cube(), [(np.full((n, 3), 1.0), np.ones(n, bool))]), T)
    assert r["first_reach"] is None and r["carry"] is None and r["rest_after"] is None
    assert r["rest_before"] == [0, n - 1] and r["static"]


def test_invalid_wrist_frames_do_not_count():
    n = 60
    w = wrist_track(n, range(10, 50))
    valid = np.ones(n, bool)
    valid[10:20] = False
    r = P.phases(P.near_frames(still_object(n), cube(), [(w, valid)]), still_object(n))
    assert (r["first_reach"], r["last_release"]) == (20, 49)
    assert (r["rest_before"], r["carry"], r["rest_after"]) == ([0, 14], [15, 54], [55, 59])


def test_windows_clip_at_the_clip_ends():
    n = 30
    r = P.phases(P.near_frames(still_object(n), cube(), [(wrist_track(n, range(2, 28)), np.ones(n, bool))]),
                 still_object(n))
    assert (r["rest_before"], r["carry"], r["rest_after"]) == (None, [0, 29], None)


def test_ep41_first_reach_is_19():
    f = np.load(FIXTURE)
    wrists = [(f["wrist_left"], f["valid_left"]), (f["wrist_right"], f["valid_right"])]
    r = P.phases(P.near_frames(f["object_to_world"], f["vertices"], wrists), f["object_to_world"])
    assert r["first_reach"] == 19  # P3's report, run 4c2eac31
    assert r["rest_before"] == [0, 13] and r["carry"][0] == 14
    # same answer as the team's stage 2 rest hold
    data = {"object_to_world_transform": f["object_to_world"]}
    for side in ("left", "right"):
        H = np.tile(np.eye(4), (len(f[f"wrist_{side}"]), 1, 1))
        H[:, :3, 3] = f[f"wrist_{side}"]
        data[f"hand_{side}_wrist_to_world_transform"], data[f"hand_{side}_is_valid"] = H, f[f"valid_{side}"]
    assert retarget_tools().reach_frame(data, f["vertices"], P.REACH_CM) == 19
