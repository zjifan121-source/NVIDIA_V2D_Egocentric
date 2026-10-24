"""S6 selection rules (pathb/selection.py): the observation cost without ground truth, and the picks."""
import numpy as np
from scipy.spatial.transform import Rotation

from pathb import selection, symmetry


def moving(n=40):
    """(n, 1, 4, 4): the object rests, is carried 30 cm up and over, and rests again."""
    T = np.tile(np.eye(4), (n, 1, 1, 1))
    s = np.clip((np.arange(n) - 10) / 20, 0, 1)
    T[:, 0, :3, 3] = np.stack([0.3 * s, 0.05 * np.sin(np.pi * s), 0.1 * np.sin(np.pi * s)], axis=1)
    T[:, 0, :3, :3] = Rotation.from_rotvec(np.outer(s, [0, 0, 0.8])).as_matrix()
    return T


def test_cost_is_zero_for_the_observation_and_ignores_a_world_offset():
    ref = moving()
    observed = np.ones(ref.shape[:2], bool)
    G = [symmetry.group("mini_dust_pan")]
    assert selection.observation_cost(ref, ref, observed, G)["cost_m"] == 0
    W = np.eye(4)
    W[:3, :3] = Rotation.from_rotvec([0, 0, 1.0]).as_matrix()
    W[:3, 3] = [1.0, -2.0, 1.0]  # the task's own world (runner shift, other yaw): frame-0 alignment removes it
    c = selection.observation_cost(W @ ref, ref, observed, G)
    assert c["cost_m"] < 1e-9 and c["termination_frames"] == 0


def test_jumps_are_termination_frames_and_unobserved_frames_do_not_count():
    ref = moving()
    roll = ref.copy()
    roll[20:25, 0, 2, 3] += 0.3  # 30 cm off on five carry frames
    observed = np.ones(ref.shape[:2], bool)
    c = selection.observation_cost(roll, ref, observed, [symmetry.group("mini_dust_pan")])
    assert c["termination_frames"] == 5 and np.isclose(c["cost_m"], 5 * selection.PENALTY_M / len(ref))
    observed[20:25] = False  # S3c did not trust those frames
    assert selection.observation_cost(roll, ref, observed, [symmetry.group("mini_dust_pan")])["cost_m"] < 1e-9


def test_symmetric_equivalents_cost_nothing():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(400, 3)) * [0.04, 0.04, 0.05]  # a cup-like cloud, axis z
    G = symmetry.group("blue_cup", v)
    ref = moving()
    roll = ref.copy()  # frame 0 as reset (the reference's); then the cup spins about its own axis
    for t in range(1, len(ref)):
        roll[t, 0] = ref[t, 0] @ G[t % len(G)]
    assert selection.observation_cost(roll, ref, np.ones(ref.shape[:2], bool), [G])["cost_m"] < 1e-6


def test_still_loses_to_a_moving_rollout_without_ground_truth():
    ref = moving()
    observed = np.ones(ref.shape[:2], bool)
    still = np.repeat(ref[:1], len(ref), axis=0)
    noisy = ref.copy()
    noisy[:, 0, :3, 3] += np.random.default_rng(1).normal(0, 0.01, (len(ref), 3))
    G = [symmetry.group("mini_dust_pan")]
    costs = {"inverted": selection.observation_cost(noisy, ref, observed, G),
             "still": selection.observation_cost(still, ref, observed, G)}
    assert selection.pick_evaluation(costs)[0] == "inverted"


def test_public_pick_breaks_ties_by_mppe():
    scores = {"keyframe": {"AUC": 0.5004, "MPPE cm": 6.0}, "inverted": {"AUC": 0.5, "MPPE cm": 5.0},
              "still": {"AUC": 0.40, "MPPE cm": 9.0}}
    assert selection.pick_public(scores)[0] == "inverted"
    scores["keyframe"]["AUC"] = 0.52
    assert selection.pick_public(scores)[0] == "keyframe"


def test_two_objects_add_rpe():
    scores = {"keyframe": {"AUC": 0.5, "MPPE cm": 5.0, "RPE cm": 9.0}, "inverted": {"AUC": 0.5005, "MPPE cm": 6.0, "RPE cm": 3.0}}
    assert selection.pick_public(scores)[0] == "keyframe" and selection.pick_public(scores, rpe=True)[0] == "inverted"
    ref = np.concatenate([moving(), moving()], axis=1)
    ref[:, 1, 0, 3] += 0.2  # the second object 20 cm beside the first
    roll = ref.copy()
    roll[10:, 1, 1, 3] += 0.03  # the second object drifts 3 cm from the first after frame 10
    observed = np.ones(ref.shape[:2], bool)
    G = [symmetry.group("mini_dust_pan")] * 2
    plain = selection.observation_cost(roll, ref, observed, G)["cost_m"]
    rel = selection.observation_cost(roll, ref, observed, G, relative=True)["cost_m"]
    assert rel > plain > 0
