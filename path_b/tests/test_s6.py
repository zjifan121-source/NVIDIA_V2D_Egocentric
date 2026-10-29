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


def test_v4_s6_settings_long_schedule_and_groups(tmp_path, monkeypatch):
    """V4 S6: the 250M schedule drops the 60M compression only; no override may replace the action or
    termination group; a run's Hydra record must show residual_hand_pose and tracking."""
    import pytest
    from pathb import policy
    monkeypatch.delenv("PATHB_OBJECT_STOP", raising=False)
    steps, short = policy.settings()
    long_steps, long = policy.settings(long=True)
    th = policy.short_thresholds()
    assert steps == th[-1] == 60_002_304 and long_steps == policy.LONG_STEPS == 250_003_456
    assert th[0] == 4_718_592 and all(t % 4096 == 0 for t in th) and th == sorted(th)
    assert any(o.startswith("curriculum.thresholds=") for o in short) and not any(o.startswith("curriculum.") for o in long)
    assert "task.motion_speed=1.0" in long and "sim.njmax=1536" in long and "sim.nconmax=384" in long
    assert not any(o.startswith("termination.wrist_position") for o in short + long)  # FlashCHORD's 0.15 m
    monkeypatch.setenv("PATHB_OBJECT_STOP", "0.05")
    assert "termination.object_position.threshold=0.05" in policy.settings()[1]
    for bad in ("action=hand_pose", "override /action=hand_pose", "+termination=none"):
        with pytest.raises(ValueError):
            policy.check_overrides([bad])
    policy.check_overrides(["termination.wrist_position.threshold=0.3"])  # a value inside the group is fine
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra/hydra.yaml").write_text("hydra:\n  runtime:\n    choices:\n      action: residual_hand_pose\n"
                                                "      termination: tracking\n      sim: rl\n")
    assert policy.chosen_groups(tmp_path) == policy.GROUPS


def test_v4_s6_long_pass_trains_the_selected_candidate(tmp_path, monkeypatch):
    import json
    from pathb import runner
    from pathb.stages import s6_select, s6_train
    root = tmp_path / "episode_000041"
    for cand in ("inverted", "still"):
        (root / f"s5_task/{cand}").mkdir(parents=True)
        (root / f"s5_task/{cand}/task.json").write_text(json.dumps({"partition": f"/p/{cand}", "mode": "A"}))
    ctx = runner.Context({"episode": 41, "split": "public", "objects": [{"name": "white_pot"}]}, root, {})
    monkeypatch.delenv("PATHB_S6_LONG", raising=False)
    assert [j[0] for j in s6_train.jobs(ctx)] == ["inverted", "still"]
    (root / "s6_select").mkdir()
    (root / "s6_select/selection.json").write_text(json.dumps({"candidate": "inverted", "task": "inverted"}))
    monkeypatch.setenv("PATHB_S6_LONG", "2")
    jobs = s6_train.jobs(ctx)
    assert [(j[0], j[2], j[3]) for j in jobs[2:]] == [("inverted_250m_s42", True, 42), ("inverted_250m_s43", True, 43)]
    assert s6_select.task_of("inverted_250m_s43") == "inverted" and s6_select.task_of("still") == "still"


def test_a_candidate_flashchord_refuses_does_not_stop_the_others(tmp_path, monkeypatch):
    import json
    import pytest
    from pathb import policy, runner
    from pathb.stages import s6_train
    root = tmp_path / "episode_000031"
    for cand in ("inverted", "still"):
        (root / f"s5_task/{cand}").mkdir(parents=True)
        (root / f"s5_task/{cand}/task.json").write_text(json.dumps({"partition": f"/p/{cand}", "mode": "A"}))
    ctx = runner.Context({"episode": 31, "split": "public", "objects": [{"name": "mini_sweeper"}]}, root, {})
    monkeypatch.delenv("PATHB_S6_LONG", raising=False)

    def train(partition, run, name, long=False, seed=42):
        if partition == "/p/inverted":
            raise RuntimeError("exit 1: python scripts/train_flash_sac.py ...; see train.log")
        return {"seconds": 3600, "steps": 60_002_304, "resumed_from": None, "nefc_overflow_lines": 0}

    monkeypatch.setattr(policy, "train", train)
    r = s6_train.run(ctx)
    assert list(r.outputs) == ["still"]
    assert [m.status for m in r.metrics if m.name == "inverted: trained"] == ["FAIL"]
    monkeypatch.setattr(policy, "train", lambda *a, **k: train("/p/inverted", None, None))
    with pytest.raises(RuntimeError, match="no candidate trained"):
        s6_train.run(ctx)


def test_pinned_thresholds_count_failure_frames():
    ref = moving()
    observed = np.ones(ref.shape[:2], bool)
    off = ref.copy()
    off[:, 0, 0, 3] += np.linspace(0, 0.14, len(ref))  # drifting away: frame 0 aligns, the rest does not
    G = [symmetry.group("mini_dust_pan")]
    loose = selection.observation_cost(off, ref, observed, G, max_position=0.10)
    tight = selection.observation_cost(off, ref, observed, G, max_position=0.05)
    assert tight["termination_frames"] > loose["termination_frames"] > 0
    assert tight["first_termination_frame"] < loose["first_termination_frame"]
