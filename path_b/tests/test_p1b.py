"""P1b (pathb/stages/p1b_masks.py): hand masks without far pieces."""
import numpy as np

from pathb.stages import p1b_masks


def test_hand_blob_drops_far_pieces_and_follows_the_hand():
    m = np.zeros((100, 200), bool)
    m[40:60, 20:40] = True     # the hand
    m[40:60, 43:46] = True     # a finger cut off by an object, 3 px away
    m[40:70, 150:190] = True   # a leak onto another region, larger than the hand
    first = p1b_masks.hand_blob(m, None, reach=8)
    assert first[50, 160] and not first[50, 30]  # no history: the largest piece
    prev = np.zeros_like(m)
    prev[42:58, 22:38] = True  # the previous frame's hand
    got = p1b_masks.hand_blob(m, prev, reach=8)
    assert got[50, 30] and got[50, 44] and not got[50, 160]


def test_a_track_that_jumps_onto_another_object_loses_those_frames(tmp_path):
    """lost_frames: obj_0 rests, obj_1 moves; from frame 10 obj_0's track jumps onto obj_1 and follows it (ep0's
    pot onto the lifted lid): obj_0 loses 10-19, obj_1 keeps every frame; object_mask returns None there."""
    import cv2
    from pathb import baseline as B
    B.lost_frames.cache_clear()
    root = tmp_path / "episode_000000"
    for k in (0, 1, "hand_L"):
        (root / f"p1b_masks/{k if isinstance(k, str) else f'obj_{k}'}").mkdir(parents=True)
    for t in range(20):
        rest, moving = np.zeros((80, 120), np.uint8), np.zeros((80, 120), np.uint8)
        rest[20:40, 10:30] = 255
        moving[40:60, 30 + 3 * t:50 + 3 * t] = 255
        cv2.imwrite(str(root / f"p1b_masks/obj_0/{t:06d}.png"), moving if t >= 10 else rest)
        cv2.imwrite(str(root / f"p1b_masks/obj_1/{t:06d}.png"), moving)
    lost = B.lost_frames(str(root))
    assert lost["obj_0"] == frozenset(range(10, 20)) and lost["obj_1"] == frozenset()
    assert B.object_mask(root, 0, 12) is None and B.object_mask(root, 0, 5) is not None and B.object_mask(root, 1, 12) is not None


def _hands(root, frames):
    """p1b_masks/hand_L, hand_R from (left, right) mask pairs per frame."""
    import cv2
    for t in ("hand_L", "hand_R"):
        (root / t).mkdir(parents=True, exist_ok=True)
    for k, (L, R) in enumerate(frames):
        cv2.imwrite(str(root / f"hand_L/{k:06d}.png"), L.astype(np.uint8) * 255)
        cv2.imwrite(str(root / f"hand_R/{k:06d}.png"), R.astype(np.uint8) * 255)


def test_hand_check_flags_one_hand_tracked_twice_and_picks_the_wrong_handed_track(tmp_path):
    """§14.5 G-A: both prompts on the left hand (ep21) are not distinct; the right track, which WiLoR votes
    left, is the one to re-seed. Two hands apart with own-side votes pass."""
    A = np.zeros((60, 80), bool)
    A[10:30, 5:25] = True
    B_ = np.zeros_like(A)
    B_[10:30, 50:70] = True
    _hands(tmp_path / "same", [(A, A)] * 4)
    got = p1b_masks.hand_check(tmp_path / "same", 4, {"hand_L": {"left": 9, "right": 1}, "hand_R": {"left": 7, "right": 2}},
                               {"hand_L": 0, "hand_R": 3})
    assert not got["distinct"] and got["reseed"] == "hand_R" and got["median_iou"] == 1.0
    _hands(tmp_path / "apart", [(A, B_)] * 4)
    got = p1b_masks.hand_check(tmp_path / "apart", 4, {"hand_L": {"left": 9, "right": 1}, "hand_R": {"left": 0, "right": 0}},
                               {"hand_L": 0, "hand_R": 0})
    assert got["distinct"] and got["reseed"] is None and got["median_iou"] == 0.0
    got = p1b_masks.hand_check(tmp_path / "apart", 4, {"hand_L": {"left": 2, "right": 8}, "hand_R": {"left": 0, "right": 5}},
                               {"hand_L": 0, "hand_R": 0})
    assert not got["distinct"] and got["reseed"] == "hand_L"  # apart, but the left track is on a right hand (ep44)


def test_reseed_instance_fills_the_wilor_box_and_avoids_the_other_hand():
    from pathb.drivers import sam3_masks as S
    A = np.zeros((60, 80), bool)
    A[10:30, 5:25] = True
    B_ = np.zeros_like(A)
    B_[10:30, 50:70] = True
    out = {"out_obj_ids": [1, 2], "out_probs": [0.9, 0.6], "out_binary_masks": np.stack([A, B_])}
    got = S.matching(out, 0.35, 0.01, A.size, [48, 8, 72, 32], A, 0.1, 0.3)
    assert got is not None and got[0] == 2 and got[3] > 0.6  # the right box's instance, not the likelier left hand
    assert S.matching(out, 0.35, 0.01, A.size, [3, 8, 27, 32], A, 0.1, 0.3) is None  # the box is the other hand's


def test_cut_blanks_repeats_of_the_other_hand_and_stops_at_a_switch():
    from pathb.drivers import sam3_masks as S
    A = np.zeros((60, 80), bool)
    A[10:30, 5:25] = True
    B_ = np.zeros_like(A)
    B_[10:30, 50:70] = True
    masks = {k: (A if k in (3, 6, 7, 8, 9) else B_) for k in range(10)}  # one stray frame, then a switch at 6
    other = {k: A for k in range(10)}
    kept, switch, blanked = S.cut(masks, other, 0, 10, 1, 0.1, 3)
    assert switch == 6 and sorted(kept) == [0, 1, 2, 4, 5] and blanked == [3, 6, 7, 8]
    kept, switch, blanked = S.cut(masks, other, 5, -1, -1, 0.1, 3)  # backward from 5
    assert switch is None and sorted(kept) == [0, 1, 2, 4, 5] and blanked == [3]


class _FakeSam:
    """Two hands: left = A on every frame; the right instance (id 2) is B, but propagated from a seed before
    frame 5 it jumps onto A at 5 (SAM's identity switch, ep21); seeded at 5 or later it stays on B."""
    def __init__(self, A, B_, n):
        self.A, self.B, self.n, self.seed = A, B_, n, {}

    def handle_request(self, r):
        if r["type"] == "start_session":
            return {"session_id": "s"}
        if r["type"] == "add_prompt":
            self.seed[r["session_id"]] = r["frame_index"]
            return {"outputs": {"out_obj_ids": [1, 2], "out_probs": [0.9, 0.8],
                                "out_binary_masks": np.stack([self.A, self.B])}}
        return {}

    def handle_stream_request(self, r):
        f, step = r["start_frame_index"], 1 if r["propagation_direction"] == "forward" else -1
        seeded = self.seed[r["session_id"]]
        for k in range(f, self.n if step == 1 else -1, step):
            right = self.A if (seeded < 5 and k >= 5) else self.B
            yield {"frame_index": k, "outputs": {"out_obj_ids": [1, 2], "out_binary_masks": np.stack([self.A, right])}}


def test_reseed_rebuilds_a_track_that_switched_to_the_other_hand(tmp_path):
    from PIL import Image
    from pathb.drivers import sam3_masks as S
    n = 12
    A = np.zeros((60, 80), bool)
    A[10:30, 5:25] = True
    B_ = np.zeros_like(A)
    B_[10:30, 50:70] = True
    (tmp_path / "hand_L").mkdir()
    for k in range(n):
        Image.fromarray(A.astype(np.uint8) * 255).save(tmp_path / f"hand_L/{k:06d}.png")
    req = {"min_prob": 0.35, "min_area": 0.01, "reseed": {
        "track": "hand_R", "other": "hand_L", "texts": ["right hand", "hand"], "boxes": {str(k): [48, 8, 72, 32] for k in range(n)},
        "max_iou": 0.1, "box_match": 0.3, "switch_run": 3, "max_segments": 8}}
    got = S.reseed(_FakeSam(A, B_, n), lambda old=None: "s", req, n, tmp_path, A.size, (80, 60))
    assert [(s["seed_frame"], s["start"], s["end"], s["switch"]) for s in got["segments"]] == [(0, 0, 4, 5), (5, 5, 11, None)]
    assert got["present_frames"] == list(range(n))
    for k in range(n):
        assert (np.asarray(Image.open(tmp_path / f"hand_R_reseed/{k:06d}.png")) > 0)[20, 60]
