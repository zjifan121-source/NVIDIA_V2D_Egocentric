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
