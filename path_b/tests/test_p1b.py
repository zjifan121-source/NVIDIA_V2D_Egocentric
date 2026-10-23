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
