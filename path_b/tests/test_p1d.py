"""P1d (pathb/stages/p1d_hands.py): what the stage does around the baseline's hand modules."""
import json

from pathb.stages import p1d_hands


def test_p1d_rejects_nonfinite_alignments(tmp_path):
    for oid, cam_t in (("2", [0.1, 0.2, 0.5]), ("3", [float("inf")] * 3)):
        f = tmp_path / f"hamer_aligned/{oid}/000007.json"
        f.parent.mkdir(parents=True)
        f.write_text(json.dumps({"camera": {"cam_t": cam_t}, "diagnostics": {"scale": 1.0 if oid == "2" else float("inf")}}))
    assert p1d_hands.reject_nonfinite(tmp_path) == {"2": [], "3": [7]}
    assert (tmp_path / "hamer_aligned/2/000007.json").is_file()
    assert (tmp_path / "hamer_aligned_rejected/3/000007.json").is_file() and not (tmp_path / "hamer_aligned/3/000007.json").exists()


def test_p1d_host_layout(tmp_path):
    f = tmp_path / "2/000005.json"
    f.parent.mkdir()
    f.write_text(json.dumps({"cam_t": [0.1, 0.2, 0.5], "intrinsics": {"fx": 1150.0}, "mano": {}}))
    assert p1d_hands.host_layout(tmp_path) == 1 and p1d_hands.host_layout(tmp_path) == 0
    r = json.loads(f.read_text())
    assert r["camera"] == {"cam_t": [0.1, 0.2, 0.5], "intrinsics": {"fx": 1150.0}} and r["cam_t"] == [0.1, 0.2, 0.5]
    assert p1d_hands.cam_t(r) == [0.1, 0.2, 0.5]


def test_p1d_rejects_depth_spikes(tmp_path):
    z = [0.45] * 10
    z[4] = 0.99  # one frame half a metre too far
    for k, zk in enumerate(z):
        f = tmp_path / f"hamer_aligned/2/{k:06d}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"camera": {"cam_t": [0.1, 0.0, zk + 0.005 * k]}}))  # a slow, real drift
    (tmp_path / "hamer_aligned/3").mkdir()
    assert p1d_hands.reject_spikes(tmp_path) == {"2": [4], "3": []}
    assert (tmp_path / "hamer_aligned_rejected/2/000004.json").is_file()


def test_p1d_hand_depth_keeps_the_valid_median(tmp_path, monkeypatch):
    import numpy as np
    from pathb import geometry
    depth = np.full((20, 20), 0.9)  # background
    depth[5:15, 5:15] = np.linspace(0.40, 0.50, 100).reshape(10, 10)  # the hand
    depth[5:15, 5:9] = 0.0  # 40 % of the hand without depth
    (tmp_path / "depth").mkdir()
    geometry.write_depth(tmp_path / "depth/000000.png", depth)
    hand = np.zeros((20, 20), bool)
    hand[5:15, 5:15] = True
    monkeypatch.setattr(p1d_hands.B, "read_mask", lambda folder, k: hand if folder == "L" else np.zeros_like(hand))
    out = p1d_hands.hand_depth(tmp_path, {"hand_L": "L", "hand_R": "R"}, 1)
    got = geometry.read_depth(out / "000000.png")
    valid = depth[hand & (depth > 0)]
    assert (got[hand] > 0).all() and abs(np.median(got[hand]) - np.median(valid)) < 1e-3
    assert np.allclose(got[~hand], depth[~hand], atol=1e-3)  # outside the hands: unchanged


def test_side_masks_follow_wilor_handedness(tmp_path):
    """Both P1b tracks on the left hand (ep0); WiLoR sees the left and the right: masks/2 gets the left
    hand's P1b pixels inside WiLoR's left box, masks/3 WiLoR's right box (no P1b pixel there). The
    files in masks/ are hard links to P1b's, which stay unchanged."""
    import json
    import os
    import cv2
    import numpy as np
    from pathb.stages import p1d_hands as P
    H, W = 120, 160
    p1b = {name: tmp_path / "p1b" / name for name in ("hand_L", "hand_R")}
    left = np.zeros((H, W), np.uint8)
    left[60:90, 20:50] = 255
    for name, folder in p1b.items():
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "000000.png"), left)  # both tracks found the left hand
    d = tmp_path / "p1d"
    for oid, name in (("2", "hand_L"), ("3", "hand_R")):
        (d / f"masks/{oid}").mkdir(parents=True)
        os.link(p1b[name] / "000000.png", d / f"masks/{oid}/000000.png")
    (d / "wilor_raw").mkdir()
    (d / "wilor_raw/000000.json").write_text(json.dumps([
        {"is_right": False, "score": 0.9, "bbox": {"x0": 18, "y0": 58, "x1": 52, "y1": 92}},
        {"is_right": True, "score": 0.8, "bbox": {"x0": 100, "y0": 50, "x1": 140, "y1": 90}}]))
    votes = {"hand_L": {"right": 0, "left": 5}, "hand_R": {"right": 1, "left": 4}}
    counts = P.side_masks(d, p1b, 1, votes)
    m2 = cv2.imread(str(d / "masks/2/000000.png"), 0) > 0
    m3 = cv2.imread(str(d / "masks/3/000000.png"), 0) > 0
    assert m2[70, 30] and not m2[70, 120] and m2.sum() == (left > 0).sum()
    assert m3[70, 120] and not m3[70, 30] and abs(m3.sum() - 36 * 36) < 80
    assert counts["2"]["wilor_and_p1b"] == 1 and counts["3"]["wilor_box"] == 1
    assert (cv2.imread(str(p1b["hand_R"] / "000000.png"), 0) > 0).sum() == (left > 0).sum()  # P1b untouched
