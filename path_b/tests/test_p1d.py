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
