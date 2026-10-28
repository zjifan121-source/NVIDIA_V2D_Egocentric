

def test_reference_trust_reads_s3c_rests(tmp_path):
    """F3, GT-free: a rest whose posed mesh misses most of its mask's rays (ep0, 11, 21, 22: 0.84-0.95), or one
    S3c held beyond the hands' reach, makes the inverted reference untrusted; the episode takes still."""
    import json
    from pathb import selection as SEL

    def report(name, rows):
        d = tmp_path / f"s3c_invert/{name}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "invert_report.json").write_text(json.dumps({"segments": rows}))
    report("cup", [{"kind": "static", "start": 0, "end": 9, "rays_missed": 0.03},
                   {"kind": "grasp", "start": 10, "end": 30}, {"kind": "static", "start": 31, "end": 50, "rays_missed": 0.2}])
    trust = SEL.reference_trust(tmp_path, ["cup"])
    assert trust["trusted"] and SEL.pick_reference(trust)[0] == "inverted"
    report("rack", [{"kind": "static", "start": 0, "end": 50, "rays_missed": 0.9}])
    trust = SEL.reference_trust(tmp_path, ["cup", "rack"])
    assert not trust["trusted"] and trust["rays_missed"]["rack"] == 0.9 and SEL.pick_reference(trust)[0] == "still"
    report("rack", [{"kind": "static", "start": 0, "end": 50, "rays_missed": 0.1,
                     "beyond_reach": {"moved_m": 1.1, "reach_m": 0.26}}])
    trust = SEL.reference_trust(tmp_path, ["cup", "rack"])
    assert not trust["trusted"] and trust["beyond_reach"]["rack"] == [[0, 50]]
