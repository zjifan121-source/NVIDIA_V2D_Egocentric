"""runner.STAGES is the spec's stage list, and a stage not built yet is skipped without a report."""
from pathb import runner

SPEC_STAGES = ("p0_ingest p1_calibrate p2_depth p1b_masks p1c_camera p1d_hands p3_scale_bridge p4_export "
               "s3a2_fuse_scan s3a2_register s3b_segments s3b_track s3c_invert s3d_settle "
               "s4a_phases s4b_reference s4c_retarget s5_task "
               "s6_train s6_eval s6_select s7_pack s23_load_retarget").split()  # V4_BUILD_SPEC.md §1


def test_stage_order_is_the_spec():
    assert list(runner.STAGES) == SPEC_STAGES


def test_unbuilt_stage_skips_without_a_report(tmp_path, monkeypatch):
    real = runner.importlib.util.find_spec
    monkeypatch.setattr(runner.importlib.util, "find_spec",
                        lambda name, *a: None if name == "pathb.stages.s5_task" else real(name, *a))
    ctx = runner.Context({"episode": 40, "split": "public", "objects": [{"name": "mini_dust_pan"}]},
                         tmp_path / "episode_000040", {})
    report, ran = runner.run_stage("s5_task", ctx, force=True)
    assert (report["status"], ran) == ("SKIP", False)
    assert "not built yet" in report["skip_reason"]
    assert not (ctx.root / "s5_task").exists()
