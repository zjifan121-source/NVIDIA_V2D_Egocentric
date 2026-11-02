"""S3e's keep rule re-applied without the GPU (Recon P4): a copy of <src run>'s episodes into <dst run> (S1 linked;
S3a2 .. S3d and s3e_refine/ with the GPU pass's records copied), then S3e (nvidia-smi hidden, so the stage
re-applies the current rule to the records), S4a and S4b. For a GPU pass made under an older rule, e.g. the
evaluation episodes' p4c pass (2026-11-02) under p4s's scorer rule (`ace62402`): run from a checkout at that
commit or later. No waterfall, so no ground truth is read; on public episodes run `run.py waterfall` after.

    uv run --project path_b python -I path_b/vm/s3e_regate.py <src run> <dst run> <episode> [<episode> ...]
"""
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "path_b"))
from pathb import runner  # noqa: E402

S1 = ("p0_ingest", "p1_calibrate", "p1b_masks", "p1c_camera", "p1d_hands", "p2_depth", "p3_scale_bridge", "p4_export",
      "s3a2_fuse_scan")
COPIED = ("s3a2_register", "s3a_mesh", "s3b_segments", "s3c_invert", "s3d_settle", "s3e_refine")
STAGES = ("s3e_refine", "s4a_phases", "s4b_reference")

which = shutil.which
shutil.which = lambda name, *a, **k: None if name == "nvidia-smi" else which(name, *a, **k)  # S3e: re-apply only


def main() -> int:
    out = REPO / "path_b/data/outputs"
    src, dst = out / sys.argv[1], out / sys.argv[2]
    for n in (int(x) for x in sys.argv[3:]):
        e = f"episode_{n:06d}"
        a, b = src / e, dst / e
        if not (a / "s3e_refine").is_dir():
            print(f"episode {n}: no s3e_refine/ in {src.name}, skipped", flush=True)
            continue
        if b.exists():
            shutil.rmtree(b)
        b.mkdir(parents=True)
        for s in S1:
            if (a / s).exists():
                os.symlink((a / s).resolve(), b / s)
        for s in COPIED:
            shutil.copytree(a / s, b / s, symlinks=True)
        for f in ("run.json", "manual_inputs.json"):
            if (a / f).is_file():
                shutil.copy2(a / f, b / f)
        runner.run_episode(n, dst, stages=STAGES, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
