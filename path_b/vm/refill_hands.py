"""A bundle set shipped before stage23.FILL_INVALID was on (2026-11-02), rebuilt with its frames without a hand filled
(James, 2026-11-02: "Yes, rebuild"); the objects untouched.

S4b writes each candidate as its source bundle (s4b_reference.solved: S3d's, S3c's, else P4's) with the hands cleaned
(stage23.clean_hands) and the object arrays set to the candidate's track. While FILL_INVALID was off, the frames without
a hand kept P4's zeros, which a loader that ignores hand_<side>_is_valid poses at the camera. Per bundle of <set> with
such frames: its source in the run of its build (BUILDS.json's, else --default), cleaned with FILL_INVALID off and given
the shipped object arrays, must reproduce the shipped result.npz key by key (floats within SOURCE_ATOL; else the tool
stops, nothing written); the
bundle then becomes that source cleaned with FILL_INVALID on, with the same object arrays. Every other file is copied
unchanged; README.md gets a note, BUILDS.json (if there) a "hands_refilled" entry, SHA256SUMS.txt its sums again.
Writes <set>_refill next to <set>; upload it under <set>'s name (vm/hf_outputs.py) once it is moved into place.

    uv run --project path_b python -I path_b/vm/refill_hands.py <set> [--run <build>=<run> ...] [--default <run>]
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "path_b"))
from pathb import stage23  # noqa: E402
from pathb.bundle import p4_bundle  # noqa: E402
from pathb.stages.s4b_reference import solved  # noqa: E402

OBJECT_KEYS = ("object_to_world_transform", "object_to_camera_transform")
SOURCE_ATOL = 1e-12  # float noise between venvs (evaluation ep20's E0 source: 5e-16, on its frames without a hand)
DAY = "2026-11-02"


def hand_less(data: dict) -> dict[str, int]:
    return {s: int((~np.asarray(data[f"hand_{s}_is_valid"], bool)).sum()) for s in stage23.SIDES}


def source(run: Path, episode: int, name: str) -> Path:
    """The bundle S4b started from (s4b_reference.run)."""
    ctx = SimpleNamespace(root=run / f"episode_{episode:06d}", number=episode)
    v4 = solved(ctx, name)
    return v4["bundle"] if v4 else p4_bundle(ctx, name)


def candidate(src: Path, fill: bool, shipped: dict) -> dict:
    """S4b's candidate from its source: the hands cleaned (FILL_INVALID = fill), the shipped object arrays."""
    keep = stage23.FILL_INVALID
    stage23.FILL_INVALID = fill
    try:
        with np.load(src / "result.npz") as z:
            data = dict(z)
        stage23.clean_hands(data)
    finally:
        stage23.FILL_INVALID = keep
    for k in OBJECT_KEYS:
        data[k] = shipped[k]
    return data


def differing(a: dict, b: dict, atol: float = 0.0) -> list[str]:
    """Keys that only one side has, or whose arrays differ (floats by more than atol, others at all)."""
    def equal(x, y):
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        if np.issubdtype(x.dtype, np.floating):
            return bool(np.allclose(x, y, rtol=0.0, atol=atol, equal_nan=True))
        return bool(np.array_equal(x, y))
    return sorted(k for k in set(a) | set(b) if k not in a or k not in b or not equal(a[k], b[k]))


def deviation(a: dict, b: dict) -> float:
    """The largest absolute difference over the float arrays both have."""
    return max((float(np.nanmax(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)), initial=0.0)) for k in a
                if k in b and np.issubdtype(a[k].dtype, np.floating) and a[k].shape == b[k].shape), default=0.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def refill(set_dir: Path, runs: dict[str, Path], default: Path | None = None) -> list[dict]:
    """<set>_refill from <set> (module doc); returns one row per rebuilt bundle."""
    builds = json.loads((set_dir / "BUILDS.json").read_text()) if (set_dir / "BUILDS.json").is_file() else {}
    plan = []
    for f in sorted(set_dir.rglob("result.npz")):
        with np.load(f) as z:
            shipped = dict(z)
        missing = hand_less(shipped)
        if not any(missing.values()):
            continue
        rel = f.relative_to(set_dir)
        if len(rel.parts) != 4 or not rel.parts[0].startswith("episode_"):  # e.g. alternatives/: no build to look up
            raise SystemExit(f"{rel}: frames without a hand outside the set's main bundles; not handled")
        episode, name = int(rel.parts[0][-6:]), rel.parts[1]
        build = builds.get("episodes", {}).get(str(episode), {}).get("build")
        run = runs.get(build) if build else default
        if run is None:
            raise SystemExit(f"{rel}: no run for build {build!r} (--run {build}=<run>)")
        src = source(run, episode, name)
        then = candidate(src, False, shipped)
        diff = differing(then, shipped, SOURCE_ATOL)
        if diff:
            raise SystemExit(f"{rel}: {src} cleaned as shipped differs in {diff}; nothing written")
        new = candidate(src, True, shipped)
        changed = differing(new, shipped)
        if [k for k in changed if not k.startswith("hand_")]:
            raise SystemExit(f"{rel}: the refill changed {changed}; nothing written")
        row = {"bundle": rel.parent.as_posix(), "build": build, "source": str(src.relative_to(run.parent)),
               "source_reproduces_within": deviation(then, shipped), "missing": missing, "changed_keys": changed}
        for side in stage23.SIDES:
            if not missing[side]:
                continue
            key = f"hand_{side}_wrist_to_world_transform"
            moved = np.abs(new[key].astype(np.float64) - shipped[key].astype(np.float64)).max(axis=(1, 2)) > 0
            invalid = ~np.asarray(shipped[f"hand_{side}_is_valid"], bool)
            edges = np.flatnonzero(np.diff(invalid.astype(int)) != 0)  # step t -> t + 1 into or out of a gap
            step = {n: round(float(np.linalg.norm(np.diff(d[key][:, :3, 3].astype(np.float64), axis=0), axis=1)[edges].max()), 4)
                    if len(edges) else None for n, d in (("before", shipped), ("after", new))}
            row[side] = {"frames_moved": int(moved.sum()), "moved_outside_missing": int((moved & ~invalid).sum()),
                         "max_gap_edge_step_m": step}
        plan.append((rel, new, row))
    dest = set_dir.with_name(set_dir.name + "_refill")
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(set_dir, dest)
    for rel, new, _ in plan:
        np.savez(dest / rel, **new)
    rows = [row for _, _, row in plan]
    note = (f"\n## Changes\n\n- {DAY}: rebuilt with the frames without a hand filled (Path B `stage23.FILL_INVALID`, on by "
            "default since that day). Those frames (`hand_<side>_is_valid` False) held zeros, which a loader that ignores "
            "`is_valid` poses at the camera; they now hold the hand interpolated between the nearest frames with one (position "
            "linear, rotation slerp, fingers linear, held at the clip's ends), `is_valid` still False. Rebuilt: "
            + ", ".join(f"`{r['bundle'].split('/result_slam')[0]}`" for r in rows)
            + ". Object arrays and every other file unchanged.\n")
    if rows and (dest / "README.md").is_file():
        (dest / "README.md").write_text((dest / "README.md").read_text().rstrip("\n") + "\n" + note)
    if rows and builds:
        builds["hands_refilled"] = {"date": DAY, "tool": "path_b/vm/refill_hands.py", "bundles": rows}
        (dest / "BUILDS.json").write_text(json.dumps(builds, indent=1) + "\n")
    sums = dest / "SHA256SUMS.txt"
    if sums.is_file():
        paths = [line.split("  ", 1)[1] for line in sums.read_text().splitlines() if line.strip()]
        sums.write_text("".join(f"{sha256(dest / p)}  {p}\n" for p in paths))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("set", help="a folder of path_b/data/outputs")
    p.add_argument("--run", action="append", default=[], help="<build>=<run>: the run of a build BUILDS.json names")
    p.add_argument("--default", help="the run of the episodes BUILDS.json does not name (a set without one: all)")
    a = p.parse_args()
    out = REPO / "path_b/data/outputs"
    runs = {b: out / r for b, r in (x.split("=", 1) for x in a.run)}
    rows = refill(out / a.set, runs, out / a.default if a.default else None)
    print(json.dumps(rows, indent=1))
    print(f"{len(rows)} bundle(s) rebuilt in {out / (a.set + '_refill')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
