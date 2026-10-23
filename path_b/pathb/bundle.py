"""Copies of reconstruction bundles with every length rescaled (stage 2's rescale_world)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .team import retarget_tools


def bundle_at(root: Path, number: int, name: str) -> Path:
    """An episode's bundle for one object, as P4 exports it (BUNDLES.md layout); S3c completes it.
    root: the episode's run folder (<out>/episode_XXXXXX)."""
    return root / f"p4_export/episode_{number:06d}/{name}/result_slam_gravity_aligned"


def p4_bundle(ctx, name: str) -> Path:
    return bundle_at(ctx.root, ctx.number, name)


def rescaled_copy(src: Path, dest: Path, factor: float, link: bool = True,
                  skip: tuple[str, ...] = ()) -> Path:
    """dest/result.npz = src's with every length times `factor`; the other files are linked (or
    copied, link=False) unchanged: mesh vertices stay put and object_scale carries the factor."""
    data = dict(np.load(src / "result.npz"))
    retarget_tools().rescale_world(data, factor)
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "result.npz" or f.name in skip or (dest / f.name).exists():
            continue
        if link:
            (dest / f.name).symlink_to(f.resolve())
        elif f.is_dir():
            shutil.copytree(f, dest / f.name)
        else:
            shutil.copy2(f, dest / f.name)
    np.savez(dest / "result.npz", **data)
    return dest


PLACEHOLDER_OBJ = "v 0 0 0\nv 0.001 0 0\nv 0 0.001 0\nf 1 2 3\n"  # the writer needs a mesh; removed after


def hand_dirs(root: Path) -> tuple[Path | None, Path | None]:
    """P1d's left and right hand folders, by the handedness P1d settled (hand_tracks.json)."""
    import json
    tracks = json.loads((root / "p1d_hands/hand_tracks.json").read_text())["tracks"]
    out = []
    for right in (False, True):
        ids = [t["object_id"] for t in tracks if bool(t["is_right"]) == right]
        d = root / f"p1d_hands/hamer_aligned_filled/{ids[0]}" if ids else None
        out.append(d if d is not None and d.is_dir() and any(d.iterdir()) else None)
    return out[0], out[1]


def write_v3_bundle(ctx, dest: Path, log: Path, mesh_path: Path | None = None,
                    object_poses_dir: Path | None = None, extra: dict | None = None) -> Path:
    """One object's v3 bundle by the baseline's own writer (v2d_common.result_bundle.
    write_result_bundle, in the baseline host venv): P4's camera in the v3 world, P1d's hands, and
    once S3 exists the object (poses as object_to_camera Transform3d files) and its mesh. Without a
    mesh (P4) the writer gets a placeholder triangle, and its mesh files are removed again."""
    from .baseline import call
    left, right = hand_dirs(ctx.root)
    if mesh_path is None:
        mesh_path = dest.parent / "_placeholder.obj"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        mesh_path.write_text(PLACEHOLDER_OBJ)
        placeholder = True
    else:
        placeholder = False
    if dest.exists():
        shutil.rmtree(dest)
    call("v2d.common.result_bundle:write_result_bundle",
         {"result_dir": dest, "frames_dir": ctx.root / "p1_calibrate/cam_a",
          "intrinsics_path": ctx.root / "p1c_camera/intrinsics.json", "mesh_path": mesh_path,
          "object_poses_dir": object_poses_dir, "object_scale": 1.0,
          "camera_to_world_dir": ctx.root / "p4_export/camera_to_world", "camera_pose_convention": "camera_to_world",
          "left_hand_dir": left, "right_hand_dir": right,
          "source_manifest": {"producer": "path_b v3", "episode": ctx.number}}, log)
    if placeholder:
        for f in dest.iterdir():
            if f.name not in ("result.npz", "manifest.json"):
                f.unlink() if f.is_file() else shutil.rmtree(f)
        mesh_path.unlink()
    if extra:
        data = dict(np.load(dest / "result.npz"))
        data.update({k: np.asarray(v) for k, v in extra.items()})
        np.savez(dest / "result.npz", **data)
    return dest
