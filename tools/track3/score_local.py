"""Score Track 3 predictions on the 20 public episodes with the kit's official metric code.

The Kaggle scorer compares a submission against a hidden "solution". This module builds the
same kind of solution for the PUBLIC episodes (ground-truth poses + scanned meshes) and runs
the unmodified metric files in assets/data/v2d_submission_kit/metric_code/track_3/.

What is approximated: the hidden solution's per-object anchor point and symmetry axis are
confidential. Here the anchor is the scan's surface centroid and the symmetry axis is fitted
from the scan (cups: axial, lid: half-turn, as the kit README states). Absolute numbers can
therefore differ slightly from Kaggle; comparisons between methods are what this is for.

Usage (repo root):
    uv run --project tools python tools/track3/score_local.py --self-test
    uv run --project tools python tools/track3/score_local.py --results <dir> --meshes <dir>
<dir> layouts: --results holds evaluator output episode_XXXXXX.parquet (or .npy, see
load_predictions); --meshes holds episode_XXXXXX/<object>.{glb,obj,ply}.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "assets/data/v2d_submission_kit"
DATA = REPO / "assets/data/track_3/public"
MANIFEST = Path(__file__).with_name("episodes.json")
sys.path.insert(0, str(KIT))

from v2dlb.mesh_budget import budget_mesh  # noqa: E402
from v2dlb.vertices import mesh_path_for, sample_vertices  # noqa: E402

POSE_METRICS = {  # metric file -> (display name, higher is better)
    "AUC": ("AUC", True),
    "SP-SR": ("SP-SR", True),
    "MP-SR": ("MP-SR", True),
    "RPE": ("RPE cm", False),
    "MPPE": ("MPPE cm", False),
}
POSE_COLUMNS = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
# Kit README: cups are axially symmetric, the lid is symmetric under a half-turn.
SYMMETRY_ORDER = {"beige_cup": 360, "blue_cup": 360, "white_pot_lid": 2}
REFERENCE_FACE_BUDGET = 10_000
LOCAL_COMMIT_URL = "https://github.com/local/scoring/commit/0000000"


@lru_cache(maxsize=None)
def metric_module(name: str):
    spec = importlib.util.spec_from_file_location(f"t3_{name}", KIT / f"metric_code/track_3/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_episodes() -> list[dict]:
    return [e for e in json.loads(MANIFEST.read_text())["episodes"] if e["split"] == "public"]


def ground_truth(episode: int) -> np.ndarray:
    """(frames, objects, 7) world_T_object poses, xyzw quaternions, in per-frame slot order."""
    table = pd.read_parquet(DATA / f"data/chunk-000/episode_{episode:06d}.parquet")
    frames = table["observation.objects"].tolist()
    if not all(o["visible"] for frame in frames for o in frame):
        raise ValueError(f"episode {episode}: invisible ground-truth frames are not handled")
    wxyz = np.array([[o["pose"] for o in frame] for frame in frames], dtype=np.float64)
    return np.concatenate([wxyz[..., :3], wxyz[..., 4:], wxyz[..., 3:4]], axis=-1)


def scan_path(name: str) -> Path:
    return mesh_path_for(DATA / "mesh", name)


def symmetry_axis(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Fit the symmetry axis: the PCA axis whose eigenvalue stands apart from the other two."""
    points = metric_module("CD-O")._sample_mesh_surface(vertices, faces, 20_000, 0)
    centred = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centred.T @ centred / len(centred))
    gaps = [abs(values[0] - values[1]) + abs(values[0] - values[2]),
            abs(values[1] - values[0]) + abs(values[1] - values[2]),
            abs(values[2] - values[0]) + abs(values[2] - values[1])]
    return vectors[:, int(np.argmax(gaps))]


@lru_cache(maxsize=None)
def reference_object(name: str) -> dict:
    cdo = metric_module("CD-O")
    path = scan_path(name)
    v, f = budget_mesh(path, REFERENCE_FACE_BUDGET, REFERENCE_FACE_BUDGET)
    anchor, _ = cdo._surface_moments(v, f)
    order = SYMMETRY_ORDER.get(name, 1)
    axis = symmetry_axis(v, f) if order > 1 else np.zeros(3)
    return {"vertices": v, "faces": f, "anchor": anchor, "points": sample_vertices(path),
            "axis": axis, "order": order}


def symmetry_residual_cm(name: str) -> float:
    """How well the fitted axis maps the scan onto itself (cm); small means the fit is right."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation
    ref = reference_object(name)
    angle = 2 * np.pi / (ref["order"] if ref["order"] < 360 else 8)
    pts = ref["points"] - ref["anchor"]
    turned = Rotation.from_rotvec(ref["axis"] * angle).apply(pts)
    return float(cKDTree(pts).query(turned)[0].mean() * 100)


def geometry_payload(names: list[str]) -> str:
    auc = metric_module("AUC")
    arrays = {"names": np.array(names)}
    for i, name in enumerate(names):
        ref = reference_object(name)
        arrays.update({f"v{i}": ref["vertices"], f"f{i}": ref["faces"], f"anchor{i}": ref["anchor"],
                       f"points{i}": ref["points"], f"axis{i}": ref["axis"], f"order{i}": np.int64(ref["order"])})
    return auc._encode_archive(**arrays)


def row_id(episode: int, frame: int, slot: int) -> str:
    return f"track_3/e{episode:06d}/w000/f{frame:06d}/o{slot}"


def pose_solution(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for e in episodes:
        gt = ground_truth(e["episode"])
        for frame in range(gt.shape[0]):
            for slot, obj in enumerate(e["objects"]):
                rows.append([row_id(e["episode"], frame, slot), e["episode"], 0, frame, slot, obj["name"],
                             *gt[frame, slot]])
    solution = pd.DataFrame(rows, columns=["row_id", "episode_index", "world_index", "frame_index",
                                           "object_slot", "object_name", *["ref_" + c for c in POSE_COLUMNS]])
    names = sorted({o["name"] for e in episodes for o in e["objects"]})
    solution["reference_geometry_b64"] = ""
    solution.loc[0, "reference_geometry_b64"] = geometry_payload(names)
    return solution


def cdo_solution(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for e in episodes:
        for slot, obj in enumerate(e["objects"]):
            v, f = budget_mesh(scan_path(obj["name"]), 4096, 4096)
            for array, values, tag in (("object_mesh_vertices", v, 0), ("object_mesh_faces", f, 1)):
                for i, xyz in enumerate(values):
                    rows.append([f"t3m_{e['episode']:06d}_{obj['name']}_{tag}_{i:06d}", e["episode"], slot,
                                 obj["name"], array, i, *map(float, xyz)])
    return pd.DataFrame(rows, columns=["row_id", "episode_index", "object_slot", "object_name", "array_name",
                                       "point_index", "ref_x", "ref_y", "ref_z"])


def mesh_payload(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    v, f = budget_mesh(path, 4096, 4096)
    return metric_module("AUC")._encode_archive(vertices=v, faces=f), v, f


def build_submission(episodes: list[dict], poses: dict[int, np.ndarray],
                     meshes: dict[tuple[int, str], Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combined pose submission + derived CD-O submission, as the kit packer would produce."""
    pose_rows, cdo_rows = [], []
    for e in episodes:
        ep, achieved = e["episode"], poses[e["episode"]]
        if achieved.shape != (e["num_frames"], len(e["objects"]), 7):
            raise ValueError(f"episode {ep}: poses {achieved.shape}, expected "
                             f"{(e['num_frames'], len(e['objects']), 7)}")
        for slot, obj in enumerate(e["objects"]):
            payload, v, f = mesh_payload(meshes[(ep, obj["name"])])
            for frame in range(achieved.shape[0]):
                pose_rows.append([row_id(ep, frame, slot), *achieved[frame, slot], payload if frame == 0 else ""])
            for tag, values in ((0, v), (1, f)):
                for i, xyz in enumerate(values):
                    cdo_rows.append([f"t3m_{ep:06d}_{obj['name']}_{tag}_{i:06d}", *map(float, xyz)])
    submission = pd.DataFrame(pose_rows, columns=["row_id", *POSE_COLUMNS, "object_mesh_b64"])
    for column in ("eval_script_sha256", "eval_code_sha256", "checkpoint_sha256",
                   "packer_script_sha256", "uploader_script_sha256"):
        submission[column] = "local"
    submission["code_commit_url"] = LOCAL_COMMIT_URL
    cdo = pd.DataFrame(cdo_rows, columns=["row_id", "x", "y", "z"])
    cdo["code_commit_url"] = LOCAL_COMMIT_URL
    return submission, cdo


def _run_metric(name: str, solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    return metric_module(name).score(solution, submission, "row_id")


def score(episodes: list[dict], poses: dict[int, np.ndarray], meshes: dict[tuple[int, str], Path]) -> dict:
    """All six official Track 3 metrics; each runs in its own process (each re-registers meshes)."""
    from concurrent.futures import ProcessPoolExecutor
    submission, cdo = build_submission(episodes, poses, meshes)
    solution = pose_solution(episodes)
    jobs = {display: (name, solution, submission) for name, (display, _) in POSE_METRICS.items()}
    jobs["CD-O cm"] = ("CD-O", cdo_solution(episodes), cdo)
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {display: pool.submit(_run_metric, *job) for display, job in jobs.items()}
        return {display: future.result() for display, future in futures.items()}


def evaluator_poses(path: Path, episode: dict) -> np.ndarray:
    """Evaluator output (one row per step, object and world) -> (frames, objects, 7), xyzw.

    Uses world 0 and the achieved (simulated) object poses. Objects go by object_id, which must
    follow episodes.json order. Rows are keyed by reference_frame, which equals the video frame
    only with the Track 3 frame-rate settings (README §6); any other numbering is rejected.
    """
    table = pd.read_parquet(path)
    table = table[table["environment_id"] == 0]
    frames, objects = episode["num_frames"], len(episode["objects"])
    ids = sorted(table["object_id"].unique())
    if len(ids) != objects:
        raise ValueError(f"{path.name}: {len(ids)} objects in the rollout, episode has {objects}")
    columns = [f"achieved_position_{a}" for a in "xyz"] + [f"achieved_quaternion_{a}" for a in "xyzw"]
    poses = np.full((frames, objects, 7), np.nan)
    for slot, object_id in enumerate(ids):
        rows = table[table["object_id"] == object_id].drop_duplicates("reference_frame", keep="last")
        index = rows["reference_frame"].to_numpy()
        if index.min() < 0 or index.max() >= frames:
            raise ValueError(f"{path.name}: reference_frame runs {index.min()}..{index.max()}, "
                             f"video has {frames} frames (check sim.fps / task.motion_speed)")
        poses[index, slot] = rows[columns].to_numpy(dtype=np.float64)
    missing = np.isnan(poses[..., 0]).any(axis=1)
    if missing.any():
        raise ValueError(f"{path.name}: no rollout pose for {int(missing.sum())} of {frames} frames "
                         f"(first {int(np.argmax(missing))})")
    return poses


def load_predictions(results: Path, episodes: list[dict]) -> dict[int, np.ndarray]:
    """Per-episode predictions from evaluator output (episode_XXXXXX.parquet) or episode_XXXXXX.npy
    of shape (frames, objects, 7), xyzw."""
    poses = {}
    for e in episodes:
        stem = results / f"episode_{e['episode']:06d}"
        if stem.with_suffix(".parquet").is_file():
            poses[e["episode"]] = evaluator_poses(stem.with_suffix(".parquet"), e)
        elif stem.with_suffix(".npy").is_file():
            poses[e["episode"]] = np.load(stem.with_suffix(".npy"))
        else:
            raise FileNotFoundError(f"no predictions for episode {e['episode']} in {results}")
    return poses


def print_table(rows: dict[str, dict]) -> None:
    metrics = list(next(iter(rows.values())))
    print(f"{'case':<34}" + "".join(f"{m:>11}" for m in metrics))
    for case, values in rows.items():
        print(f"{case:<34}" + "".join(f"{values[m]:>11.4f}" for m in metrics))


def self_test(episode_ids: list[int]) -> None:
    """Sanity cases with known answers, using ground truth + scans as the 'prediction'."""
    from scipy.spatial.transform import Rotation
    episodes = [e for e in public_episodes() if e["episode"] in episode_ids]
    scans = {(e["episode"], o["name"]): scan_path(o["name"]) for e in episodes for o in e["objects"]}
    gt = {e["episode"]: ground_truth(e["episode"]) for e in episodes}
    for name in sorted({o["name"] for e in episodes for o in e["objects"]} & set(SYMMETRY_ORDER)):
        print(f"symmetry fit {name}: order {SYMMETRY_ORDER[name]}, residual {symmetry_residual_cm(name):.3f} cm")

    def moved(poses, rotation, offset, scale=1.0):
        out = poses.copy()
        out[..., :3] = (poses[..., :3] * scale) @ rotation.as_matrix().T + offset
        out[..., 3:] = (rotation * Rotation.from_quat(poses[..., 3:].reshape(-1, 4))).as_quat().reshape(poses[..., 3:].shape)
        return out

    rng = np.random.default_rng(0)
    world = Rotation.from_euler("zyx", [70, 20, -35], degrees=True)
    cases = {
        "ground truth": gt,
        "GT in another world frame": {k: moved(v, world, [0.4, -1.2, 0.3]) for k, v in gt.items()},
        "GT, trajectory scaled x1.2": {k: moved(v, Rotation.identity(), 0.0, 1.2) for k, v in gt.items()},
        "GT + 2 cm position noise": {k: v + np.concatenate([rng.normal(0, 0.02, v[..., :3].shape),
                                                             np.zeros(v[..., 3:].shape)], -1) for k, v in gt.items()},
        "GT frozen at first frame": {k: np.repeat(v[:1], len(v), axis=0) for k, v in gt.items()},
    }
    rows = {case: score(episodes, poses, scans) for case, poses in cases.items()}
    print_table(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="score known cases built from ground truth")
    parser.add_argument("--episodes", type=int, nargs="*", help="public episode indices (default: all 20)")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--meshes", type=Path)
    args = parser.parse_args()
    ids = args.episodes or [e["episode"] for e in public_episodes()]
    if args.self_test:
        self_test(ids)
        return
    if not (args.results and args.meshes):
        parser.error("--results and --meshes are required (or use --self-test)")
    episodes = [e for e in public_episodes() if e["episode"] in ids]
    meshes = {}
    for e in episodes:
        for o in e["objects"]:
            stem = args.meshes / f"episode_{e['episode']:06d}" / o["name"]
            found = [p for s in (".glb", ".obj", ".ply", ".stl", ".off") if (p := stem.with_suffix(s)).is_file()]
            if not found:
                raise FileNotFoundError(f"no mesh for {stem}.*")
            meshes[(e["episode"], o["name"])] = found[0]
    print_table({"submission": score(episodes, load_predictions(args.results, episodes), meshes)})


if __name__ == "__main__":
    main()
