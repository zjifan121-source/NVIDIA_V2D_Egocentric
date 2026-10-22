"""S4 scene: one episode's stage 2-3 output in SPIDER's two-object bimanual layout, table at z = 0.

SPIDER (facebookresearch/spider, CC BY-NC 4.0, a pinned dependency, never copied here) already runs
two objects in one bimanual scene: generate_xml.py adds `right_object` and `left_object` as free
bodies, makes them collide with each other and with the floor, and the optimizer tracks both poses
(qpos ends with right pos+quat, left pos+quat). It has no table, only a floor plane 1 mm below z = 0.
This module supplies what SPIDER has no processor for:

  * the reference, read from the team's stage 2-3 output (the ManoSharpa partition: MANO joints,
    object poses and contacts in FlashCHORD's world), so S4 and S5 share one frame;
  * the table: the support top under the objects becomes z = 0, so SPIDER's floor is the table;
  * slots: each object goes to the hand that touches it most (SPIDER places a hand's contact sites
    on its own slot's object), and contact guidance is marked unsafe when one hand handles both;
  * a support check: an object that neither rests on the table nor on the other object cannot be
    held up by SPIDER's single floor, so the episode is flagged.

The functions below need numpy only. `build` (the CLI) runs in the stage 2-3 environment
(RETARGET_VENV), which has pyarrow, trimesh and pxr for the team's readers.

    $RETARGET_VENV/bin/python path_b/pathb/spider_scene.py --run <s23 run dir> --episode 13 \
        --out <dataset dir>
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DATASET = "v2d"  # SPIDER dataset name for Path B episodes
EMBODIMENT = "bimanual"
SLOTS = ("right", "left")
TIPS = (4, 8, 12, 16, 20)  # MANO_JOINTS_ORDER (robotic_grounding/retarget/params.py): thumb4 .. pinky4
CONTACT_M = 0.03  # a fingertip within 3 cm of the object counts as touching it
REST_M = 0.02  # an object whose bottom is within 2 cm of a support top rests on it
UNSAFE_SHARE = 0.10  # guidance is unsafe when >10% of a hand's contact frames are on the other slot


@dataclass
class Reference:
    fps: float
    names: list[str]  # object names, in the partition's body order
    obj_pos: np.ndarray  # (T, B, 3) world, metres
    obj_wxyz: np.ndarray  # (T, B, 4)
    joints: dict[str, np.ndarray]  # side -> (T, 21, 3) MANO joints, world
    vertices: list[np.ndarray] = field(default_factory=list)  # per object, (V, 3) body frame
    faces: list[np.ndarray] = field(default_factory=list)  # per object, (F, 3)

    @property
    def frames(self) -> int:
        return self.obj_pos.shape[0]


# ---------------------------------------------------------------- geometry
def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd's method, batched; w >= 0."""
    R = np.asarray(R, dtype=float)
    m = R.reshape(-1, 3, 3)
    q = np.empty((len(m), 4))
    tr = np.trace(m, axis1=1, axis2=2)
    for i, (r, t) in enumerate(zip(m, tr)):
        if t > 0:
            s = 2 * np.sqrt(t + 1)
            q[i] = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
        else:
            k = int(np.argmax(np.diag(r)))
            a, b = (k + 1) % 3, (k + 2) % 3
            s = 2 * np.sqrt(1 + r[k, k] - r[a, a] - r[b, b])
            v = np.empty(3)
            v[k] = 0.25 * s
            v[a] = (r[a, k] + r[k, a]) / s
            v[b] = (r[b, k] + r[k, b]) / s
            q[i] = ((r[b, a] - r[a, b]) / s, *v)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q.reshape(*R.shape[:-2], 4)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def hand_pose(joints: np.ndarray, side: str) -> np.ndarray:
    """(T, 7) wrist position + wxyz in SPIDER's hand-frame convention: z from the wrist (joint 0)
    to the middle knuckle (9), y across the palm from the ring (13) to the index knuckle (5),
    x = y x z; the left hand's x and y are flipped so both hands share one frame layout."""
    z = _unit(joints[:, 9] - joints[:, 0])
    y_aux = _unit(joints[:, 5] - joints[:, 13])
    x = _unit(np.cross(y_aux, z))
    y = _unit(np.cross(z, x))
    if side == "left":
        x, y = -x, -y
    R = np.stack([x, y, z], axis=-1)  # columns
    return np.concatenate([joints[:, 0], matrix_to_quat_wxyz(R)], axis=-1)


def fingertips(joints: np.ndarray) -> np.ndarray:
    """(T, 5, 7) fingertip positions with identity orientation (SPIDER's keypoint layout)."""
    tips = np.zeros((joints.shape[0], 5, 7))
    tips[:, :, :3] = joints[:, list(TIPS)]
    tips[:, :, 3] = 1.0
    return tips


def world_vertices(ref: Reference, b: int, frame: int) -> np.ndarray:
    R = quat_wxyz_to_matrix(ref.obj_wxyz[frame, b])
    return ref.vertices[b] @ R.T + ref.obj_pos[frame, b]


def surface_samples(vertices: np.ndarray, faces: np.ndarray, n: int = 2000, seed: int = 0) -> np.ndarray:
    """n points spread over the mesh surface by area (fixed seed), plus the vertices themselves, so
    coarse meshes (a few large faces) are measured by their faces, not only their corners."""
    tri = vertices[faces]  # (F, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), size=n, p=area / area.sum()) if area.sum() > 0 else np.zeros(0, int)
    u, v = rng.random((2, len(pick)))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[pick]
    points = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    keep = vertices if len(vertices) <= n else vertices[np.linspace(0, len(vertices) - 1, n).astype(int)]
    return np.concatenate([points, keep])


def tip_distances(ref: Reference, b: int, side: str) -> np.ndarray:
    """(T,) distance from the nearest fingertip of `side` to object b's surface (area samples)."""
    R = quat_wxyz_to_matrix(ref.obj_wxyz[:, b])  # (T, 3, 3)
    tips = ref.joints[side][:, list(TIPS)]  # (T, 5, 3)
    local = np.einsum("tji,tkj->tki", R, tips - ref.obj_pos[:, b, None, :])  # into the body frame
    verts = surface_samples(ref.vertices[b], ref.faces[b])
    out = np.empty(ref.frames)
    for t in range(ref.frames):
        out[t] = np.sqrt(((local[t, :, None, :] - verts[None]) ** 2).sum(-1)).min()
    return out


# ---------------------------------------------------------------- slots and support
def assign_slots(ref: Reference) -> dict:
    """Object -> SPIDER slot by contact frames. One object: the right slot (SPIDER puts the left
    hand's contact sites on it when no left object exists). Two: the pairing with more matched
    contact frames, roster order on a tie."""
    B = len(ref.names)
    if B not in (1, 2):
        raise ValueError(f"SPIDER's bimanual scene holds one or two objects, this episode has {B}")
    touch = {name: {side: int((tip_distances(ref, b, side) < CONTACT_M).sum()) for side in SLOTS}
             for b, name in enumerate(ref.names)}
    if B == 1:
        slots = {ref.names[0]: "right"}
        safe = True
    else:
        a, c = ref.names
        straight = touch[a]["right"] + touch[c]["left"]
        crossed = touch[a]["left"] + touch[c]["right"]
        slots = {a: "right", c: "left"} if straight >= crossed else {a: "left", c: "right"}
        safe = True
        for side in SLOTS:
            own = next(n for n, s in slots.items() if s == side)
            other = next(n for n in ref.names if n != own)
            total = touch[own][side] + touch[other][side]
            if total and touch[other][side] > UNSAFE_SHARE * total:
                safe = False
    return {"slots": slots, "contact_frames": touch, "guidance_safe": safe}


def support_plan(ref: Reference, disks: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """z0 = the lowest support top under any object at frame 0 (the table); each object then rests
    on the table, rests on the other object, or is raised (SPIDER's one floor cannot hold it)."""
    bottoms, centres, boxes = [], [], []
    for b in range(len(ref.names)):
        w = world_vertices(ref, b, 0)
        bottoms.append(float(w[:, 2].min()))
        centres.append(w[:, :2].mean(axis=0))
        boxes.append((w.min(axis=0), w.max(axis=0)))
    tops = []
    for centre in centres:
        under = [float(hi[2]) for lo, hi in disks if np.all(lo[:2] <= centre) and np.all(centre <= hi[:2])]
        tops.append(max(under) if under else None)
    known = [t for t in tops if t is not None]
    source = "support disks"
    if known:
        z0 = min(known)
    else:
        z0, source = min(bottoms), "lowest object bottom (no support disk under an object)"
    states = {}
    for b, name in enumerate(ref.names):
        gap = bottoms[b] - z0
        if abs(gap) <= REST_M:
            states[name] = {"rests_on": "table", "gap_cm": round(gap * 100, 2)}
            continue
        on_other = None
        for o in range(len(ref.names)):
            if o == b:
                continue
            lo, hi = boxes[o]
            inside = np.all(lo[:2] <= centres[b]) and np.all(centres[b] <= hi[:2])
            if inside and lo[2] - REST_M <= bottoms[b] <= hi[2] + REST_M:
                on_other = ref.names[o]
        states[name] = {"rests_on": on_other or "nothing", "gap_cm": round(gap * 100, 2)}
    return {"z0": z0, "source": source, "objects": states,
            "supported": all(s["rests_on"] != "nothing" for s in states.values())}


# ---------------------------------------------------------------- SPIDER layout
def task_name(episode: int) -> str:
    return f"episode_{episode:06d}"


def processed_dir(dataset_dir: Path, robot_type: str, task: str, data_id: int = 0) -> Path:
    """spider.io.get_processed_data_dir's layout."""
    return Path(dataset_dir) / "processed" / DATASET / robot_type / EMBODIMENT / task / str(data_id)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")


def keypoints(ref: Reference, slots: dict, z0: float) -> dict[str, np.ndarray]:
    """trajectory_keypoints.npz arrays, world shifted so the table top is z = 0."""
    shift = np.array([0.0, 0.0, z0])
    out = {}
    for side in SLOTS:
        pose = hand_pose(ref.joints[side], side)
        pose[:, :3] -= shift
        tips = fingertips(ref.joints[side])
        tips[:, :, :3] -= shift
        out[f"qpos_wrist_{side}"] = pose
        out[f"qpos_finger_{side}"] = tips
        obj = np.zeros((ref.frames, 7))
        obj[:, 3] = 1.0  # placeholder for an empty slot, as SPIDER's processors write it
        name = next((n for n, s in slots.items() if s == side), None)
        if name is not None:
            b = ref.names.index(name)
            obj[:, :3] = ref.obj_pos[:, b] - shift
            obj[:, 3:] = ref.obj_wxyz[:, b] / np.linalg.norm(ref.obj_wxyz[:, b], axis=-1, keepdims=True)
        out[f"qpos_obj_{side}"] = obj
    out["timestamps"] = np.round(np.arange(ref.frames) / ref.fps * 1e9).astype(np.int64)
    return out


def write_spider_inputs(ref: Reference, dataset_dir: Path, episode: int, support: dict,
                        extra: dict | None = None) -> dict:
    """Writes SPIDER's processed layout for one episode and returns its task_info.
    support: support_plan(ref, disks)."""
    dataset_dir = Path(dataset_dir).resolve()
    task = task_name(episode)
    slot_info = assign_slots(ref)
    data = keypoints(ref, slot_info["slots"], support["z0"])
    kp_dir = processed_dir(dataset_dir, "mano", task)
    kp_dir.mkdir(parents=True, exist_ok=True)
    np.savez(kp_dir / "trajectory_keypoints.npz", **data)
    mesh_dirs = {}
    for name, side in slot_info["slots"].items():
        b = ref.names.index(name)
        rel = Path("processed") / DATASET / "assets/objects" / f"{task}_{name}"
        write_obj(dataset_dir / rel / "visual.obj", ref.vertices[b], ref.faces[b])
        mesh_dirs[side] = str(rel)
    info = {
        "task": task, "dataset_name": DATASET, "robot_type": "mano", "embodiment_type": EMBODIMENT,
        "data_id": 0, "ref_dt": 1.0 / ref.fps, "dataset_dir": str(dataset_dir),
        "right_object_mesh_dir": mesh_dirs.get("right"), "left_object_mesh_dir": mesh_dirs.get("left"),
        "v2d": {"episode": episode, "objects": ref.names, "frames": ref.frames, "fps": ref.fps,
                "table_z_world_m": support["z0"], "table_source": support["source"],
                "support": support["objects"], "supported": support["supported"], **slot_info,
                **(extra or {})},
    }
    (kp_dir.parent / "task_info.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


# ---------------------------------------------------------------- reading stage 2-3 output
def reference_from_partition(d: dict, meshes: list[tuple[np.ndarray, np.ndarray]], names: list[str]) -> Reference:
    """d: one ManoSharpa row (run_load_retarget.read_partition); meshes in body order."""
    pos = np.asarray(d["object_body_position"], dtype=float)
    wxyz = np.asarray(d["object_body_wxyz"], dtype=float)
    if pos.ndim == 2:  # one body stored without its body axis
        pos, wxyz = pos[:, None], wxyz[:, None]
    joints = {side: np.asarray(d[f"mano_{side}_joints"], dtype=float) for side in SLOTS}
    ref = Reference(fps=float(d["fps"]), names=list(names), obj_pos=pos, obj_wxyz=wxyz, joints=joints,
                    vertices=[np.asarray(v, dtype=float) for v, _ in meshes],
                    faces=[np.asarray(f, dtype=int) for _, f in meshes])
    if pos.shape[1] != len(names):
        raise ValueError(f"{pos.shape[1]} object bodies, {len(names)} names")
    for side, j in joints.items():
        if j.shape != (ref.frames, 21, 3):
            raise ValueError(f"mano_{side}_joints has shape {j.shape}, expected ({ref.frames}, 21, 3)")
    return ref


def build(run_dir: Path, episode: int, out: Path) -> dict:
    """CLI body; needs the stage 2-3 environment (pyarrow, trimesh, pxr)."""
    import sys

    import trimesh

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/track3"))
    import run_load_retarget as R

    report = json.loads((run_dir / "report.json").read_text())
    processed = Path(report["processed"])
    seq = report["sequence_id"]
    names = report["merge"]["objects"] if report.get("merge") else [report["object"]]
    bodies = [f"{seq}_{n}" for n in names] if report.get("merge") else [seq]
    obj_dir = processed.parents[1]  # .../ego_recon/processed
    meshes = []
    for body in bodies:
        mesh = trimesh.load(obj_dir / f"{body}.obj", force="mesh", process=False)
        meshes.append((np.asarray(mesh.vertices), np.asarray(mesh.faces)))
    ref = reference_from_partition(R.read_partition(processed), meshes, names)
    usda = Path(report["support_usd"])
    disks = R.support_tops(usda) if usda.is_file() else []
    return write_spider_inputs(ref, out, episode, support_plan(ref, disks),
                               {"source_run": str(run_dir), "support_disks": len(disks)})


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run_load_retarget output dir (has report.json)")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="SPIDER dataset dir to write into")
    args = ap.parse_args()
    print(json.dumps(build(args.run, args.episode, args.out), indent=1))
