"""S3d driver: settle one object on its support in MuJoCo (CPU), in a venv with mujoco and trimesh
(the VM's daid venv: mujoco 3.4.0; the Mac's ~/V2D/venvs/sim).

    $SETTLE_PY path_b/pathb/drivers/settle.py <request.json> <result.json>

request: {"seconds", "timestep", "free": {"name", "pieces": [obj files, canonical frame], "pose": 4x4
canonical -> world, "mass_kg"}, "fixed": [{"name", "pieces", "pose"}] (the support, if it is not the
table), "table": true}. The table is the plane z = 0. The free body's frame is the canonical frame, so
its pose after `seconds` of gravity is the settled canonical -> world pose.
result: {"pose": 4x4, "moved_m", "turned_deg", "penetration_m" (deepest contact of the free body with
the table or the support at the end), "contacts", "steps"}. Contacts: multiccd (several points per
convex pair, so a flat bottom rests instead of rocking on one) and solref SOLREF.
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import trimesh


SOLREF = "0.004 1"  # default: stiffer than MuJoCo's 0.02 (a 0.1 kg cup sank 1.8 mm into a rack floor), 2 x the timestep


def quat_wxyz(R: np.ndarray) -> list[float]:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).reshape(-1))
    return q.tolist()


def body_xml(name: str, pieces: list[str], pose: np.ndarray, free: bool, density: float | None) -> tuple[str, str]:
    assets = "".join(f'<mesh name="{name}_{i}" file="{p}"/>' for i, p in enumerate(pieces))
    pos = " ".join(f"{x:.9f}" for x in pose[:3, 3])
    quat = " ".join(f"{x:.9f}" for x in quat_wxyz(pose[:3, :3]))
    geoms = "".join(f'<geom name="{name}_g{i}" type="mesh" mesh="{name}_{i}" condim="3" friction="1 0.005 0.0001" solref="{SOLREF}"'
                    + (f' density="{density:.6f}"' if density else "") + "/>" for i in range(len(pieces)))
    return assets, f'<body name="{name}" pos="{pos}" quat="{quat}">' + ("<freejoint/>" if free else "") + geoms + "</body>"


def main(request: str, result: str) -> int:
    req = json.loads(Path(request).read_text())
    free = req["free"]
    volume = sum(abs(trimesh.load(p, force="mesh", process=False).volume) for p in free["pieces"])
    density = free["mass_kg"] / max(volume, 1e-9)
    pose0 = np.asarray(free["pose"], dtype=np.float64)
    assets, bodies = body_xml("free", free["pieces"], pose0, True, density)
    for f in req.get("fixed", []):
        a, b = body_xml(f["name"], f["pieces"], np.asarray(f["pose"], dtype=np.float64), False, None)
        assets, bodies = assets + a, bodies + b
    table = f'<geom name="table" type="plane" size="5 5 0.1" condim="3" solref="{SOLREF}"/>' if req.get("table", True) else ""
    xml = (f'<mujoco model="settle"><compiler inertiafromgeom="true"/>'
           f'<option timestep="{req.get("timestep", 0.002)}" gravity="0 0 -9.81"><flag multiccd="enable"/></option>'
           f"<asset>{assets}</asset><worldbody>{table}{bodies}</worldbody></mujoco>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    steps = int(round(req.get("seconds", 0.5) / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "free")
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, data.xquat[bid])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = R.reshape(3, 3), data.xpos[bid]
    mine = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}
    pen, n = 0.0, 0
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in mine) != (c.geom2 in mine):
            n += 1
            pen = max(pen, -float(c.dist))
    turned = np.degrees(np.arccos(np.clip((np.trace(pose0[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1)))
    Path(result).write_text(json.dumps({"pose": pose.tolist(), "moved_m": float(np.linalg.norm(pose[:3, 3] - pose0[:3, 3])),
                                        "turned_deg": float(turned), "penetration_m": pen, "contacts": n,
                                        "steps": steps}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
