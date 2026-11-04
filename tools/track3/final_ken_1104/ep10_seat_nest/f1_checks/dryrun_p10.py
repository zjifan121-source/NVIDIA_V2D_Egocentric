"""k10 F1 dry run: execute p10seat.py's level_scene wrapper (original levelling replaced by identity, since the
arrays are already levelled) on bob1's ep10 bundle_cleaned arrays (h5_ep10_respinf_v4 = bob1 ep10 scene).
Also: loader transform A (parquet vs bundle_cleaned) rotation, and comparison with the plan's rule simulation.
usage (retarget venv python, V2D_REPO set): dryrun_p10.py <scene_dir> <p10seat.py> <rlr_h5.py> <nestplan.json> <out.json>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

scene, p10, rlr, nestplan, out = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("rlr_h5_dry", rlr)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
g = vars(m)
g["level_scene"] = lambda data, bundle, *a, **k: {"dryrun": "identity levelling (bundle_cleaned is levelled)"}
exec(compile(open(p10).read(), p10, "exec"), g)
bdir = Path(scene) / "bundle_cleaned"
data = dict(np.load(bdir / "result.npz", allow_pickle=True))
T0 = data["object_to_world_transform"].astype(np.float64).copy()
T1 = data["object_1_to_world_transform"].astype(np.float64).copy()
info = g["level_scene"](data, bdir)
dz = np.array(info["p10_dz_cm"])
Tn = data["object_to_world_transform"].astype(np.float64)
chk = {"max_abs_xy_change_m": float(np.abs(Tn[:, :2, 3] - T0[:, :2, 3]).max()),
       "max_abs_rot_change": float(np.abs(Tn[:, :3, :3] - T0[:, :3, :3]).max()),
       "beige_untouched": bool(np.array_equal(data["object_1_to_world_transform"].astype(np.float64), T1))}
# loader transform A: parquet body pose @ inv(bundle pose); constant rigid if the loader only offsets
import pyarrow.parquet as pq  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

pqf = next((Path(scene) / "assets/human_motion_data/ego_recon/processed").glob("*/robot_name=sharpa_wave/*.parquet"))
t = pq.read_table(pqf).to_pylist()[0]
P = np.array(t["object_body_position"], float)
Q = np.array(t["object_body_wxyz"], float)
N = len(P)
R = Rot.from_quat(Q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(N, -1, 3, 3)
As = []
for k, TT in enumerate((T0, T1)):
    for f in range(N):
        Tp = np.eye(4)
        Tp[:3, :3] = R[f, k]
        Tp[:3, 3] = P[f, k]
        As.append(Tp @ np.linalg.inv(TT[f]))
As = np.array(As)
A = As.mean(0)
U, _, Vt = np.linalg.svd(A[:3, :3])
rotA = Rot.from_matrix(U @ Vt)
chk["loader_A_translation_m"] = [round(float(x), 5) for x in A[:3, 3]]
chk["loader_A_rotation_deg"] = round(float(np.degrees(rotA.magnitude())), 5)
chk["loader_A_spread_max"] = float(np.abs(As - A).max())
chk["loader_A_z_axis_tilt_deg"] = round(float(np.degrees(np.arccos(np.clip((U @ Vt)[2, 2], -1, 1)))), 5)
sim = json.load(open(nestplan))
exp = sim.get("A2_dz_cm_every5", {})
cmp_rows = {f: [round(float(dz[int(f)]), 2), exp[f]] for f in exp}
res = dict(dz_cm=info["p10_dz_cm"], dz_raw_cm=info["p10_dz_raw_cm"], rise5_cm=info["p10_rise5_cm"],
           lowest_before_cm=info["p10_lowest_before_cm"],
           info={k: v for k, v in info.items() if not isinstance(v, list) or len(v) < 10},
           checks=chk, dryrun_vs_planned_sim_every5=cmp_rows)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res["info"]))
print("checks", json.dumps(chk))
print("frame  dz_dry  dz_plan_sim  lowest_before  rise5")
for f in range(0, N, 5):
    print(f"{f:4d} {dz[f]:7.2f} {str(exp.get(str(f))):>8} {info['p10_lowest_before_cm'][f]:8.2f} {info['p10_rise5_cm'][f]}")
print("f36-58 dz:", [round(float(x), 2) for x in dz[36:59]])
print("f10-24 dz:", [round(float(x), 3) for x in dz[10:25]])
print("DRYRUN_DONE")
