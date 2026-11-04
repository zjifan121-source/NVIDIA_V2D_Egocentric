"""Per-frame / per-window cup and pot errors of rf_ep8_ds vs the adopted sc_ep8 family (GT-free, our reference REF6 =
adopted reference; the rf reference has the same cup centroid/axis and the same pot). Cup: centroid distance (spin-free);
also axis-tilt error. Pot: centroid distance + rotation error. Uses emu8 meshes (T6 cup, pot). Read-only."""
import os, sys, json
import numpy as np
sys.path.insert(0, r"C:\v2d_demo\diag\ep_status_1104\ep8_work")
os.chdir(r"C:\v2d_demo\diag\ep_status_1104\ep8_work")
import emu8 as E
from scipy.spatial.transform import Rotation as Rot
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RO = os.path.join(H, "rollouts")
NF = 219
cc, _ = E.M._surface_moments(*E.MESH["t6cup"])
pc, _ = E.M._surface_moments(*E.MESH["pot"])
ax = E.AXIS["t6cup"]


def cen(P, c):
    return P[:, :3] + Rot.from_quat(P[:, 3:]).apply(c)


def errs(path):
    A = E.rollout_poses(path, "achieved")
    R = E.REF6
    cup = 100 * np.linalg.norm(cen(A[:, 0], cc) - cen(R[:, 0], cc), axis=1)
    a1 = Rot.from_quat(A[:, 0, 3:]).apply(ax); a2 = Rot.from_quat(R[:, 0, 3:]).apply(ax)
    tilt = np.degrees(np.arccos(np.clip(np.sum(a1 * a2, -1), -1, 1)))
    pot = 100 * np.linalg.norm(cen(A[:, 1], pc) - cen(R[:, 1], pc), axis=1)
    pz = 100 * (cen(A[:, 1], pc) - cen(R[:, 1], pc))[:, 2]
    prot = np.degrees((Rot.from_quat(A[:, 1, 3:]) * Rot.from_quat(R[:, 1, 3:]).inv()).magnitude())
    return dict(cup=cup, tilt=tilt, pot=pot, pz=pz, prot=prot)


WIN = {"rest f0-31": (0, 32), "grasp f32-45": (32, 46), "blend-in f46-54": (46, 55), "spin1+pour f55-72": (55, 73), "burst f62-69": (62, 70),
       "carry f73-127": (73, 128), "pour2+spin2 f128-155": (128, 156), "hold f156-175": (156, 176), "blend-out+setdown+rest f176-218": (176, 219)}
runs = ["h_ep8_sc_K_s42", "h_ep8_sc_K_s43", "h_ep8_sc_K_s44", "h_ep8_sc_K_s45", "h_ep8_sc_W_s42",
        "rf_ep8_ds_K_s42", "rf_ep8_ds_K_s43", "rf_ep8_ds_K_s44", "rf_ep8_ds_K_s42_100M", "rf_ep8_ds_K_s43_100M", "rf_ep8_ds_K_s44_100M"]
res = {}
for r in runs:
    e = errs(os.path.join(RO, r, "episode_000008.parquet"))
    res[r] = {w: {"cup_med": round(float(np.median(e["cup"][a:b])), 2), "cup_max": round(float(e["cup"][a:b].max()), 2),
                  "cup_tilt_med": round(float(np.median(e["tilt"][a:b])), 1),
                  "pot_med": round(float(np.median(e["pot"][a:b])), 2), "pot_dz_med": round(float(np.median(e["pz"][a:b])), 2),
                  "pot_rot_med": round(float(np.median(e["prot"][a:b])), 2)} for w, (a, b) in WIN.items()}
    res[r]["_frames_cup_gt3cm"] = int((e["cup"] > 3).sum())
    res[r]["_frames_cup_gt5cm"] = int((e["cup"] > 5).sum())
    res[r]["_cup_series"] = [round(float(x), 2) for x in e["cup"]]
    res[r]["_pot_dz_series"] = [round(float(x), 2) for x in e["pz"]]
json.dump(res, open(os.path.join(H, "work", "perframe_rf.json"), "w"), indent=1)
fam = ["h_ep8_sc_K_s42", "h_ep8_sc_K_s43", "h_ep8_sc_K_s44", "h_ep8_sc_K_s45"]
rf = ["rf_ep8_ds_K_s42", "rf_ep8_ds_K_s43", "rf_ep8_ds_K_s44"]
print("window                          | cup med cm T6 / KEN4 mean / RF3 mean | cup tilt deg KEN4/RF3 | pot med cm KEN4/RF3 | pot dz KEN4/RF3")
for w in WIN:
    k = lambda rs, f: np.mean([res[x][w][f] for x in rs])
    print(f"{w:31s} | {res['h_ep8_sc_K_s42'][w]['cup_med']:5.2f} / {k(fam, 'cup_med'):5.2f} / {k(rf, 'cup_med'):5.2f} | "
          f"{k(fam, 'cup_tilt_med'):4.1f} / {k(rf, 'cup_tilt_med'):4.1f} | {k(fam, 'pot_med'):4.2f} / {k(rf, 'pot_med'):4.2f} | {k(fam, 'pot_dz_med'):5.2f} / {k(rf, 'pot_dz_med'):5.2f}")
for r in runs:
    print(r, "frames cup>3cm", res[r]["_frames_cup_gt3cm"], ">5cm", res[r]["_frames_cup_gt5cm"])
