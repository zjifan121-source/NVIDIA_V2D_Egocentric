"""d2_bottom.py (refine1104 ep8): does a despin (twist about the base cup axis) interact with T6's stretched cup bottom?
(1) stretch direction of T6's cup mesh vs bob1's in the body frame (per-vertex displacement), angle to the cup axis;
(2) at the start/end rests: angle between world up and the stretched-bottom normal under the reference pose, the
achieved pose, and a locally computed despun reference (twist vs f0 removed about the base axis through the base centroid);
(3) cumulative reference twist vs f0 at rest frames. GT-free; reads only. Run: python -B -I d2_bottom.py"""
import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, r"C:\v2d_demo\diag\ep_status_1104\ep8_work")
os.chdir(r"C:\v2d_demo\diag\ep_status_1104\ep8_work")
import emu8 as E

V6, F6 = E.MESH["t6cup"]; V1, F1 = E.MESH["b1cup"]
print("verts", V6.shape, V1.shape)
same_n = V6.shape == V1.shape
D = V6 - V1 if same_n else None
mv = np.linalg.norm(D, axis=1)
moved = mv > 1e-5
print("moved verts", int(moved.sum()), "max disp cm", round(100 * mv.max(), 3))
d_dir = (D[moved] / mv[moved, None])
dmean = d_dir.mean(0); dmean /= np.linalg.norm(dmean)
a_b1 = E.AXIS["b1cup"]  # base axis (sign arbitrary)
a_b1 = a_b1 if np.dot(a_b1, dmean) < 0 else -a_b1  # point "up" (away from the bottom)
print("stretch dir (body)", dmean.round(4), "angle to base axis (deg)", round(float(np.degrees(np.arccos(abs(np.dot(dmean, a_b1))))), 2))
c_b1, _ = E.M._surface_moments(V1, F1)
REF = E.REF6; ACH = E.ACH6
Rr = Rot.from_quat(REF[:, 0, 3:]); Ra = Rot.from_quat(ACH[:, 0, 3:])
n_b = -dmean  # outward... the bottom normal pointing down is +dmean (stretch goes down); use 'down' = dmean
z = np.array([0, 0, -1.0])


def ang(u, v):
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1), -1, 1)))


def twist_about(rrel, axis):
    q = rrel.as_quat(); p = q[..., :3] @ axis
    tw = np.concatenate([p[..., None] * axis, q[..., 3:4]], -1); tw /= np.linalg.norm(tw, axis=-1, keepdims=True)
    return Rot.from_quat(tw), 2 * np.degrees(np.arctan2(tw[..., :3] @ axis, tw[..., 3]))


rel = Rr[0].inv() * Rr
Tw, tw_deg = twist_about(rel, a_b1)
Rd = Rr * Tw.inv()  # despun: R_t * twist^-1  (body-frame twist removed)
out = {}
for name, fr in (("start f0-31", range(0, 32)), ("end f200-218", range(200, 219)), ("f60-70", range(60, 71)), ("f190-199", range(190, 200))):
    fr = np.array(list(fr))
    out[name] = {
        "ref_twist_vs_f0_med_deg": round(float(np.median((tw_deg[fr] + 180) % 360 - 180)), 1),
        "bottom_tilt_ref_deg": round(float(np.median(ang(Rr[fr].apply(dmean), z))), 2),
        "bottom_tilt_ach_deg": round(float(np.median(ang(Ra[fr].apply(dmean), z))), 2),
        "bottom_tilt_despun_ref_deg": round(float(np.median(ang(Rd[fr].apply(dmean), z))), 2),
        "axis_tilt_ref_deg": round(float(np.median(ang(Rr[fr].apply(-a_b1), z))), 2),
        "axis_tilt_ach_deg": round(float(np.median(ang(Ra[fr].apply(-a_b1), z))), 2),
    }
    print(name, out[name])
# lowest point of the T6 mesh relative to the reference / despun reference at the end rest (z of min vertex)
for name, R in (("ref", Rr), ("despun", Rd), ("ach", Ra)):
    P = ACH[:, 0, :3] if name == "ach" else REF[:, 0, :3]
    if name == "despun":
        P = REF[:, 0, :3] + Rr.apply(c_b1) - Rd.apply(c_b1)
    zmin = np.array([(R[t].apply(V6) + P[t])[:, 2].min() for t in (0, 10, 20, 205, 210, 218)])
    print(name, "T6-mesh lowest z at f0,10,20,205,210,218:", zmin.round(4))
json.dump(out, open(r"C:\v2d_demo\diag\refine1104\ep8\work\d2_out.json", "w"), indent=1)
