"""d3_gain.py (refine1104 ep8): modelled gain term for gate G7 (computed BEFORE the build, from the adopted rollout only):
T6's lost scorer credit (REF judge, verbatim AUC.py via emu8) in frames f46-185 with |reference twist step| >= 10 deg/frame
plus the 3 following frames. Run: python -B -I d3_gain.py"""
import os, sys, json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, r"C:\v2d_demo\diag\ep_status_1104\ep8_work"); os.chdir(r"C:\v2d_demo\diag\ep_status_1104\ep8_work")
import emu8 as E
E.M._METRIC = "add_auc"
sol = E.solution_df(E.judge("REF"), "t6cup", "centroid")
s = float(E.M.score(sol.copy(), E.SUBS["T6"].copy(), "row_id"))
add = 100 * np.asarray(E._LAST["spe"][0]).reshape(219, -1).mean(1)
a = E.AXIS["t6cup"]
Rr = Rot.from_quat(E.REF6[:, 0, 3:])
tw = np.zeros(219)
for t in range(1, 219):
    q = (Rr[t - 1].inv() * Rr[t]).as_quat(); p = q[:3] @ a
    tw[t] = 2 * np.degrees(np.arctan2(p, q[3])); tw[t] = (tw[t] + 180) % 360 - 180
sel = np.zeros(219, bool)
for t in range(46, 186):
    if abs(tw[t]) >= 10: sel[t:t + 4] = True
sel[186:] = False
x = np.linspace(0, 1, 9)
cred = np.array([np.trapezoid((e < np.arange(1, 10)).astype(float), x=x) for e in add])
gain = float((1 - cred[sel]).sum() / 438)
out = {"episode_auc_T6_REF": round(s, 5), "frames": np.nonzero(sel)[0].tolist(), "twist_steps": {int(t): round(float(tw[t]), 1) for t in np.nonzero(np.abs(tw) >= 10)[0]},
       "lost_credit_gain_model": round(gain, 5), "add_cm_sel": [round(float(v), 2) for v in add[sel]]}
print(json.dumps(out)); json.dump(out, open(r"C:\v2d_demo\diag\refine1104\ep8\work\d3_gain.json", "w"), indent=1)
