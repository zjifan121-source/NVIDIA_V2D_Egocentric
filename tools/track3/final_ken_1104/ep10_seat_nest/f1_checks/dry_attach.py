"""k10: dry-run the build's cleaning stage (rlr_h5.cleaned_bundle, same flags as h5_build.sh: rest hold, hand and
object clean, level, attach, smooth-hands 2) with a given H5 patch on a scene's bundle_merged, into a NEW dir of
ours (no loader/retarget, no scene). Then compare with a reference bundle_cleaned and report attach/hand shifts.
usage (retarget venv, cwd = v2d_train, V2D_REPO set):
  dry_attach.py <patch.py> <bundle_merged dir> <dest dir (new)> <reference bundle_cleaned/result.npz> <out.json>"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

patch, bundle, dest, ref_npz, outj = sys.argv[1:6]
assert not os.path.exists(dest), f"dest exists: {dest}"
rlr = os.path.expanduser("~/v2d_ops/kenc/hand5/rlr_h5.py")
spec = importlib.util.spec_from_file_location("rlr_h5_dryattach", rlr)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
g = vars(m)
exec(compile(open(patch).read(), patch, "exec"), g)
out_dir, info = g["cleaned_bundle"](Path(bundle).resolve(), Path(dest), g["TRACK3_FPS"], rest_hold=True, hand_clean=True,
                                    object_clean=True, level=True, attach=True, hand_smooth=2.0)
new = dict(np.load(Path(dest) / "result.npz", allow_pickle=True))
ref = dict(np.load(ref_npz, allow_pickle=True))
mer = dict(np.load(Path(bundle) / "result.npz", allow_pickle=True))
res = {"patch": patch, "info": {k: v for k, v in info.items() if not (isinstance(v, list) and len(v) > 20)}}
diff = {}
for k in sorted(set(new) | set(ref)):
    if k not in new or k not in ref:
        diff[k] = "missing"
        continue
    x, y = new[k], ref[k]
    if x.shape != y.shape:
        diff[k] = "shape"
    elif x.dtype.kind in "fc":
        dd = np.abs(x.astype(np.float64) - y.astype(np.float64))
        if dd.size and dd.max() > 0:
            per = dd.reshape(x.shape[0], -1).max(1) if x.ndim > 1 else dd
            fr = np.flatnonzero(per > 1e-6)
            diff[k] = {"maxabs": float(dd.max()), "frames_gt_1e-6": [int(fr[0]), int(fr[-1]), int(len(fr))] if len(fr) else []}
    elif not np.array_equal(x, y):
        diff[k] = "differs"
res["vs_reference_bundle_cleaned"] = diff
sh = {}
for lab, D in (("new", new), ("reference", ref)):
    for side in ("left", "right"):
        d = np.linalg.norm(D[f"hand_{side}_wrist_trans_in_camera"].astype(float) -
                           mer[f"hand_{side}_wrist_trans_in_camera"].astype(float), axis=1) * 100
        sh[f"{lab}_{side}"] = {"max_cm": round(float(d.max()), 2), "argmax": int(d.argmax()),
                               "f45_81_max_cm": round(float(d[45:82].max()), 2), "f82_124_max_cm": round(float(d[82:125].max()), 2),
                               "every4_f40_128": {int(f): round(float(d[f]), 2) for f in range(40, 129, 4)}}
res["wrist_shift_vs_merged_camera_cm"] = sh
T0n, T0r = new["object_to_world_transform"].astype(float), ref["object_to_world_transform"].astype(float)
res["blue_z_diff_vs_reference_cm_maxabs"] = round(float(np.abs(T0n[:, 2, 3] - T0r[:, 2, 3]).max()) * 100, 4)
json.dump(res, open(outj, "w"), indent=1)
print(json.dumps(res["info"]))
print("diff vs reference:", json.dumps(diff))
print("wrist shift:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "every4_f40_128"} for k, v in sh.items()}))
print("new_left every4:", sh["new_left"]["every4_f40_128"])
print("DRYATTACH_DONE")
