"""SUPPLEMENTARY risk check (not part of the decision rule): audit item #6 - what if the kit scores beige_cup as NOT symmetric?
Same verbatim AUC.py path as judge_ep8.py, but the cup's geometry gets axis 0 / order 1 (like the pot). Two truths:
REF (cup spun as FoundationPose says) and REF_DS (cup spin-free, video-wrist-consistent). add_auc only.
Usage: python -B -I nonsym_check.py OUT.json ROLLOUT_DIR [...]"""
import os, sys, json, functools
import numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.argv_saved = list(sys.argv)
sys.path.insert(0, H)
os.environ["JUDGE_METRICS"] = "add_auc"
import judge_ep8 as JE   # chdirs into ep8_work and imports emu8 (read-only)
E = JE.E


@functools.lru_cache(maxsize=None)
def geometry_nonsym(cupmesh, anchor_mode):
    arrays = {"names": np.array(E.NAMES)}
    for i, key in enumerate((cupmesh, "pot")):
        V, F = E.MESH[key]
        c, _ = E.M._surface_moments(V, F)
        arrays[f"v{i}"] = V; arrays[f"f{i}"] = F
        arrays[f"anchor{i}"] = c if anchor_mode == "centroid" else np.zeros(3)
        arrays[f"points{i}"] = E.M._sample_mesh_surface(V, F, 1500, 12345 + i)
        arrays[f"axis{i}"] = np.zeros(3)
        arrays[f"order{i}"] = np.array(1)
    return E.M._encode_archive(**arrays)


E.geometry = geometry_nonsym
outp, dirs = sys.argv_saved[1], sys.argv_saved[2:]
res = {}
for d in dirs:
    name = os.path.basename(os.path.normpath(d))
    sub = JE.submission(os.path.join(d, "episode_000008.parquet"))
    res[name] = {}
    for jn in ("REF", "REF_DS"):
        sol = E.solution_df(JE.judge(jn), "t6cup", "centroid")
        E.M._METRIC = "add_auc"
        s = float(E.M.score(sol.copy(), sub.copy(), "row_id"))
        add = np.asarray(E._LAST["spe"][0]).reshape(JE.NF, -1)
        res[name][jn + "_nonsym"] = {"add_auc": round(s, 5), "cup_med_cm": round(float(100 * np.median(add)), 2),
                                     "cup_end_f186_218_med_cm": round(float(100 * np.median(add[186:219])), 2)}
        print(name, jn, "nonsym", res[name][jn + "_nonsym"], flush=True)
    json.dump(res, open(outp, "w"), indent=1)
print("DONE", flush=True)
