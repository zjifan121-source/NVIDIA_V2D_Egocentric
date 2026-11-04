"""SUPPLEMENTARY (not part of the decision rule): can the two Kaggle-measured ep8 changes (T4-bob1 = +0.047, T6-bob1 = +0.068,
ledger) tell whether beige_cup is scored symmetric? Score the PACKED bob1 / T4 / T6 rows (emu8 SUBS, exactly as sent to Kaggle)
with a NON-symmetric cup (axis 0, order 1) against REF (FP-spun truth) and REF_DS (spin-free truth). The symmetric case is
emu8_out.json's 'REF, T6-cup shape, centroid anchor' row (bob1 0.899, T4 0.934, T6 0.957).
Usage: python -B -I nonsym_calib.py OUT.json NAME [NAME ...]   (NAME in bob1 T4 T6)"""
import os, sys, json, functools
import numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
argv = list(sys.argv)
sys.path.insert(0, H)
os.environ["JUDGE_METRICS"] = "add_auc"
import judge_ep8 as JE
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
outp, names = argv[1], argv[2:]
res = {}
for n in names:
    sub = E.SUBS[n].copy()
    res[n] = {}
    for jn in ("REF", "REF_DS"):
        sol = E.solution_df(JE.judge(jn), "t6cup", "centroid")
        E.M._METRIC = "add_auc"
        s = float(E.M.score(sol.copy(), sub.copy(), "row_id"))
        res[n][jn + "_nonsym"] = round(s, 5)
        print(n, jn, "nonsym", round(s, 5), flush=True)
    json.dump(res, open(outp, "w"), indent=1)
print("DONE", flush=True)
