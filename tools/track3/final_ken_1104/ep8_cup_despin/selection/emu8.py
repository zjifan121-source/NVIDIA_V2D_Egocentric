"""emu8.py: GT-free Kaggle Track 3 emulation for episode 8 (beige_cup slot 0, white_pot slot 1), calling the VERBATIM
kit scorer AUC.py (sha256 3f1a9899..., copied from C:/v2d_demo/diag/ep10_verify/scorer_2/AUC.py). Pattern = the ep10
emulator (scorer_2/emu.py): each pose board = AUC.score(solution, submission) with M._METRIC switched.

Submissions (packed rows exactly as sent to Kaggle):
  bob1 = ep8 rows of the T2 full file (bob1 + ep1 swap; ep8 = bob1)   archive/o_ep1_hold_s44/babel/track3_submission.parquet
  T4   = archive/f_ep8_cf_K_s42/babel/one.parquet   (Kaggle ep8 change vs bob1: +0.047)
  T6   = archive/h_ep8_sc_K_s42/babel/one.parquet   (Kaggle ep8 change vs bob1: +0.068)
GT stand-ins (hidden truth replaced; GT-free): reference poses = our scene reference (identical in bob1/T4/T6 scenes),
optionally corrected by the FS-depth measurements of C:/v2d_demo/depthtab/ep8/dt8.log (pot only); stand-in meshes = our
full-resolution OBJ meshes (cup: T6 stretched or bob1 original; pot: donor). Cup symmetric (PCA axis, order 360), pot
non-symmetric (inferred from the kit README, not verified).
Run: python -B emu8.py OUT.json
"""
import os, sys, json, hashlib, importlib.util, functools, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("auc_scorer_ep8", os.path.join(HERE, "AUC.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
assert M._VERTICES == {} and M._REGISTER_OBJECT_MESH and M._ALIGN_INITIAL_POSE

NAMES = ["beige_cup", "white_pot"]
NF = 219
FRAMES = np.arange(NF)
ROW_IDS = [f"track_3/e{8:06d}/w000/f{f:06d}/o{s}" for f in FRAMES for s in (0, 1)]
METRICS = ["add_auc", "spider_sr", "maniptrans_sr", "rpe_cm", "mppe_cm"]
SEGMENTS = {"rest f0-31": (0, 32), "grasp f32-44": (32, 45), "lifted f45-185": (45, 186),
            "setdown f186-199": (186, 200), "rest_end f200-218": (200, 219)}
A = r"C:\v2d_demo\archive"


def load_obj(path):
    V, F = [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) for t in line.split()[1:]]
            for i in range(1, len(idx) - 1):
                F.append([idx[0] - 1, idx[i] - 1, idx[i + 1] - 1])
    return np.array(V, float), np.array(F, np.int64)


MESH = {"t6cup": load_obj(os.path.join(HERE, "t6_beige_cup.obj")),
        "b1cup": load_obj(os.path.join(HERE, "bob1_beige_cup.obj")),
        "pot": load_obj(os.path.join(HERE, "white_pot.obj"))}


def cup_axis(V, F):
    P = M._sample_mesh_surface(V, F, 20000, 7)
    c = P - P.mean(0)
    w, U = np.linalg.eigh(c.T @ c / len(c))
    d = [abs(w[0] - w[1]), abs(w[1] - w[2]), abs(w[0] - w[2])]
    k = {0: 2, 1: 0, 2: 1}[int(np.argmin(d))]  # eigenvalue not in the closest pair
    return U[:, k] / np.linalg.norm(U[:, k]), w


AXIS = {k: cup_axis(*MESH[k])[0] for k in ("t6cup", "b1cup")}


def rollout_poses(path, kind):
    d = pq.read_table(path).to_pandas()
    d = d[d.environment_id == 0].sort_values(["reference_frame", "object_body_id"])
    a = np.stack([d[f"{kind}_position_{c}"].to_numpy(np.float64) for c in "xyz"] +
                 [d[f"{kind}_quaternion_{c}"].to_numpy(np.float64) for c in "xyzw"], -1)
    assert len(a) == 2 * NF and (d.reference_frame.to_numpy() == np.repeat(FRAMES, 2)).all()
    return a.reshape(NF, 2, 7)


def packed(path):
    d = pd.read_parquet(path)
    return d[d.row_id.str.startswith("track_3/e000008/")].set_index("row_id").loc[ROW_IDS].reset_index()


SUBS = {"bob1": packed(os.path.join(A, "o_ep1_hold_s44", "babel", "track3_submission.parquet")),
        "T4": packed(os.path.join(A, "f_ep8_cf_K_s42", "babel", "one.parquet")),
        "T6": packed(os.path.join(A, "h_ep8_sc_K_s42", "babel", "one.parquet"))}
REF6 = M._normalise_quaternions(rollout_poses(os.path.join(A, "h_ep8_sc_K_s42", "results", "episode_000008.parquet"), "reference"), "r", ValueError)
REF4 = M._normalise_quaternions(rollout_poses(os.path.join(A, "f_ep8_cf_K_s42", "results", "episode_000008.parquet"), "reference"), "r", ValueError)
ACH6 = rollout_poses(os.path.join(A, "h_ep8_sc_K_s42", "results", "episode_000008.parquet"), "achieved")


def ramp_piecewise(knots):
    """knots: list of (frame, value-vector); linear between knots, constant outside."""
    f = np.array([k[0] for k in knots], float); v = np.array([k[1] for k in knots], float)
    return np.stack([np.interp(FRAMES, f, v[:, j]) for j in range(v.shape[1])], -1)


# pot corrections from depthtab/ep8/dt8.log SUMMARY (cm): phases start_0_30 / pot_contact_16_94 / pot_late_166_196 / end_200_218
SEAT_DZ = ramp_piecewise([(0, [1.16]), (13, [1.16]), (16, [0.55]), (94, [0.55]), (130, [0.5]), (166, [0.42]), (196, [0.42]),
                          (200, [1.57]), (218, [1.57])])[:, 0] / 100.0
RAY_DW = ramp_piecewise([(0, [-0.01, -0.75, 0.76]), (13, [-0.01, -0.75, 0.76]), (16, [0.0, -0.19, 0.25]), (94, [0.0, -0.19, 0.25]),
                         (166, [0.03, -0.2, 0.25]), (196, [0.03, -0.2, 0.25]), (200, [0.06, -0.66, 0.71]), (218, [0.06, -0.66, 0.71])]) / 100.0


def judge(name):
    P = REF6.copy()
    if name.startswith("SEAT"):
        P[:, 1, 2] += SEAT_DZ
    elif name.startswith("RAY"):
        P[:, 1, :3] += RAY_DW
    return P


@functools.lru_cache(maxsize=None)
def geometry(cupmesh, anchor_mode):
    arrays = {"names": np.array(NAMES)}
    for i, key in enumerate((cupmesh, "pot")):
        V, F = MESH[key]
        c, _ = M._surface_moments(V, F)
        arrays[f"v{i}"] = V; arrays[f"f{i}"] = F
        arrays[f"anchor{i}"] = c if anchor_mode == "centroid" else np.zeros(3)
        arrays[f"points{i}"] = M._sample_mesh_surface(V, F, 1500, 12345 + i)
        arrays[f"axis{i}"] = AXIS[cupmesh] if i == 0 else np.zeros(3)
        arrays[f"order{i}"] = np.array(360 if i == 0 else 1)
    return M._encode_archive(**arrays)


def solution_df(ref, cupmesh, anchor_mode):
    flat = ref.reshape(-1, 7)
    sol = pd.DataFrame({"row_id": ROW_IDS, "episode_index": 8, "world_index": 0, "frame_index": np.repeat(FRAMES, 2),
                        "object_slot": np.tile([0, 1], NF), "object_name": np.tile(NAMES, NF)})
    for i, c in enumerate(M._REF_COLUMNS):
        sol[c] = flat[:, i]
    sol["reference_geometry_b64"] = ""
    sol.loc[0, "reference_geometry_b64"] = geometry(cupmesh, anchor_mode)
    return sol


# ---- observers (no behaviour change), as in scorer_2/emu.py
_REG = {}
_orig_register = M._register_object_body


def _register_memo(candidate, reference, anchor, seed):
    key = (hashlib.md5(candidate[0].tobytes() + candidate[1].tobytes()).hexdigest(),
           hashlib.md5(reference[0].tobytes() + reference[1].tobytes()).hexdigest(), tuple(np.round(anchor, 12)), seed)
    if key not in _REG:
        _REG[key] = _orig_register(candidate, reference, anchor, seed)
    ca, rot = _REG[key]
    return ca.copy(), rot.copy()


M._register_object_body = _register_memo
_LAST = {}
_orig_align = M._align_registered_scene
_orig_metric = M._registered_pose_metric
_orig_spe = M._symmetry_point_errors


def _align_observed(achieved, reference, axes):
    out = _orig_align(achieved, reference, axes)
    qa, qo = achieved[0, 0, 0, 3:], out[0, 0, 0, 3:]
    q = M._quaternion_multiply(qo, qa * np.array([-1, -1, -1, 1.0]))
    _LAST["align_deg"] = float(np.degrees(2 * np.arccos(min(1.0, abs(q[3]) / np.linalg.norm(q)))))
    return out


def _metric_observed(metric, achieved, reference, vertices, axes, orders):
    _LAST.update(achieved=achieved, reference=reference, axes=axes, orders=orders, spe=[])
    return _orig_metric(metric, achieved, reference, vertices, axes, orders)


def _spe_observed(achieved, reference, points, axis, order=360):
    out = _orig_spe(achieved, reference, points, axis, order)
    if len(points) > 6:
        _LAST.setdefault("spe", []).append(out.copy())
    return out


M._align_registered_scene = _align_observed
M._registered_pose_metric = _metric_observed
M._symmetry_point_errors = _spe_observed


def diagnostics():
    a, r, ax, od = (_LAST[k] for k in ("achieved", "reference", "axes", "orders"))
    out = {"align_deg": round(_LAST["align_deg"], 3)}
    for i, n in enumerate(NAMES):
        add = _LAST["spe"][i]
        out[n] = {"auc": round(M._add_auc(add[..., None]), 4),
                  "seg_auc": {s: round(M._add_auc(add[b0:b1, :, None]), 3) for s, (b0, b1) in SEGMENTS.items()},
                  "seg_add_med_cm": {s: round(float(100 * np.median(add[b0:b1])), 2) for s, (b0, b1) in SEGMENTS.items()},
                  "mean_pos_err_cm": round(float(100 * np.linalg.norm(a[:, :, i, :3] - r[:, :, i, :3], axis=-1).mean()), 3),
                  "mean_orient_err_deg": round(float(np.degrees(M._symmetry_orientation_errors(a, r, ax, od)[:, :, i].mean())), 3)}
    return out


def score_all(ref, sub, cupmesh, anchor_mode):
    sol = solution_df(ref, cupmesh, anchor_mode)
    res = {}
    for m in METRICS:
        M._METRIC = m
        res[m] = round(float(M.score(sol.copy(), sub.copy(), "row_id")), 5)
        if m == "add_auc":
            res["diag"] = diagnostics()
    M._METRIC = "add_auc"
    return res


if __name__ == "__main__":
    out = sys.argv[1]
    info = {"ref_T4_vs_T6_max_abs": float(np.abs(REF4 - REF6).max()),
            "ach6_vs_packed_T6_max_abs": float(np.abs(ACH6.reshape(-1, 7) - SUBS["T6"][M._POSE_COLUMNS].to_numpy()).max()),
            "cup_axis_t6": AXIS["t6cup"].round(5).tolist(), "cup_axis_b1": AXIS["b1cup"].round(5).tolist(),
            "seat_dz_cm_f0_f50_f150_f210": [round(100 * SEAT_DZ[f], 2) for f in (0, 50, 150, 210)]}
    print(json.dumps(info), flush=True)
    runs = [("REF", "t6cup", "centroid"), ("REF", "b1cup", "centroid"), ("REF", "t6cup", "origin"),
            ("SEAT", "t6cup", "centroid"), ("RAY", "t6cup", "centroid"), ("SEAT", "b1cup", "centroid")]
    res = {"info": info, "runs": {}}
    for jn, cm, am in runs:
        key = f"{jn}|{cm}|{am}"; res["runs"][key] = {}
        for cand, sub in SUBS.items():
            t = time.time()
            r = score_all(judge(jn), sub, cm, am)
            res["runs"][key][cand] = r
            print(f"{key:24s} {cand:5s} " + " ".join(f"{m}={r[m]}" for m in METRICS) +
                  f" | cup {r['diag']['beige_cup']['auc']} pot {r['diag']['white_pot']['auc']} align {r['diag']['align_deg']}"
                  f" | pos err cm {r['diag']['beige_cup']['mean_pos_err_cm']}/{r['diag']['white_pot']['mean_pos_err_cm']}"
                  f" orient deg {r['diag']['beige_cup']['mean_orient_err_deg']}/{r['diag']['white_pot']['mean_orient_err_deg']} ({time.time() - t:.0f}s)", flush=True)
        json.dump(res, open(out, "w"), indent=1)
    print("DONE", flush=True)
