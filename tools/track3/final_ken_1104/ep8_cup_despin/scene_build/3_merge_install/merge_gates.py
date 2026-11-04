"""merge_gates.py O RAWBUNDLE BASEOBJ GAIN (refine1104 ep8): merge ONLY the cup-pose + right-hand columns of the prepared
despin build (O/despin_src/despin.parquet, from babel-t9-28 /scratch/haisenk-v2d/diag/ep8_plan/despin) into O/rf_ep8_ds
(a copy of the adopted scene sc_ep8), then evaluate the pre-registered gates G1-G8 (O/gates.json).
Writes O/gate_results.json. Never touches anything outside O."""
import sys, os, glob, json, hashlib, shutil
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, trimesh
from scipy.spatial.transform import Rotation as Rot

O, RAW, BASEOBJ, GAIN = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
SRC, OUT, DS = f"{O}/src_sc_ep8", f"{O}/rf_ep8_ds", f"{O}/despin_src"
STRETCH = np.array([0.0293, 0.7586, 0.6509])  # T6 stretch direction in the body frame (local d2_bottom.py)
assert os.path.realpath(OUT).startswith("/data/user_data/haisenk/v2d/refine1104/ep8/")


def pqpath(s):
    return glob.glob(s + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet")[0]


fa, fo, fd = pqpath(SRC), pqpath(OUT), f"{DS}/despin.parquet"
A = pq.ParquetFile(fa).read()
D = pq.ParquetFile(fd).read()
rep_full = [c for c in A.column_names if c.startswith("mano_right_") or c.startswith("robot_right_")] + ["object_root_position", "object_root_axis_angle"]
body = ["object_body_position", "object_body_wxyz"]
miss = [c for c in rep_full + body if c not in D.column_names]
assert not miss, miss
arrs = []
for c in A.column_names:
    ft = A.schema.field(c).type
    if c in rep_full:
        arrs.append(pa.array([D.column(c)[0].as_py()], type=ft))
    elif c in body:
        a = A.column(c)[0].as_py()
        d = D.column(c)[0].as_py()
        arrs.append(pa.array([[[d[t][0]] + list(a[t][1:]) for t in range(len(a))]], type=ft))
    else:
        arrs.append(A.column(c).combine_chunks())
pq.write_table(pa.Table.from_arrays(arrs, schema=A.schema), fo)
Mt = pq.ParquetFile(fo).read()
a_, m_, d_ = (t.to_pylist()[0] for t in (A, Mt, D))
R = {"gates": {}, "info": {}}


def gate(k, ok, **v):
    R["gates"][k] = {"pass": bool(ok), **v}
    print(k, "PASS" if ok else "FAIL", json.dumps(v, default=str)[:900], flush=True)


def P(t, k):
    return np.array(t[k], float)


# G8 shape
gate("G8_shape", Mt.column_names == A.column_names and Mt.num_rows == A.num_rows
     and P(m_, "object_body_position").shape == P(a_, "object_body_position").shape and m_["fps"] == a_["fps"],
     n_cols=len(Mt.column_names), frames=int(P(m_, "object_body_position").shape[0]))
# G2 untouched columns
bad, changed = [], []
for c in A.column_names:
    same = m_[c] == a_[c]
    if c in rep_full or c in body:
        if not same:
            changed.append(c)
    elif not same:
        bad.append(c)
pot_same = all(np.array_equal(P(m_, c)[:, 1], P(a_, c)[:, 1]) for c in body)
rep_ok = all(m_[c] == d_[c] for c in rep_full) and all(np.array_equal(P(m_, c)[:, 0], P(d_, c)[:, 0]) for c in body)
gate("G2_untouched_columns", not bad and pot_same and rep_ok, unexpected_diffs=bad, changed=changed, pot_identical=pot_same,
     replaced_equal_despin=rep_ok)
R["info"]["root_eq_cup_adopted_max_cm"] = float(100 * np.abs(P(a_, "object_root_position") - P(a_, "object_body_position")[:, 0]).max())
R["info"]["root_eq_cup_merged_max_cm"] = float(100 * np.abs(P(m_, "object_root_position") - P(m_, "object_body_position")[:, 0]).max())
# bundle_cleaned npz: replace by the despin build's iff every differing key is a right-hand / object key
na = np.load(f"{SRC}/bundle_cleaned/result.npz", allow_pickle=True)
nd = np.load(f"{DS}/bundle_cleaned_result.npz", allow_pickle=True)
kd = []
for k in na.files:
    try:
        x, y = na[k], nd[k]
        if x.shape != y.shape or not np.array_equal(x, y):
            kd.append(k)
    except Exception:
        kd.append(k + "?")
extra = [k for k in nd.files if k not in na.files]
okrep = bool(kd) and all(("right" in k.lower() or "object" in k.lower() or "obj" in k.lower()) for k in kd) and not extra
R["info"]["bundle_cleaned_diff_keys"] = kd
R["info"]["bundle_cleaned_extra_keys"] = extra
R["info"]["bundle_cleaned_replaced"] = okrep
if okrep:
    shutil.copyfile(f"{DS}/bundle_cleaned_result.npz", f"{OUT}/bundle_cleaned/result.npz")
print("bundle_cleaned diff keys", kd, "extra", extra, "replaced", okrep, flush=True)


# G1 untouched files
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


skip = {os.path.relpath(fo, OUT), "bundle_cleaned/result.npz"}
fl = sorted(os.path.relpath(os.path.join(r, f), SRC) for r, _, fs in os.walk(SRC) for f in fs)
fl2 = sorted(os.path.relpath(os.path.join(r, f), OUT) for r, _, fs in os.walk(OUT) for f in fs)
diff = [f for f in fl if f not in skip and md5(f"{SRC}/{f}") != md5(f"{OUT}/{f}")]
gate("G1_untouched_files", fl == fl2 and not diff, n_files=len(fl), differing=diff, same_file_list=fl == fl2)
# G3 pure spin (base mesh axis/centroid exactly as chk_scene.py)
m = trimesh.load(BASEOBJ, force="mesh", process=False)
pts, _ = trimesh.sample.sample_surface(m, 20000, seed=0)
c = pts.mean(0)
w, U = np.linalg.eigh(np.cov((pts - c).T))
sep = [min(abs(w[j] - w[(j + 1) % 3]), abs(w[j] - w[(j + 2) % 3])) for j in range(3)]
ax = U[:, int(np.argmax(sep))]


def rot(t):
    q = P(t, "object_body_wxyz")[:, 0]
    return Rot.from_quat(q[:, [1, 2, 3, 0]])


Ra, Rm = rot(a_), rot(m_)
pa_, pm_ = P(a_, "object_body_position")[:, 0], P(m_, "object_body_position")[:, 0]
cen = 100 * np.linalg.norm((Rm.apply(c) + pm_) - (Ra.apply(c) + pa_), axis=1)
axd = np.degrees(np.arccos(np.clip(np.sum(Rm.apply(ax) * Ra.apply(ax), 1), -1, 1)))
q = (Rm[0].inv() * Rm).as_quat()
tw = np.degrees(2 * np.arctan2(q[:, :3] @ ax, q[:, 3]))
tw = (tw + 180) % 360 - 180
gate("G3_pure_spin", cen.max() <= 0.05 and axd.max() <= 0.5 and np.abs(tw).max() <= 15, centroid_max_cm=round(float(cen.max()), 4),
     axis_max_deg=round(float(axd.max()), 3), twist_vs_f0_maxabs_deg=round(float(np.abs(tw).max()), 2),
     base_axis=np.round(ax, 4).tolist(), base_centroid=np.round(c, 5).tolist())
# G4 hand: chk_attach metric on the despin build's cleaned bundle (its right hand = the merged right-hand columns) + wrist rot step
raw = np.load(RAW + "/result.npz")
Wr = raw["hand_right_wrist_to_world_transform"].astype(float)
Wc = np.load(f"{DS}/bundle_cleaned_result.npz")["hand_right_wrist_to_world_transform"].astype(float)
Wa = np.load(f"{SRC}/bundle_cleaned/result.npz")["hand_right_wrist_to_world_transform"].astype(float)
dd = np.linalg.norm(Wc[:, :3, 3] - Wr[:, :3, 3], axis=1) * 100
da = np.linalg.norm(Wa[:, :3, 3] - Wr[:, :3, 3], axis=1) * 100


def wsteps(t):
    w_ = P(t, "robot_right_wrist_wxyz")
    r = Rot.from_quat(w_[:, [1, 2, 3, 0]])
    return np.r_[0, np.degrees((r[:-1].inv() * r[1:]).magnitude())]


sm, sa = wsteps(m_), wsteps(a_)
n15 = int((dd[44:201] > 15).sum())
gate("G4_hand_plausible", n15 == 0 and sm[55:73].max() <= 15, frames_gt15cm=n15, frames_gt15cm_adopted=int((da[44:201] > 15).sum()),
     max_dist_cm=round(float(dd[44:201].max()), 2), wrist_rotstep_max_f55_72=round(float(sm[55:73].max()), 1),
     adopted_rotstep_max_f55_72=round(float(sa[55:73].max()), 1), wrist_rotstep_max_all=round(float(sm.max()), 1))


# G5 contacts
def lab(t, s):
    ids = np.array(t[f"mano_{s}_object_contact_part_ids"], float)
    n = (ids > 0).sum(1)
    fr_ = np.nonzero(n > 0)[0]
    return int(fr_.min()), int(fr_.max()), int(n.sum()), sorted(set(np.unique(ids[ids > 0]).astype(int).tolist()))


lm, la = lab(m_, "right"), lab(a_, "right")
gate("G5_contacts", lm[0] == la[0] and lm[1] == la[1] and abs(lm[2] - la[2]) <= 0.05 * la[2] and lm[3] == la[3]
     and m_["mano_left_object_contact_part_ids"] == a_["mano_left_object_contact_part_ids"], merged=lm, adopted=la)
# G6 rest seating with the T6 mesh
t6 = trimesh.load(glob.glob(OUT + "/assets/human_motion_data/ego_recon/processed/*_beige_cup.obj")[0], force="mesh", process=False)
V = np.asarray(t6.vertices)
fr = list(range(0, 32)) + list(range(200, 219))
zm = np.array([(Rm[t].apply(V) + pm_[t])[:, 2].min() for t in fr])
za = np.array([(Ra[t].apply(V) + pa_[t])[:, 2].min() for t in fr])
gate("G6_rest_seating", np.abs(zm - za).max() <= 0.003, max_abs_diff_cm=round(float(100 * np.abs(zm - za).max()), 3),
     merged_lowest_end_min=round(float(zm[32:].min()), 4), adopted_lowest_end_min=round(float(za[32:].min()), 4))
# G7 side-effect budget (stretched-bottom tilt at f186-218)
Vb = np.asarray(trimesh.load(BASEOBJ, force="mesh", process=False).vertices)
S = STRETCH / np.linalg.norm(STRETCH)
if Vb.shape == V.shape:
    Dv = V - Vb
    mv = np.linalg.norm(Dv, axis=1)
    s2 = (Dv[mv > 1e-5] / mv[mv > 1e-5, None]).mean(0)
    s2 /= np.linalg.norm(s2)
    R["info"]["stretch_dir_babel"] = np.round(s2, 4).tolist()
    if np.degrees(np.arccos(np.clip(s2 @ S, -1, 1))) < 1.0:
        S = s2
else:
    R["info"]["stretch_dir_babel"] = "vertex lists differ; local constant used"
down = np.array([0, 0, -1.0])


def tilt(Rs, fr_):
    return np.degrees(np.arccos(np.clip(Rs[fr_].apply(S) @ down, -1, 1)))


fr = np.arange(186, 219)
tm, ta = tilt(Rm, fr), tilt(Ra, fr)
cost = float(np.sum(np.radians(np.maximum(0, tm - ta))) * 8.5 / 8 / 438)
gate("G7_side_effect_budget", cost <= 0.5 * GAIN, cost=round(cost, 5), gain_model=GAIN, limit=round(0.5 * GAIN, 5),
     tilt_merged_med=round(float(np.median(tm)), 2), tilt_adopted_med=round(float(np.median(ta)), 2), tilt_merged_max=round(float(tm.max()), 2))
R["all_pass"] = all(g["pass"] for g in R["gates"].values())
R["merged_parquet"] = fo
R["merged_md5"] = md5(fo)
json.dump(R, open(f"{O}/gate_results.json", "w"), indent=1, default=str)
print("ALL_PASS", R["all_pass"], flush=True)
