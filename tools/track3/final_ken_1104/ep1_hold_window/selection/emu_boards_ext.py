"""ep1 status audit (4 Nov): emulated non-AUC board metrics (SP-SR, MP-SR margin, RPE, MPPE) and AUC for the adopted T2
rollout, its own 90M/95M checkpoint draws, the re-levelled candidate r43@95M and bob1, using the ep1 verification
synthesis emulator (C:\\v2d_demo\\diag\\ep1_verify\\synthesis\\py\\syn_lib.py = verbatim kit AUC.py functions, md5 93a20d94).
Truth stand-ins are rebuilt exactly as in s1_core.py (ref / jb / seat; GT pot = our pot trimmed t cm rim-fixed).
Read-only on all inputs; writes only ep1_work\\emu_boards.json and prints a table."""
import os, sys, json, time
import numpy as np
SYN = r"C:\v2d_demo\diag\ep1_verify\synthesis\py"
sys.path.insert(0, SYN)
from syn_lib import *  # noqa: F401,F403  (IN, load_rollout, load_packed, load_obj, GT, add_series, auc, other_metrics, ...)

OUTJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emu_boards_ext.json")
O44 = r"C:\v2d_demo\diag\skep1_A1\orch\o_ep1_hold_s44"
t0 = time.time()
achA, refA = load_rollout(os.path.join(IN, "adopt_roll.parquet"))
achC, refC = load_rollout(os.path.join(IN, "cand_roll.parquet"))
achB, refB = load_rollout(os.path.join(IN, "bob1_roll.parquet"))
ach95, _ = load_rollout(os.path.join(IN, "o44_95006720.parquet"))
ach90, _ = load_rollout(os.path.join(IN, "o44_90005504.parquet"))
ach100r, _ = load_rollout(os.path.join(IN, "o44_100003840.parquet"))
_, MA, _, _ = load_packed(os.path.join(IN, "adopt_one.parquet"))
_, MC, _, _ = load_packed(os.path.join(IN, "cand_one.parquet"))
_, MB, _, _ = load_packed(os.path.join(IN, "sub_T3_ep5.parquet"))

# --- same construction as s1_core.py
TA, TC = X_to_T(refA), X_to_T(refC)
G = TC[0, 0] @ np.linalg.inv(TA[0, 0])
nz = np.linalg.inv(G)[:3, :3] @ np.array([0, 0, 1.0]); nz /= np.linalg.norm(nz)
Vw, Fw = load_obj(os.path.join(IN, "w_pot.obj"))
Vr, Fr = load_obj(os.path.join(IN, "rl_pot.obj"))
Vl, Fl = load_obj(os.path.join(IN, "w_lid.obj"))
dV = Vr - Vw
FULLTRIM = 2.07
def standin(t):
    return (Vw + (t / FULLTRIM) * dV, Fw)
Vl_c = Vl.mean(0)
_c = Vl - Vl_c; _w, _E = np.linalg.eigh(_c.T @ _c); LAX = _E[:, 0]
TRUTH_JB = np.stack([T_to_X(np.asarray(json.load(open(os.path.join(IN, "base_truth.json")))["objects"][n]["T"], float))
                     for n in NAMES], 1)
def seat(truthX, Vpot, win=(67, 249)):
    T = X_to_T(truthX).copy()
    h0 = (xf(T[0, 0], Vpot) @ nz).min()
    for f in range(win[0], win[1] + 1):
        Tl = T[f, 1]; up = Tl[:3, :3] @ LAX; up = up if up @ nz > 0 else -up
        Rl = Rot.align_vectors([nz], [up])[0].as_matrix(); cw = Tl[:3, :3] @ Vl_c + Tl[:3, 3]
        Tn = np.eye(4); Tn[:3, :3] = Rl @ Tl[:3, :3]; Tn[:3, 3] = Rl @ (Tl[:3, 3] - cw) + cw
        h = (xf(Tn, Vl) @ nz).min(); Tn[:3, 3] -= (h - h0) * nz; T[f, 1] = Tn
    return T_to_X(T)
def truths_for(t):
    return {"ref": refA, "jb": TRUTH_JB, "seat": seat(refA, standin(t)[0])}

RUNS = {"adopt_T2_100M": (achA, MA), "o44_95M": (ach95, MA), "o44_90M": (ach90, MA), "o44_100M_reeval": (ach100r, MA),
        "cand_r43_95M": (achC, MC), "bob1": (achB, MB)}
R = {}
for t in (0.0, 2.07):
    gt = GT(standin(t), (Vl, Fl))
    for tn, truth in truths_for(t).items():
        key = f"t{t:.2f}_{tn}"; R[key] = {}
        for lab, (a, m) in RUNS.items():
            add, regs, al, rf = add_series(a, m, truth, gt)
            o = other_metrics(al, rf)
            R[key][lab] = dict(auc=round(float(auc(add)), 4), **{k: (round(float(v), 4) if not isinstance(v, list) else v)
                                                                  for k, v in o.items()})
        line = " | ".join(f"{lab} auc {v['auc']:.3f} SP {v['spider_sr']:.0f} MP {v['maniptrans_sr']:.0f} pos {v['mean_pos_cm']} "
                          f"rpe {v['rpe_cm']:.2f} mppe {v['mppe_cm']:.2f}" for lab, v in R[key].items())
        print(f"{key:<11} {line}  ({time.time() - t0:.0f}s)", flush=True)
json.dump(R, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
