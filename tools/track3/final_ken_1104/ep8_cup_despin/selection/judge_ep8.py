"""judge_ep8.py (refine1104 ep8): GT-free judge of ep8 rollouts with the VERBATIM kit scorer (AUC.py via
C:/v2d_demo/diag/ep_status_1104/ep8_work/emu8.py machinery; nothing there is modified, run with -B).
Each rollout (episode_000008.parquet of a training run, env 0 achieved poses = the packed rows, verified for T6) is scored
against common stand-in truths:
  REF     = the adopted sc_ep8 reference (FP-spun cup), T6 cup mesh, centroid anchor     (primary)
  REF_DS  = the despun reference of rf_ep8_ds (cup spin removed about its axis; video-closer spin), T6 mesh
  REF_B1  = REF with bob1's cup shape;  SEAT / RAY = pot moved by the dt8.log depth readings (emu8 stand-ins)
Cup symmetric (order 360 about the T6 PCA axis), pot not (kit README inference, as emu8). Outputs per rollout: the five
pose-board metrics + per-window cup/pot ADD (median cm, episode-AUC share lost).
Usage: python -B -I judge_ep8.py OUT.json ROLLOUT_DIR [ROLLOUT_DIR ...]   (ROLLOUT_DIR holds episode_000008.parquet)"""
import os, sys, json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\v2d_demo\diag\ep_status_1104\ep8_work")
os.chdir(r"C:\v2d_demo\diag\ep_status_1104\ep8_work")
import emu8 as E

NF = 219
DS_SCENE = os.path.join(HERE, "rollouts", "rf_ep8_ds_scene.parquet")


def ref_from_scene(path):
    r = pq.ParquetFile(path).read().to_pylist()[0]
    P = np.array(r["object_body_position"], float)
    Q = np.array(r["object_body_wxyz"], float)[..., [1, 2, 3, 0]]
    return E.M._normalise_quaternions(np.concatenate([P, Q], -1), "r", ValueError)


REF_DS = ref_from_scene(DS_SCENE)
assert np.abs(REF_DS[:, 1] - E.REF6[:, 1]).max() < 1e-6, "pot must be identical"


def judge(name):
    if name == "REF_DS":
        return REF_DS.copy()
    return E.judge(name)


STANDINS = [("REF", "t6cup"), ("REF_DS", "t6cup"), ("REF", "b1cup"), ("SEAT", "t6cup"), ("RAY", "t6cup")]
# optional speed-ups (same scoring code): JUDGE_STANDINS="REF|t6cup,REF_DS|t6cup"  JUDGE_METRICS="add_auc"
if os.environ.get("JUDGE_STANDINS"):
    STANDINS = [tuple(s.split("|")) for s in os.environ["JUDGE_STANDINS"].split(",")]
if os.environ.get("JUDGE_METRICS"):
    E.METRICS = [m for m in E.METRICS if m in os.environ["JUDGE_METRICS"].split(",")]
    assert E.METRICS[0] == "add_auc"
WIN = {"cup": {"blend-in f46-54": (46, 55), "spin1+pour f55-72": (55, 73), "burst f62-69": (62, 70), "carry f73-127": (73, 128),
               "pour2+spin2 f128-155": (128, 156), "f147-152": (147, 153), "blend-out+setdown+rest f176-218": (176, 219), "rest f0-31": (0, 32)},
       "pot": {"held f14-94": (14, 95), "lift f95-127": (95, 128), "lifted f128-164": (128, 165), "end f165-218": (165, 219)}}


def submission(rollout):
    """T6's packed rows (same meshes: rf_ep8_ds keeps T6's cup/pot meshes -> same object_mesh_b64) with the pose columns
    replaced by this rollout's env-0 achieved poses (for T6 itself this reproduces the packed rows exactly)."""
    a = E.rollout_poses(rollout, "achieved").reshape(-1, 7)
    sub = E.SUBS["T6"].copy()
    assert list(sub.row_id) == E.ROW_IDS
    for i, c in enumerate(E.M._POSE_COLUMNS):
        sub[c] = a[:, i]
    return sub


def windows():
    out = {}
    for i, n in enumerate(E.NAMES):
        add = np.asarray(E._LAST["spe"][i]).reshape(NF, -1)
        key = "cup" if i == 0 else "pot"
        out[key] = {w: {"med_cm": round(float(100 * np.median(add[a:b])), 2),
                        "lost": round((1 - E.M._add_auc(add[a:b, :, None])) * (b - a) / NF / 2, 4)} for w, (a, b) in WIN[key].items()}
    return out


if __name__ == "__main__":
    outp, dirs = sys.argv[1], sys.argv[2:]
    res = {}
    for d in dirs:
        name = os.path.basename(os.path.normpath(d))
        roll = os.path.join(d, "episode_000008.parquet")
        sub = submission(roll)
        own = json.load(open(os.path.join(d, "episode_000008.json")))["metrics"] if os.path.exists(os.path.join(d, "episode_000008.json")) else {}
        res[name] = {"own_ref": own.get("add_auc"), "step": open(os.path.join(d, "step.txt")).read().strip() if os.path.exists(os.path.join(d, "step.txt")) else None}
        for jn, cm in STANDINS:
            t = time.time()
            sol = E.solution_df(judge(jn), cm, "centroid")
            r = {}
            for m in E.METRICS:
                E.M._METRIC = m
                r[m] = round(float(E.M.score(sol.copy(), sub.copy(), "row_id")), 5)
                if m == "add_auc":
                    r["win"] = windows()
            E.M._METRIC = "add_auc"
            res[name][f"{jn}|{cm}"] = r
            print(f"{name:22s} {jn:6s} {cm:5s} " + " ".join(f"{m}={r[m]}" for m in E.METRICS)
                  + f" | cup f55-72 {r['win']['cup']['spin1+pour f55-72']['med_cm']} burst {r['win']['cup']['burst f62-69']['med_cm']}"
                  f" f128-155 {r['win']['cup']['pour2+spin2 f128-155']['med_cm']} end {r['win']['cup']['blend-out+setdown+rest f176-218']['med_cm']}"
                  f" | pot lift {r['win']['pot']['lift f95-127']['med_cm']} ({time.time() - t:.0f}s)", flush=True)
        json.dump(res, open(outp, "w"), indent=1)
    print("DONE", flush=True)
