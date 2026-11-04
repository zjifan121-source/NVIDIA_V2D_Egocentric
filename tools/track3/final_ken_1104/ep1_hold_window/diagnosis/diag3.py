"""ep1: grip stability of the true MANO wrist in the object frame (raw unattached hands vs Wentao-delivered hands),
and object vs hand high-frequency jitter. Bundle world. usage: diag3.py <W bundle npz> <raw prep npz>"""
import sys, numpy as np
sys.path.insert(0, "/scratch/haisenk-v2d/hand5w/repo/tools/track3")
import run_load_retarget as rlr
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
W = dict(np.load(sys.argv[1], allow_pickle=True)); Rw = dict(np.load(sys.argv[2], allow_pickle=True))
mirror = np.diag([1.0, -1, -1])
def wrist(d, side):
    c2w = d["camera_to_world_transform"].astype(np.float64); Rc, tc = c2w[:, :3, :3], c2w[:, :3, 3]
    j0 = rlr.mano_rest_wrist(side, d[f"hand_{side}_betas"].reshape(10))
    Rm = R.from_rotvec(d[f"hand_{side}_wrist_orient_in_camera"].astype(np.float64)).as_matrix()
    if side == "left": Rm = mirror @ Rm @ mirror
    H = np.tile(np.eye(4), (len(c2w), 1, 1)); H[:, :3, :3] = Rc @ Rm
    H[:, :3, 3] = np.einsum("nij,nj->ni", Rc, d[f"hand_{side}_wrist_trans_in_camera"] + j0) + tc
    return H
obj = {"pot": W["object_to_world_transform"].astype(np.float64), "lid": W["object_1_to_world_transform"].astype(np.float64)}
print("camera identical W vs prep:", np.array_equal(W["camera_to_world_transform"], Rw["camera_to_world_transform"]))
for side, on, wins in (("left", "pot", [(30, 100), (100, 127), (127, 221), (221, 250), (250, 305), (30, 305)]),
                       ("right", "lid", [(36, 47), (47, 59), (59, 70), (70, 97), (241, 250), (250, 272), (272, 303)])):
    T = obj[on]
    for tag, d in (("raw", Rw), ("delivered", W)):
        H = wrist(d, side)
        L = np.einsum("nji,nj->ni", T[:, :3, :3], H[:, :3, 3] - T[:, :3, 3])  # wrist in object frame
        Ro = R.from_matrix(np.einsum("nji,njk->nik", T[:, :3, :3], H[:, :3, :3]))
        row = []
        for a, b in wins:
            med = np.median(L[a:b], axis=0); dev = np.linalg.norm(L[a:b] - med, axis=1) * 100
            mr = Ro[a:b].mean(); rd = np.degrees((Ro[a:b] * mr.inv()).magnitude())
            row.append(f"[{a},{b}) pos dev med {np.median(dev):.1f} p90 {np.percentile(dev,90):.1f} cm, rot dev med {np.median(rd):.0f} p90 {np.percentile(rd,90):.0f} deg")
        print(f"{side} wrist in {on} frame, {tag}:\n   " + "\n   ".join(row))
        # grip consistency between phases: median pose per phase vs the lift phase
        if on == "pot":
            ref = np.median(L[127:221], axis=0)
            print("   phase medians vs lift median (cm):", {f"{a}-{b}": round(float(np.linalg.norm(np.median(L[a:b], axis=0) - ref) * 100), 1) for a, b in wins})
# HF jitter during the pot lift: world positions minus gaussian(sigma 3) smooth
a, b = 127, 221
def hf(x): return np.linalg.norm(x - gaussian_filter1d(x, 3, axis=0), axis=1)[a:b] * 1000
pp = obj["pot"][:, :3, 3]
print(f"lift [{a},{b}) HF residual RMS mm: pot origin {np.sqrt((hf(pp)**2).mean()):.1f} | raw left wrist {np.sqrt((hf(wrist(Rw,'left')[:, :3, 3])**2).mean()):.1f} | delivered left wrist {np.sqrt((hf(wrist(W,'left')[:, :3, 3])**2).mean()):.1f}")
for a2, b2 in ((30, 127), (221, 305)):
    print(f"[{a2},{b2}) HF RMS mm: pot {np.sqrt((np.linalg.norm(pp - gaussian_filter1d(pp, 3, axis=0), axis=1)[a2:b2]*1000)**2).mean()**0.5:.1f}")
lp = obj["lid"][:, :3, 3]
print("lid HF RMS mm [40,75):", round(float(np.sqrt(((np.linalg.norm(lp - gaussian_filter1d(lp, 3, axis=0), axis=1)[40:75]*1000)**2).mean())), 1),
      "[245,280):", round(float(np.sqrt(((np.linalg.norm(lp - gaussian_filter1d(lp, 3, axis=0), axis=1)[245:280]*1000)**2).mean())), 1))
# per-frame lid height and pot speed around the carries
for t in list(range(44, 72, 2)) + list(range(244, 280, 2)):
    print(f"f{t} lid z {100*(lp[t,2]-lp[0,2]):5.1f} cm  lid speed {100*np.linalg.norm(lp[t]-lp[t-1]):4.1f} cm/f")
