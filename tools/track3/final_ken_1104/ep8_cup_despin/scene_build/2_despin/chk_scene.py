"""ep8 planner (GT-free): compare a rebuilt scene B with the base scene A (parquet only).
Cup twist about its mesh symmetry axis (swing-twist, relative to frame 0 and cumulative), cup/pot pose diffs,
robot wrist step sizes, contact-label windows. usage: python -I chk_scene.py <sceneA> <sceneB> [more scenes]"""
import glob, sys, numpy as np, pyarrow.parquet as pq, trimesh
from scipy.spatial.transform import Rotation as Rot
def load(sc):
    f = glob.glob(sc + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet")[0]
    return pq.read_table(f).to_pylist()[0]
def R(q): q = np.asarray(q, float); return Rot.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))
def axis(mp):
    m = trimesh.load(mp, force="mesh", process=False); pts, _ = trimesh.sample.sample_surface(m, 20000, seed=0)
    c = pts.mean(0); w, U = np.linalg.eigh(np.cov((pts - c).T))
    sep = [min(abs(w[j] - w[(j + 1) % 3]), abs(w[j] - w[(j + 2) % 3])) for j in range(3)]
    return U[:, int(np.argmax(sep))], c
def twist(Rrel, a):
    q = Rot.from_matrix(Rrel).as_quat(); th = 2 * np.arctan2(q[..., :3] @ a, q[..., 3]); return (th + np.pi) % (2 * np.pi) - np.pi
def steps_deg(Rs): return np.r_[0, np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("tji,tjk->tik", Rs[:-1], Rs[1:])).as_rotvec(), axis=1))]
def labels(t, s):
    ids = t.get(f"mano_{s}_object_contact_part_ids")
    if ids is None: return "n/a"
    fr = [i for i, x in enumerate(ids) if x is not None and np.any(np.asarray(x, float) >= 0)]
    return f"{fr[0]}-{fr[-1]} ({len(fr)} fr)" if fr else "none"
A = load(sys.argv[1]); a, c = axis(A["object_mesh_paths"][0])
print("cup symmetry axis (object frame)", np.round(a, 3), "surface centroid", np.round(c, 4))
for sc in sys.argv[1:]:
    t = load(sc); P = np.array(t["object_body_position"], float); RO = R(np.array(t["object_body_wxyz"], float))
    Rc = RO[:, 0]; tw0 = np.degrees(twist(np.einsum("ji,tjk->tik", Rc[0], Rc), a))
    inc = np.degrees(twist(np.einsum("tji,tjk->tik", Rc[:-1], Rc[1:]), a)); cum = np.r_[0, np.cumsum(inc)]
    st = steps_deg(Rc)
    print(f"\n== {sc}\n cup twist vs f0: min {tw0.min():.1f} max {tw0.max():.1f} last {tw0[-1]:.1f}; cumulative last {cum[-1]:.1f}, sum|inc| {np.abs(inc).sum():.0f}"
          f"; rot step max {st.max():.1f} deg at f{int(st.argmax())}, frames >10 deg: {int((st > 10).sum())}")
    for s in ("right", "left"):
        WP = np.array(t[f"robot_{s}_wrist_position"], float); WR = R(np.array(t[f"robot_{s}_wrist_wxyz"], float))
        tr = np.r_[0, np.linalg.norm(np.diff(WP, axis=0), axis=1) * 100]; rs = steps_deg(WR)
        win = {"f40-56": slice(40, 57), "f56-70": slice(56, 71), "f125-156": slice(125, 157), "f170-190": slice(170, 191)}
        print(f" {s} wrist: rot step max {rs.max():.1f} deg (f{int(rs.argmax())}), trans step max {tr.max():.2f} cm (f{int(tr.argmax())}); "
              + "; ".join(f"{k} rot {rs[v].max():.1f} trans {tr[v].max():.2f}" for k, v in win.items()) + f"; labels {labels(t, s)}")
    if sc != sys.argv[1]:
        PA = np.array(A["object_body_position"], float); RA = R(np.array(A["object_body_wxyz"], float))
        for k, n in ((0, "cup"), (1, "pot")):
            dp = np.linalg.norm(P[:, k] - PA[:, k], axis=1) * 100
            ax_b = RO[:, k] @ a; ax_a = RA[:, k] @ a; dax = np.degrees(np.arccos(np.clip((ax_a * ax_b).sum(1), -1, 1)))
            dang = np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("tji,tjk->tik", RA[:, k], RO[:, k])).as_rotvec(), axis=1))
            cpt = np.linalg.norm((RO[:, k] @ c + P[:, k]) - (RA[:, k] @ c + PA[:, k]), axis=1) * 100
            print(f" {n} vs base: origin diff max {dp.max():.3f} cm, centroid-point diff max {cpt.max():.3f} cm, axis-dir diff max {dax.max():.2f} deg, full rot diff max {dang.max():.1f} deg")
        for s in ("right", "left"):
            d = np.linalg.norm(np.array(t[f"robot_{s}_wrist_position"], float) - np.array(A[f"robot_{s}_wrist_position"], float), axis=1) * 100
            print(f" {s} wrist vs base: pos diff max {d.max():.2f} cm (f{int(d.argmax())}), frames >1 cm: {int((d > 1).sum())}")
