"""anchorseat.py RESULTS.parquet [...]: per object, achieved-vs-reference rotation angle (deg) and position offset (cm) at
frames 0,1,2,5,10,20 and the mean over frames 0-30 (first environment). Shows whether the sim re-seats objects at reset."""
import sys, numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
for f in sys.argv[1:]:
    t = pq.read_table(f); c = {n: np.asarray(t.column(n).to_pylist()) for n in t.column_names}
    env0 = c["environment_id"] == c["environment_id"].min()
    print("==", f.split("/runs/")[-1])
    for b in sorted(set(c["object_body_name"][env0])):
        m = env0 & (c["object_body_name"] == b); o = np.argsort(c["reference_frame"][m])
        g = lambda k: np.stack([c[k][m][o] for k in k], 1)
        qa = g(["achieved_quaternion_x", "achieved_quaternion_y", "achieved_quaternion_z", "achieved_quaternion_w"]).astype(float)
        qr = g(["reference_quaternion_x", "reference_quaternion_y", "reference_quaternion_z", "reference_quaternion_w"]).astype(float)
        pa = g(["achieved_position_x", "achieved_position_y", "achieved_position_z"]).astype(float)
        pr = g(["reference_position_x", "reference_position_y", "reference_position_z"]).astype(float)
        ang = np.degrees((R.from_quat(qa) * R.from_quat(qr).inv()).magnitude()); dp = 100 * np.linalg.norm(pa - pr, axis=1)
        pick = [0, 1, 2, 5, 10, 20]
        print(f"  {'_'.join(b.split('_')[-2:]):16s} rot err deg " + " ".join(f"f{i}:{ang[i]:.1f}" for i in pick if i < len(ang))
              + f" | mean f0-30 {ang[:31].mean():.1f} | pos err cm f0 {dp[0]:.2f} f10 {dp[10]:.2f} mean f0-30 {dp[:31].mean():.2f}")
