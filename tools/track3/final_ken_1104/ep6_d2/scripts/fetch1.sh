set -u; D=/data/user_data/haisenk/v2d/refine1104/ep6_d2; PY=/home/haisenk/anaconda3/bin/python
cat > $D/scene/rf_ep6_d2.FAILED_G6c_NOT_INSTALLED.txt << '__M_EOF__'
rf_ep6_d2 (refine1104 ep6 round 2, D2) was BUILT but FAILED pre-registered gate G6(c) (grip distance 1.6 vs bob1 2.2 cm,
limit 0.5 cm; right wrist in the spoon frame 4.0 cm / 21.4 deg off bob1's for the whole hold f32-185).
Not installed, not queued, not endorsed. See ../FAILED.json.
__M_EOF__
$PY - << 'PY' > $D/work/out/g6_extra.json
import json, numpy as np
from scipy.spatial.transform import Rotation as Rot
D = "/data/user_data/haisenk/v2d/refine1104/ep6_d2"
a = np.load(D + "/scene/rf_ep6_d2/bundle_cleaned/result.npz"); b = np.load("/data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep6/bundle_cleaned/result.npz")
Ha, Hb = a["hand_right_wrist_to_world_transform"].astype(float), b["hand_right_wrist_to_world_transform"].astype(float)
Ta, Tb = a["object_1_to_world_transform"].astype(float), b["object_1_to_world_transform"].astype(float)
dp = np.linalg.norm(Ha[:, :3, 3] - Hb[:, :3, 3], axis=1) * 100
dr = np.degrees(Rot.from_matrix(np.einsum("nji,njk->nik", Ha[:, :3, :3], Hb[:, :3, :3])).magnitude())
# D2b preview: hand follows the spoon correction rigidly (keeps bob1's grip): H = C(t) H_bob1, C = Ta Tb^-1
C = np.einsum("nij,njk->nik", Ta, np.linalg.inv(Tb))
Hc = np.einsum("nij,njk->nik", C, Hb)
dpc = np.linalg.norm(Hc[:, :3, 3] - Hb[:, :3, 3], axis=1) * 100
def seg(x, a_, b_): return dict(median=round(float(np.median(x[a_:b_])), 3), max=round(float(x[a_:b_].max()), 3))
out = dict(right_wrist_world_shift_cm={k: seg(dp, *v) for k, v in dict(f32_35=(32, 36), f36_171=(36, 172), f172_185=(172, 186)).items()},
           right_wrist_rot_change_deg={k: seg(dr, *v) for k, v in dict(f32_35=(32, 36), f36_171=(36, 172), f172_185=(172, 186)).items()},
           D2b_preview_hand_follows_spoon_shift_cm={k: seg(dpc, *v) for k, v in dict(f36_171=(36, 172)).items()},
           D2b_preview_outside_window_shift_cm=float(np.concatenate([dpc[:36], dpc[172:]]).max()))
print(json.dumps(out, indent=1))
PY
cat $D/work/out/g6_extra.json
cd $D && tar czf - gates.json work/d2.py work/g6.py work/kit_AUC.py work/interpen.py work/out/d2_result.json work/out/d2.log work/out/g6_result.json work/out/g6.log work/out/g6_extra.json work/out/interpen_new.json work/out/interpen_bob1.json work/out/d2_targets.npz work/out/pseudo_truth_V1_scene_world.npy build/build.log build/ep6_fixpatch.py scene/rf_ep6_d2/report.json | base64 -w0 > $D/work/out/fetch1.b64
md5sum $D/work/out/fetch1.b64 | cut -c1-8; wc -c < $D/work/out/fetch1.b64
