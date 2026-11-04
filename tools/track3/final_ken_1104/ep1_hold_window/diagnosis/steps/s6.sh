P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python; O=/scratch/haisenk-v2d/hand5w/r2/ep1; S=/scratch/haisenk-v2d/hand5w/scenes
W=/scratch/haisenk-v2d/wentao_mix_20261103/experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103
H5=~/v2d_ops/kenc/hand5
head -5 $H5/contact_audit2.py | grep -i usage; head -5 $H5/cmp_scene.py | grep -i usage
cd ~/v2d_ops/kenc/hand5w/r2/ep1
$P ev.py $S/w_hold_ep1 $O/ev_w_hold_ep1.npz > $O/ev_w_hold_ep1.txt 2>&1; echo ev rc $?
$P diag2.py $O/ev_w_ctl_ep1.npz $O/ev_w_hold_ep1.npz $S/w_ctl_ep1 $S/w_hold_ep1 $S/w_hold_ep1/bundle_cleaned/result.npz $S/w_raw_ep1/bundle_cleaned/result.npz 2>&1 | grep -v Warn > $O/diag2_hold.txt
grep -n "left:\|right:\|contact labels\|ctl vs raw\|phantom\|on pot\|on lid" $O/diag2_hold.txt | sed -n '1,40p' | cut -c1-400
$P diag2.py $O/ev_w_raw_ep1.npz $O/ev_w_hold_ep1.npz $S/w_raw_ep1 $S/w_hold_ep1 $S/w_hold_ep1/bundle_cleaned/result.npz $S/w_raw_ep1/bundle_cleaned/result.npz 2>&1 | grep "ctl vs raw"
echo "== geo hold lift"; $P geo_b.py $S/w_hold_ep1 130 215 2>&1 | grep -v Warn | head -9
echo "== geo hold table 40-120"; $P geo_b.py $S/w_hold_ep1 40 120 2>&1 | grep -v Warn | sed -n 2,8p
echo "== geo hold 225-300"; $P geo_b.py $S/w_hold_ep1 225 300 2>&1 | grep -v Warn | sed -n 2,8p
echo "== geo ctl 225-300"; $P geo_b.py $S/w_ctl_ep1 225 300 2>&1 | grep -v Warn | sed -n 2,8p
# array identity: bundle_cleaned hold vs raw vs Wentao
$P - <<'PY'
import numpy as np
S="/scratch/haisenk-v2d/hand5w/scenes"; W="/scratch/haisenk-v2d/wentao_mix_20261103/experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103/scenes/episode_000001/bundle/result.npz"
h=np.load(f"{S}/w_hold_ep1/bundle_cleaned/result.npz",allow_pickle=True); r=np.load(f"{S}/w_raw_ep1/bundle_cleaned/result.npz",allow_pickle=True); w=np.load(W,allow_pickle=True)
for ref,nm in ((r,"w_raw"),(w,"wentao")):
    diff=[k for k in h.files if k in ref.files and not np.array_equal(h[k],ref[k])]
    print(f"hold vs {nm}: differing keys {diff}; missing {[k for k in ref.files if k not in h.files]}")
    obj=[k for k in h.files if k.startswith("object") or k.startswith("camera")]
    print(f"   object/camera keys identical to {nm}: {all(np.array_equal(h[k],ref[k]) for k in obj if k in ref.files)} ({len(obj)} keys)")
PY
