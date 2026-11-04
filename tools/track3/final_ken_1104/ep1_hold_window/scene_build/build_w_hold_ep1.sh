R=/scratch/haisenk-v2d/hand5w/repo/tools/track3/run_load_retarget.py
grep -n "attach_hands(\|^def mask_contact\|^def hold_owner\|^def holds\|^def lifted_segments\|^def object_vertices\|^def mano_rest_wrist" $R
cp ~/v2d_ops/kenc/hand5w/r2/ep1/patch_whold.py /scratch/haisenk-v2d/hand5w/r2/ep1/patch_whold.py
md5sum /scratch/haisenk-v2d/hand5w/r2/ep1/patch_whold.py
H5_PATCH=/scratch/haisenk-v2d/hand5w/r2/ep1/patch_whold.py bash ~/v2d_ops/kenc/hand5w/wbuild2.sh w_hold_ep1 1 rawhands 2>&1 | tail -15
grep -a "ep1n attach\|hand5: patch" /scratch/haisenk-v2d/hand5w/scenes/w_hold_ep1.log | cut -c1-600
/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python - <<'PY'
import json; r=json.load(open("/scratch/haisenk-v2d/hand5w/scenes/w_hold_ep1/report.json"))
c=r.get("cleaning",{})
for k,v in c.items():
    if any(s in k for s in ("attach","ep1n","refill","rule","smooth")): print(" ",k,v)
print(" h5_patch", r.get("h5_patch"))
PY
