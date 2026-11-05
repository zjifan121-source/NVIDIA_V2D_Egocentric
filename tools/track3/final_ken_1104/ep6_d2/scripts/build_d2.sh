set -u; D=/data/user_data/haisenk/v2d/refine1104/ep6_d2; B=$D/build; hostname; date
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
SRC=/scratch/haisenk-v2d/diag/ep6_plan; OUTS=$D/scene/rf_ep6_d2
BOB1=/data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep6
[ -e $OUTS ] && { echo "scene exists $OUTS - refusing"; exit 3; }
/home/haisenk/anaconda3/bin/python - <<'PY'
import json, sys
r = json.load(open('/data/user_data/haisenk/v2d/refine1104/ep6_d2/work/out/d2_result.json'))
g = r['gates']; print('G5', json.dumps(g['G5'])); print('sigma', r['sigma_chosen'], 'all_pass_G1_G5', r['all_pass_G1_G5'])
print({k: g[k]['pass'] for k in g})
sys.exit(0 if r['all_pass_G1_G5'] is True else 4)
PY
[ $? = 0 ] || { echo "GATES G1-G5 NOT ALL PASS - no build"; exit 4; }
mkdir -p $B
[ -e $B/b_d2 ] || cp -a $SRC/b_ctl $B/b_d2
[ -e $B/repo ] || cp -a $SRC/repo $B/repo
du -sh $B/b_d2 $B/repo; ls -la $B/repo $B/repo/tools/track3 $B/b_d2 | head -40
md5sum $B/repo/tools/track3/run_load_retarget.py $SRC/repo/tools/track3/run_load_retarget.py $B/b_d2/result.npz $SRC/b_ctl/result.npz $BOB1/bundle_merged/result.npz | cut -c1-8,33-
grep source_run $B/b_d2/manifest.json
cat > $B/ep6_fixpatch.py << '__P_EOF__'
# ep6_fixpatch.py - H5_PATCH for refine1104 ep6 round 2 (D2 spoon de-smoothing).  exec'd by run_load_retarget.py
# (md5 a88f12fb) into its globals right before cleaned_bundle(); it wraps attach_hands only:
#   1. asserts that the pot (slot 0) and spoon (slot 1) tracks reaching attach_hands equal bob1's bundle_cleaned
#      (EP6_BOB1_CLEANED) within 1e-6, i.e. every earlier cleaning step (rest hold, levelling 1.4 / 15.4 deg) is bob1's;
#   2. writes the D2 spoon poses (EP6_TARGETS: object_1_to_world_transform in the bundle frame, object_1_mask) on the
#      masked frames only (f36-171), so bob1's levelled spoon is kept everywhere else;
#   3. calls the unchanged team attach rule (the right hand re-attaches to the corrected spoon);
#   4. with EP6_NO_LEFT_ATTACH=1 restores every hand_left_* array and drops left holds from the info, so the left
#      hand and the pot stay bob1's (bob1 had no left attach).
import os as _ep6_os

import numpy as _ep6_np

_ep6_orig_attach_hands = attach_hands  # noqa: F821  (run_load_retarget.attach_hands)


def attach_hands(data, bundle, *args, **kwargs):  # noqa: F811
    tg = _ep6_np.load(_ep6_os.environ["EP6_TARGETS"])
    ref = _ep6_np.load(_ep6_os.environ["EP6_BOB1_CLEANED"])
    for key in ("object_to_world_transform", "object_1_to_world_transform"):
        dmax = float(_ep6_np.abs(data[key].astype(_ep6_np.float64) - ref[key].astype(_ep6_np.float64)).max())
        print(f"== ep6 D2 patch: pre-attach {key} vs bob1 bundle_cleaned max |diff| {dmax:.3g}", flush=True)
        assert dmax <= 1e-6, f"{key} differs from bob1 bundle_cleaned by {dmax}"
    m = tg["object_1_mask"].astype(bool)
    T = data["object_1_to_world_transform"].astype(_ep6_np.float64)
    T[m] = tg["object_1_to_world_transform"][m]
    data["object_1_to_world_transform"] = T.astype(data["object_1_to_world_transform"].dtype)
    if "object_1_to_camera_transform" in data:
        ci = _ep6_np.linalg.inv(data["camera_to_world_transform"].astype(_ep6_np.float64))
        data["object_1_to_camera_transform"] = (ci @ data["object_1_to_world_transform"].astype(_ep6_np.float64)).astype(
            data["object_1_to_camera_transform"].dtype)
    fr = _ep6_np.flatnonzero(m)
    print(f"== ep6 D2 patch: spoon frames replaced: {int(m.sum())} [{int(fr.min())}..{int(fr.max())}]", flush=True)
    keep_left = _ep6_os.environ.get("EP6_NO_LEFT_ATTACH") == "1"
    saved = {k: data[k].copy() for k in list(data) if k.startswith("hand_left_")} if keep_left else {}
    info = _ep6_orig_attach_hands(data, bundle, *args, **kwargs)
    if keep_left:
        for k, v in saved.items():
            data[k] = v
        for k in list(info):
            if k.startswith("attached_") and isinstance(info[k], list):
                info[k] = [h for h in info[k] if h[0] != "left"]
                if not info[k]:
                    del info[k]
        info.pop("hand_left_attached_frames", None)
        info["ep6_no_left_attach"] = True
        print(f"== ep6 D2 patch: {len(saved)} hand_left_* arrays restored", flush=True)
    info["ep6_d2_spoon_frames"] = int(m.sum())
    print(f"== ep6 D2 patch: attach info {info}", flush=True)
    return info
__P_EOF__
cp $D/work/out/d2_targets.npz $B/d2_targets.npz
md5sum $B/ep6_fixpatch.py $B/d2_targets.npz | cut -c1-8,33-
/home/haisenk/anaconda3/bin/python -c "import numpy as np; z=np.load('$B/b_d2/result.npz'); print([k for k in z.files if k.startswith('object')])"
P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
T0=$(date +%s)
(cd $B/repo && H5_PATCH=$B/ep6_fixpatch.py EP6_TARGETS=$B/d2_targets.npz EP6_NO_LEFT_ATTACH=1 EP6_BOB1_CLEANED=$BOB1/bundle_cleaned/result.npz timeout 540 $P tools/track3/run_load_retarget.py --bundle $B/b_d2 --episode 6 --attach-hands --smooth-hands 2 --skip-video --device cpu --sequence-id ep6_bm_scored_attach --out $OUTS > $B/build.log 2>&1); echo "build rc $? in $(( $(date +%s)-T0 )) s"
grep "^==" $B/build.log | head -30; tail -5 $B/build.log; date
