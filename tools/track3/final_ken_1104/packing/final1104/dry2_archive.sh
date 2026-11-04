set -u
F=$HOME/v2d_ops/kenc/final1104; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python; B=/tmp/haisenk_v2d
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
echo "== $(hostname) $(date)"
cd $K
$PY submit/v2d_submit.py $F/local_best/track3_submission.parquet --track track_3 --only add_auc spider_sr maniptrans_sr rpe_cm --dry-run -m "local_best final1104 (dry-run only, NOT SUBMITTED)" > $F/local_best/dryrun_only_auc_sp_mp_rpe.log 2>&1; echo "kit exit (auc/sp/mp/rpe) $?"
$PY submit/v2d_submit.py $F/local_best/track3_submission.parquet --track track_3 --only mppe_cm cd_o_cm --dry-run -m "local_best final1104 (dry-run only, NOT SUBMITTED)" > $F/local_best/dryrun_only_mppe_cdo.log 2>&1; echo "kit exit (mppe/cd-o) $?"
cat $F/local_best/dryrun_only_auc_sp_mp_rpe.log $F/local_best/dryrun_only_mppe_cdo.log
sha256sum $F/local_best/track3_submission.parquet
cd $HOME
echo "== scene evidence"
for c in b_ep1_wh_K_s48 rf_ep8_ds_K_s43; do grep -rlE "w_hold_ep1|rf_ep8_ds" $B/runs/$c --include=*.yaml --include=*.log --include=*.txt --include=*.json 2>/dev/null | head -3; grep -rhoE "(w_hold_ep1|rf_ep8_ds)[^ \"']{0,40}" $B/runs/$c/train.log $B/runs/$c/early_*/results/hydra 2>/dev/null | sort | uniq -c | head -3; done
grep -rh "b_ep1_wh_K_s48\|rf_ep8_ds_K_s43" $B/*.txt $B/kq/*.txt ~/v2d_ops/*tasks*.txt 2>/dev/null | head -4
echo "== build archive dirs (staged in final1104/CAND/archive)"
arch() { # CAND SCENE EP STEP NOTE
  local C=$1 SC=$2 EP=$3 S=$4 NOTE=$5 A=$F/$1/archive/$1 R=$B/runs/$1
  [ -e $A ] && { echo "$A exists - skip"; return; }
  mkdir -p $A/babel
  tar czf $A/scene.tgz -C $B/scenes $SC
  cp -a $R/train/policy_$S.safetensors $A/
  cp -a $R/early_$S/results $A/results
  for f in sweep.json best.txt recipe.txt train.log checkpoints.json; do [ -f $R/$f ] && cp -a $R/$f $A/; done
  cp -a $F/$C/one.parquet $F/$C/pack.log $F/$C/pack_args.txt $A/babel/
  python3 - $A "$C" "$SC" "$EP" "$S" "$R/early_$S/results" "$NOTE" "$F/$C/one.parquet" <<'PY'
import json, sys, glob, hashlib, os
A, cand, scene, ep, step, res, note, one = sys.argv[1:9]
m = json.load(open(glob.glob(f"{A}/results/episode_*.json")[0]))["metrics"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = dict(candidate=cand, episode=int(ep), scene=scene, scene_tgz="scene.tgz", checkpoint_step=int(step), step_arg=step,
           results_source=res, own_ref=m.get("add_auc"), own_ref_per_body=m.get("add_auc_per_body"),
           policy_file=f"policy_{step}.safetensors", policy_sha256=sha(f"{A}/policy_{step}.safetensors"),
           rollout_sha256=sha(glob.glob(f"{A}/results/episode_*.parquet")[0]),
           recipe=open(f"{A}/recipe.txt").read().strip() if os.path.exists(f"{A}/recipe.txt") else None,
           scene_tgz_sha256=sha(f"{A}/scene.tgz"), packed_one_parquet=one, packed_one_sha256=sha(one),
           packed_commit_url="https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6",
           source_host="babel-s5-16 /tmp/haisenk_v2d (job 10774619)", status="NOT SUBMITTED (local best, 4 Nov final1104)", note=note)
json.dump(man, open(f"{A}/manifest.json", "w"), indent=1)
print(json.dumps({k: man[k] for k in ("candidate","scene","checkpoint_step","own_ref","own_ref_per_body","policy_sha256","rollout_sha256","packed_one_sha256")}))
PY
  (cd $A && find . -type f ! -name SHA256SUMS -exec sha256sum {} \; | sort -k2 > SHA256SUMS)
  du -sh $A; ls -la $A
}
arch b_ep1_wh_K_s48 w_hold_ep1 1 100003840 "ep1 local best (4 Nov judge: beats T2 o_ep1_hold_s44@100M in 16/16 Kaggle-calibrated emulator cells, median AUC 0.887 vs 0.847); scene = adopted T2 scene w_hold_ep1; NOT SUBMITTED"
arch rf_ep8_ds_K_s43 rf_ep8_ds 8 95006720 "ep8 local best (despin scene rf_ep8_ds; beats T6 h_ep8_sc_K_s42@100M under all 5 GT-free stand-ins, ep8/JUDGED.json; review approve_with_caveats); NOT SUBMITTED"
date
