set -u
F=$HOME/v2d_ops/kenc/final1104; TS=$HOME/v2d_ops/kenc/tests; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
echo "== $(hostname) $(date)"
echo "-- strays from the CRLF attempt:"; ls -la $F $F/local_best | cat -A | grep -F "^M"; CR=$(printf ""); rm -f "$F/local_best/manifest.txt$CR"; rmdir "$F/local_best$CR" 2>/dev/null; echo "strays left: $(ls -la $F $F/local_best | cat -A | grep -cF "^M")"
cat > $F/combine_final.sh <<'CF'
#!/bin/bash
# combine_final.sh OUTNAME EP:ONE_PARQUET [EP:ONE_PARQUET ...]  (Babel, compute step). Copy of ~/v2d_ops/kenc/combine.sh logic
# (bob1 + packed one-episode files chained with swap_episode.py --adopt-commit-url), but takes full one.parquet paths and
# writes under ~/v2d_ops/kenc/final1104/OUTNAME. No kit call here (dry-run is a separate script). Never submits.
set -u
OUT=$1; shift
PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
T=$HOME/v2d_ops/kenc/final1104/$OUT; mkdir -p $T
[ -e $T/track3_submission.parquet ] && { echo "$T/track3_submission.parquet exists - not overwriting"; exit 1; }
cur=$HOME/v2d_ops/kenc/bob1/candidate_bob1.parquet; i=0; : > $T/manifest.txt
echo "base $cur $(sha256sum $cur | cut -c1-64)" >> $T/manifest.txt
for spec in "$@"; do ep=${spec%%:*}; one=${spec#*:}; i=$((i+1))
  $PY $HOME/v2d_ops/swap_episode.py $cur $one $ep $T/step$i.parquet --adopt-commit-url 2>&1 | tee -a $T/swap.log | grep -E "wrote|refuse|REFUSED|ERROR|Error|checkpoint"
  [ -f $T/step$i.parquet ] || { echo "SWAP_FAILED ep$ep"; exit 1; }
  cur=$T/step$i.parquet
  echo "$ep $one $(sha256sum $one | cut -c1-64)" >> $T/manifest.txt
done
cp $cur $T/track3_submission.parquet
sha256sum $T/track3_submission.parquet | tee $T/track3_submission.parquet.sha256
CF
bash $F/combine_final.sh local_best 1:$F/b_ep1_wh_K_s48/one.parquet 3:$TS/h_ep3_sc_K_s42/one.parquet 5:$TS/r_ep5_rl_K_s43/one.parquet 8:$F/rf_ep8_ds_K_s43/one.parquet 10:$TS/n_ep10_nb_K_s42/one.parquet
cat $F/local_best/manifest.txt
echo "== verify vs base4_n10 (team base + ep10 NEST) and the one-episode files"
$PY - $F/local_best/track3_submission.parquet $HOME/v2d_ops/kenc/combos/base4_n10/track3_submission.parquet $F/b_ep1_wh_K_s48/one.parquet $F/rf_ep8_ds_K_s43/one.parquet <<'PY'
import sys, pandas as pd
new, base, o1, o8 = [pd.read_parquet(p) for p in sys.argv[1:5]]
print("rows", len(new), len(base), "cols equal", list(new.columns) == list(base.columns), "row_id order equal", (new.row_id.values == base.row_id.values).all())
ep = new.row_id.str.slice(8, 15)
eb = base.row_id.str.slice(8, 15)
for e in sorted(ep.unique()):
    a = new[ep == e].reset_index(drop=True); b = base[eb == e].reset_index(drop=True)
    same = a.equals(b)
    if not same or e in ("e000001", "e000008"): print(e, len(a), "identical to base4_n10" if same else "DIFFERS from base4_n10", a.checkpoint_sha256.iloc[0][:12], "base", b.checkpoint_sha256.iloc[0][:12])
for e, o in (("e000001", o1), ("e000008", o8)):
    a = new[ep == e].reset_index(drop=True).drop(columns=["code_commit_url"]); oo = o.reset_index(drop=True).drop(columns=["code_commit_url"])
    print(e, "rows equal to its one.parquet (except commit URL):", a.equals(oo))
print("uniform cols:", {c: new[c].astype(str).nunique() for c in ["eval_script_sha256","eval_code_sha256","packer_script_sha256","uploader_script_sha256","code_commit_url"]})
print("commit url:", new.code_commit_url.iloc[0])
PY
echo "== v2d_submit.py preflight + main (read-only check)"
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
sed -n '38,57p;232,363p;444,575p' $K/submit/v2d_submit.py
