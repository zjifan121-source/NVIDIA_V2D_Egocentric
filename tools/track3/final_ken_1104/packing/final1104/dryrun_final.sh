set -u
F=$HOME/v2d_ops/kenc/final1104/local_best; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
echo "== $(hostname) $(date)"
cd $K && $PY submit/v2d_submit.py $F/track3_submission.parquet --track track_3 --dry-run -m "local_best final1104 (dry-run only, NOT SUBMITTED)" > $F/dryrun.log 2>&1; echo "kit exit $?"
cat $F/dryrun.log
sha256sum $F/track3_submission.parquet $F/track3_submission.cd-o.parquet 2>&1 | tee $F/sha256.txt
cmp $F/track3_submission.cd-o.parquet $HOME/v2d_ops/kenc/combos/base4_n10/track3_submission.cd-o.parquet && echo "CD-O derived file byte-identical to base4_n10's"
ls -la $F
echo "== sizes for archive"
B=/tmp/haisenk_v2d
du -sh $B/scenes/w_hold_ep1 $B/scenes/rf_ep8_ds 2>&1
ls -la $B/runs/b_ep1_wh_K_s48/train/policy_100003840.safetensors $B/runs/rf_ep8_ds_K_s43/train/policy_95006720.safetensors
du -sh $B/runs/b_ep1_wh_K_s48/early_100003840/results $B/runs/rf_ep8_ds_K_s43/early_95006720/results
ls -la $B/runs/b_ep1_wh_K_s48/ $B/runs/rf_ep8_ds_K_s43/ | grep -v "^d\|^total"
date
