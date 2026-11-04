#!/bin/bash
# combine.sh OUTNAME EP:CAND [EP:CAND ...]  (Babel, compute step): bob1 + the packed one-episode files of the given
# candidates (~/v2d_ops/kenc/tests/CAND/one.parquet), chained with swap_episode.py; then kit dry-run. Never submits.
set -u
OUT=$1; shift
PY=/home/haisenk/v2d_ops/kaggle_env/bin/python; K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
T=$HOME/v2d_ops/kenc/combos/$OUT; mkdir -p $T; cur=$HOME/v2d_ops/kenc/bob1/candidate_bob1.parquet; i=0
[ -e $T/track3_submission.parquet ] && { echo "exists: $T - not overwriting"; exit 1; }
for spec in "$@"; do ep=${spec%%:*}; cand=${spec#*:}; i=$((i+1))
  $PY $HOME/v2d_ops/swap_episode.py $cur $HOME/v2d_ops/kenc/tests/$cand/one.parquet $ep $T/step$i.parquet --adopt-commit-url 2>&1 | grep -E "wrote|refuse|ERROR|Error" ; cur=$T/step$i.parquet
  echo "$ep $cand" >> $T/manifest.txt
done
cp $cur $T/track3_submission.parquet
cd $K && $PY submit/v2d_submit.py $T/track3_submission.parquet --track track_3 --dry-run -m "$OUT" > $T/dryrun.log 2>&1; grep -iE "valid|rows|left|error|identity|refus" $T/dryrun.log | head -14
sha256sum $T/track3_submission.parquet
