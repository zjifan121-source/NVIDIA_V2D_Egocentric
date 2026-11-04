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
