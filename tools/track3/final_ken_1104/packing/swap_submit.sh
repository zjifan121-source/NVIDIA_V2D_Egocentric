#!/bin/bash
# swap_submit.sh EP NAME RESULTS_DIR MESH_DIR MSG [SUBMIT=0|1]   (Babel, inside a compute step)
# Pack ONE episode (team relabel + kit packer, pack_one.sh), splice it into bob1's exact Kaggle file (other 19 episodes
# byte-identical, bob1's commit URL adopted), validate with the kit's v2d_submit.py --dry-run; SUBMIT=1 then submits for
# real (fans out to all 6 Track 3 boards). Output: ~/v2d_ops/kenc/tests/NAME/track3_submission.parquet + logs.
set -u
EP=$1 NAME=$2 RES=$3 MESH=$4 MSG=$5 SUB=${6:-0}
OPT=""; [ -n "${ONLY:-}" ] && OPT="--only $ONLY"   # e.g. ONLY=add_auc: test on the AUC board only (kit option)
PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
T=$HOME/v2d_ops/kenc/tests/$NAME; mkdir -p $T
URL=https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6
BASE=$HOME/v2d_ops/kenc/bob1/candidate_bob1.parquet
rm -f $T/one.parquet $T/track3_submission.parquet
bash $HOME/v2d_ops/pack_one.sh $EP $RES $MESH $URL $T/one.parquet > $T/pack.log 2>&1; tail -6 $T/pack.log
[ -f $T/one.parquet ] || { echo PACK_FAILED; exit 1; }
$PY $HOME/v2d_ops/swap_episode.py $BASE $T/one.parquet $EP $T/track3_submission.parquet --adopt-commit-url 2>&1 | tail -4
[ -f $T/track3_submission.parquet ] || { echo SWAP_FAILED; exit 1; }
cd $K
$PY submit/v2d_submit.py $T/track3_submission.parquet --track track_3 $OPT --dry-run -m "$MSG" > $T/dryrun.log 2>&1; tail -14 $T/dryrun.log
if [ "$SUB" = 1 ]; then
  grep -qi "error\|invalid\|fail" $T/dryrun.log && { echo "DRY-RUN REPORTED A PROBLEM - not submitting"; exit 1; }
  LEFT=$(grep -o "[0-9]* left today" $T/dryrun.log | awk '{print $1}' | sort -n | head -1)
  # Ken 4 Nov 01:45: stop swap tests once the team counter is down to 3 (judge locally after that)
  if [ -z "$LEFT" ] || [ "$LEFT" -le 3 ]; then echo "KAGGLE COUNTER AT ${LEFT:-unknown} LEFT (<= 3): NOT SUBMITTING (Ken's rule)"; exit 3; fi
  $PY submit/v2d_submit.py $T/track3_submission.parquet --track track_3 $OPT -m "$MSG" > $T/submit.log 2>&1; tail -20 $T/submit.log
  date '+%F %T' > $T/SUBMITTED
fi
