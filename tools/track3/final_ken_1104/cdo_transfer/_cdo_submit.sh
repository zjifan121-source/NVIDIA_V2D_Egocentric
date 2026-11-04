set -u
C=$HOME/v2d_ops/kenc/cdo; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
F=$C/final/cdo_w33_ep10beigeT6.parquet
MSG="Ken CD-O test: team best (w33 = base + ep33 probe) + ep10 beige_cup <- T6 ep8 beige_cup mesh (same object; depth-measured +2.8 cm, rim fixed)"
cd $K
$PY submit/v2d_submit.py $F --track track_3_mesh --dry-run -m "$MSG" > $C/final/dryrun.log 2>&1; cat $C/final/dryrun.log | tail -4
LEFT=$(grep -o "[0-9]* left today" $C/final/dryrun.log | awk '{print $1}' | sort -n | head -1)
if [ -z "$LEFT" ] || [ "$LEFT" -le 3 ]; then echo "COUNTER ${LEFT:-unknown} <= 3: NOT SUBMITTING"; exit 3; fi
grep -qi "error\|invalid\|fail" $C/final/dryrun.log && { echo "DRY-RUN PROBLEM - not submitting"; exit 1; }
$PY submit/v2d_submit.py $F --track track_3_mesh -m "$MSG" > $C/final/submit.log 2>&1; tail -8 $C/final/submit.log
date '+%F %T' > $C/final/SUBMITTED
