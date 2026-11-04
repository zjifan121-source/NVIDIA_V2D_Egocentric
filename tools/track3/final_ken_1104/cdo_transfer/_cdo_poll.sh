K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit; PY=/home/haisenk/v2d_ops/kaggle_env/bin/python
R=/home/haisenk/v2d_ops/kenc/cdo/final/cdo_w33_ep10beigeT6.parquet.3985f46a1cbc.kaggle-receipt.json
cd $K
for i in $(seq 1 30); do
  out=$($PY submit/v2d_submit.py --track track_3_mesh --poll-receipt $R 2>&1 | tail -4)
  echo "$out" | grep -q "COMPLETE\|ERROR\|FAILED" && { echo "$out"; exit 0; }
  sleep 30
done
echo "still pending: $out"
