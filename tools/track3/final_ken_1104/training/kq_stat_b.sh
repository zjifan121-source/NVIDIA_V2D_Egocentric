# kq_stat.sh (inside a step on babel-s5-16): GPU use, driver log tail, queue, per-run progress/best
B=/tmp/haisenk_v2d; cd $B
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "-- kq log"; tail -12 ~/v2d_ops/logs/kq_babel.out 2>/dev/null
echo "-- queue"; grep -v '^#' kq_babel.txt 2>/dev/null | awk '{print $1, $5, $6}'
for r in $(grep -v '^#' kq_babel.txt 2>/dev/null | awk '{print $1}'); do
  [ -d runs/$r ] || { echo "$r not started"; continue; }
  p=$(tail -c 3000 runs/$r/train.log 2>/dev/null | tr '\r' '\n' | grep "training progress" | tail -1 | grep -o "([0-9.]*%)")
  echo "$r $p best=$(cat runs/$r/best.txt 2>/dev/null) $(ls runs/$r/FAILED 2>/dev/null)"
done
