# kq_start.sh (run on the Babel LOGIN node via ssh babel): start the kq driver as one background step of the H100 job on
# babel-s5-16, all 8 GPUs, queue /tmp/haisenk_v2d/kq_babel.txt (append lines any time). Refuses if a v2d-kq step already runs.
set -u
J=10774619
if squeue -s -j $J -h -o '%.20j' 2>/dev/null | grep -q v2d-kq; then echo "v2d-kq step already running"; squeue -s -j $J; exit 0; fi
mkdir -p ~/v2d_ops/logs
setsid nohup srun --jobid=$J --overlap --whole -n1 --job-name=v2d-kq bash -c '
  B=/tmp/haisenk_v2d; [ -e $B/v2d_train ] || ln -s /data/user_data/haisenk/v2d/v2d_train $B/v2d_train
  touch $B/kq_babel.txt; exec bash $B/kq.sh $B/kq_babel.txt 0 1 2 3 4 5 6 7' > ~/v2d_ops/logs/kq_babel.out 2>&1 < /dev/null &
sleep 8; squeue -s -j $J -o '%.18i %.12j %.10M'; tail -3 ~/v2d_ops/logs/kq_babel.out
