for t in a b c d; do echo "-- $t $(grep -c 'rc=0' ~/v2d_ops/logs/push1104_taskA_$t.out) ok, $(grep -c 'rc=[1-9]' ~/v2d_ops/logs/push1104_taskA_$t.out) failed"; tail -n 2 ~/v2d_ops/logs/push1104_taskA_$t.out; done
srun --jobid=11035729 --overlap -n1 -c1 --mem=2G python3 ~/v2d_ops/kenc/push1104/taskA/agg.py < /dev/null 2>&1 | cut -c1-420
K=/data/user_data/haisenk/v2d/v2d_train/assets/data/v2d_submission_kit
