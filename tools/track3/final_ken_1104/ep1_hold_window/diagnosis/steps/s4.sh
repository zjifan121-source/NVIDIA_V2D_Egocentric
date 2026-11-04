P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python; O=/scratch/haisenk-v2d/hand5w/r2/ep1
W=/scratch/haisenk-v2d/wentao_mix_20261103/experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103
cd ~/v2d_ops/kenc/hand5w/r2/ep1
$P diag3.py $W/scenes/episode_000001/bundle/result.npz /scratch/haisenk-v2d/hand5w/bundles/w_raw_ep1/result.npz 2>&1 | grep -v Warn | tee $O/diag3.txt
