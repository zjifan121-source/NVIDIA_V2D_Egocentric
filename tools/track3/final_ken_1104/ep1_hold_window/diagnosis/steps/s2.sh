O=/scratch/haisenk-v2d/hand5w/r2/ep1; S=/scratch/haisenk-v2d/hand5w/scenes; P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
W=/scratch/haisenk-v2d/wentao_mix_20261103/experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103
cd ~/v2d_ops/kenc/hand5w/r2/ep1
ls /scratch/haisenk-v2d/hand5w/bundles/w_raw_ep1/
$P diag2.py $O/ev_w_ctl_ep1.npz $O/ev_w_raw_ep1.npz $S/w_ctl_ep1 $S/w_raw_ep1 /scratch/haisenk-v2d/hand5w/bundles/w_raw_ep1/result.npz $W/scenes/episode_000001/bundle/result.npz 2>&1 | tee $O/diag2.txt
echo ---- raw ev
sed -n '25,70p' $O/ev_w_raw_ep1.txt | cut -c1-200
cat $W/scenes/episode_000001/run.json | head -c 1500; echo
cat $W/scenes/episode_000001/support_v1/*.json | head -c 1200
