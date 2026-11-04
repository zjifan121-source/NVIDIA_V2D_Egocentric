O=/scratch/haisenk-v2d/hand5w/r2/ep1; S=/scratch/haisenk-v2d/hand5w/scenes; P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
head -22 $O/diag2.txt | cut -c1-900
cd ~/v2d_ops/kenc/hand5w/r2/ep1
for n in w_ctl_ep1 w_raw_ep1; do echo "== geo $n lift 130-215"; $P geo_b.py $S/$n 130 215; echo "== geo $n table 40-120"; $P geo_b.py $S/$n 40 120; done 2>&1 | grep -v Warn
