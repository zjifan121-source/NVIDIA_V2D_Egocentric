set -u
O=/scratch/haisenk-v2d/hand5w/r2/ep1; mkdir -p $O; S=/scratch/haisenk-v2d/hand5w/scenes
P=/data/user_data/haisenk/v2d/v2d_train/.venv_retarget/bin/python
W=/scratch/haisenk-v2d/wentao_mix_20261103/experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103
grep -n -i "episode_000001\b\|ep1\b\|episode 1\b\|ep01" $W/README.md | head -20
ls $W/scenes/episode_000001/ $W/scenes/episode_000001/bundle | head -30
for n in w_ctl_ep1 w_raw_ep1; do echo "== $n"; $P - $S/$n/report.json <<'PY'
import json,sys; r=json.load(open(sys.argv[1]))
def walk(d,p=""):
    for k,v in d.items():
        if isinstance(v,dict): walk(v,p+k+".")
        elif any(s in k for s in ("attach","rest_hold","refill","smooth","level","object_refill","rule","h5")): print(" ",p+k,v)
walk(r)
PY
done
cd ~/v2d_ops/kenc/hand5w/r2/ep1
for n in w_ctl_ep1 w_raw_ep1; do $P ev.py $S/$n $O/ev_$n.npz > $O/ev_$n.txt 2>&1; echo "ev $n rc=$?"; done
cat $O/ev_w_ctl_ep1.txt | head -80
