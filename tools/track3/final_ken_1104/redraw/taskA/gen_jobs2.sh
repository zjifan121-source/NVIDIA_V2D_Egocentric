# gen_jobs2.sh OUT D_FROM D_TO EP... : like gen_jobs.sh, ep20 uses the original HF checkpoint (its stored scene path exists)
W=/data/user_data/haisenk/v2d/push1104/taskA; OUT=$1 A=$2 Z=$3; shift 3
: > $OUT
for d in $(seq $A $Z); do for ep in "$@"; do
  st=$(awk -v e=$ep '$1==e{print $2}' ~/v2d_ops/kenc/push1104/taskA/eps.txt)
  P=$W/pol/ep$ep/policy_$st.safetensors
  [ $ep = 20 ] && P=$W/hf/experiments/edmond/bob1_submission_20261103/episode_000020/policy/policy_100003840.safetensors
  printf "%s %s d%02d\n" $ep $P $d >> $OUT; done; done
wc -l $OUT
