# gen_jobs.sh OUT D_FROM D_TO EP... : lines "EP POLICY dNN", draws outer loop (balanced partial results)
W=/data/user_data/haisenk/v2d/push1104/taskA; OUT=$1 A=$2 Z=$3; shift 3
: > $OUT
for d in $(seq $A $Z); do for ep in "$@"; do
  st=$(awk -v e=$ep '$1==e{print $2}' ~/v2d_ops/kenc/push1104/taskA/eps.txt)
  printf "%s %s d%02d\n" $ep $W/pol/ep$ep/policy_$st.safetensors $d >> $OUT; done; done
wc -l $OUT
