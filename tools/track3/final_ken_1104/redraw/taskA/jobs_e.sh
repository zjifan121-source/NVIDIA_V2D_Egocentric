W=/data/user_data/haisenk/v2d/push1104/taskA; : > $W/jobs_e.txt
for d in 1 2 3 4; do for ep in 28 19 14; do
  st=$(awk -v e=$ep '$1==e{print $2}' ~/v2d_ops/kenc/push1104/taskA/eps.txt)
  printf "%s %s n%02d\n" $ep $W/pol/ep$ep/policy_$st.safetensors $d >> $W/jobs_e.txt; done; done
cat $W/jobs_e.txt | head -3
