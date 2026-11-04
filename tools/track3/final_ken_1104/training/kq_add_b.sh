# kq_add.sh (inside a step on babel-s5-16): append queue lines given on stdin to /tmp/haisenk_v2d/kq_babel.txt (skips names already queued)
B=/tmp/haisenk_v2d; Q=$B/kq_babel.txt; touch $Q
while read -r n ep pq s rec prio; do [ -z "$n" ] && continue
  case "$n" in \#*) continue;; esac
  [ -d "$pq" ] || { echo "NO SCENE DIR for $n: $pq"; continue; }
  grep -q "^$n " $Q && { echo "already queued $n"; continue; }
  [ -e $B/runs/$n ] && { echo "run dir exists $n - skipped"; continue; }
  flock $Q.lock bash -c "echo '$n $ep $pq $s $rec ${prio:-50}' >> $Q"; echo "queued $n"
done
