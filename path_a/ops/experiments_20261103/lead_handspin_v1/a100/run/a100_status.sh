for r in $(ls /root/runs | grep -v ABANDONED); do
  L=/root/runs/$r/train.log
  S=$(grep "training progress" $L | tail -1 | cut -c20-52)
  E=$(grep -c -E "Traceback|Killed|out of memory|RESOURCE_EXHAUSTED" $L)
  X=""; [ -f /root/runs/$r/TRAIN_EXITED ] && X="EXITED:$(cut -d" " -f2 /root/runs/$r/TRAIN_EXITED)"; [ -f /root/runs/$r/EVAL_DONE ] && X="$X EVAL_DONE"
  B=""; [ -f /root/runs/$r/checkpoint_selection.json ] && B=$(/root/venv-cpu/bin/python -c "import json;d=json.load(open('/root/runs/$r/checkpoint_selection.json'));print('best',d[0]['step'],round(d[0]['auc'],4),[round(x,3) for x in d[0]['per_body']],'n=%d'%len(d))")
  [ "$E" != "0" ] && X="$X TRAIN_ERROR"
  echo "$r | $S | $B | $X"
done
echo "GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) RAM_avail $(free -g | awk 'NR==2{print $7}')G"
true
