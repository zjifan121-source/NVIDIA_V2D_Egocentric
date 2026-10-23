#!/usr/bin/env bash
# On the Mac, before deleting a GPU instance: copy everything on it that exists nowhere else.
#
#     path_b/vm/save_instance.sh [--remove-deploy-key]
#
# - code: the instance's checkout must be clean and its commit known to the Mac (code is never
#   edited on the instance, so this is a check, not a copy)
# - outputs: for every run under path_b/data/outputs/, the files Hugging Face does not hold
#   (hf_outputs.py pending: all of a run never uploaded; results added or changed since an upload)
#   are copied to the Mac's path_b/data/outputs/<run>/ (regular files; symlinks are listed in
#   <run>/SYMLINKS.txt). What is on Hugging Face is restorable with hf_outputs.py download.
# - logs: /root/v2d/logs/ to ~/V2D/vm_logs/<instance>_<date>/, with SAVED.md (what was saved)
# - --remove-deploy-key: delete the instance's deploy key from GitHub; use it on the last run,
#   right before deleting the instance (it must still be reachable; the key is useless after)
# Rebuilt by new_instance.sh, so not copied: dataset, team bundles, environments, Docker images,
# the stereo engine, MANO files and path_b/.env (both come from the Mac).
set -euo pipefail

HOST=l40s
GH_REPO=zjifan121-source/NVIDIA_V2D_Egocentric
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VM_REPO=/root/v2d/NVIDIA_V2D_Egocentric
OUT=$VM_REPO/path_b/data/outputs
REMOVE_KEY=""
[ "${1:-}" = "--remove-deploy-key" ] && REMOVE_KEY=1

die() { printf 'save_instance: %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

VM_NAME=$(ssh -o ConnectTimeout=10 "$HOST" hostname) || die "cannot reach $HOST"
DEST=$HOME/V2D/vm_logs/${VM_NAME}_$(date +%Y-%m-%d)
mkdir -p "$DEST"
SAVED=$DEST/SAVED.md
printf '# Saved from %s (%s) on %s\n\n' "$VM_NAME" "$(ssh -G "$HOST" | awk '/^hostname /{print $2}')" "$(date)" > "$SAVED"
safe=1

step "code"
vm_head=$(ssh "$HOST" "git -C $VM_REPO rev-parse HEAD")
vm_branch=$(ssh "$HOST" "git -C $VM_REPO branch --show-current")
dirty=$(ssh "$HOST" "git -C $VM_REPO status --porcelain")
if git -C "$REPO" cat-file -e "$vm_head^{commit}" 2>/dev/null; then
  echo "   $vm_branch @ ${vm_head:0:8}: known to the Mac"
  printf -- '- code: %s @ %s, known to the Mac\n' "$vm_branch" "${vm_head:0:8}" >> "$SAVED"
else
  echo "   $vm_branch @ ${vm_head:0:8} is NOT on the Mac"; safe=""
  printf -- '- code: %s @ %s is NOT on the Mac\n' "$vm_branch" "${vm_head:0:8}" >> "$SAVED"
fi
if [ -n "$dirty" ]; then
  echo "   local changes on the instance (copied to $DEST/vm_changes.txt):"; echo "$dirty" | sed 's/^/     /'
  ssh "$HOST" "cd $VM_REPO && git status --porcelain && git diff" > "$DEST/vm_changes.txt"
  printf -- '- the instance had uncommitted changes: vm_changes.txt\n' >> "$SAVED"; safe=""
fi

step "run outputs"
# the Mac's hf_outputs.py, so an instance on an older commit is checked the same way
scp -q "$REPO/path_b/vm/hf_outputs.py" "$HOST:/tmp/hf_outputs.py"
runs=$(ssh "$HOST" "ls $OUT 2>/dev/null" || true)
printf '\n| Run | Copied to the Mac |\n|---|---|\n' >> "$SAVED"
for run in $runs; do
  # the run's files Hugging Face does not hold (all of them for a run never uploaded)
  ssh "$HOST" "cd $VM_REPO && /usr/local/bin/uv run -q --project tools python /tmp/hf_outputs.py pending --run $run --outputs $OUT" \
    > "$DEST/pending_$run.txt" || die "hf_outputs.py pending --run $run failed"
  n=$(wc -l < "$DEST/pending_$run.txt" | tr -d ' ')
  if [ "$n" = 0 ]; then
    echo "   $run: all on Hugging Face"
    printf '| %s | nothing: all on Hugging Face |\n' "$run" >> "$SAVED"
    rm -f "$DEST/pending_$run.txt"
    continue
  fi
  mkdir -p "$REPO/path_b/data/outputs/$run"
  scp -q "$DEST/pending_$run.txt" "$HOST:/tmp/pending_$run.txt"
  ssh "$HOST" "cd $OUT/$run && tar -cf - -T /tmp/pending_$run.txt && rm -f /tmp/pending_$run.txt" \
    | tar -xf - -C "$REPO/path_b/data/outputs/$run"
  ssh "$HOST" "cd $OUT && find $run -type l -printf '%p -> %l\n'" > "$REPO/path_b/data/outputs/$run/SYMLINKS.txt"
  [ -s "$REPO/path_b/data/outputs/$run/SYMLINKS.txt" ] || rm -f "$REPO/path_b/data/outputs/$run/SYMLINKS.txt"
  missing=0
  while IFS= read -r rel; do [ -f "$REPO/path_b/data/outputs/$run/$rel" ] || missing=$((missing + 1)); done < "$DEST/pending_$run.txt"
  size=$(ssh "$HOST" "cd $OUT/$run && du -ch \$(cat) 2>/dev/null | tail -1 | cut -f1" < "$DEST/pending_$run.txt")
  if [ "$missing" -gt 0 ]; then
    echo "   $run: $missing of $n files did not arrive"; safe=""
  fi
  echo "   $run: $n files ($size) copied to path_b/data/outputs/$run/"
  printf '| %s | %s files, %s → path_b/data/outputs/%s/ (list: pending_%s.txt) |\n' "$run" "$n" "$size" "$run" "$run" >> "$SAVED"
done

step "logs -> $DEST"
ssh "$HOST" "tar -C /root/v2d -cf - logs" | tar -xf - -C "$DEST"
echo "   $(ls "$DEST/logs" | wc -l | tr -d ' ') log files"
printf '\n- logs: %s files in logs/\n' "$(ls "$DEST/logs" | wc -l | tr -d ' ')" >> "$SAVED"

step "deploy key"
key_id=$(gh repo deploy-key list -R "$GH_REPO" --json id,title -q ".[] | select(.title == \"$VM_NAME\") | .id" || true)
if [ -z "$key_id" ]; then
  echo "   none on GitHub for $VM_NAME"
elif [ -n "$REMOVE_KEY" ]; then
  gh repo deploy-key delete "$key_id" -R "$GH_REPO" && echo "   removed $VM_NAME's key ($key_id)"
  printf -- '- deploy key %s (%s) removed from GitHub\n' "$key_id" "$VM_NAME" >> "$SAVED"
else
  echo "   $VM_NAME's key ($key_id) is still on GitHub; right before deleting the instance, rerun with --remove-deploy-key"
fi

step "result"
if [ -n "$safe" ]; then
  echo "   everything is saved: the instance can be deleted (summary: $SAVED)"
  printf '\n**Safe to delete.**\n' >> "$SAVED"
else
  echo "   NOT everything is saved; see above and $SAVED"
  printf '\n**Not safe to delete yet** (see above).\n' >> "$SAVED"
  exit 1
fi
