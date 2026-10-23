#!/usr/bin/env bash
# On the Mac: turn a new Verda GPU instance into a ready Path B machine with one command.
#
#     path_b/vm/new_instance.sh <ip> [--restore RUN | --no-restore] [--no-frames] [--branch BRANCH]
#
# 1. points the ssh alias l40s at <ip> (~/.ssh/config; a stale host key for that IP is replaced)
# 2. waits for SSH, copies setup.sh, and puts the MANO files and path_b/.env in /root/v2d/inbox/
# 3. gives the instance a Hugging Face token: HF_WRITE_TOKEN from path_b/.env if set (uploads work),
#    else its read-only HF_TOKEN (downloads only; `hf auth login` with a write token before uploads)
# 4. creates the instance's read-only GitHub deploy key and adds it with gh (your login)
# 5. runs setup.sh in tmux (dataset, bundles, environments, stereo engine, restore of RUN) and
#    follows its log until it ends
# 6. pushes BRANCH (default: this checkout's branch) through the instance's bare repo, checks it
#    out there and runs the unit tests
# Default restore: run 4c2eac31 with frames (V3_BUILD_SPEC.md §10). Nothing secret is printed.
# Takes about an hour on a fresh instance; rerunning it on a ready one takes about a minute.
set -euo pipefail

HOST=l40s
GH_REPO=zjifan121-source/NVIDIA_V2D_Egocentric
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ENV_FILE=$REPO/path_b/.env
MANO_DIR=${MANO_DIR:-$HOME/V2D/mano_v1_2/models}
VM_REPO=/root/v2d/NVIDIA_V2D_Egocentric

die() { printf 'new_instance: %s\n' "$*" >&2; exit 1; }
step() { printf '\n== %s\n' "$*"; }

[ $# -ge 1 ] || die "usage: $0 <ip> [--restore RUN | --no-restore] [--no-frames] [--branch BRANCH]"
IP=$1; shift
RESTORE=4c2eac31 FRAMES=1 BRANCH=$(git -C "$REPO" branch --show-current)
while [ $# -gt 0 ]; do
  case $1 in
    --restore) RESTORE=$2; shift 2 ;;
    --no-restore) RESTORE=""; shift ;;
    --no-frames) FRAMES=""; shift ;;
    --branch) BRANCH=$2; shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done

step "checks on the Mac"
for f in "$MANO_DIR/MANO_RIGHT.pkl" "$MANO_DIR/MANO_LEFT.pkl" "$ENV_FILE" "$HOME/.ssh/id_ed25519"; do
  [ -f "$f" ] || die "missing $f"
done
gh auth status -h github.com >/dev/null 2>&1 || die "gh is not logged in (gh auth login)"
if git -C "$REPO" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  CLONE_BRANCH=$BRANCH
else
  CLONE_BRANCH=main_buffer  # BRANCH reaches the instance through its bare repo (step 6)
fi
echo "   branch $BRANCH (first clone: $CLONE_BRANCH); restore: ${RESTORE:-none}${RESTORE:+${FRAMES:+ with frames}}"

step "ssh alias $HOST -> $IP"
python3 - "$HOME/.ssh/config" "$HOST" "$IP" <<'EOF'
import re, sys
from pathlib import Path
path, host, ip = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text() if path.exists() else ""
block = re.search(rf"(?ms)^Host {re.escape(host)}\n(.*?)(?=^Host |\Z)", text)
if block:
    body = re.sub(r"(?m)^(\s*)HostName\s+\S+", rf"\g<1>HostName {ip}", block.group(1))
    text = text[:block.start(1)] + body + text[block.end(1):]
else:
    text += (f"\nHost {host}\n    HostName {ip}\n    User root\n    IdentityFile ~/.ssh/id_ed25519\n"
             "    IdentitiesOnly yes\n")
path.write_text(text)
EOF
for i in $(seq 60); do
  out=$(ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$HOST" true 2>&1) && break
  if echo "$out" | grep -q "REMOTE HOST IDENTIFICATION HAS CHANGED"; then
    # a new instance on an IP used before: its host key is new, so the stored one is replaced
    ssh-keygen -R "$IP" >/dev/null 2>&1
    echo "   host key for $IP changed (new instance): replaced"
    continue
  fi
  [ "$i" = 60 ] && die "no SSH on $IP after 10 min: $out"
  sleep 10
done
VM_NAME=$(ssh "$HOST" hostname)
echo "   reachable: $VM_NAME ($(ssh "$HOST" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))"

step "setup.sh, MANO files and path_b/.env to the instance"
ssh "$HOST" "mkdir -p /root/v2d/inbox /root/v2d/logs && chmod 700 /root/v2d/inbox"
scp -q "$REPO/path_b/vm/setup.sh" "$HOST:/root/setup.sh"
if ssh "$HOST" "test -f $VM_REPO/baseline/reconstruction/data/weights/hand/models/MANO_LEFT.pkl"; then
  echo "   MANO files already in place"
else
  scp -q "$MANO_DIR/MANO_RIGHT.pkl" "$MANO_DIR/MANO_LEFT.pkl" "$HOST:/root/v2d/inbox/"
fi
if ssh "$HOST" "test -f $VM_REPO/path_b/.env"; then
  echo "   path_b/.env already in place"
else
  scp -q "$ENV_FILE" "$HOST:/root/v2d/inbox/path_b.env"
fi
ssh "$HOST" "chmod 600 /root/v2d/inbox/* 2>/dev/null || true"

step "Hugging Face token"
if ssh "$HOST" "test -s /root/.cache/huggingface/token"; then
  echo "   the instance already has one"
else
  token=$(grep -m1 '^HF_WRITE_TOKEN=' "$ENV_FILE" | cut -d= -f2- || true)
  kind="write (HF_WRITE_TOKEN)"
  if [ -z "$token" ]; then
    token=$(grep -m1 '^HF_TOKEN=' "$ENV_FILE" | cut -d= -f2- || true)
    kind="read-only (HF_TOKEN): before an upload, run  ssh -t $HOST /root/.local/bin/hf auth login"
  fi
  [ -n "$token" ] || die "no HF_WRITE_TOKEN or HF_TOKEN in $ENV_FILE"
  printf '%s' "$token" | ssh "$HOST" "umask 077; mkdir -p /root/.cache/huggingface; cat > /root/.cache/huggingface/token"
  unset token
  echo "   stored a $kind token"
fi

step "GitHub deploy key (read-only)"
ssh "$HOST" '[ -f /root/.ssh/v2d_github ] || ssh-keygen -q -t ed25519 -N "" -C "v2d deploy key $(hostname)" -f /root/.ssh/v2d_github'
pub=$(ssh "$HOST" cat /root/.ssh/v2d_github.pub)
if gh repo deploy-key list -R "$GH_REPO" --json key -q '.[].key' | grep -qF "$(echo "$pub" | awk '{print $2}')"; then
  echo "   already on GitHub"
else
  keyfile=$(mktemp)
  echo "$pub" > "$keyfile"
  gh repo deploy-key add "$keyfile" -R "$GH_REPO" -t "$VM_NAME"
  rm -f "$keyfile"
fi

# ACCEPT_NVIDIA_MODEL_EULA=1: James accepted the NVIDIA Open Model License for TAO FoundationPose (2026-10-22, 2026-10-23)
step "setup.sh in tmux (session 'setup'; about two hours on a fresh instance, mostly downloads)"
LOG=/root/v2d/logs/setup_$(date -u +%Y%m%d_%H%M%S).log
ssh "$HOST" "command -v tmux >/dev/null || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null)
  loginctl enable-linger root 2>/dev/null || true
  tmux kill-session -t setup 2>/dev/null || true
  tmux new -d -s setup 'ACCEPT_NVIDIA_MODEL_EULA=1 BRANCH=$CLONE_BRANCH RESTORE_RUN=$RESTORE ${FRAMES:+RESTORE_FRAMES=1} bash /root/setup.sh > $LOG 2>&1; echo SETUP_EXIT \$? >> $LOG'"
echo "   log: $HOST:$LOG  (watch: ssh -t $HOST tmux attach -t setup)"
seen=0
while :; do
  sleep 20
  lines=$(ssh -o ConnectTimeout=10 "$HOST" "tail -n +$((seen + 1)) $LOG 2>/dev/null" || true)
  [ -n "$lines" ] || continue
  seen=$((seen + $(printf '%s\n' "$lines" | wc -l)))
  printf '%s\n' "$lines" | grep -E '^== |ok:|NEEDS YOU|warning|Stopped|Ready|^   - |SETUP_EXIT' || true
  code=$(printf '%s\n' "$lines" | awk '/^SETUP_EXIT/{print $2}')
  [ -n "$code" ] && break
done
[ "$code" = 0 ] || die "setup.sh ended with exit $code; see $HOST:$LOG and /root/v2d/logs/setup_<step>.log"

step "code: $BRANCH through the instance's bare repo"
[ -z "$(ssh "$HOST" "git -C $VM_REPO status --porcelain --untracked-files=no")" ] \
  || die "the instance's checkout has local changes; code is never edited there"
git -C "$REPO" push --no-verify -q --force "$HOST:/root/v2d/repo.git" "$BRANCH"  # the bare repo only
ssh "$HOST" "cd $VM_REPO && git fetch -q /root/v2d/repo.git $BRANCH && git checkout -q -B $BRANCH FETCH_HEAD"
mac=$(git -C "$REPO" rev-parse --short=8 "$BRANCH")
vm=$(ssh "$HOST" "git -C $VM_REPO rev-parse --short=8 HEAD")
[ "$mac" = "$vm" ] || die "instance is at $vm, the Mac's $BRANCH at $mac"
ssh "$HOST" "cd $VM_REPO && /usr/local/bin/uv run --project path_b pytest -q path_b/tests 2>&1 | tail -1"

step "ready: $VM_NAME at $IP, $BRANCH @ $vm"
echo "   ssh $HOST; cd $VM_REPO   (save before deleting it: path_b/vm/save_instance.sh)"
