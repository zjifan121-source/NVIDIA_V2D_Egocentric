#!/usr/bin/env bash
# One-time setup of a fresh GPU instance for Path B (and the shared stages 2-3), as root on Ubuntu
# with the NVIDIA driver, Docker and the NVIDIA Container Toolkit already installed (Verda images
# ship them). Safe to rerun: every step checks what is already there and skips it.
#
# First run: copy this file to the instance and run it there (it clones the repo on the way):
#     scp path_b/vm/setup.sh l40s:  &&  ssh -t l40s bash setup.sh
# Later runs: bash /root/v2d/NVIDIA_V2D_Egocentric/path_b/vm/setup.sh
#
# It stops and says what to do when it needs you: adding its GitHub deploy key (once per
# instance), `hf auth login`, copying the MANO files. Settings (environment variables):
#     BRANCH=main_buffer        branch to check out on the first clone
#     RESTORE_RUN=4c2eac31      Path B outputs to restore from Hugging Face (path_b/vm/hf_outputs.py)
#     RESTORE_FRAMES=1          also restore that run's per-frame intermediates (~18 GB)
#     SELF_TEST=1               also run the local scorer's self-test (~17 min, CPU)
# Logs: /root/v2d/logs/setup_<step>.log
set -euo pipefail

V2D=/root/v2d
REPO=$V2D/NVIDIA_V2D_Egocentric
RECON=$REPO/baseline/reconstruction
LOGS=$V2D/logs
BRANCH=${BRANCH:-main_buffer}
GITHUB=git@github.com:zjifan121-source/NVIDIA_V2D_Egocentric.git
UV_VERSION=0.11.6
RETARGET=$V2D/venvs/retarget
KEY=/root/.ssh/v2d_github
# github.com's published ed25519 host key fingerprint (docs.github.com, "GitHub's SSH key fingerprints")
GITHUB_ED25519=SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU
export PATH=/root/.local/bin:/usr/local/bin:$PATH
NEEDS_YOU=()

mkdir -p "$V2D" "$LOGS"
say() { printf '\n== %s\n' "$*"; }
ok() { printf '   ok: %s\n' "$*"; }
need() { NEEDS_YOU+=("$*"); printf '   NEEDS YOU: %s\n' "$*"; }
logged() { local name=$1; shift; "$@" >"$LOGS/setup_$name.log" 2>&1 || { tail -20 "$LOGS/setup_$name.log"; return 1; }; }

check_host() {
  say "host"
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
  command -v docker >/dev/null || { echo "Docker is missing: use an image with Docker + NVIDIA Container Toolkit"; exit 1; }
  logged docker_gpu docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L
  ok "GPU visible inside Docker"
  local free_gb; free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  [ "$free_gb" -ge 120 ] && ok "${free_gb} GB free" || echo "   warning: ${free_gb} GB free; a full setup with frames uses ~100 GB"
}

tools() {
  say "tools: git-lfs, ffmpeg, uv $UV_VERSION, Python 3.11, hf"
  local missing=()
  for p in git-lfs ffmpeg unzip tmux; do command -v "$p" >/dev/null || missing+=("$p"); done
  if [ ${#missing[@]} -gt 0 ]; then
    logged apt bash -c "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ${missing[*]}"
  fi
  if [ "$(uv --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ]; then
    local d; d=$(mktemp -d)
    local base=https://github.com/astral-sh/uv/releases/download/$UV_VERSION
    (cd "$d" && curl -fsSLO "$base/uv-x86_64-unknown-linux-gnu.tar.gz" \
      && curl -fsSLO "$base/uv-x86_64-unknown-linux-gnu.tar.gz.sha256" \
      && sha256sum -c uv-x86_64-unknown-linux-gnu.tar.gz.sha256 \
      && tar xzf uv-x86_64-unknown-linux-gnu.tar.gz \
      && install -m 0755 uv-x86_64-unknown-linux-gnu/uv uv-x86_64-unknown-linux-gnu/uvx /usr/local/bin/)
    rm -rf "$d"
  fi
  uv python find 3.11 >/dev/null 2>&1 || logged python uv python install 3.11
  command -v hf >/dev/null || logged hf_cli uv tool install huggingface_hub
  ok "$(uv --version), $(uv python find 3.11), hf $(hf version 2>/dev/null | awk '/version:/{print $2}')"
}

github_access() {
  say "GitHub access (read-only deploy key)"
  if [ ! -f "$KEY" ]; then
    ssh-keygen -q -t ed25519 -N "" -C "v2d deploy key $(hostname)" -f "$KEY"
  fi
  if ! grep -q "^Host github.com" /root/.ssh/config 2>/dev/null; then
    printf 'Host github.com\n  IdentityFile %s\n  IdentitiesOnly yes\n' "$KEY" >> /root/.ssh/config
    chmod 600 /root/.ssh/config
  fi
  if ! ssh-keygen -F github.com >/dev/null 2>&1; then
    local scan; scan=$(ssh-keyscan -t ed25519 github.com 2>/dev/null)
    if [ "$(echo "$scan" | ssh-keygen -lf - | awk '{print $2}')" != "$GITHUB_ED25519" ]; then
      echo "github.com host key does not match GitHub's published fingerprint; stopping"; exit 1
    fi
    echo "$scan" >> /root/.ssh/known_hosts
  fi
  if git ls-remote "$GITHUB" HEAD >/dev/null 2>&1; then
    ok "deploy key works"
    return 0
  fi
  need "add this instance's deploy key on GitHub, then rerun:
         github.com/zjifan121-source/NVIDIA_V2D_Egocentric -> Settings -> Deploy keys -> Add deploy key
         (title: $(hostname); leave 'Allow write access' unticked); key:
         $(cat "$KEY.pub")"
  return 1
}

code() {
  say "code ($REPO)"
  if [ ! -d "$REPO/.git" ]; then
    logged clone env GIT_LFS_SKIP_SMUDGE=1 git clone -q -b "$BRANCH" "$GITHUB" "$REPO"
  fi
  git -C "$REPO" lfs install --local >/dev/null
  logged lfs git -C "$REPO" lfs pull  # from NVIDIA's public LFS store (.lfsconfig)
  # Unpushed work arrives from the Mac with: git push --no-verify l40s:/root/v2d/repo.git <branch>
  [ -d "$V2D/repo.git" ] || git init -q --bare -b main_buffer "$V2D/repo.git"
  ok "$(git -C "$REPO" log --oneline -1) on $(git -C "$REPO" branch --show-current)"
}

dataset() {
  say "Track 3 dataset"
  if [ ! -f "$V2D/data/track_3/public/meta/camera_calibration.json" ]; then
    logged dataset hf download nvidia/video_to_data_challenge --repo-type dataset --include "track_3/*" --local-dir "$V2D/data"
  fi
  [ -e "$REPO/assets/data/track_3" ] || ln -s "$V2D/data/track_3" "$REPO/assets/data/track_3"
  grep -qx "assets/data/track_3" "$REPO/.git/info/exclude" || echo "assets/data/track_3" >> "$REPO/.git/info/exclude"
  ok "$(ls "$V2D/data/track_3/public/videos/chunk-000/observation.images.ego_cam_a" | wc -l) public and \
$(ls "$V2D/data/track_3/evaluation/videos/chunk-000/observation.images.ego_cam_a" | wc -l) evaluation episodes"
}

hf_login() {
  say "Hugging Face login"
  if hf auth whoami >/dev/null 2>&1; then
    ok "logged in as $(hf auth whoami 2>/dev/null | awk '/user/{print $2}')"
    return 0
  fi
  need "log in to Hugging Face (team bundles; a Write token also lets you upload outputs), then rerun:
         hf auth login     (answer n to 'Add token as git credential?')"
  return 1
}

envs() {
  say "Python environments"
  logged env_tools uv sync --project "$REPO/tools"
  logged env_path_b uv sync --project "$REPO/path_b"
  if [ ! -x "$RECON/.venv/bin/python" ]; then
    logged env_host_venv uv venv -q --seed --python 3.11 "$RECON/.venv"
  fi
  if ! "$RECON/.venv/bin/python" -c "import v2d.foundation_stereo.docker._config" 2>/dev/null; then
    (cd "$RECON" && logged env_host env PYTHON=.venv/bin/python ./scripts/install_ego_reconstruction_packages.sh \
      && logged env_host_stereo .venv/bin/python -m pip install -q -e modules/v2d_foundation_stereo/docker)
  fi
  if ! "$RETARGET/bin/python" -c "import torch, mujoco, pinocchio, manotorch, chumpy" 2>/dev/null; then
    # README §5.12, verbatim
    local P=$RETARGET/bin/python
    logged env_retarget bash -c "cd '$REPO' && uv venv -q --python 3.11 '$RETARGET' \
      && uv pip install -q --python '$P' torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128 \
      && uv pip install -q --python '$P' -r tools/track3/load_retarget_requirements.txt \
      && uv pip install -q --python '$P' pip setuptools wheel \
      && uv pip install -q --python '$P' --no-build-isolation 'chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da' \
      && uv pip install -q --python '$P' --no-deps 'manotorch @ git+https://github.com/lixiny/manotorch@a2a70c591f91551078b7bb2af9b5d9f275b626e0' \
      && uv pip install -q --python '$P' --no-deps -e baseline/robotic_grounding/source/robotic_grounding \
           -e baseline/reconstruction/modules/v2d_common -e baseline/reconstruction/modules/v2d_task_library_loader/lib"
  fi
  ok "tools, path_b, baseline host ($RECON/.venv), stages 2-3 ($RETARGET: $("$RETARGET/bin/python" -c 'import torch; print(torch.__version__, torch.cuda.is_available())'))"
}

bundles() {
  say "team bundles (rose-tea-cafe/v2d-t3-recon)"
  # on a fresh instance hub/ does not exist yet, and a failing find would end the script (pipefail)
  mkdir -p "$RECON/data/outputs/hub"
  local have; have=$(find "$RECON/data/outputs/hub" -name result.npz | wc -l)
  if [ "$have" -lt 69 ]; then
    local episodes
    episodes=$(cd "$REPO" && uv run --project tools python -c \
      'import json; print(" ".join(str(e["episode"]) for e in json.load(open("tools/track3/episodes.json"))["episodes"]))')
    logged bundles bash -c "cd '$REPO' && for e in $episodes; do uv run --project tools python \
      tools/track3/bundles.py --repo rose-tea-cafe/v2d-t3-recon download --episode \$e || exit 1; done"
  fi
  ok "$(find "$RECON/data/outputs/hub" -name result.npz | wc -l) object bundles"
}

mano() {
  say "MANO hand model (your license; never in git)"
  local dir=$RECON/data/weights/hand/models
  mkdir -p "$dir"
  if [ -f "$dir/MANO_RIGHT.pkl" ] && [ -f "$dir/MANO_LEFT.pkl" ]; then
    chmod 600 "$dir"/MANO_*.pkl
    ok "MANO_RIGHT.pkl and MANO_LEFT.pkl present"
    return 0
  fi
  need "copy your MANO files from the Mac, then rerun:
         scp ~/V2D/mano_v1_2/models/MANO_RIGHT.pkl ~/V2D/mano_v1_2/models/MANO_LEFT.pkl <host>:$dir/"
  return 1
}

stereo() {
  say "stereo model (baseline v2d_foundation_stereo, read-only)"
  docker image inspect v2d_foundation_stereo >/dev/null 2>&1 \
    || (cd "$RECON" && logged stereo_image python3 modules/v2d_foundation_stereo/docker/build.py)
  local weights=$RECON/data/weights/foundation_stereo
  ls "$weights"/*.onnx >/dev/null 2>&1 \
    || (cd "$RECON" && logged stereo_onnx .venv/bin/python modules/v2d_foundation_stereo/docker/run_download_weights.py --output_dir "$weights")
  # the engine is built per GPU architecture and TensorRT version; the module names it accordingly
  (cd "$RECON" && logged stereo_engine .venv/bin/python modules/v2d_foundation_stereo/docker/run_export_engine.py --model_dir "$weights")
  ok "engines: $(cd "$weights" && ls *.engine | tr '\n' ' ')"
}

outputs() {
  [ -n "${RESTORE_RUN:-}" ] || return 0
  say "Path B outputs: run $RESTORE_RUN from Hugging Face"
  if [ -d "$REPO/path_b/data/outputs/$RESTORE_RUN" ]; then
    ok "already here"
    return 0
  fi
  (cd "$REPO" && logged restore uv run --project tools python path_b/vm/hf_outputs.py download \
    --run "$RESTORE_RUN" ${RESTORE_FRAMES:+--frames})
  ok "restored into path_b/data/outputs/$RESTORE_RUN"
}

smoke() {
  say "smoke tests"
  if (cd "$REPO" && logged pytest uv run --project path_b pytest -q path_b/tests); then ok "Path B unit tests"
  else need "Path B unit tests failed: $LOGS/setup_pytest.log"; fi
  local sample=$RECON/modules/v2d_foundation_stereo/assets out=$V2D/smoke/stereo_sample
  rm -rf "$out"
  if (cd "$RECON" && logged stereo_sample .venv/bin/python modules/v2d_foundation_stereo/docker/run_image_list_to_depth.py \
    --left_dir "$sample/left" --right_dir "$sample/right" --calibration_file "$sample/calibration.json" \
    --model_dir "$RECON/data/weights/foundation_stereo" --depth_folder "$out/depth" --intrinsics_folder "$out/intrinsics"); then
    ok "stereo depth on the module's sample: $(ls "$out/depth" | wc -l) frames"
  else need "stereo sample failed: $LOGS/setup_stereo_sample.log"; fi
  if [ -n "${SELF_TEST:-}" ]; then
    if (cd "$REPO" && logged self_test uv run --project tools python tools/track3/score_local.py --self-test); then
      ok "scorer self-test: $(grep -c . "$LOGS/setup_self_test.log") lines in $LOGS/setup_self_test.log"
    else need "scorer self-test failed: $LOGS/setup_self_test.log"; fi
  fi
}

check_host
tools
if ! github_access && [ ! -d "$REPO/.git" ]; then
  printf '\nStopped: the first clone needs the deploy key.\n'; exit 2
fi
code
dataset
if hf_login; then bundles; fi
envs
mano || true
stereo
if hf auth whoami >/dev/null 2>&1; then outputs; fi
smoke

if [ ${#NEEDS_YOU[@]} -gt 0 ]; then
  printf '\n== Done, except what needs you (then rerun this script):\n'
  printf '   - %s\n' "${NEEDS_YOU[@]}"
  exit 2
fi
printf '\n== Ready. Run Path B with:\n   cd %s && RETARGET_VENV=%s uv run --project path_b python path_b/run.py run --episodes 41\n' "$REPO" "$RETARGET"
