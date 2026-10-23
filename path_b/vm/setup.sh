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
# instance), `hf auth login`, copying the MANO files. path_b/vm/new_instance.sh (on the Mac) does
# all three before it starts this script, so a new instance needs one command (vm/README.md).
# Files the Mac drops in /root/v2d/inbox/ are moved into place: MANO_{RIGHT,LEFT}.pkl and
# path_b.env (-> path_b/.env). Settings (environment variables):
#     BRANCH=main_buffer        branch to check out on the first clone
#     RESTORE_RUN=4c2eac31      Path B outputs to restore from Hugging Face (path_b/vm/hf_outputs.py)
#     RESTORE_FRAMES=1          also restore that run's per-frame intermediates (~18 GB)
#     SELF_TEST=1               also run the local scorer's self-test (~17 min, CPU)
#     V3_ENVS=0                 skip Path B v3's environments (default: build them; ~30 GB, ~30 min):
#                               the team's 12 reconstruction images and their weights, venvs stereo,
#                               flash_chord, sam3 and daid; SAM 3 and Do as I Do clones in third_party/
#     ACCEPT_NVIDIA_MODEL_EULA=1  also download TAO FoundationPose (NVIDIA Open Model License; only
#                               after James has accepted it)
# Logs: /root/v2d/logs/setup_<step>.log
set -euo pipefail

V2D=/root/v2d
REPO=$V2D/NVIDIA_V2D_Egocentric
RECON=$REPO/baseline/reconstruction
LOGS=$V2D/logs
INBOX=$V2D/inbox  # filled by new_instance.sh: MANO files, path_b.env
BRANCH=${BRANCH:-main_buffer}
GITHUB=git@github.com:zjifan121-source/NVIDIA_V2D_Egocentric.git
UV_VERSION=0.11.6
RETARGET=$V2D/venvs/retarget
THIRD=$V2D/third_party
GHCR=ghcr.io/ryanzhou123232
# the team's published images (built at afe7dad; FoundationPose with the TensorRT runtime fix)
RECON_IMAGES=(foundation_pose grounding_dino sam2 sam3d wilor hamer geocalib droid_slam gsplat_refinement mediapipe moge anycalib)
# third-party code for Path B v3, never in the repo (V3_BUILD_SPEC.md §2); commits pinned on first clone
SAM3_URL=https://github.com/facebookresearch/sam3.git
SAM3_COMMIT=${SAM3_COMMIT:-}
DAID_URL=https://github.com/malik-group/do-as-i-do.git
DAID_COMMIT=${DAID_COMMIT:-}
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
  # without lingering, systemd stops root's user manager (and the tmux panes under it) whenever no
  # SSH session is open; long jobs run in tmux
  loginctl enable-linger root 2>/dev/null && ok "lingering on: tmux jobs outlive SSH sessions" || true
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
  # Seeded with the clone's GitHub branches, so a push from the Mac sends only the new commits
  # (an empty bare repo makes the Mac upload the whole ~900 MB history)
  if [ -z "$(git -C "$V2D/repo.git" for-each-ref --count=1 refs/heads)" ]; then
    git -C "$V2D/repo.git" fetch -q "$REPO" "+refs/remotes/origin/*:refs/heads/*"
    git -C "$V2D/repo.git" update-ref -d refs/heads/HEAD 2>/dev/null || true
  fi
  if [ -f "$INBOX/path_b.env" ]; then
    install -m 600 "$INBOX/path_b.env" "$REPO/path_b/.env" && rm -f "$INBOX/path_b.env"
  fi
  [ -f "$REPO/path_b/.env" ] && ok "path_b/.env present (git-ignored; source it only for v2d_submit.py)"
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
  for f in MANO_RIGHT.pkl MANO_LEFT.pkl; do
    [ -f "$INBOX/$f" ] && install -m 600 "$INBOX/$f" "$dir/$f" && rm -f "$INBOX/$f"
  done
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

recon_images() {
  say "reconstruction images ($GHCR, ${#RECON_IMAGES[@]} modules)"
  local m tag missing=()
  for m in "${RECON_IMAGES[@]}"; do
    docker image inspect "v2d_$m" >/dev/null 2>&1 && continue
    tag=afe7dad; [ "$m" = foundation_pose ] && tag=afe7dad-trt-fix
    # ghcr drops long streams now and then; finished layers survive a retry
    if logged "pull_$m" bash -c "for t in 1 2 3; do docker pull -q $GHCR/v2d_$m:$tag && exit 0; sleep 15; done; exit 1"; then
      docker tag "$GHCR/v2d_$m:$tag" "v2d_$m"
    else
      missing+=("$m")
    fi
  done
  if [ ${#missing[@]} -gt 0 ]; then need "image pulls failed: ${missing[*]} ($LOGS/setup_pull_<module>.log)"; return 1; fi
  ok "all ${#RECON_IMAGES[@]} present"
}

recon_weights() {
  say "reconstruction weights (data/weights/<module>)"
  local py=$RECON/.venv/bin/python w=$RECON/data/weights
  fetch() {  # name, then the download command; a .done marker makes reruns skip it
    local name=$1; shift
    [ -e "$w/$name/.done" ] && return 0
    if ! (cd "$RECON" && logged "weights_$name" "$@"); then
      need "weights for $name: download failed ($LOGS/setup_weights_$name.log)"; return 1
    fi
    mkdir -p "$w/$name" && touch "$w/$name/.done"
  }
  fetch moge $py -m v2d.moge.docker.run_download_weights --output_dir data/weights/moge
  fetch sam2 $py -m v2d.sam2.docker.run_download_weights --output_dir data/weights/sam2
  fetch anycalib $py -m v2d.anycalib.docker.run_download_weights --output_dir data/weights/anycalib
  fetch grounding_dino $py -m v2d.grounding_dino.docker.run_download_weights --output_dir data/weights/grounding_dino
  fetch sam3d $py -m v2d.sam3d.docker.run_download_weights --output_dir data/weights/sam3d  # gated: HF login
  fetch wilor $py -m v2d.wilor.docker.run_download_weights --weights_dir data/weights/wilor
  fetch hamer $py -m v2d.hamer.docker.run_download_weights --weights_dir data/weights/hamer
  fetch droid_slam $py -m v2d.droid_slam.docker.run_download_weights --output_dir data/weights/droid_slam
  fetch geocalib $py -m v2d.geocalib.docker.run_download_weights --output_dir data/weights/geocalib
  fetch gsplat_refinement $py -m v2d.gsplat_refinement.docker.run_download_weights --weights_path data/weights/gsplat_refinement
  if [ -n "${ACCEPT_NVIDIA_MODEL_EULA:-}" ]; then
    fetch foundation_pose $py -m v2d.foundation_pose.docker.run_download_weights --output_dir data/weights/foundation_pose \
      --accept_nvidia_model_eula
  elif [ ! -e "$w/foundation_pose/.done" ]; then
    need "FoundationPose weights need the NVIDIA Open Model License (NGC TAO FoundationPose): once accepted, rerun with ACCEPT_NVIDIA_MODEL_EULA=1"
  fi
  local have; have=$(find "$w" -maxdepth 2 -name .done | wc -l)
  ok "$have modules downloaded: $(cd "$w" && ls -d */ 2>/dev/null | tr -d / | tr '\n' ' ')"
}

clone_pinned() {  # url, folder, commit (empty: keep the clone's HEAD and record it)
  local url=$1 dir=$2 commit=$3
  [ -d "$dir/.git" ] || logged "clone_$(basename "$dir")" git clone -q --recurse-submodules "$url" "$dir"
  if [ -n "$commit" ] && [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
    git -C "$dir" fetch -q origin && git -C "$dir" checkout -q "$commit" && git -C "$dir" submodule update -q --init --recursive
  fi
  printf '%s %s %s\n' "$(basename "$dir")" "$url" "$(git -C "$dir" rev-parse HEAD)" >> "$THIRD/COMMITS.txt.new"
}

v3_envs() {
  say "Path B v3 environments (V3_BUILD_SPEC.md §2)"
  local V=$V2D/venvs
  mkdir -p "$THIRD" "$V"
  rm -f "$THIRD/COMMITS.txt.new"
  # stereo: tools/track3/stereo_seed.py (jobs/README.md, verbatim pins)
  if ! "$V/stereo/bin/python" -c "import cv2, onnxruntime, scipy" 2>/dev/null; then
    logged env_stereo bash -c "uv venv -q --python 3.11 '$V/stereo' && uv pip install -q --python '$V/stereo/bin/python' \
      numpy scipy opencv-python-headless onnxruntime-gpu==1.20.0"
  fi
  # flash_chord: FlashCHORD's own uv.lock (the container's pins), train + dev extras; editable
  # install with hatchling writes nothing into baseline/
  local FC=$REPO/baseline/robotic_grounding/flash_chord
  if ! "$V/flash_chord/bin/python" -c "import flash_chord, jax, mujoco_warp" 2>/dev/null; then
    logged env_flash_chord bash -c "uv venv -q --python 3.11 '$V/flash_chord' \
      && uv export -q --project '$FC' --frozen --extra train --extra dev --no-emit-project --no-hashes -o '$V/flash_chord.req.txt' \
      && uv pip install -q --python '$V/flash_chord/bin/python' -r '$V/flash_chord.req.txt' \
      && uv pip install -q --python '$V/flash_chord/bin/python' --no-deps -e '$FC'"
  fi
  # sam3: SAM 3 / 3.1 (facebook/sam3.1, gated: HF login), its README's torch
  clone_pinned "$SAM3_URL" "$THIRD/sam3" "$SAM3_COMMIT"
  if ! "$V/sam3/bin/python" -c "import sam3, torch" 2>/dev/null; then
    logged env_sam3 bash -c "uv venv -q --python 3.12 '$V/sam3' \
      && uv pip install -q --python '$V/sam3/bin/python' torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 \
      && uv pip install -q --python '$V/sam3/bin/python' -e '$THIRD/sam3'"
  fi
  # daid: Do as I Do retargeting (MIT; MuJoCo Warp, mink, viser, coacd), its README: pip install -e .
  clone_pinned "$DAID_URL" "$THIRD/do-as-i-do" "$DAID_COMMIT"
  if ! "$V/daid/bin/python" -c "import mujoco_warp, mink" 2>/dev/null; then
    logged env_daid bash -c "uv venv -q --python 3.12 '$V/daid' \
      && uv pip install -q --python '$V/daid/bin/python' -e '$THIRD/do-as-i-do/retargeting'"
  fi
  mv "$THIRD/COMMITS.txt.new" "$THIRD/COMMITS.txt"
  local e failed=()
  for e in stereo flash_chord sam3 daid; do [ -x "$V/$e/bin/python" ] || failed+=("$e"); done
  "$V/flash_chord/bin/python" -c "import flash_chord, jax, mujoco_warp" 2>/dev/null || failed+=("flash_chord imports")
  "$V/sam3/bin/python" -c "import sam3, torch; assert torch.cuda.is_available()" 2>/dev/null || failed+=("sam3 imports")
  "$V/daid/bin/python" -c "import mujoco_warp, mink" 2>/dev/null || failed+=("daid imports")
  if [ ${#failed[@]} -gt 0 ]; then need "v3 environments incomplete: ${failed[*]} ($LOGS/setup_env_*.log)"; return 1; fi
  ok "venvs stereo, flash_chord, sam3, daid in $V; clones pinned in $THIRD/COMMITS.txt"
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
if [ "${V3_ENVS:-1}" != 0 ]; then
  recon_images && recon_weights || true
  v3_envs || true
fi
smoke

if [ ${#NEEDS_YOU[@]} -gt 0 ]; then
  printf '\n== Done, except what needs you (then rerun this script):\n'
  printf '   - %s\n' "${NEEDS_YOU[@]}"
  exit 2
fi
printf '\n== Ready. Run Path B with:\n   cd %s && RETARGET_VENV=%s uv run --project path_b python path_b/run.py run --episodes 41\n' "$REPO" "$RETARGET"
