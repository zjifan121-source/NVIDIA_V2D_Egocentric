# Added for V2D Track 3 by Edmond Wang, 2026-10: opt-in Apptainer backend for
# hosts without Docker (e.g. Slurm clusters). Apache-2.0, see NOTICE.
"""Run a ``docker run ...`` command with the selected container runtime.

Docker stays the default and is passed through unchanged. Set
``V2D_CONTAINER_RUNTIME=apptainer`` to translate the same command to
``apptainer exec``; images are then read from
``$V2D_APPTAINER_IMAGE_DIR/<image>.sif`` (e.g. made with
``apptainer pull v2d_moge.sif docker://<registry>/v2d_moge:<tag>``).
Containers start in ``/app`` (the images' WORKDIR); override with
``V2D_APPTAINER_PWD``. ``V2D_APPTAINER_FLAGS`` adds host-specific options, e.g.
``--no-mount /opt`` where the site config binds the host's /opt over the images' /opt/conda.

Docker anonymous volumes (``-v /path``, used under ``dev=True`` to keep an image's
compiled files visible beneath the mounted source tree) have no Apptainer
equivalent, so each one is copied out of the image once, into
``$V2D_APPTAINER_IMAGE_DIR/<image>.preserve/<path>``, and bound back at ``/path``.

Docker gives every container a private /tmp (the images' HOME); Apptainer would share the
host's, where another user's /tmp/.cache breaks Warp and Hugging Face. So /tmp (and HOME when it
is /tmp) is a private host folder per Slurm job (or process): <tempdir>/v2d-tmp-<user>-<id>.
"""
import getpass
import os
import shlex
import tempfile
import subprocess
import sys
from pathlib import Path

RUNTIME_ENV = "V2D_CONTAINER_RUNTIME"
IMAGE_DIR_ENV = "V2D_APPTAINER_IMAGE_DIR"

PWD_ENV = "V2D_APPTAINER_PWD"
FLAGS_ENV = "V2D_APPTAINER_FLAGS"

# Every module image sets WORKDIR /app; Apptainer ignores WORKDIR, so pass it.
_DEFAULT_PWD = "/app"


def container_runtime() -> str:
    runtime = os.environ.get(RUNTIME_ENV, "docker").strip().lower() or "docker"
    if runtime not in ("docker", "apptainer"):
        raise ValueError(f"{RUNTIME_ENV}={runtime!r}; expected 'docker' or 'apptainer'")
    return runtime


def _sif_path(image: str) -> str:
    image_dir = os.environ.get(IMAGE_DIR_ENV)
    if not image_dir:
        raise RuntimeError(f"{RUNTIME_ENV}=apptainer requires {IMAGE_DIR_ENV} (directory of <image>.sif files)")
    name = image.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]
    sif = Path(image_dir) / f"{name}.sif"
    if not sif.is_file():
        raise FileNotFoundError(f"Apptainer image for {image!r} not found: {sif}")
    return str(sif)


def _private_tmp(env: dict[str, str]) -> str:
    """One private /tmp per Slurm job (shared by its steps, so caches persist between them)."""
    ident = env.get("SLURM_JOB_ID") or str(os.getpid())
    path = Path(tempfile.gettempdir()) / f"v2d-tmp-{getpass.getuser()}-{ident}"
    path.mkdir(mode=0o700, exist_ok=True)
    return str(path)


def _preserved_copy(sif: str, path: str, flags: list[str]) -> str:
    """Host copy of `path` as built inside `sif` (made once; atomic rename for concurrent jobs)."""
    dst = Path(sif).with_suffix(".preserve") / path.lstrip("/")
    if dst.is_dir():
        return str(dst)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}")
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["apptainer", "exec", "--cleanenv", "--no-home", *flags, "--bind", f"{tmp}:/preserve_out",
                    sif, "cp", "-a", f"{path}/.", "/preserve_out/"], check=True)
    try:
        tmp.rename(dst)
    except OSError:  # another job finished first
        subprocess.run(["rm", "-rf", str(tmp)], check=False)
    return str(dst)


def to_apptainer(cmd: list[str], env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Translate a ``docker run`` argv to ``apptainer exec``.

    Returns the new argv and the subprocess environment. Bare ``-e NAME``
    (pass the host's value, e.g. HF_TOKEN) becomes ``APPTAINERENV_NAME`` in the
    environment rather than an argument, so secrets stay off the command line.
    """
    if cmd[:2] != ["docker", "run"]:
        raise ValueError(f"expected a 'docker run' command, got: {cmd[:2]}")
    flags = shlex.split(env.get(FLAGS_ENV, ""))
    out = ["apptainer", "exec", "--cleanenv", "--pwd", env.get(PWD_ENV, _DEFAULT_PWD), *flags]
    run_env = dict(env)
    binds: list[tuple[str, str]] = []
    anonymous: list[str] = []
    nv = False
    home = None
    i = 2
    while i < len(cmd) and cmd[i].startswith("-"):
        opt = cmd[i]
        if opt == "--rm":
            i += 1
        elif opt == "--runtime=nvidia":
            nv = True
            i += 1
        elif opt == "--gpus":
            nv = True
            i += 2
        elif opt == "--user":  # Apptainer always runs as the calling user.
            i += 2
        elif opt == "-e":
            var = cmd[i + 1]
            if var.startswith("HOME="):
                # Apptainer forbids setting HOME via --env; --home sets it (and
                # binds that host path, e.g. /tmp, which is already shared).
                home = var.split("=", 1)[1]
            elif "=" in var:
                out += ["--env", var]
            elif var in run_env:
                run_env[f"APPTAINERENV_{var}"] = run_env[var]
            i += 2
        elif opt == "-v":
            spec = cmd[i + 1]
            if ":" in spec:
                host, target = spec.split(":", 1)
                binds.append((host, target))
                out += ["--bind", spec]
            else:
                anonymous.append(spec)
            i += 2
        else:
            raise ValueError(f"docker option {opt!r} has no Apptainer translation")
    if i >= len(cmd):
        raise ValueError("docker run command has no image")
    sif = _sif_path(cmd[i])
    for path in anonymous:
        # The mount point must exist in whatever host tree is bound above it.
        for host, target in binds:
            if path.startswith(target.rstrip("/") + "/"):
                Path(host, os.path.relpath(path, target)).mkdir(parents=True, exist_ok=True)
        out += ["--bind", f"{_preserved_copy(sif, path, flags)}:{path}"]
    tmp = _private_tmp(env)
    out += ["--no-mount", "tmp", "--bind", f"{tmp}:/tmp"]
    if home:
        out += ["--home", f"{tmp}:{home}"] if home == "/tmp" else ["--home", home]
    else:
        out.append("--no-home")
    if nv:
        out.append("--nv")
    out.append(sif)
    out += cmd[i + 1:]
    return out, run_env


def run_container_cmd(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` for a ``docker run`` argv, honouring V2D_CONTAINER_RUNTIME."""
    if container_runtime() == "docker":
        return subprocess.run(cmd, **kwargs)
    env = kwargs.pop("env", None)
    cmd, kwargs["env"] = to_apptainer(cmd, dict(os.environ if env is None else env))
    print("[apptainer] " + " ".join(cmd), file=sys.stderr, flush=True)
    return subprocess.run(cmd, **kwargs)
