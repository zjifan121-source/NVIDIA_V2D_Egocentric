# Added for V2D Track 3 by Edmond Wang, 2026-10: opt-in Apptainer backend for
# hosts without Docker (e.g. Slurm clusters). Apache-2.0, see NOTICE.
"""Run a ``docker run ...`` command with the selected container runtime.

Docker stays the default and is passed through unchanged. Set
``V2D_CONTAINER_RUNTIME=apptainer`` to translate the same command to
``apptainer exec``; images are then read from
``$V2D_APPTAINER_IMAGE_DIR/<image>.sif`` (e.g. made with
``apptainer pull v2d_moge.sif docker://<registry>/v2d_moge:<tag>``).
Containers start in ``/app`` (the images' WORKDIR); override with
``V2D_APPTAINER_PWD``.
"""
import os
import subprocess
import sys
from pathlib import Path

RUNTIME_ENV = "V2D_CONTAINER_RUNTIME"
IMAGE_DIR_ENV = "V2D_APPTAINER_IMAGE_DIR"

PWD_ENV = "V2D_APPTAINER_PWD"

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


def to_apptainer(cmd: list[str], env: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Translate a ``docker run`` argv to ``apptainer exec``.

    Returns the new argv and the subprocess environment. Bare ``-e NAME``
    (pass the host's value, e.g. HF_TOKEN) becomes ``APPTAINERENV_NAME`` in the
    environment rather than an argument, so secrets stay off the command line.
    """
    if cmd[:2] != ["docker", "run"]:
        raise ValueError(f"expected a 'docker run' command, got: {cmd[:2]}")
    out = ["apptainer", "exec", "--cleanenv", "--pwd", env.get(PWD_ENV, _DEFAULT_PWD)]
    run_env = dict(env)
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
            if ":" not in spec:
                # Docker anonymous volume (used under --dev to keep image-built
                # artifacts visible beneath the source mount). Apptainer has no
                # equivalent, so the image's compiled files would be hidden.
                raise RuntimeError(
                    f"anonymous volume {spec!r} is not supported with Apptainer; "
                    "run this step without --dev"
                )
            out += ["--bind", spec]
            i += 2
        else:
            raise ValueError(f"docker option {opt!r} has no Apptainer translation")
    if i >= len(cmd):
        raise ValueError("docker run command has no image")
    out += ["--home", home] if home else ["--no-home"]
    if nv:
        out.append("--nv")
    out.append(_sif_path(cmd[i]))
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
