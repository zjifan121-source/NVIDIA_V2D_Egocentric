"""S7 preflight and pack (V3_BUILD_SPEC.md S7, task 16): run.py pack --split evaluation --out <run>.

Preflight, every failure listed (any one stops the pack):
1. every evaluation episode has s6_select/selection.json and its rollout (s6_eval/<candidate>/results/
   episode_XXXXXX.parquet + .json);
2. per object, S3a's mesh, the mesh inside S5's URDF (the runner's processed/<body>.obj) and the mesh
   handed to the packer (S5's score_input/meshes/) are one geometry: the same vertices (rounded to 1 µm,
   sorted) hashed with SHA-256. The spec compares file hashes; the three are different files (S3a's
   OBJ / GLB, the loader's OBJ, the runner's score OBJ), so the geometry is what can be compared;
3. no evaluation episode's report.json lists a ground-truth or scan file in files_read (GT firewall);
4. git: the working tree is clean and HEAD is on origin (the code commit URL must resolve);
5. the kit's own tools/preflight.py --track track_3 --results <packed results> --skip-kaggle passes.
Pack: the team's tools/track3/pack_submission.py with a plan per episode (run = the selected
candidate's s6_eval folder, scene = its s5_task runner folder), the code commit of HEAD; every packer
line with "MESH WARNING" is collected. Writes <run>/pack/{plan.json, preflight.json, pack.log,
track3_submission.parquet}. Never uploads (Kaggle stays on hold until James says go).
--split public is the dry pack (spec task 16's test): the same checks without the GT firewall (public
episodes are scored against ground truth by design), written to <run>/pack_dry/.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from . import gt
from .paths import REPO, manifest

REPO_URL = "https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric"


def geometry_hash(path: Path) -> tuple[str, int]:
    """SHA-256 of a mesh's vertices rounded to 1 µm and sorted (file format and vertex order do not
    matter); and the vertex count."""
    m = trimesh.load(path, force="mesh", process=False)
    v = np.round(np.asarray(m.vertices, dtype=np.float64) * 1e6).astype(np.int64)
    v = v[np.lexsort(v.T[::-1])]
    return hashlib.sha256(np.unique(v, axis=0).tobytes()).hexdigest(), len(v)


def selected(root: Path) -> tuple[str | None, list[str]]:
    sel = root / "s6_select/selection.json"
    if not sel.is_file():
        return None, [f"{root.name}: no s6_select/selection.json"]
    cand = json.loads(sel.read_text())["candidate"]  # the rollout's run (s6_eval/<candidate>/)
    problems = [f"{root.name}: no {p.relative_to(root)}" for p in
                (root / f"s6_eval/{cand}/results/{root.name}.parquet", root / f"s6_eval/{cand}/results/{root.name}.json")
                if not p.is_file()]
    return cand, problems


def mesh_checks(root: Path, names: list[str], cand: str) -> tuple[dict, list[str]]:
    cand = cand.split("_250m")[0]  # a 250M run trained on its candidate's S5 task
    task = json.loads((root / f"s5_task/{cand}/task.json").read_text())
    seq = task["sequence_id"].removesuffix("_daid")
    processed = Path(task["partition"]).parent.parent
    out, problems = {}, []
    for name in names:
        body = seq if len(names) == 1 else f"{seq}_{name}"
        files = {"s3a": root / f"s3a_mesh/{name}/mesh.obj", "urdf": processed / f"{body}.obj",
                 "packer": root / f"s5_task/{cand}/run/score_input/meshes/{root.name}/{name}.obj"}
        hashes = {}
        for k, f in files.items():
            if not f.is_file():
                problems.append(f"{root.name}/{name}: no {k} mesh {f}")
            else:
                hashes[k] = geometry_hash(f)[0]
        if len(set(hashes.values())) > 1:
            problems.append(f"{root.name}/{name}: meshes differ ({', '.join(f'{k} {h[:12]}' for k, h in hashes.items())})")
        out[name] = hashes
    return out, problems


def firewall(root: Path) -> list[str]:
    bad = []
    for rep in root.glob("*/report.json"):
        for f in json.loads(rep.read_text()).get("files_read", []):
            if gt.is_ground_truth_path(f):
                bad.append(f"{root.name}/{rep.parent.name}: read {f}")
    return bad


def git_state() -> tuple[str, list[str]]:
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    head, branch = run("rev-parse", "HEAD"), run("branch", "--show-current")
    problems = []
    if run("status", "--porcelain", "--untracked-files=no"):
        problems.append("git: uncommitted changes")
    if not run("branch", "-r", "--contains", head):
        problems.append(f"git: HEAD {head[:8]} is not on origin (push {branch} first)")
    return head, problems


def main(out: Path, numbers: list[int]) -> Path:
    pack = out / ("pack" if all(manifest()[n]["split"] == "evaluation" for n in numbers) else "pack_dry")
    pack.mkdir(parents=True, exist_ok=True)
    plan, details, problems = {}, {}, []
    for n in numbers:
        root = out / f"episode_{n:06d}"
        names = [o["name"] for o in manifest()[n]["objects"]]
        cand, p = selected(root)
        problems += p
        if cand is None:
            continue
        hashes, p = mesh_checks(root, names, cand)
        problems += p + (firewall(root) if manifest()[n]["split"] == "evaluation" else [])
        plan[str(n)] = {"run": str(root / f"s6_eval/{cand}"), "scene": str(root / f"s5_task/{cand.split('_250m')[0]}/run")}
        details[str(n)] = {"candidate": cand, "mesh_geometry_sha256": hashes}
    if len(plan) < len(numbers):
        problems.append(f"{len(plan)} of {len(numbers)} episodes are ready")
    head, p = git_state()
    problems += p
    (pack / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    report = {"head": head, "episodes": details, "problems": problems, "packed": False, "mesh_warnings": []}
    if not problems:
        tools_py = str(REPO / "tools/.venv/bin/python")
        with open(pack / "pack.log", "w") as log:
            rc = subprocess.run([tools_py, str(REPO / "tools/track3/pack_submission.py"), "--plan", str(pack / "plan.json"),
                                 "--work", str(pack), "--code-commit", f"{REPO_URL}/commit/{head}", "--python", tools_py,
                                 "-o", str(pack / "track3_submission.parquet")], cwd=REPO, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        text = (pack / "pack.log").read_text()
        report["mesh_warnings"] = [ln for ln in text.splitlines() if "MESH WARNING" in ln]
        if rc:
            report["problems"].append(f"packer exit {rc} (pack.log)")
        else:
            report["packed"] = True
            kit = subprocess.run([tools_py, "tools/preflight.py", "--track", "track_3", "--results", str(pack / "results"),
                                  "--skip-kaggle"], cwd=REPO / "assets/data/v2d_submission_kit", capture_output=True, text=True)
            report["kit_preflight"] = {"exit": kit.returncode, "tail": kit.stdout.splitlines()[-15:]}
            if kit.returncode:
                report["problems"].append("the kit's preflight failed (preflight.json: kit_preflight)")
    (pack / "preflight.json").write_text(json.dumps(report, indent=1) + "\n")
    return pack / "preflight.json"
