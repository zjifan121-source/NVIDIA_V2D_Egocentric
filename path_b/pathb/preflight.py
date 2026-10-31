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
Pack (spec §14 L11): the kit's own tools/pack_submission.py, run in the kit, on <pack>/results (each
selected rollout, its bodies renamed to the roster's object names: our scene names them
<sequence>[_<object>], the packer wants the roster's) and <pack>/meshes (each object's S5 score mesh,
in the rollout's body coordinates), with the code commit of HEAD; every packer line with "MESH
WARNING" is collected. Writes <run>/pack/{plan.json, preflight.json, pack.log, results/, meshes/,
track3_submission.parquet}. Never uploads (Kaggle stays on hold until James says go).
--split public is the dry pack: the same checks without the GT firewall (public episodes are scored
against ground truth by design), in <run>/pack_dry/ and through a copy of the kit (pack_dry/kit/) whose
roster, row grid and scored episodes are the packed public episodes' (the kit's own lists only the
evaluation split; the kit itself is never changed).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from . import gt
from .paths import REPO, manifest

KIT = REPO / "assets/data/v2d_submission_kit"
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


def here(path: str, root: Path) -> Path:
    """A path S5 wrote on another host (task.json holds the L40S's absolute paths), re-rooted at this episode
    folder when it does not exist here: its part after the episode folder's name. Packing a copy on the Mac."""
    p = Path(path)
    if p.exists() or root.name not in p.parts:
        return p
    return root.joinpath(*p.parts[p.parts.index(root.name) + 1:])


def mesh_checks(root: Path, names: list[str], cand: str) -> tuple[dict, list[str]]:
    cand = cand.split("_250m")[0]  # a 250M run trained on its candidate's S5 task
    task = json.loads((root / f"s5_task/{cand}/task.json").read_text())
    seq = task["sequence_id"].removesuffix("_daid")
    processed = here(task["partition"], root).parent.parent
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


def relabel(src: Path, dst: Path, names: list[str]) -> list[str]:
    """Copy a rollout with object_body_name set to the roster's names by object_id. Each body must sit
    in its roster slot: named <sequence>_<object> (or, alone in its scene, ending in its object's name)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pq.read_table(src)
    ids, bodies = table.column("object_id").to_pylist(), table.column("object_body_name").to_pylist()
    if sorted(set(ids)) != list(range(len(names))):
        return [f"{src.name}: object ids {sorted(set(ids))} for roster {names}"]
    wrong = sorted({b for i, b in zip(ids, bodies) if not b.endswith("_" + names[i])})
    if wrong:
        return [f"{src.name}: bodies {wrong} do not match roster {names} by slot"]
    i = table.schema.get_field_index("object_body_name")
    field = table.schema.field(i)
    pq.write_table(table.set_column(i, field, pa.array([names[k] for k in ids]).cast(field.type)), dst)
    return []


def gather(pack: Path, plan: dict, names: dict[str, list[str]]) -> list[str]:
    """<pack>/results/episode_XXXXXX.{parquet,json} (relabelled) and <pack>/meshes/episode_XXXXXX/<object>.obj."""
    problems = []
    for sub in ("results", "meshes"):
        shutil.rmtree(pack / sub, ignore_errors=True)
        (pack / sub).mkdir(parents=True)
    for n, p in plan.items():
        e = f"episode_{int(n):06d}"
        results = Path(p["run"]) / "results"
        problems += relabel(results / f"{e}.parquet", pack / f"results/{e}.parquet", names[n])
        shutil.copy2(results / f"{e}.json", pack / f"results/{e}.json")
        (pack / f"meshes/{e}").mkdir()
        for name in names[n]:
            src = Path(p["scene"]) / f"score_input/meshes/{e}/{name}.obj"
            if src.is_file():
                shutil.copy2(src, pack / f"meshes/{e}/{name}.obj")
            else:
                problems.append(f"{e}/{name}: no score mesh {src}")
    return problems


def dry_kit(pack: Path, numbers: list[int]) -> Path:
    """A copy of the kit whose Track 3 roster, row grid (world 0, every frame, every object) and scored
    episodes are these public episodes'; its tools are copied (they find the kit at their own parent),
    the rest linked."""
    import pandas as pd
    kit = pack / "kit"
    shutil.rmtree(kit, ignore_errors=True)
    kit.mkdir(parents=True)
    for f in KIT.iterdir():
        if f.name not in ("config", "data", "tools"):
            (kit / f.name).symlink_to(f)
    shutil.copytree(KIT / "tools", kit / "tools")
    (kit / "config").mkdir()
    (kit / "data").mkdir()
    config = json.loads((KIT / "config/tracks.json").read_text())
    config["tracks"]["track_3"]["scored_episodes"] = sorted(numbers)
    (kit / "config/tracks.json").write_text(json.dumps(config, indent=1) + "\n")
    m = manifest()
    roster = {str(n): [o["name"] for o in m[n]["objects"]] for n in sorted(numbers)}
    (kit / "data/track_3_evaluation_objects.json").write_text(json.dumps({"episodes": roster}, indent=1) + "\n")
    sample = pd.read_parquet(KIT / "data/track_3_sample_submission.parquet")
    row = sample.iloc[0]
    ids = [f"track_3/e{n:06d}/w000/f{f:06d}/o{o}" for n in sorted(numbers) for f in range(int(m[n]["num_frames"]))
           for o in range(len(roster[str(n)]))]
    pd.DataFrame({c: [row[c]] * len(ids) for c in sample.columns} | {"row_id": ids}).to_parquet(
        kit / "data/track_3_sample_submission.parquet", index=False)
    return kit


def main(out: Path, numbers: list[int]) -> Path:
    evaluation = all(manifest()[n]["split"] == "evaluation" for n in numbers)
    pack = out / ("pack" if evaluation else "pack_dry")
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
        problems += gather(pack, plan, {str(n): [o["name"] for o in manifest()[n]["objects"]] for n in numbers if str(n) in plan})
    if not problems:
        tools_py = str(REPO / "tools/.venv/bin/python")  # an interpreter with the kit's requirements.txt
        kit = KIT if evaluation else dry_kit(pack, [int(n) for n in plan])
        report["kit"] = str(kit)
        with open(pack / "pack.log", "w") as log:
            rc = subprocess.run([tools_py, "tools/pack_submission.py", "--track", "track_3", "--results", str(pack / "results"),
                                 "--meshes", str(pack / "meshes"), "--code-commit", f"{REPO_URL}/commit/{head}",
                                 "-o", str(pack / "track3_submission.parquet")], cwd=kit, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        text = (pack / "pack.log").read_text()
        report["mesh_warnings"] = [ln for ln in text.splitlines() if "MESH WARNING" in ln]
        if rc:
            report["problems"].append(f"packer exit {rc} (pack.log)")
        else:
            report["packed"] = True
            pre = subprocess.run([tools_py, "tools/preflight.py", "--track", "track_3", "--results", str(pack / "results"),
                                  "--skip-kaggle"], cwd=kit, capture_output=True, text=True)
            report["kit_preflight"] = {"exit": pre.returncode, "tail": pre.stdout.splitlines()[-15:]}
            if pre.returncode:
                report["problems"].append("the kit's preflight failed (preflight.json: kit_preflight)")
    (pack / "preflight.json").write_text(json.dumps(report, indent=1) + "\n")
    return pack / "preflight.json"
