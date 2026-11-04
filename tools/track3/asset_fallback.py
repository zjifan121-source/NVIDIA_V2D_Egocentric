"""Select frozen, input-reviewed reconstruction bundles; never optimize or score.

An explicit visual review is evidence supplied by the caller, not an automatic
vision classifier. Complete baselines are always retained. See asset_fallback.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np


CONVENTION = "T_world_from_camera @ T_camera_from_original_metric_mesh"
ALLOWED = {".npz", ".obj", ".mtl", ".png", ".jpg", ".jpeg"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def resolve(base, path):
    p = Path(path)
    return p if p.is_absolute() else base / p


def verify_evidence(base, evidence):
    require(bool(evidence), "Missing evidence receipts")
    for item in evidence:
        require(sha256(resolve(base, item["path"])) == item["sha256"],
                "Evidence receipt hash mismatch")


def validate_bundle(base, arm, n, objects):
    bundle = arm["bundle"]
    root = resolve(base, bundle["path"]).resolve()
    files = bundle["files"]
    require("result.npz" in files, "Missing result.npz binding")
    require(arm["binding"]["convention"] == CONVENTION, "Unsupported coordinates")
    require(arm["binding"]["bundle_sha256"] == fingerprint(files),
            "Mesh/pose bundle binding mismatch")
    verify_evidence(base, arm["binding"]["evidence"])
    for name, digest in files.items():
        p = root / name
        require(Path(name).name == name and p.suffix.lower() in ALLOWED,
                "Unsafe or unsupported bundle filename")
        require(p.resolve().parent == root, "Bundle file escapes root")
        require(sha256(p) == digest, f"Bundle hash mismatch: {name}")
    mesh_names = ["mesh.obj" if i == 0 else f"mesh_{i}.obj"
                  for i in range(len(objects))]
    require(all(name in files for name in mesh_names), "Missing object mesh")
    # Check external material/texture references too: no omitted asset is silently reused.
    for name in files:
        if Path(name).suffix.lower() not in {".obj", ".mtl"}:
            continue
        for line in (root / name).read_text().splitlines():
            words = line.split()
            if words and (words[0] == "mtllib" or words[0].startswith("map_")):
                require(len(words) == 2 and words[1] in files,
                        f"Unsupported or unbound material reference in {name}")
    with np.load(root / "result.npz", allow_pickle=False) as data:
        require(data["object_names"].tolist() == objects, "Object identity/order mismatch")
        camera = data["camera_to_world_transform"]
        require(camera.shape == (n, 4, 4) and np.isfinite(camera).all(),
                "Incomplete or nonfinite camera")
        require(data["camera_is_valid"].shape == (n,) and
                data["camera_is_valid"].all(), "Invalid camera frames")
        for i in range(len(objects)):
            prefix = "object" if i == 0 else f"object_{i}"
            cam = data[prefix + "_to_camera_transform"]
            world = data[prefix + "_to_world_transform"]
            valid = data[prefix + "_is_valid"]
            require(cam.shape == world.shape == (n, 4, 4) and
                    np.isfinite(cam).all() and np.isfinite(world).all(),
                    "Incomplete or nonfinite object poses")
            require(valid.shape == (n,) and valid.all(), "Invalid object frames")
            require(np.allclose(world, camera @ cam, atol=2e-6, rtol=2e-6),
                    "Object/camera coordinate mismatch")
    return root, files


def choose(case, base):
    """No case ID, quality score, or GT value participates in this decision."""
    a = case["baseline"]
    if a["state"] == "COMPLETE":
        chosen, reason = "baseline", "Retain complete baseline"
    elif a["state"] == "FAIL_ANCHOR_ALIGNMENT":
        gate = a.get("gate", {})
        require(gate.get("state") == "FAIL" and gate.get("GT_used") is False,
                "Missing input-only failure review")
        require(gate.get("reason") == "ASYMMETRIC_STRUCTURE_DIRECTION",
                "Failure is outside this fallback mechanism")
        verify_evidence(base, gate.get("evidence"))
        c = case.get("candidate")
        if not c or c["state"] != "COMPLETE":
            return None, "Alternative missing or incomplete"
        cg = c.get("gate", {})
        require(cg.get("GT_used") is False and cg.get("state") in {"PASS", "FAIL"},
                "Missing input-only candidate review")
        verify_evidence(base, cg.get("evidence"))
        if cg["state"] != "PASS":
            return None, "Alternative failed structure/anchor review"
        require(bool(a.get("input_signature")) and
                a["input_signature"] == c.get("input_signature"),
                "Observation/FP input signature mismatch")
        chosen, reason = "candidate", "Input-reviewed initialization recovery"
    else:
        return None, "Unresolved failure; this mechanism does not apply"
    validate_bundle(base, case[chosen], case["frames"], case["objects"])
    return chosen, reason


def run(manifest_path, output=None):
    manifest_path = Path(manifest_path).resolve()
    base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema_version"] == 1, "Unsupported manifest version")
    require(bool(manifest["cases"]), "Empty cohort")
    rows, seen = [], set()
    for case in manifest["cases"]:
        ident = case["id"]
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", ident)) and ident not in seen,
                "Invalid or duplicate case ID")
        require(case["frames"] > 0 and len(case["objects"]) > 0, "Empty sequence")
        seen.add(ident)
        arm, reason = choose(case, base)
        rows.append({"id": ident, "selected_arm": arm, "reason": reason,
                     "bundle_sha256": case[arm]["binding"]["bundle_sha256"] if arm else None})
    result = {"schema_version": 1, "manifest_sha256": sha256(manifest_path),
              "complete": all(row["selected_arm"] is not None for row in rows),
              "GT_selection": False, "visual_review_required": True, "rows": rows}
    if output is not None:
        require(result["complete"], "Unresolved cases: refuse incomplete delivery")
        output = Path(output).resolve()
        require(not output.exists(), "Output exists; never overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".fallback-", dir=output.parent))
        try:
            for case, row in zip(manifest["cases"], rows):
                arm = case[row["selected_arm"]]
                source, files = validate_bundle(base, arm, case["frames"], case["objects"])
                target = temporary / case["id"]
                target.mkdir()
                for name, digest in files.items():
                    shutil.copy2(source / name, target / name)
                    require(sha256(target / name) == digest, "Export hash mismatch")
                metadata = dict(row, frames=case["frames"], objects=case["objects"],
                                convention=CONVENTION, files=files, experimental=True)
                (target / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
            (temporary / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
            require(not output.exists(), "Output appeared during export")
            temporary.rename(output)
        except BaseException:
            shutil.rmtree(temporary)
            raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional new directory; otherwise validate only")
    args = parser.parse_args()
    try:
        result = run(args.manifest, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"asset_fallback: {error}\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["complete"] else 2)
