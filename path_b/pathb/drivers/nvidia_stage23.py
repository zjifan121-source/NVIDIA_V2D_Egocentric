"""Path B's stages 2-3 on NVIDIA's code as imported (pathb/nvidia.py's mirrors; no team code), run in the
stages 2-3 venv ($RETARGET_VENV: torch, manotorch, pxr, robotic_grounding's dependencies).

    $RETARGET_VENV/bin/python path_b/pathb/drivers/nvidia_stage23.py <request.json>

request.json: {"loader": mirror of v2d_task_library_loader, "robotic": mirror of robotic_grounding,
"mano_dir": dir with models/, "device", "fps", "ground_z_offset", "sequence_id", "objects": [{"name",
"bundle"}, ...] (every bundle in the same world, with the same camera and hands: S4b's), "out"}.

1. load    NVIDIA's ego loader (EgoReconDatasetLoader) subclassed here (`Scene`): one body per object, each
           from its own bundle; the clip's frame rate; the scene aligned as NVIDIA aligns a bundle with
           --no_ground_align (object 0 centred in xy, its bottom at ground_z_offset). Each hand is scaled
           about its mesh centroid by the bundle's hand_<side>_scale, as NVIDIA's hand alignment defines that
           scale (v2d_hamer/lib/align_hands.py: "the multiplicative size correction the renderer applies
           around the mesh centroid"); NVIDIA's loader leaves it out
2. retarget scripts/retarget/ego_recon_to_sharpa.py (IK onto the Sharpa Wave hands)
3. urdf    scripts/generate_rigid_urdfs.py --dataset ego_recon
4. support scripts/reconstruct_support_surfaces.py through drivers/support_surfaces.py (a dropped disk at the
           table's height, `table_height`, that the kept disks do not cover comes back)
Writes <out>/report.json {sequence_id, processed (the partition), support_usd, meshes}; pathb/stage23.py
checks the output in Path B's venv.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROBOT = "sharpa_wave"


class Mirror(importlib.abc.MetaPathFinder):
    """Packages from the mirrors, ahead of the venv's editable installs of baseline/ (their submodules then
    resolve through the package's __path__); `namespaces` are made namespace packages."""

    def __init__(self, packages: dict[str, str], namespaces: tuple = ()):
        self.packages, self.namespaces = packages, namespaces

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.namespaces:
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        if fullname in self.packages:
            d = Path(self.packages[fullname])
            return importlib.util.spec_from_file_location(fullname, d / "__init__.py", submodule_search_locations=[str(d)])
        return None


def env_for(req: dict, out: Path) -> dict:
    hmd = out / "assets/human_motion_data"  # FlashCHORD re-roots paths through /assets/human_motion_data/
    rg_pkg = Path(req["robotic"]) / "source/robotic_grounding"
    return dict(os.environ, HUMAN_MOTION_DATA_DIR=str(hmd), ROBOTIC_GROUNDING_INTERMEDIATE_DIR=str(out / "intermediate"),
                PYTHONPATH=os.pathsep.join([str(rg_pkg), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
                PYOPENGL_PLATFORM="egl", PYTHONUNBUFFERED="1")


def load(req: dict, out: Path) -> Path:
    """Step 1 in this process; returns the loaded partition's parent (loader --output_dir)."""
    os.environ.update(env_for(req, out))
    lib = Path(req["loader"]) / "lib"
    pkg = Path(req["robotic"]) / "source/robotic_grounding/robotic_grounding"
    sys.meta_path.insert(0, Mirror({"robotic_grounding": str(pkg), "v2d.task_library_loader.lib": str(lib)},
                                   ("v2d", "v2d.task_library_loader")))
    import numpy as np
    import torch
    from robotic_grounding.retarget.naming import make_usd_safe
    from v2d.task_library_loader.lib import dataset_loader_base as base
    from v2d.task_library_loader.lib import ego_recon_loader as ego
    from v2d.task_library_loader.lib.read_mano import MANO
    assert Path(base.__file__).parent == lib and Path(ego.__file__).parent == lib, (base.__file__, ego.__file__)

    class ScaledMANO(MANO):
        """NVIDIA's MANO wrapper; joints and vertices scaled about the vertex centroid (`scales`: per side, (n,))."""
        scales: dict = {}

        def forward(self, side, *args, **kwargs):
            out = super().forward(side, *args, **kwargs)
            s = self.scales.get(side)
            if s is None or len(s) != out["vertices"].shape[0]:
                return out
            s = torch.as_tensor(s, dtype=out["vertices"].dtype, device=out["vertices"].device)[:, None, None]
            c = out["vertices"].mean(dim=1, keepdim=True)
            return dict(out, vertices=c + s * (out["vertices"] - c), joints=c + s * (out["joints"] - c))

    base.MANO = ScaledMANO  # the base loader's FK only; the ego loader's own re-pose keeps NVIDIA's MANO
    seq, objects = req["sequence_id"], req["objects"]
    bodies = [make_usd_safe(seq)] if len(objects) == 1 else [make_usd_safe(f"{seq}_{o['name']}") for o in objects]

    def source(o):
        d = Path(o["bundle"])
        with np.load(d / "result.npz") as z:
            scale = float(z["object_scale"])
        return ego.EgoReconSequenceSource(result_dir=d, npz_path=d / "result.npz", mesh_path=d / "mesh.obj",
                                          object_scale=scale)

    sources = [source(o) for o in objects]

    class Scene(ego.EgoReconDatasetLoader):
        def list_sequences(self, args):
            return [base.SequenceInfo(sequence_id=seq, raw_motion_file=str(sources[0].npz_path), object_name=seq,
                                      object_body_names=bodies, source=sources[0])]

        def load_mano_data(self, sequence_info, device):
            out = super().load_mano_data(sequence_info, device)
            with np.load(sources[0].npz_path) as z:
                ScaledMANO.scales = {s: np.asarray(z[f"hand_{s}_scale"], np.float64) for s in ("right", "left")
                                     if f"hand_{s}_scale" in z.files}
            return out

        def load_object_data(self, sequence_info):
            R_align, t_align = self._scene_align(sequence_info)
            data = {}
            for body, src in zip(bodies, sources):
                with np.load(src.npz_path) as z:
                    poses = z["object_to_world_transform"].astype(np.float64)
                aligned = poses.copy()
                aligned[:, :3, :3] = np.einsum("ij,njk->nik", R_align, poses[:, :3, :3])
                aligned[:, :3, 3] = (R_align @ poses[:, :3, 3].T).T + t_align
                root_pos, root_aa = base.poses_to_root_position_and_axis_angle(aligned)
                data[body] = (aligned, root_pos, root_aa, None)
            return data

        def load_object_meshes(self, sequence_info, device):
            scales = {s.object_scale for s in sources}
            if len(scales) != 1:
                raise ValueError(f"objects with different object_scale {scales}: the loader takes one")
            return base.load_meshes_to_device({b: str(s.mesh_path) for b, s in zip(bodies, sources)}, device,
                                              vertex_scale=scales.pop())

        def get_fps(self):
            return float(req["fps"])

        def get_object_mesh_paths(self, sequence_info):
            return [str(self._ensure_object_asset(replace(sequence_info, object_body_names=[b], source=s)))
                    for b, s in zip(bodies, sources)]

        def get_object_urdf_paths(self, sequence_info):
            return [str(Path(self._args.object_model_root) / f"{b}_rigid.urdf") for b in bodies]

    loaded = out / "intermediate/ego_recon/loaded"
    sys.argv = ["ego_recon_loader", "--dataset_root", str(sources[0].result_dir), "--output_dir", str(loaded),
                "--mano_model_dir", str(req["mano_dir"]), "--no_ground_align",
                "--ground_z_offset", str(req["ground_z_offset"]), "--sequence_name", seq, "--object_name", seq,
                "--sequence_id", seq, "--device", req["device"], "--save"]
    args = ego.parse_args()
    args.dataset = "ego_recon"
    Scene().run(args)
    return loaded


def run(cmd: list, cwd: Path, env: dict, log: Path) -> None:
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        rc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"exit {rc}: {' '.join(map(str, cmd[:3]))}; see {log}")


def partition(root: Path, seq: str) -> Path:
    return root / f"sequence_id={seq}" / f"robot_name={ROBOT}"


def table_height(req: dict) -> float:
    """The table (z = 0 in S3d's world) in the task scene. NVIDIA's loader with --no_ground_align moves object 0's
    frame-0 bottom to ground_z_offset (ego_recon_loader._scene_align), so the table sits that bottom's height under
    it: the same height where object 0 starts on the table, higher where its first rest is fitted into the table
    (public ep12: the cup in the rack 3.8 cm down, the rack's disk then 2.7 cm over ground_z_offset)."""
    import numpy as np
    import trimesh
    d = Path(req["objects"][0]["bundle"])
    with np.load(d / "result.npz") as z:
        T0, scale = z["object_to_world_transform"][0].astype(np.float64), float(z["object_scale"])
    mesh = trimesh.load(str(d / "mesh.obj"))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    v = np.asarray(mesh.vertices, dtype=np.float64) * scale
    return float(req["ground_z_offset"]) - float(((T0[:3, :3] @ v.T).T + T0[:3, 3])[:, 2].min())


def main(request: str) -> int:
    req = json.loads(Path(request).read_text())
    out = Path(req["out"])
    out.mkdir(parents=True, exist_ok=True)
    log = out / "pipeline.log"
    seq = req["sequence_id"]
    hmd = out / "assets/human_motion_data"
    processed = hmd / "ego_recon/processed"
    usda = hmd / f"ego_recon/reconstructed_stage/{seq}_{ROBOT}_support.usda"
    env, rg, py = env_for(req, out), Path(req["robotic"]), sys.executable
    loaded = out / "intermediate/ego_recon/loaded"
    if not any(partition(loaded, seq).glob("*.parquet")):
        load(req, out)
    if not any(partition(processed, seq).glob("*.parquet")):
        run([py, "scripts/retarget/ego_recon_to_sharpa.py", "--input_dir", loaded, "--output_dir", processed,
             "--sequence_id", seq, "--device", req["device"], "--save"], rg, env, log)
    bodies = [seq] if len(req["objects"]) == 1 else [f"{seq}_{o['name']}" for o in req["objects"]]
    if not all((processed / f"{b}_rigid.urdf").is_file() for b in bodies):
        run([py, "scripts/generate_rigid_urdfs.py", "--dataset", "ego_recon"], rg, env, log)
    if not usda.is_file():
        run([py, Path(__file__).with_name("support_surfaces.py"), table_height(req),
             "scripts/reconstruct_support_surfaces.py", "--dataset", "ego_recon", "--input_dir", processed,
             "--sequence_id", seq, "--robot_name", ROBOT, "--output", usda], rg, env, log)
    report = {"sequence_id": seq, "processed": str(partition(processed, seq)), "support_usd": str(usda),
              "meshes": [str(processed / f"{b}.obj") for b in bodies], "objects": req["objects"], "fps": req["fps"]}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
