"""Copy a built scene from another machine into ours and repoint its absolute paths (no code from the source is run).

The processed parquet stores absolute paths of the machine that built it (raw_motion_file, object_mesh_paths,
object_urdf_paths). This copies SRC to DST, points those columns at the copied files, and, when the scene has no
score_input/meshes (the packer needs it), writes them from the scene's own processed meshes under the roster names.

    python tools/track3/import_scene.py SRC DST --episode N [--raw-bundle DIR_WITH_result.npz]
"""
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROC = "assets/human_motion_data/ego_recon/processed"
ROSTER = Path(__file__).resolve().parent / "episodes.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--raw-bundle", type=Path, help="bundle the scene was built from, when the scene has no bundle_cleaned")
    a = ap.parse_args()
    assert not a.dst.exists(), f"{a.dst} exists"
    shutil.copytree(a.src, a.dst, symlinks=False)
    proc = a.dst / PROC
    files = glob.glob(str(proc / "sequence_id=*/robot_name=sharpa_wave/*.parquet"))
    assert len(files) == 1, files
    t = pq.read_table(files[0])
    info = {}

    def local(p):
        q = proc / os.path.basename(p)
        assert q.is_file(), f"missing {q}"
        return str(q)

    cols = {}
    for c in ("object_mesh_paths", "object_urdf_paths"):
        old = t.column(c).combine_chunks().to_pylist()[0]
        cols[c] = [local(p) for p in old]
        info[c] = cols[c]
    raw = a.dst / "bundle_cleaned" / "result.npz"
    if not raw.is_file() and a.raw_bundle:
        (a.dst / "bundle_raw").mkdir()
        for f in a.raw_bundle.iterdir():
            if f.is_file():
                shutil.copy2(f, a.dst / "bundle_raw" / f.name)
        raw = a.dst / "bundle_raw" / "result.npz"
    assert raw.is_file(), f"no bundle for raw_motion_file: {raw}"
    cols["raw_motion_file"] = str(raw)
    info["raw_motion_file"] = str(raw)
    for c, v in cols.items():
        i = t.column_names.index(c)
        t = t.set_column(i, t.field(i), pa.array([v], type=t.schema.field(c).type))
    os.rename(files[0], files[0] + ".orig")
    pq.write_table(t, files[0])

    # score_input/meshes/episode_N/<roster name>.obj for the packer
    sdir = a.dst / "score_input" / "meshes" / f"episode_{a.episode:06d}"
    if not sdir.is_dir():
        names = [o["name"] for o in {e["episode"]: e for e in json.loads(ROSTER.read_text())["episodes"]}[a.episode]["objects"]]
        sdir.mkdir(parents=True)
        made = {}
        for n in names:
            objs = [p for p in cols["object_mesh_paths"] if os.path.basename(p).endswith(f"_{n}.obj")]
            assert len(objs) == 1, (n, cols["object_mesh_paths"])
            shutil.copy2(objs[0], sdir / f"{n}.obj")
            mtl = objs[0][:-4] + ".mtl"
            if os.path.isfile(mtl):
                shutil.copy2(mtl, sdir / os.path.basename(mtl))  # the obj's mtllib names it
                for line in open(mtl):
                    if line.split()[:1] == ["map_Kd"]:
                        tex = proc / line.split()[1]
                        if tex.is_file():
                            shutil.copy2(tex, sdir / tex.name)
            made[n] = os.path.basename(objs[0])
        info["score_input_written"] = made
    (a.dst / "IMPORTED.json").write_text(json.dumps(dict(source=str(a.src), **info), indent=1))
    print(json.dumps(dict(dst=str(a.dst), **{k: (v if isinstance(v, (str, dict)) else len(v)) for k, v in info.items()})))


if __name__ == "__main__":
    main()
