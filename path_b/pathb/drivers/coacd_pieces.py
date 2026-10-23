"""S3a driver: convex pieces of a mesh with CoACD (threshold 0.05), run in a venv that has coacd and
trimesh (FlashCHORD's: coacd 1.0.11).

    $FLASH_CHORD_VENV/bin/python path_b/pathb/drivers/coacd_pieces.py <mesh> <out dir> [threshold]
Writes <out>/piece_%03d.obj and pieces.json.
"""
import json
import sys
from pathlib import Path

import coacd
import trimesh


def main(mesh_path: str, out: str, threshold: float = 0.05) -> int:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), threshold=threshold)
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    info = []
    for i, (v, f) in enumerate(parts):
        piece = trimesh.Trimesh(v, f, process=False)
        piece.export(d / f"piece_{i:03d}.obj")
        info.append({"file": f"piece_{i:03d}.obj", "vertices": len(v), "volume_m3": float(abs(piece.volume))})
    (d / "pieces.json").write_text(json.dumps({"threshold": threshold, "pieces": info}, indent=1) + "\n")
    print(f"{len(info)} convex pieces", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], *(float(a) for a in sys.argv[3:4])))
