"""Point a copied training scene at its own files.

A scene's training parquet stores absolute paths to the machine that built it (object mesh, object URDF,
source bundle). FlashCHORD falls back to the scene's own assets/ folder only when that path does not exist;
on a shared cluster where the builder's folder exists but is private, loading fails with PermissionError.
This rewrites the builder's scene root to the scene's current location in every string cell of the
processed parquet (in place, keeping the schema), so a scene downloaded from the team dataset trains as is.
Usage: python tools/track3/relocate_scene.py <scene dir> [<scene dir> ...]
  scene dir: the folder holding assets/ (e.g. .../scenes/episode_000038/ken_c_scored_attach)
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def old_root(table: pa.Table) -> str:
    """The builder's scene root: an object mesh path up to its last '/assets/'."""
    mesh = table.column('object_mesh_paths')[0].as_py()[0]
    return mesh[: mesh.rindex('/assets/')]


def rewrite(value, old: str, new: str):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [rewrite(v, old, new) for v in value]
    return value


def relocate(scene: Path) -> None:
    for parquet in sorted(scene.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet')):
        table = pq.read_table(parquet)
        old, new = old_root(table), str(scene.resolve())
        if old == new:
            print(f'{parquet}: already local')
            continue
        columns = []
        for name in table.column_names:
            col = table.column(name)
            if pa.types.is_string(col.type) or pa.types.is_large_string(col.type) or (
                    pa.types.is_list(col.type) and pa.types.is_string(col.type.value_type)):
                col = pa.array([rewrite(v, old, new) for v in col.to_pylist()], type=col.type)
            columns.append(col)
        pq.write_table(pa.Table.from_arrays(columns, schema=table.schema), parquet)
        print(f'{parquet}: {old} -> {new}')


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        relocate(Path(arg))
