"""S3c's static target (pathb/stages/s3c_invert.largest_above_table): the TSDF's largest piece above the table."""
import numpy as np
import open3d as o3d

from pathb import scan as S
from pathb.stages.s3c_invert import largest_above_table


def box(corner, size, splits):
    m = o3d.geometry.TriangleMesh.create_box(*size)
    m.translate(corner)
    return m.subdivide_midpoint(splits)


def test_a_floor_slab_under_the_table_is_never_the_target():
    cup = box((0.0, 0.0, 0.0), (0.08, 0.08, 0.1), 2)
    floor = box((0.0, 0.0, -0.75), (0.5, 0.5, 0.01), 4)  # ep21: the floor through the mask's edge, more triangles
    both = cup + floor
    assert np.asarray(S.largest_part(both).vertices)[:, 2].mean() < -0.7  # the plain largest piece is the slab
    assert np.asarray(largest_above_table(both).vertices)[:, 2].min() >= 0.0
    # without a slab the target is the largest piece exactly as before
    a, b = largest_above_table(cup), S.largest_part(cup)
    assert len(a.triangles) == len(b.triangles) and np.allclose(np.asarray(a.vertices), np.asarray(b.vertices))
