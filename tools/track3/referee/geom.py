"""Small geometry helpers for the referee (pose JSON I/O, OBJ reading, surface sampling, the mesh's twin turns)."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def depth_m(path):
    """Depth in metres from the pipeline's 16-bit inverse-depth PNG (65535 / raw - 1); NaN where raw == 65535.
    Read with Pillow (in the tools environment); same pixel values as OpenCV's IMREAD_UNCHANGED."""
    from PIL import Image
    if not Path(path).is_file():
        return None
    p = np.asarray(Image.open(path)).astype(np.float64)
    return np.where(p < 65535, 65535.0 / np.maximum(p, 1) - 1.0, np.nan)


def load_T(path):
    """Pose JSON (rotation wxyz, translation) -> (4x4 mesh-to-camera matrix, the JSON dict)."""
    d = json.loads(Path(path).read_text())
    w, x, y, z = d['rotation']
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = d['translation']
    return T, d


def save_T(path, T, d):
    """Write T in the same JSON layout as d (other keys such as scale are kept)."""
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    out = dict(d, rotation=[float(w), float(x), float(y), float(z)], translation=[float(v) for v in T[:3, 3]])
    Path(path).write_text(json.dumps(out, indent=4))


def obj_mesh(path):
    """Vertices and triangles of an OBJ file (polygons fanned into triangles)."""
    v, f = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith('v '):
            v.append([float(a) for a in line.split()[1:4]])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) for t in line.split()[1:]]
            f += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    v, f = np.array(v), np.array(f)
    return v, np.where(f < 0, f + len(v), f - 1)


def surface_points(v, f, n, seed=0):
    """n points uniform over the triangle surface (area-weighted, sqrt(r1) barycentric)."""
    rng = np.random.default_rng(seed)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    tri = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n))[:, None], rng.random(n)[:, None]
    return (1 - r1) * a[tri] + r1 * (1 - r2) * b[tri] + r1 * r2 * c[tri]


def klein(v, f, n=200000, seed=0):
    """The identity and the three 180-degree turns about the mesh's PCA axes through its area-weighted surface centroid
    (4x4, mesh frame). They form a Klein four-group: composing two labels is their XOR."""
    pts = surface_points(v, f, n, seed)
    cen = pts.mean(0)
    _, vec = np.linalg.eigh(np.cov((pts - cen).T))  # ascending: minor, mid, major
    F = [np.eye(4)]
    for ax in vec.T:
        M = np.eye(4)
        M[:3, :3] = 2 * np.outer(ax, ax) - np.eye(3)
        M[:3, 3] = cen - M[:3, :3] @ cen
        F.append(M)
    ext = np.ptp((pts - cen) @ vec, axis=0) * 100
    return np.stack(F), {'axes': ['minor', 'mid', 'major'], 'extent_cm': ext.round(2).tolist()}
