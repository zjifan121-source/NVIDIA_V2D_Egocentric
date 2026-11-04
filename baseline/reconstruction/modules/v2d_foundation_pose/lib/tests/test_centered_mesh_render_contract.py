"""CPU call-site contract tests: public poses must match the rendered geometry.

Extract the real method bodies to avoid importing CUDA/model dependencies. The
renderer stand-in checks camera-space vertices, not an implementation-specific
matrix formula. Rasterization and network behavior require separate GPU tests.
"""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'foundation_pose_tracker.py'


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __matmul__(self, other):
        return Tensor(self.value @ other.value)

    def __getitem__(self, key):
        return Tensor(self.value[key])

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def method(name, render):
    cls = next(n for n in ast.parse(SOURCE.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'FoundationPoseTracker')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    torch = SimpleNamespace(as_tensor=lambda x, **kw: Tensor(x), float=np.float32,
                            no_grad=contextlib.nullcontext,
                            linalg=SimpleNamespace(inv=lambda x: Tensor(np.linalg.inv(x.value))))
    ns = {'np': np, 'torch': torch, 'nvdiffrast_render': render}
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), 'exec'), ns)
    return ns[name]


def check_render_matches_original_mesh_in_camera(name, scale, offset):
    vertices = scale * (np.array([[-.04, -.03, -.02], [.04, .03, .02], [.01, -.02, .015]]) + offset)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    A = np.eye(4); A[:3, 3] = -center
    # Noncommuting rotation and translation catch inverse/sign/side errors.
    T = np.eye(4); T[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]; T[:3, 3] = [.13, -.06, 1.1]
    expected = vertices @ T[:3, :3].T + T[:3, 3]
    calls = []

    def render(K, H, W, pose, **kw):
        v = kw['mesh_tensors']['pos']; M = pose.value[0]
        actual = v @ M[:3, :3].T + M[:3, 3]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        calls.append(M.copy())
        return None, Tensor(np.ones((1, H, W))), None

    est = SimpleNamespace(mesh_tensors={'pos': vertices-center}, get_tf_to_centered_mesh=lambda: Tensor(A))
    tracker = SimpleNamespace(_est=est, _glctx=None)
    camera = SimpleNamespace(width=4, height=4, to_matrix=lambda: np.eye(3))
    mask = SimpleNamespace(mask=np.ones((4, 4), bool)); depth = SimpleNamespace(depth=np.ones((4, 4)))
    pose = SimpleNamespace(to_matrix=lambda: T)
    T_before = T.copy(); vertices_before = est.mesh_tensors['pos'].copy()
    fn = method(name, render)
    result = fn(tracker, mask, camera, pose) if name == '_mask_iou' else fn(tracker, depth, mask, camera, pose)
    assert len(calls) == 1 and result > .99
    np.testing.assert_array_equal(T, T_before)
    np.testing.assert_array_equal(est.mesh_tensors['pos'], vertices_before)


class CenteredMeshRenderContractTest(unittest.TestCase):
    def test_public_pose_matches_original_geometry(self):
        for name in ['_mask_iou', '_score_scale']:
            for scale in [0.5, 1.0, 2.0]:
                for offset in [[0, 0, 0], [.12, -.08, .03]]:
                    with self.subTest(method=name, scale=scale, offset=offset):
                        check_render_matches_original_mesh_in_camera(name, scale, offset)


if __name__ == '__main__':
    unittest.main()
