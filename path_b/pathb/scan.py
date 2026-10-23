"""Observed geometry for phase A2 (V3_BUILD_SPEC.md §11): an object's surface fused from stereo depth,
checked against held-out depth, and rigid registration (scale fixed: every input is metric).

Open3D: ScalableTSDFVolume (voxel 4 mm, sdf_trunc 16 mm, depth only) per set of frames, each frame's
depth limited to the object's mask; marching cubes to a mesh. Point-to-plane ICP; FPFH + RANSAC for a
start when two clouds share no frame (a scan against a fused patch).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

VOXEL = 0.004  # m (spec A2.1 default)
SDF_TRUNC = 0.016  # m
DEPTH_TRUNC = 3.0  # m: the head camera sees the table within ~1 m


@dataclass(eq=False)  # compared by identity (it holds arrays)
class Frame:
    index: int
    depth: np.ndarray  # (H, W) metres, 0 = none; only the object's pixels
    K: np.ndarray  # 3x3 at the depth's resolution
    camera_to_world: np.ndarray  # 4x4


def volume() -> o3d.pipelines.integration.ScalableTSDFVolume:
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL, sdf_trunc=SDF_TRUNC, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)


def intrinsic(K: np.ndarray, shape: tuple[int, int]) -> o3d.camera.PinholeCameraIntrinsic:
    h, w = shape
    return o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])


def integrate(frames: list[Frame]) -> o3d.geometry.TriangleMesh:
    """One TSDF over the frames, in the world; its marching-cubes mesh (empty if nothing fused)."""
    vol = volume()
    for f in frames:
        depth = o3d.geometry.Image(np.ascontiguousarray(f.depth, dtype=np.float32))
        color = o3d.geometry.Image(np.zeros(f.depth.shape + (3,), np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0,
                                                                  depth_trunc=DEPTH_TRUNC, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intrinsic(f.K, f.depth.shape), np.linalg.inv(f.camera_to_world))
    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def largest_part(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """The mesh's largest connected piece (stray depth fused off the object goes)."""
    if len(mesh.triangles) == 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    keep = int(np.argmax(np.asarray(counts)))
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(labels != keep)
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


def rendered_depth(mesh: o3d.geometry.TriangleMesh, K: np.ndarray, camera_to_world: np.ndarray,
                   shape: tuple[int, int]) -> np.ndarray:
    """(H, W) z-depth of the mesh seen by that camera; 0 where it is not seen."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    rays_cam = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], axis=-1)
    R, t = camera_to_world[:3, :3], camera_to_world[:3, 3]
    dirs = rays_cam.reshape(-1, 3) @ R.T  # z = 1 in the camera: t_hit is the z-depth
    rays = np.concatenate([np.broadcast_to(t, dirs.shape), dirs], axis=1).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(h, w)
    return np.where(np.isfinite(hit), hit, 0.0)


def depth_rms(mesh: o3d.geometry.TriangleMesh, frames: list[Frame]) -> dict:
    """The mesh against held-out depth: RMS (m) of rendered minus stereo depth where both exist."""
    diffs = []
    for f in frames:
        r = rendered_depth(mesh, f.K, f.camera_to_world, f.depth.shape)
        both = (r > 0) & (f.depth > 0)
        diffs.append((r - f.depth)[both])
    d = np.concatenate(diffs) if diffs else np.array([])
    return {"rms_m": float(np.sqrt(np.mean(d ** 2))) if len(d) else None, "pixels": int(len(d)),
            "median_abs_m": float(np.median(np.abs(d))) if len(d) else None}


def cloud(points: np.ndarray, normals: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)))
    if normals is not None:
        pc.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    else:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * VOXEL, max_nn=30))
    return pc


def mesh_points(mesh: o3d.geometry.TriangleMesh, n: int = 20000, seed: int = 0) -> o3d.geometry.PointCloud:
    o3d.utility.random.seed(seed)
    return mesh.sample_points_uniformly(number_of_points=n, use_triangle_normal=True)


def icp(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, init: np.ndarray = np.eye(4),
        max_dist: float = 0.01, iterations: int = 60) -> dict:
    """Point-to-plane ICP, scale fixed: T maps source onto target."""
    r = o3d.pipelines.registration.registration_icp(
        source, target, max_dist, init, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations))
    return {"T": np.asarray(r.transformation), "fitness": float(r.fitness), "rmse_m": float(r.inlier_rmse)}


def global_start(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, voxel: float = 0.005,
                 seed: int = 0) -> np.ndarray:
    """FPFH features + RANSAC: a start for ICP between clouds in unrelated frames."""
    o3d.utility.random.seed(seed)

    def prep(pc):
        down = pc.voxel_down_sample(voxel)
        down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
        feat = o3d.pipelines.registration.compute_fpfh_feature(
            down, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))
        return down, feat

    (s, fs), (t, ft) = prep(source), prep(target)
    r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        s, t, fs, ft, True, 1.5 * voxel, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(1.5 * voxel)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
    return np.asarray(r.transformation)


def one_sided_cm(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> float:
    """Mean distance (cm) from points to the surface (vertices, faces)."""
    import trimesh
    surface = trimesh.Trimesh(vertices, faces, process=False)
    return float(np.mean(surface.nearest.on_surface(np.asarray(points))[1]) * 100)
