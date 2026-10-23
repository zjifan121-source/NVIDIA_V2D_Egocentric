"""Symmetry groups of the roster's objects (V3_BUILD_SPEC.md S3b recovery rule, S3c fusion).

A pose T (object -> world) and T @ S for S in the object's group look the same, so a tracker may
report either. Groups follow the spec: the cups -> 36 rotations about their axis (10 degree steps);
white_pot_lid -> {identity, half turn about its axis}; every other object -> {identity}.

The axis is the mesh's own: the principal axis of the vertex cloud whose variance stands apart from
the other two (a body of revolution has two equal ones), through the vertex centroid. Transforms are
4x4 in the object frame, identity first.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

CUPS = {"blue_cup", "beige_cup"}
HALF_TURN = {"white_pot_lid"}
CUP_STEPS = 36


def axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(unit axis, centroid) of a body of revolution's vertices. The sign is fixed so the axis's
    largest component is positive, so the same mesh always gives the same axis."""
    v = np.asarray(vertices, dtype=np.float64)
    c = v.mean(axis=0)
    w, e = np.linalg.eigh(np.cov((v - c).T))
    # the eigenvalue farthest from the other two
    gaps = [abs(w[i] - np.mean(np.delete(w, i))) for i in range(3)]
    a = e[:, int(np.argmax(gaps))]
    return (a if a[np.argmax(np.abs(a))] > 0 else -a), c


def about_axis(a: np.ndarray, c: np.ndarray, angle: float) -> np.ndarray:
    """4x4 rotation by `angle` (rad) about the line through c along a."""
    S = np.eye(4)
    S[:3, :3] = Rotation.from_rotvec(a / np.linalg.norm(a) * angle).as_matrix()
    S[:3, 3] = c - S[:3, :3] @ c
    return S


def group(name: str, vertices: np.ndarray | None = None) -> np.ndarray:
    """(K, 4, 4) symmetry transforms of a roster object, identity first."""
    if name in CUPS or name in HALF_TURN:
        if vertices is None:
            raise ValueError(f"{name} is symmetric about its axis: pass its mesh vertices")
        a, c = axis(vertices)
        steps = CUP_STEPS if name in CUPS else 2
        return np.stack([about_axis(a, c, 2 * np.pi * k / steps) for k in range(steps)])
    return np.eye(4)[None]


def angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Rotation distance in radians."""
    return float(Rotation.from_matrix(Ra.T @ Rb).magnitude())


def closest(reference: np.ndarray, measured: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, int]:
    """The symmetric equivalent measured @ G[k] whose rotation is closest to reference's; (pose, k)."""
    candidates = measured @ G
    k = int(np.argmin([angle(reference[:3, :3], T[:3, :3]) for T in candidates]))
    return candidates[k], k
