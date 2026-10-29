"""Step 4 of drivers/nvidia_stage23.py: NVIDIA's scripts/reconstruct_support_surfaces.py as imported, with one
change to its phantom filter (support_recon._filter_phantom_supports), run in $RETARGET_VENV:

    $RETARGET_VENV/bin/python path_b/pathb/drivers/support_surfaces.py <table_z> <script> <script args ...>

NVIDIA's filter drops body N's still disk when another body's origin sits inside the disk's circle, within 10 cm
of its height, for 90 % of the clip (a tool resting on its target). Our canonical meshes have their origin at the
footprint's centre on the table, so a dish rack with a cup or a sweeper standing in it loses its disk, and on
ep31 both objects lose theirs (no support surface: FlashCHORD refuses the scene; the still scene's rack has
nothing under it). A disk NVIDIA drops comes back when it is at the table's height (within TABLE_TOL_M of
<table_z>, the scene's ground_z_offset) and less than COVERED of it lies on the disks NVIDIA kept (within
support_recon's consolidation height): at the table it can only be the table, and a covered one adds nothing.
Disks above the table stay filtered. Every restored disk is printed; NVIDIA's consolidation then merges them.
"""
from __future__ import annotations

import runpy
import sys

import numpy as np

TABLE_TOL_M = 0.02  # default: a rest's lowest point is within 2 cm of the table in S3d's settled bundles
COVERED = 0.5  # default: a disk half on kept disks is already held up there
GRID = 41  # sample points across a disk's diameter for its covered fraction


def covered(disk, kept: list, z_tol: float) -> float:
    """Fraction of the disk's area that lies inside a kept disk at a height within z_tol."""
    cx, cy, z, r = disk
    u = np.linspace(-r, r, GRID)
    x, y = np.meshgrid(cx + u, cy + u)
    inside = np.hypot(x - cx, y - cy) <= r
    on = np.zeros_like(inside)
    for kx, ky, kz, kr in kept:
        if abs(kz - z) <= z_tol:
            on |= np.hypot(x - kx, y - ky) <= kr
    return float((on & inside).sum() / max(inside.sum(), 1))


def restore(filtered: dict, all_disks: dict, table_z: float, z_tol: float) -> dict:
    """NVIDIA's kept disks plus each dropped disk at the table's height that the kept ones do not cover."""
    kept = [d for disks in filtered.values() for d in disks]
    out = {b: list(d) for b, d in filtered.items()}
    for body, disks in all_disks.items():
        for d in disks:
            if d in filtered.get(body, []) or abs(d[2] - table_z) > TABLE_TOL_M:
                continue
            c = covered(d, kept, z_tol)
            if c < COVERED:
                out.setdefault(body, []).append(d)
                print(f"    restored table support on '{body}' (center=({d[0]:.3f},{d[1]:.3f},{d[2]:.3f}), "
                      f"r={d[3]:.3f}): {d[2] - table_z:+.3f} m from the table, {c:.0%} on kept disks")
    return out


def main() -> None:
    from robotic_grounding.retarget import support_recon as SR
    table_z, script = float(sys.argv[1]), sys.argv[2]
    nvidia = SR._filter_phantom_supports

    def table_kept(all_disks, names, positions):
        return restore(nvidia(all_disks, names, positions), all_disks, table_z, SR.SURFACE_CONSOLIDATE_Z_TOL_M)

    SR._filter_phantom_supports = table_kept  # reconstruct_support_for_sequence looks it up in the module
    sys.argv = [script] + sys.argv[3:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
