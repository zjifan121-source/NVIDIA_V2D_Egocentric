"""NVIDIA's reconstruct_support_surfaces.py with the phantom-support filter pointed the other way.

support_recon drops body N's table disk when another body sits inside N's disk circle near its
height, meant for a tool resting on its target. In our scenes it drops the wrong one: the table
under ep1's pot (the lid rests on it) and under the dish rack of ep15, ep16 and ep30 (the other
object lies inside the rack's footprint), so that object falls in simulation. Here a disk is
dropped only when it lies more than 1 cm above another body's disk and inside its circle: the
upper object rests on the lower one, which keeps its table. Everything else is NVIDIA's code.

Same arguments as scripts/reconstruct_support_surfaces.py; run it from baseline/robotic_grounding.
"""
from __future__ import annotations

import runpy
import sys

import numpy as np
from robotic_grounding.retarget import support_recon

SCRIPT = "scripts/reconstruct_support_surfaces.py"


def drop_stacked_supports(all_disks, object_body_names, object_body_position):
    flat = [(body, disk) for body, disks in all_disks.items() for disk in disks]
    kept_all = {}
    for body, disks in all_disks.items():
        kept = []
        for cx, cy, z, r in disks:
            under = [other for other, (ox, oy, oz, orad) in flat
                     if other != body and z > oz + 0.01 and np.hypot(cx - ox, cy - oy) < orad]
            if under:
                print(f"    dropped support of '{body}' at z={z:.3f}: it rests on '{under[0]}'")
            else:
                kept.append((cx, cy, z, r))
        if kept:
            kept_all[body] = kept
    return kept_all


if __name__ == "__main__":
    support_recon._filter_phantom_supports = drop_stacked_supports
    sys.argv = [SCRIPT, *sys.argv[1:]]
    runpy.run_path(SCRIPT, run_name="__main__")
