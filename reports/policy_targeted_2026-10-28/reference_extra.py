"""Prepare the next targeted cases on CPU while the first GPU trials finish."""
import trimesh
import diagnose

diagnose.PLAN=diagnose.HERE/'baseline_plan.json'
trimesh.proximity.closest_point=trimesh.proximity.closest_point_naive
for ep in (20,10,14,1,3):
    diagnose.reference(ep)
