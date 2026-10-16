#!/bin/bash
# Write back the table disk under ep1's pot that support_recon's phantom filter dropped (the lid rests
# on the pot). Values are support_recon's own: center (-0.002, -0.019), top z 1.000, radius 0.111.
set -euo pipefail
OUT=/data/user_data/$USER/v2d/recon_data/outputs/load_retarget
U=$OUT/episode_000001/merged_still/assets/human_motion_data/ego_recon/reconstructed_stage/ep1_merged_still_sharpa_wave_support.usda
[ -f "$U.empty_orig" ] || cp "$U" "$U.empty_orig"
cat > "$U" <<'USD'
#usda 1.0
(
    defaultPrim = "support_surfaces"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "support_surfaces"
{
    def Cylinder "support_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        uniform token axis = "Z"
        double height = 0.01
        color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]
        double radius = 0.111
        double3 xformOp:translate = (-0.002, -0.019, 0.995)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
USD
echo "wrote $U"
