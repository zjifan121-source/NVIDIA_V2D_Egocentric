"""Effective densities (kg/m^3) of the roster's objects for S3a's mass.json: the reconstructed mesh is
closed, the real objects are mostly hollow or thin-walled, so the density is the object's mass over
its closed volume. Fixed before any evaluation run (V3_BUILD_SPEC.md S3a); the values are estimates
from the objects' kind (no object was weighed)."""
DENSITY = {
    "white_pot": 250.0,  # metal pot: ~0.6 kg in ~2.5 l
    "white_pot_lid": 600.0,  # thin metal lid
    "blue_cup": 150.0,  # plastic cup, hollow
    "beige_cup": 150.0,
    "water_pitcher": 120.0,  # plastic pitcher, hollow
    "plastic_dish_rack": 80.0,  # wire-like plastic rack
    "mini_dust_pan": 150.0,  # thin plastic
    "mini_sweeper": 250.0,  # plastic handle and bristles
    "wooden_piece_1": 600.0,  # solid wood
    "wooden_piece_2": 600.0,
    "wooden_spoon": 650.0,
}
DEFAULT = 300.0


def density(name: str) -> float:
    return DENSITY.get(name, DEFAULT)
