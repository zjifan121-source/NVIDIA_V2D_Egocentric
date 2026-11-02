"""pathb/trackability.py: a carry with the hand on the object is supported and one 5 cm off is not; a thin
sheet keeps the 1.5 cm touch distance; frame-0 support by a disk or another object; a rest 1 cm above its
disk settles about 1 cm (needs the sim venv)."""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from pathb import trackability as T

N = 30
SIM = Path.home() / "V2D/venvs/sim/bin/python"  # a venv with mujoco and coacd


def carry(offset_m: float) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """A 6 cm box (delta 3 cm) moving 2 cm per frame along x; the right hand's thumb and index on its +y / -y
    faces pushed `offset_m` out; the left hand far away."""
    box = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
    poses = np.tile(np.eye(4), (N, 1, 1))
    poses[:, 0, 3] = 0.02 * np.arange(N)
    poses[:, 2, 3] = 1.0
    tips = np.full((N, 2, 5, 3), 5.0)
    centre = poses[:, :3, 3]
    tips[:, 0, 0] = centre + [0, 0.03 + offset_m, 0]
    tips[:, 0, 1] = centre + [0, -0.03 - offset_m, 0]
    return box, poses, tips


def test_carry_with_the_hand_on_the_object_is_held():
    box, poses, tips = carry(0.0)
    moving = T.moving_frames(box, poses, T.labels(None, N))
    hs = T.hand_support(box, poses, tips, T.delta_m(box))
    assert moving[1:].all() and not moving[0]
    assert hs["held"].all() and hs["held_opp"].all()
    assert float((moving & ~hs["held"]).sum()) == 0


def test_carry_5cm_off_the_object_is_unsupported():
    box, poses, tips = carry(0.05)
    moving = T.moving_frames(box, poses, T.labels(None, N))
    hs = T.hand_support(box, poses, tips, T.delta_m(box))
    free = moving & ~hs["held"]
    assert not hs["held"].any()
    assert free.sum() / moving.sum() == 1.0 and T.longest_run(free) == N - 1
    assert np.allclose(hs["nearest_m"], 0.05, atol=1e-4)


def test_grasp_and_bridge_labels_make_frames_moving():
    box, poses, _ = carry(0.0)
    poses[:, 0, 3] = 0.0  # standing still
    report = {"segments": [{"kind": "static", "start": 0, "end": 9}, {"kind": "grasp", "start": 10, "end": 14, "failed": True},
                           {"kind": "unstable", "start": 15, "end": 19, "bridge": [15, 22]}]}
    lab = T.labels(report, N)
    moving = T.moving_frames(box, poses, lab)
    assert np.nonzero(moving)[0].tolist() == list(range(10, 23))
    assert lab["failed"][10:15].all() and lab["static"] == [(0, 9)]


def test_thin_sheet_keeps_the_minimum_touch_distance():
    sheet = trimesh.creation.box(extents=(0.2, 0.2, 0.004))
    assert T.delta_m(sheet) == pytest.approx(T.DELTA_MIN_M)
    poses = np.tile(np.eye(4), (2, 1, 1))
    tips = np.full((2, 2, 5, 3), 5.0)
    tips[0, 0, :2] = [[0, 0, 0.002 + 0.010], [0.05, 0, 0.002 + 0.010]]  # 1 cm over the sheet: touching
    tips[1, 0, :2] = [[0, 0, 0.002 + 0.020], [0.05, 0, 0.002 + 0.020]]  # 2 cm over: not
    hs = T.hand_support(sheet, poses, tips, T.delta_m(sheet))
    assert hs["held"].tolist() == [True, False]
    assert not hs["held_opp"].any()  # both tips over the same face


def rest(z_gap: float) -> tuple[dict, trimesh.Trimesh, dict]:
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    pose = np.eye(4)
    pose[2, 3] = 1.0 + 0.05 + z_gap
    disk = {"name": "d", "radius": 0.1, "height": 0.01, "centre": [0.0, 0.0, 0.995]}
    return {"name": "box", "poses": pose[None]}, box, disk


def test_support_by_a_disk_or_an_object():
    o, box, disk = rest(0.0)
    assert T.support_ok([o], [box], [disk])["ok"]
    high, _, _ = rest(0.05)
    assert not T.support_ok([high], [box], [disk])["ok"]
    cup = {"name": "cup", "poses": np.eye(4)[None].copy()}
    cup["poses"][0, 2, 3] = 1.0 + 0.1 + 0.025  # on top of the box
    small = trimesh.creation.box(extents=(0.04, 0.04, 0.05))
    s = T.support_ok([o, cup], [box, small], [disk])
    assert s["ok"] and s["objects"]["cup"]["object"] and not s["objects"]["cup"]["disk"]


def test_body_names_map_to_the_roster():
    assert T.object_name("ep40_inverted_mini_dust_pan", ["mini_dust_pan"]) == "mini_dust_pan"  # one object: the sequence id
    names = ["white_pot", "white_pot_lid"]
    assert T.object_name("ep0_still_white_pot_lid", names) == "white_pot_lid"
    assert T.object_name("ep0_still_white_pot", names) == "white_pot"


def test_support_disks_read_from_usd(tmp_path):
    usd = tmp_path / "s.usda"
    usd.write_text('def Xform "support_surfaces"\n{\n    def Cylinder "ep_0" (\n    )\n    {\n        double height = 0.01\n'
                   '        double radius = 0.2\n        double3 xformOp:translate = (0.1, -0.2, 0.995)\n    }\n}\n')
    assert T.support_disks(usd) == [{"name": "ep_0", "radius": 0.2, "height": 0.01, "centre": [0.1, -0.2, 0.995]}]


@pytest.mark.skipif(not SIM.exists(), reason="needs a venv with mujoco and coacd (~/V2D/venvs/sim)")
def test_rest_1cm_above_its_disk_settles_about_1cm(tmp_path, monkeypatch):
    monkeypatch.setenv("SETTLE_PY", str(SIM))
    monkeypatch.setenv("PATHB_CACHE", str(tmp_path / "cache"))
    out = {}
    for gap in (0.0, 0.01):
        o, box, disk = rest(gap)
        mesh = tmp_path / f"box_{gap}.obj"
        box.export(mesh)
        o.update(mesh=mesh, body="box")
        usd = tmp_path / "s.usda"
        usd.write_text('def Cylinder "d" (\n)\n{\n    double height = 0.01\n    double radius = 0.1\n'
                       '    double3 xformOp:translate = (0, 0, 0.995)\n}\n')
        processed = tmp_path / "processed"
        processed.mkdir(exist_ok=True)
        (processed / "box_rigid.urdf").write_text('<robot><link name="object"><inertial><mass value="0.3"/></inertial></link></robot>')
        work = tmp_path / f"work_{gap}"
        work.mkdir()
        out[gap] = T.rest_settle({"support_usd": usd, "processed": processed}, [o], {"box": [0]}, work)
    assert out[0.0]["rss0_cm"] < 0.2 and out[0.0]["unsettled_rests"] == 0
    assert 0.8 < out[0.01]["rss0_cm"] < 1.4 and out[0.01]["rss0_deg"] < 1.0
