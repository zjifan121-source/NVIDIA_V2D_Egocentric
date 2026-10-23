"""Forward kinematics of the Sharpa Wave hands in numpy (V3_BUILD_SPEC.md S5): the robot_{side}_frames
of FlashCHORD's ManoSharpa parquet from a wrist pose and the 22 finger joints.

One kinematic tree, two readers: urdf() reads FlashCHORD's sharpawave URDF (the hands FlashCHORD
simulates), mjcf() an MJCF hand. The frame list every Sharpa ManoSharpa parquet carries (67 names:
universe, root_joint, then per body its joint, the body and its sites) is Pinocchio's for the
retargeter's MJCF (robotic_grounding/assets/xmls/sharpawave, BuildFromMJCF with a free-flyer root,
hand_kinematics.py); the URDF has no sites. So S5 lists the frames with the MJCF and
tests/test_fk.py checks that the URDF puts every shared frame in the same place. The wrist is the
root body (right_hand_C_MC / left_hand_C_MC): robot_{side}_wrist_position/_wxyz = frame root_joint.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
FLASH_URDF = REPO / "baseline/robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/sharpawave/{side}_sharpawave.urdf"
RETARGET_MJCF = (REPO / "baseline/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/xmls/sharpawave/"
                 "{side}_sharpawave.xml")


def transform(pos=None, R=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if pos is not None:
        T[:3, 3] = pos
    return T


def floats(text: str | None, default) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.asarray(default, dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v  # fixed joints may carry a zero axis


@dataclass
class Node:
    """A frame: parent node (-1 = the wrist), fixed offset from the parent, then the joint motion."""
    name: str
    parent: int
    offset: np.ndarray
    joint: str | None = None  # movable joint driving this node
    kind: str = "fixed"  # fixed | hinge | slide
    axis: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame: bool = True  # listed in frames()


@dataclass
class Hand:
    nodes: list[Node]

    @property
    def names(self) -> list[str]:
        return ["universe", "root_joint"] + [n.name for n in self.nodes if n.frame]

    @property
    def joints(self) -> list[str]:
        return [n.joint for n in self.nodes if n.joint]

    def poses(self, wrist_pos: np.ndarray, wrist_wxyz: np.ndarray, q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """(T, 4, 4) world transform of every node; q maps joint name -> (T,) angle (rad) or offset (m)."""
        wrist_pos = np.atleast_2d(wrist_pos)
        n = len(wrist_pos)
        wrist = np.tile(np.eye(4), (n, 1, 1))
        wrist[:, :3, :3] = Rotation.from_quat(np.atleast_2d(wrist_wxyz)[:, [1, 2, 3, 0]]).as_matrix()
        wrist[:, :3, 3] = wrist_pos
        out, world = {}, []
        for node in self.nodes:
            T = (wrist if node.parent < 0 else world[node.parent]) @ node.offset
            if node.kind != "fixed":
                v = np.broadcast_to(np.asarray(q[node.joint], dtype=float), (n,))
                M = np.tile(np.eye(4), (n, 1, 1))
                if node.kind == "hinge":
                    M[:, :3, :3] = Rotation.from_rotvec(v[:, None] * node.axis).as_matrix()
                else:
                    M[:, :3, 3] = v[:, None] * node.axis
                T = T @ M
            world.append(T)
            out[node.name] = T
        out["root_joint"] = wrist
        out["universe"] = np.tile(np.eye(4), (n, 1, 1))
        return out

    def frames(self, wrist_pos: np.ndarray, wrist_wxyz: np.ndarray, q: dict[str, np.ndarray]) -> np.ndarray:
        """(T, F, 7) position + wxyz of the frames in self.names (the parquet's robot_{side}_frames)."""
        poses = self.poses(wrist_pos, wrist_wxyz, q)
        T = np.stack([poses[name] for name in self.names], axis=1)
        wxyz = Rotation.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
        return np.concatenate([T[..., :3, 3], wxyz.reshape(T.shape[:2] + (4,))], axis=-1)


def mjcf(path: str | Path) -> Hand:
    """An MJCF hand whose first worldbody body is the wrist (Pinocchio's free-flyer child)."""
    root = ET.parse(path).getroot().find("worldbody").find("body")
    nodes: list[Node] = []

    def body_offset(el) -> np.ndarray:
        wxyz = floats(el.get("quat"), [1, 0, 0, 0])
        return transform(floats(el.get("pos"), [0, 0, 0]), Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix())

    def add(el, parent: int, is_root: bool) -> None:
        joints = el.findall("joint")
        if len(joints) > 1:
            raise ValueError(f"{el.get('name')}: one joint per body expected")
        offset = np.eye(4) if is_root else body_offset(el)
        if joints:
            j = joints[0]
            jpos = floats(j.get("pos"), [0, 0, 0])
            axis = floats(j.get("axis"), [0, 0, 1])
            kind = {"hinge": "hinge", "slide": "slide", None: "hinge"}[j.get("type")]
            nodes.append(Node(j.get("name"), parent, offset @ transform(jpos), j.get("name"), kind, unit(axis)))
            parent, offset = len(nodes) - 1, transform(-jpos)
        nodes.append(Node(el.get("name"), parent, offset))
        me = len(nodes) - 1
        for s in el.findall("site"):  # the root body's sites are not Pinocchio frames
            nodes.append(Node(s.get("name"), me, body_offset(s), frame=not is_root))
        for child in el.findall("body"):
            add(child, me, False)

    add(root, -1, True)
    return Hand(nodes)


def urdf(path: str | Path) -> Hand:
    """A URDF hand: the root link is the wrist; per joint its frame (= the child link's) and the link."""
    robot = ET.parse(path).getroot()
    joints = robot.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    root = next(link.get("name") for link in robot.findall("link") if link.get("name") not in children)
    nodes = [Node(root, -1, np.eye(4))]
    index = {root: 0}
    pending = list(joints)
    while pending:
        j = next(j for j in pending if j.find("parent").get("link") in index)
        pending.remove(j)
        o = j.find("origin")
        R = Rotation.from_euler("xyz", floats(o.get("rpy") if o is not None else None, [0, 0, 0])).as_matrix()
        offset = transform(floats(o.get("xyz") if o is not None else None, [0, 0, 0]), R)
        kind = {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide", "fixed": "fixed"}[j.get("type")]
        axis = floats(j.find("axis").get("xyz"), [1, 0, 0]) if j.find("axis") is not None else np.array([1.0, 0, 0])
        parent = index[j.find("parent").get("link")]
        nodes.append(Node(j.get("name"), parent, offset, j.get("name") if kind != "fixed" else None, kind, unit(axis)))
        nodes.append(Node(j.find("child").get("link"), len(nodes) - 1, np.eye(4)))
        index[j.find("child").get("link")] = len(nodes) - 1
    return Hand(nodes)


def sharpa(side: str, source: str = "mjcf") -> Hand:
    return mjcf(str(RETARGET_MJCF).format(side=side)) if source == "mjcf" else urdf(str(FLASH_URDF).format(side=side))
