"""GSmini cutting mechanics adapted from EmbodiedSWE's Isaac Sim slice scene.

Apache-2.0 upstream: EmbodiedSWE/EmbodiedSWE, commit
046f7bf48865f3d3b48e64614350777fb5d5d163, slice_food.py.
Rigid pieces, contact and separation are physical; cohesion release is a rule.
"""

from __future__ import annotations

import json
import math

from benchmark.catalog import ASSETS


def seam_clearance_m(left_vertices, right_vertices, normal):
    """Conservative signed gap between the two collision hulls along the cut normal."""
    import numpy as np

    left, right, axis = map(np.asarray, (left_vertices, right_vertices, normal))
    length = np.linalg.norm(axis)
    if (not np.isfinite(left).all() or not np.isfinite(right).all()
            or not np.isfinite(length) or length <= 0):
        raise RuntimeError("Non-finite or degenerate cutting collision geometry")
    axis = axis/length
    return float((right @ axis).min() - (left @ axis).max())


def point_velocity(com_velocity, angular_velocity, point, com_position):
    """Rigid-point velocity using a position and velocity at the same COM frame."""
    import numpy as np

    return np.asarray(com_velocity) + np.cross(angular_velocity, np.asarray(point)-com_position)


def cut_gate(*, plane_distance_m, flesh_distance_m, normal_alignment, edge_height_m,
             release_height_m, minimum_height_m, food_speed_m_s, blade_vz_m_s,
             contact_force_n, grasped, config):
    """Only the real edge, aligned and pressed into the flesh, may release a seam."""
    values = (plane_distance_m, flesh_distance_m, normal_alignment, edge_height_m,
              release_height_m, minimum_height_m, food_speed_m_s, blade_vz_m_s, contact_force_n)
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("Non-finite cutting gate telemetry")
    return (grasped and abs(plane_distance_m) < config["plane_tolerance_m"]
            and abs(flesh_distance_m) < config["flesh_radius_m"]
            and abs(normal_alignment) > math.cos(math.radians(config["blade_align_deg"]))
            and minimum_height_m <= edge_height_m < release_height_m
            and food_speed_m_s < config["food_settle_speed_m_s"]
            and blade_vz_m_s < 0.0
            and contact_force_n >= config["knife_food_contact_threshold_n"])


class CuttingRuntime:
    """Native fixed welds and measured geometry; never controls task-object poses."""

    def __init__(self, env):
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from benchmark.catalog import object_asset

        self.env = env
        self.config = env.task["episode_config"]
        self.manifest = json.loads((ASSETS / "objects" / f"extra_{self.config['food']}" / "manifest.json").read_text())
        self.names = ["mechanism" if i == 4 else f"piece_{i}" for i in range(8)]
        self.pieces = [env.scene[name] for name in self.names]
        self.centroids = np.array([p["centroid"] for p in self.manifest["pieces"]])
        self.knife = env.scene["knife"]
        stage = env.sim.stage
        self.joints = []
        for i, plane in enumerate(self.manifest["planes_x"]):
            joint = UsdPhysics.FixedJoint.Define(stage, f"/World/envs/env_0/CutWelds/plane_{i}")
            joint.CreateBody0Rel().SetTargets([f"/World/envs/env_0/{self.names[i]}/Piece"])
            joint.CreateBody1Rel().SetTargets([f"/World/envs/env_0/{self.names[i+1]}/Piece"])
            middle = [plane, 0, 0]
            joint.CreateLocalPos0Attr(Gf.Vec3f(*(np.array(middle)-self.centroids[i])))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*(np.array(middle)-self.centroids[i+1])))
            joint.CreateJointEnabledAttr(True)
            self.joints.append(joint)
        source = Usd.Stage.Open(str(object_asset("extra_slice_knife22r_0")))
        points = np.array(UsdGeom.Mesh(source.GetPrimAtPath("/Asset/Piece/Collision")).GetPointsAttr().Get())
        # Retain the upstream edge samples, rather than using the mesh AABB's tip.
        self.edge_local = np.array([(x, float(points[abs(points[:, 0]-x) < .008, 1].min()), 0)
                                    for x in np.linspace(0, .20, 21) if (abs(points[:, 0]-x) < .008).any()])
        self.bounds = []
        for name in self.names:
            source = Usd.Stage.Open(str(object_asset(env.task["objects"][name]["type"])))
            points = np.array(UsdGeom.Mesh(source.GetPrimAtPath("/Asset/Piece/Collision")).GetPointsAttr().Get())
            self.bounds.append(points)
        self.reset()

    @staticmethod
    def rotation(q):
        from scipy.spatial.transform import Rotation
        w, x, y, z = q
        return Rotation.from_quat([x, y, z, w])

    def reset(self):
        self.cut = [False] * len(self.joints)
        for joint in self.joints:
            joint.GetJointEnabledAttr().Set(True)
        self.tool_lifted = False
        self.last_gate = {}
        self.reference_distance = float(((self.centroids[4]-self.centroids[3])**2).sum()**.5)
        self.knife_reset_z = self.env.task["objects"]["knife"]["initial_position"][2]

    def initialize_grasp(self):
        """Only reset may place the tool; the episode uses frictional grasp alone."""
        import torch
        from scipy.spatial.transform import Rotation

        env = self.env
        robot = env.scene["robot"]
        hi = robot.body_names.index("panda_hand")
        hand_r = self.rotation(robot.data.body_quat_w[0, hi].cpu().tolist())
        chop = self.rotation(env.task["objects"]["knife"]["initial_quat"])
        nominal_hand = Rotation.from_euler("z", math.pi/2) * Rotation.from_euler("x", math.pi)
        knife_r = hand_r * nominal_hand.inv() * chop
        center = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
        self.reset_grip_expected = center.tolist()
        self.reset_normal = knife_r.apply([0, 0, 1]).tolist()
        contract = env.task["reset_contract"]
        position = center - knife_r.apply(contract["tool_grip_local_m"])
        x, y, z, w = knife_r.as_quat()
        root = torch.tensor([[*position, w, x, y, z]], dtype=torch.float32, device=env.device)
        self.knife.write_root_pose_to_sim(root)
        self.knife.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
        joints = robot.data.joint_pos.clone()
        for name in ("panda_finger_joint1", "panda_finger_joint2"):
            joints[:, robot.joint_names.index(name)] = contract["tool_grasp_initial_joint_pos"]
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.action[:, -1] = contract["tool_gripper_command"]
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(0.0)
        self.knife_reset_z = float(position[2])

    def grasp_forces(self):
        import torch
        knife_index = list(self.env.task["objects"]).index("knife")
        return [float(torch.linalg.vector_norm(
            self.env.scene[f"gsmini_{side}_contact"].data.force_matrix_w[..., knife_index, :],
            dim=-1).sum()) for side in ("left", "right")]

    def grasp_load_path(self):
        """Measure only the knife's normal load on gels versus hard housings."""
        import torch

        gel = self.grasp_forces()
        index = list(self.env.task["objects"]).index("knife")
        case = [float(torch.linalg.vector_norm(
            self.env.scene[f"gsmini_{side}_case_contact"].data.force_matrix_w[..., index, :],
            dim=-1).sum()) for side in ("left", "right")]
        if not all(math.isfinite(v) and v >= 0 for v in (*gel, *case)):
            raise RuntimeError("Non-finite cutting grasp load path")
        fraction = sum(case)/max(sum(gel)+sum(case), 1e-12)
        return {"gel_normal_force_n": gel, "housing_normal_force_n": case,
                "housing_force_fraction": fraction,
                "valid": min(gel) >= self.env.task["success_config"]["contact_threshold_n"]
                and fraction <= self.env.task["success_config"]["max_tool_housing_force_fraction"]}

    def food_contact_force(self):
        """Peak native knife-food normal reaction in the last control interval."""
        import torch

        history = self.env.scene["knife_food_contact"].data.force_matrix_w_history
        return float(torch.linalg.vector_norm(history, dim=-1).sum(dim=(-1, -2)).max())

    def validate_grasp(self):
        import numpy as np
        load = self.grasp_load_path()
        forces = load["gel_normal_force_n"]
        if not load["valid"]:
            raise RuntimeError(f"Cutting reset grasp bypasses gel through housing or lost pad contact: {load}")
        p, r = self.pose(self.knife)
        grip = p + r.apply(self.env.task["reset_contract"]["tool_grip_local_m"])
        slip = np.linalg.norm(grip-np.array(self.reset_grip_expected))
        if (not np.isfinite(forces).all() or not np.isfinite(slip)
                or min(forces) < self.env.task["success_config"]["contact_threshold_n"]
                or slip > self.env.task["reset_contract"]["max_tool_reset_slip_m"]):
            from benchmark.backends.gsmini.sensors import _pad_centers_world
            pads = _pad_centers_world(self.env, ("gelpad_left", "gelpad_right"),
                                      self.env.cfg.gsmini_pad_centers_local)
            raise RuntimeError(f"Cutting reset could not establish a two-pad grasp: {forces} N; "
                               f"pads={[v.cpu().tolist() for v in pads]}; "
                               f"grip={grip.tolist()}; slip={slip}; "
                               f"initial_grip={self.reset_grip_expected}; normal={self.reset_normal}")

    def validate_reset(self):
        self.validate_grasp()
        state = self.metrics()
        if (any(self.cut) or not all(j.GetJointEnabledAttr().Get() for j in self.joints)
                or not state["all_pieces_on_board"]
                or state["max_piece_speed_m_s"] > self.env.task["goals"][0]["max_piece_speed_m_s"]):
            raise RuntimeError(f"Cutting reset did not produce an intact settled food assembly: {state}")

    def pose(self, asset):
        return (asset.data.root_pos_w[0].cpu().numpy(),
                self.rotation(asset.data.root_quat_w[0].cpu().tolist()))

    def edge_world(self):
        p, r = self.pose(self.knife)
        return p + r.apply(self.edge_local)

    def aim(self):
        import numpy as np
        i = self.config["cut_plane_index"]
        # Each seam uses its still-connected neighbour as its own live frame.
        p, r = self.pose(self.pieces[i])
        local = (self.centroids[i] + self.centroids[i+1]) / 2
        local[0] = self.manifest["planes_x"][i]
        return p + r.apply(local-self.centroids[i]), r, local

    def update(self):
        import numpy as np
        env = self.env
        grasped = self.grasp_load_path()["valid"]
        kp, kr = self.pose(self.knife)
        self.tool_lifted = bool(self.tool_lifted or (grasped and float(kp[2]) > self.knife_reset_z + .03))
        i = self.config["cut_plane_index"]
        if self.cut[i] or env._resetting:
            return
        aim, r, local = self.aim()
        edge = self.edge_world()
        nearest = edge[np.linalg.norm(edge[:, :2]-aim[:2], axis=1).argmin()]
        delta = r.inv().apply(nearest-aim)
        normal = r.inv().apply(kr.apply([0, 0, 1]))
        speed = max(float(p.data.root_lin_vel_w[0].norm()) for p in self.pieces)
        # Include angular motion of the measured edge, not just COM velocity.
        velocity = self.knife.data.root_lin_vel_w[0].cpu().numpy()
        omega = self.knife.data.root_ang_vel_w[0].cpu().numpy()
        com = self.knife.data.root_com_pos_w[0].cpu().numpy()
        edge_vz = float(point_velocity(velocity, omega, nearest, com)[2])
        # History covers the entire control interval, including transient impacts.
        contact_force = self.food_contact_force()
        board_p, board_r = self.pose(env.scene["board"])
        board_top = float((board_p + board_r.apply([0, 0, .012]))[2])
        self.last_gate = dict(plane_distance_m=float(delta[0]), flesh_distance_m=float(delta[1]),
                             normal_alignment=float(normal[0]), edge_height_m=float(nearest[2]),
                             release_height_m=max(float(aim[2])-self.config["depth_past_center_m"],
                                                  board_top+.0015),
                             minimum_height_m=board_top-.0002,
                             food_speed_m_s=speed, blade_vz_m_s=edge_vz,
                             contact_force_n=contact_force,
                             grasped=bool(grasped and self.tool_lifted))
        if cut_gate(**self.last_gate, config=self.config):
            self.joints[i].GetJointEnabledAttr().Set(False)
            self.cut[i] = True

    def metrics(self):
        import numpy as np
        load = self.grasp_load_path()
        board_p, board_r = self.pose(self.env.scene["board"])
        board_inverse = board_r.inv()
        on_board = True
        board_clearance = math.inf
        poses, world_vertices = [], []
        for asset, points in zip(self.pieces, self.bounds):
            p, r = self.pose(asset)
            poses.append((p, r))
            world_vertices.append(p + r.apply(points))
            vertices = board_inverse.apply(world_vertices[-1] - board_p)
            board_clearance = min(board_clearance, float(vertices[:, 2].min()-.012))
            on_board &= (abs(vertices[:, 0]).max() < .23 and abs(vertices[:, 1]).max() < .17
                         and vertices[:, 2].min() >= .012-self.config["max_food_board_penetration_m"])
        a, ar = poses[3]
        b, _ = poses[4]
        separation = seam_clearance_m(world_vertices[3], world_vertices[4],
                                     ar.apply([1, 0, 0]))
        food_top = max(vertices[:, 2].max() for vertices in world_vertices)
        return {"target_plane_released": self.cut[self.config["cut_plane_index"]],
                "cut_count": sum(self.cut), "gap_increase_m": float(np.linalg.norm(a-b)-self.reference_distance),
                "separation_m": separation,
                "minimum_board_clearance_m": board_clearance,
                "all_pieces_on_board": bool(on_board), "tool_lifted": bool(self.tool_lifted),
                "tool_grasped": load["valid"],
                "tool_housing_force_fraction": load["housing_force_fraction"],
                "knife_food_contact_force_n": self.food_contact_force(),
                "knife_clearance_m": float(self.edge_world()[:, 2].min()-food_top),
                "max_piece_speed_m_s": max(float(p.data.root_lin_vel_w[0].norm()) for p in self.pieces),
                "cut_mask": list(self.cut), "cut_gate": dict(self.last_gate),
                "knife_position_w_m": self.pose(self.knife)[0].tolist()}
