"""Shared Franka lift control configuration for collection, training and evaluation.

Import after launching Isaac Sim. Arm action units and asset physics are preserved;
the gripper input is continuous in [-1, 1].
"""

import isaaclab.utils.string as string_utils
from isaaclab.managers import ActionTerm, EventTermCfg
from isaaclab.envs.mdp.actions import BinaryJointPositionAction, BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

from benchmark.backends.gsmini.workspace import ResetReachableObject
from benchmark.catalog import ASSETS


def _pad_centers_world(env, links, local_centers):
    """Transform delivered pad centers using the current rigid-body poses."""
    import torch
    from isaaclab.utils.math import quat_apply

    robot = env.scene["robot"]
    body_ids = [robot.body_names.index(link) for link in links]
    offsets = torch.tensor(local_centers, device=env.device,
                           dtype=robot.data.body_pos_w.dtype)
    centers = robot.data.body_pos_w.transpose(0, 1)[body_ids] + quat_apply(
        robot.data.body_quat_w.transpose(0, 1)[body_ids],
        offsets.unsqueeze(1).expand(-1, env.num_envs, -1))
    return list(centers.unbind(dim=0))


def _lift_test_object_spawn_kwargs(rigid_props):
    import isaaclab.sim as sim_utils

    return {
        "rigid_props": rigid_props.copy(),
        "collision_props": sim_utils.CollisionPropertiesCfg(),
        "mass_props": sim_utils.MassPropertiesCfg(density=1200.0),
        "physics_material": sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5, dynamic_friction=0.5, restitution=0.0,
        ),
        "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.1)),
    }


class ContinuousGripperPositionAction(BinaryJointPositionAction):
    """Interpolate gripper joint targets from -1 closed to +1 open."""

    def __init__(self, cfg, env):
        import torch

        # Isaac Lab allocates binary clipping by action_dim (1). Continuous
        # targets need one bound per resolved finger, so defer clipping setup.
        super().__init__(cfg if cfg.clip is None else cfg.replace(clip=None), env)
        self.cfg = cfg
        self._clip = None
        if self.cfg.clip is not None:
            self._clip = torch.tensor(
                [[-float("inf"), float("inf")]], device=self.device,
            ).repeat(self.num_envs, self._num_joints, 1)
            indices, _, values = string_utils.resolve_matching_names_values(
                self.cfg.clip, self._joint_names,
            )
            self._clip[:, indices] = torch.tensor(values, device=self.device)

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        fraction = (self._raw_actions.clamp(-1.0, 1.0) + 1.0) * 0.5
        self._processed_actions[:] = (self._close_command + fraction
                                     * (self._open_command - self._close_command))
        if self._clip is not None:
            self._processed_actions[:] = self._processed_actions.clamp(
                min=self._clip[:, :, 0], max=self._clip[:, :, 1],
            )


@configclass
class ContinuousGripperPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for continuous gripper position control."""

    class_type: type[ActionTerm] = ContinuousGripperPositionAction


class GravityCompensatedJointPositionAction(JointPositionAction):
    """Add PhysX gravity feedforward at each physics step through the existing drives."""

    def position_actions(self, positions):
        """Invert this action term's resolved scale/offset in its resolved joint order."""
        return (positions - self._offset) / self._scale

    def apply_actions(self) -> None:
        super().apply_actions()
        gravity = self._asset.root_physx_view.get_gravity_compensation_forces()
        self._asset.set_joint_effort_target(gravity[:, self._joint_ids], joint_ids=self._joint_ids)


@configclass
class FrankaLiftEnvCfg(FrankaCubeLiftEnvCfg):
    """Official lift task with the same gravity compensation for every caller."""

    grasp_width_offset: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        # The bundled Grid asset is byte-identical to IsaacLab's online default.
        self.scene.plane.spawn.usd_path = str(
            ASSETS / 'objects/ground/default_ground.usd')
        # The standalone lift example also uses local assets. GSmini and task
        # builders replace these defaults with their own published robot/scene.
        self.scene.robot.spawn.usd_path = str(
            ASSETS / 'robots/Franka_gsmini/panda_arm_hand/panda_arm_hand.usd')
        self.scene.table.spawn.usd_path = str(ASSETS / "objects/seattle_lab_table/table_instanceable.usd")
        self.scene.object.spawn.usd_path = str(ASSETS / "objects/dex_cube/dex_cube_instanceable.usd")
        frame = str(ASSETS / "objects/frame_prim/frame_prim.usd")
        self.scene.ee_frame.visualizer_cfg.markers["frame"].usd_path = frame
        self.commands.object_pose.goal_pose_visualizer_cfg.markers["frame"].usd_path = frame
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].usd_path = frame
        gripper_action = self.actions.gripper_action
        self.actions.gripper_action = ContinuousGripperPositionActionCfg(
            asset_name=gripper_action.asset_name,
            joint_names=gripper_action.joint_names,
            open_command_expr=gripper_action.open_command_expr,
            close_command_expr=gripper_action.close_command_expr,
            debug_vis=gripper_action.debug_vis,
            clip=gripper_action.clip,
        )
        self.sim.render.rendering_mode = "quality"
        self.sim.render.antialiasing_mode = "DLAA"
        self.events.reset_object_position = EventTermCfg(
            func=ResetReachableObject, mode="reset", params={"margin": 0.05},
        )
        # Fixed arm gains shared by collection and policy execution.
        for name in ("panda_shoulder", "panda_forearm"):
            self.scene.robot.actuators[name].stiffness = 800.0
            self.scene.robot.actuators[name].damping = 80.0
        self.actions.arm_action.class_type = GravityCompensatedJointPositionAction
