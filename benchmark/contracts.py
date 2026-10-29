"""Shared pre-action observations and absolute joint targets for the Panda RGB profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from benchmark.scene.cameras import CAMERAS

ACTION_REPRESENTATION = "absolute target joint positions; continuous gripper"
GRIPPER_REPRESENTATION = "continuous [-1, 1]; -1 closed, +1 open"
VISION_CAMERAS = tuple(CAMERAS)


def camera_image(env, name: str):
    """Read the same raw pre-action RGB sample for recording."""
    observations = env.obs_buf
    if name in observations["vision"]:
        return observations["vision"][name][0].detach()
    return observations["tactile"][name.rsplit("_", 1)[-1]][0].detach()


def joint_action_metadata(env) -> dict:
    """Read the resolved native controller mapping without changing its settings."""
    import torch

    robot, arm = env.scene["robot"], env.action_manager.get_term("arm_action")
    _, names = robot.find_joints(arm.cfg.joint_names, preserve_order=arm.cfg.preserve_order)
    mapping = {}
    for name, value in (("arm_offset", arm._offset), ("arm_scale", arm._scale)):
        value = torch.as_tensor(value).reshape(-1)
        value = value.expand(arm.action_dim) if value.numel() == 1 else value.reshape(-1, arm.action_dim)[0]
        mapping[name] = value.detach().cpu().tolist()
    return {"state_names": list(robot.joint_names), "arm_joint_names": list(names),
            **mapping,
            "gripper": GRIPPER_REPRESENTATION, "observations": "pre-action"}


def gripper_action_metadata(env) -> dict:
    """Record actual finger order and endpoints, including task-specific closed targets."""
    import torch

    gripper = env.action_manager.get_term("gripper_action")
    names = gripper._joint_names
    return {"representation": GRIPPER_REPRESENTATION, "joint_names": list(names),
            **{key: torch.as_tensor(value).reshape(-1, len(names))[0].detach().cpu().tolist()
               for key, value in (("open", gripper._open_command), ("closed", gripper._close_command))}}


def recording_contract(env_args: dict) -> dict:
    """Carry the recorded control, camera and task metadata through conversion."""
    keys = ("control_dt", "cameras", "joint_action", "gripper_action", "task_definition")
    result = {key: env_args[key] for key in keys if key in env_args}
    if "gsmini_contact" in env_args:
        result["physics_dt"] = env_args["gsmini_contact"]["physics_dt"]
    return result


@dataclass(frozen=True)
class ObservationActionSpec:
    camera_names: tuple[str, ...] = (*VISION_CAMERAS, "gsmini_left", "gsmini_right")
    state_names: tuple[str, ...] = (
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1", "panda_finger_joint2",
    )
    arm_joint_names: tuple[str, ...] = (
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    )
    arm_offset: tuple[float, ...] = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)
    arm_scale: tuple[float, ...] = (0.5,) * 7
    image_shape: tuple[int, ...] = (240, 320, 3)
    vision_shape: tuple[int, ...] = (*reversed(CAMERAS["agentview"]["resolution"]), 3)
    vision_shapes: tuple[tuple[int, ...], ...] = ()
    fps: int = 20
    action_space: str = "joint"
    state_space: str = "matching_action"
    tactile_keys: tuple[str, ...] = ()
    tactile_shape: tuple[int, ...] = (6, 48, 64)

    @classmethod
    def from_dict(cls, value: dict) -> ObservationActionSpec:
        value = dict(value)
        for name in ("camera_names", "state_names", "arm_joint_names", "arm_offset", "arm_scale", "image_shape", "vision_shape", "tactile_keys", "tactile_shape"):
            if name in value:
                value[name] = tuple(value[name])
        if "vision_shapes" in value:
            value["vision_shapes"] = tuple(tuple(shape) for shape in value["vision_shapes"])
        return cls(**value)

    @classmethod
    def from_env_args(cls, env_args: dict) -> ObservationActionSpec:
        """Read the controller/timing contract from already loaded recording metadata."""
        action = env_args.get("joint_action")
        control_dt = env_args["control_dt"]
        kwargs = {key: action[key] for key in (
            "state_names", "arm_joint_names", "arm_offset", "arm_scale")} if action else {}
        kwargs["fps"] = round(1.0 / control_dt)
        cameras = env_args.get("cameras", {})
        if cameras:
            resolutions = [tuple(cameras[name]["resolution"]) for name in VISION_CAMERAS]
            kwargs["vision_shape"] = (*reversed(resolutions[0]), 3)
            if any(value != resolutions[0] for value in resolutions):
                kwargs["vision_shapes"] = tuple((*reversed(value), 3) for value in resolutions)
        return cls.from_dict(kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    def features(self) -> dict:
        features = {
            "observation.state": {"dtype": "float32", "shape": (len(self.state_names),),
                                  "names": list(self.state_names)},
            "action": {"dtype": "float32", "shape": (len(self.arm_joint_names) + 1,),
                       "names": [f"target_{name}" for name in self.arm_joint_names]
                                + ["continuous_gripper_command"]},
        }
        for camera in self.camera_names:
            features[f"observation.images.{camera}"] = {
                "dtype": "video", "shape": self.camera_shape(camera),
                "names": ["height", "width", "channels"],
            }
        return features


    def camera_shape(self, camera: str) -> tuple[int, ...]:
        if camera in VISION_CAMERAS and self.vision_shapes:
            return self.vision_shapes[VISION_CAMERAS.index(camera)]
        return self.vision_shape if camera in VISION_CAMERAS else self.image_shape

    def target_action(self, command) -> np.ndarray:
        """Convert recorded controller commands to physical joint targets."""
        command = np.asarray(command, np.float32)
        n_arm = len(self.arm_joint_names)
        target = command[..., :n_arm] * np.asarray(self.arm_scale, np.float32) + np.asarray(self.arm_offset, np.float32)
        return np.concatenate((target, command[..., -1:]), axis=-1)
