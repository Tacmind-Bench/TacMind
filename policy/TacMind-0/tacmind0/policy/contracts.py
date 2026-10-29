"""Serialized CLS-token policy contract and validators; no model or CLI imports."""

import numpy as np

from tacmind0.constants.robot import ActionMode, RobotStateDesc
from tacmind0.constants.tactile import TACTILE_TOKEN_ID

CONTRACT = {
    "version": 2,
    "source": "marker",
    "color": "bgr",
    "history_size": 8,
    "history_stride": 5,
    "frame_size": 42,
    "patch_size": 14,
    "resize": "area56_bilinear42_round_uint8",
    "readout": "cls",
    "streaming": False,
    "vision_views": ["agentview", "wrist"],
    "fusion": "cls_token",
    "token_id": TACTILE_TOKEN_ID,
    "visual_augmentation": {
        "agentview": "none",
        "wrist": "none",
        "probability": 0.5,
    },
    "action_mode": "relative",
    "output_action_dim": 8,
    "state_desc": ["joint"] * 7 + ["gripper"],
    "add_state": True,
    "n_bins": 256,
    "image_prompts": ["Agent view", "Wrist"],
}


def validate_visual_augmentation(value: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError("TacMind-0 visual_augmentation must be an object")
    for view in ("agentview", "wrist"):
        if value.get(view) not in ("none", "light"):
            raise ValueError(f"TacMind-0 {view} augmentation must be 'none' or 'light'")
    probability = value.get("probability")
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not np.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("TacMind-0 augmentation probability must be in [0, 1]")


def validate_contract(contract: dict) -> None:
    """Validate the dimensions and options used by action and image transforms."""
    if contract.get("output_action_dim") != 8:
        raise ValueError("TacMind-0 requires eight action dimensions")
    descriptors = contract.get("state_desc")
    if not isinstance(descriptors, list) or len(descriptors) != 8:
        raise ValueError("TacMind-0 state_desc must describe exactly eight dimensions")
    for value in descriptors:
        RobotStateDesc(value)
    ActionMode(contract.get("action_mode"))
    if not isinstance(contract.get("add_state"), bool):
        raise ValueError("TacMind-0 add_state must be boolean")
    bins = contract.get("n_bins")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("TacMind-0 n_bins must be an integer >= 2")
    prompts = contract.get("image_prompts")
    if (
        not isinstance(prompts, list)
        or len(prompts) != 2
        or not all(isinstance(value, str) and value.strip() for value in prompts)
    ):
        raise ValueError("TacMind-0 requires two nonempty image prompts")
    validate_visual_augmentation(
        contract.get("visual_augmentation", CONTRACT["visual_augmentation"])
    )
