"""RoboTwin 2.0 dataset registration."""

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotType
from opendm.dataset.register import register_dataset

_ROBOTWIN2_DATA_ROOT = "./data/robotwin2.0"

register_dataset(
    {
        "generalist": {
            "jsonl_dir": _ROBOTWIN2_DATA_ROOT,
            "image_dir": f"{_ROBOTWIN2_DATA_ROOT}/video",
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "robot_type": RobotType.ALOHA_ROBOTWIN2,
            "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2],
        },
    },
    prefix="robotwin2",
)


# Task/setting views over the public robotwin2_generalist archive. Keeping the
# components separate allows the SFT recipe to apply its task-aware weights
# without introducing a second data source.
ROBOTWIN2_COMPONENT_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
ROBOTWIN2_COMPONENT_SETTINGS = ("clean", "randomized")

_ROBOTWIN2_COMPONENT_DATASETS = {
    f"{task}_{setting}": {
        "jsonl_dir": f"{_ROBOTWIN2_DATA_ROOT}/jsonl/{task}/{setting}/",
        "image_dir": f"{_ROBOTWIN2_DATA_ROOT}/video",
        "image_keys": ["images_1", "images_2", "images_3"],
        "image_prompts": ["Head", "Left wrist", "Right wrist"],
        "robot_type": RobotType.ALOHA_ROBOTWIN2,
        "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2],
        "fps": 30,
    }
    for task in ROBOTWIN2_COMPONENT_TASKS
    for setting in ROBOTWIN2_COMPONENT_SETTINGS
}

register_dataset(_ROBOTWIN2_COMPONENT_DATASETS, prefix="robotwin2_component")

# Preserve existing dataset names used by external configs and checkpoints.
register_dataset(_ROBOTWIN2_COMPONENT_DATASETS, prefix="robotwin2_example")
