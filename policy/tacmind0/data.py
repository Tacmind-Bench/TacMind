"""Export prepared benchmark episodes in the native TacMind-0 JSONL format."""

import json
from pathlib import Path

import numpy as np

def native_state(state):
    """Convert measured joints to the native seven joints and gripper aperture."""
    state = np.asarray(state)
    return np.concatenate((state[..., :7], state[..., 7:9].sum(axis=-1, keepdims=True)), axis=-1)


def export_jsonl(cfg, manifest):
    """Map Deformation images to the native model's historical marker slots."""
    from policy.tacmind0.representation import validate
    validate(manifest, cfg["prepared_root"])
    root = Path(cfg["prepared_root"])
    destination = Path(cfg["output_dir"]) / "tacmind0_data"
    destination.mkdir()
    for row in manifest["episodes"]:
        episode = row["index"]
        with np.load(root / f"episode_{episode:06d}.npz", allow_pickle=False) as data:
            with (destination / f"episode_{episode:06d}.jsonl").open("w") as stream:
                for frame in range(row["length"]):
                    record = dict(
                        state=native_state(data["state"][frame]).tolist(),
                        action=data["action"][frame].tolist(),
                        prompt=row["prompts"][frame],
                        is_robot=True,
                    )
                    for key, camera in (
                        ("images_1", "agentview"),
                        ("images_2", "wristview"),
                        ("marker_left", "deformation_left"),
                        ("marker_right", "deformation_right"),
                    ):
                        path = (
                            root
                            / "images"
                            / f"{episode:06d}"
                            / camera
                            / f"{frame:06d}.png"
                        )
                        record[key] = dict(type="image", url=str(path))
                    stream.write(json.dumps(record) + "\n")
    return destination
