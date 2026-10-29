"""Convert collected GSmini episodes and Deformation views to one LeRobot dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_tasks(index_path):
    return json.loads(Path(index_path).read_text())


def convert(index_path, output, *, repo_id, sample=False, task_index=None):
    """Write lossless camera and native Deformation frames with original task IDs."""
    import h5py
    import hdf5plugin  # Register filters used by the source recordings.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from benchmark.contracts import ObservationActionSpec
    from benchmark.io.image_storage import rgb_sequence
    from benchmark.backends.gsmini.markers import deformation_image

    index_path, output = Path(index_path).resolve(), Path(output).resolve()
    tasks = load_tasks(index_path)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(output)
        output.rmdir()  # LeRobot creates its root; accept user-created empty directories.
    selected = tasks if task_index is None else [x for x in tasks if x["task_index"] == task_index]
    if not selected:
        raise ValueError(f"Unknown task index: {task_index}")
    if sample:
        selected = selected[:1]
    first_manifest_path = Path(selected[0]["manifest"])
    first_manifest = json.loads(first_manifest_path.read_text())
    first_entry = first_manifest["episodes"][str(selected[0]["recorded_task_id"])][0]
    with h5py.File(first_manifest_path.parent / first_entry["file"], "r") as source:
        spec = ObservationActionSpec.from_env_args(json.loads(source["data"].attrs["env_args"]))
    features = {
        "observation.state": {"dtype": "float32", "shape": (9,), "names": None},
        "action": {"dtype": "float32", "shape": (8,), "names": None},
        "task_id": {"dtype": "int32", "shape": (1,), "names": None},
        **{
            f"observation.images.{name}": {
                "dtype": "image", "shape": shape, "names": ["height", "width", "channel"]
            }
            for name, shape in (
                ("agentview", spec.camera_shape("agentview")), ("wristview", spec.camera_shape("wristview")),
                ("deformation_left", spec.camera_shape("gsmini_left")),
                ("deformation_right", spec.camera_shape("gsmini_right")),
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(repo_id=repo_id, root=output, fps=20,
                                    robot_type="franka_gsmini", features=features,
                                    use_videos=False, image_writer_threads=4)
    lineage = []
    try:
        for task in selected:
            manifest_path = Path(task["manifest"])
            manifest = json.loads(manifest_path.read_text())
            entries = manifest["episodes"][str(task["recorded_task_id"])]
            if sample:
                entries = entries[:1]
            instruction = manifest["language_instruction"]
            for episode, entry in enumerate(entries):
                source_path = (manifest_path.parent / entry["file"]).resolve()
                with h5py.File(source_path, "r") as source:
                    env = json.loads(source["data"].attrs["env_args"])
                    demo = source["data/demo_0"]
                    state = np.asarray(demo["obs/joint_pos"][:], dtype=np.float32)
                    action = np.asarray(ObservationActionSpec.from_env_args(env).target_action(demo["actions"][:]), dtype=np.float32)
                    length = len(action)
                    cameras = {key: rgb_sequence(demo["obs/" + key])
                               for key in ("agentview", "wristview", "gsmini_left", "gsmini_right")}
                    motions = {side: demo[f"obs/gsmini_{side}_marker_motion"] for side in ("left", "right")}
                    for t in range(length):
                        dataset.add_frame({
                            "observation.state": state[t], "action": action[t],
                            "task_id": np.asarray([task["task_index"]], dtype=np.int32),
                            "observation.images.agentview": cameras["agentview"][t],
                            "observation.images.wristview": cameras["wristview"][t],
                            **{f"observation.images.deformation_{side}": deformation_image(
                                cameras[f"gsmini_{side}"][t], motions[side][t]) for side in motions},
                            "task": instruction,
                        })
                dataset.save_episode()
                lineage.append(dict(episode_index=dataset.num_episodes - 1, task_index=task["task_index"],
                                    suite=task["suite"], task_id=task["recorded_task_id"],
                                    seed=entry["seed"], frames=length, source=str(source_path),
                                    manifest=str(manifest_path)))
                print(f"[CONVERT] {len(lineage)} episodes: {task['suite']}/{task['recorded_task_id']} {episode + 1}/{len(entries)}", flush=True)
    finally:
        dataset.finalize()
    (output / "tacmind_lineage.json").write_text(json.dumps({
        "format": "tacmind.gsmini.multitask.lerobot.v1", "repo_id": repo_id,
        "source_index": str(index_path),
        "episodes": lineage, "frames": sum(x["frames"] for x in lineage),
        "tactile_input": "fots-deformation-arrows-v1",
    }, indent=2) + "\n")
    (output / "conversion_metadata.json").write_text(json.dumps({
        "source_index": str(index_path), "episodes": len(lineage),
        "frames": sum(x["frames"] for x in lineage),
        "tactile_input": "fots-deformation-arrows-v1",
    }, indent=2) + "\n")
    print(f"[CONVERT] Complete: {len(lineage)} episodes, {sum(x['frames'] for x in lineage)} frames", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    convert(args.source_index, args.output, repo_id=args.repo_id, sample=args.sample,
            task_index=args.task_index)


if __name__ == "__main__":
    main()
