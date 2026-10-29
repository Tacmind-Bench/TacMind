"""TacMind-0 input adapter for the lossless GSmini LeRobot v3 dataset."""
import json
from pathlib import Path

import numpy as np


def export_index(root, destination):
    """Export native JSONL action records; visual payloads remain in LeRobot."""
    import pyarrow.parquet as pq
    from policy.tacmind0.data import native_state
    from policy.tacmind0.representation import require_dataset

    root, destination = Path(root), Path(destination)
    require_dataset(root)
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    task_names = dict(zip(tasks["task_index"].astype(int), tasks.index))
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    episode = -1
    stream = None
    try:
        for file in sorted((root / "data").rglob("*.parquet")):
            parquet = pq.ParquetFile(file)
            columns = ["episode_index", "task_index",
                       "observation.state", "action"]
            for batch in parquet.iter_batches(batch_size=2048, columns=columns):
                for row in batch.to_pylist():
                    current = int(row["episode_index"])
                    if current != episode:
                        if stream is not None:
                            stream.close()
                        episode = current
                        stream = (destination / f"episode_{episode:06d}.jsonl").open("w")
                    state = np.asarray(row["observation.state"], dtype=np.float32)
                    action = np.asarray(row["action"], dtype=np.float32)
                    record = {"state": native_state(state).tolist(), "action": action.tolist(),
                              "prompt": task_names[int(row["task_index"])], "is_robot": True,
                              "lerobot_index": count}
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    count += 1
    finally:
        if stream is not None:
            stream.close()
    return count


def install(root):
    """Connect TacMind-0's independent transforms to LeRobot image payloads."""
    from policy.lerobot_images import LeRobotImages
    from policy.tacmind0.representation import require_dataset

    require_dataset(root)
    import tacmind0.policy.config as config
    from tacmind0.tactile.preprocessing import history_indices, prepare_history

    class LoadImages:
        def __init__(self, image_keys, image_dir, **kwargs):
            self.reader = LeRobotImages(root)

        def __call__(self, data):
            data["images"] = self.reader.get(int(data["lerobot_index"]), ("agentview", "wristview"))
            return data

    class LoadTactileHistory:
        def __init__(self, image_dir, cache_size=256):
            self.reader = LeRobotImages(root)

        def __call__(self, data):
            frames = []
            for i in history_indices(data["meta_data"]["frame_index"]):
                index = json.loads(data["raw_lines"][i])["lerobot_index"]
                frames.append(self.reader.get(index, ("deformation_left", "deformation_right")))
            left, right = zip(*frames)
            data["tactile_pixel_values"] = prepare_history(left, right).unsqueeze(0)
            return data

    config.LoadImages = LoadImages
    config.LoadTactileHistory = LoadTactileHistory
