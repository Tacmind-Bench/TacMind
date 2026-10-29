"""Read lossless LeRobot GSmini episodes for Tac-LeWM pretraining."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from policy.lerobot_images import LeRobotImages


class LeRobotWorldDataset:
    """Expose episode windows and full action statistics to the native trainer."""

    def __init__(self, root, *, num_steps, frameskip, history_size, keys_to_load, **_):
        import pyarrow.parquet as pq
        from policy.tacmind0.representation import require_dataset

        self.root = Path(root).resolve()
        require_dataset(self.root)
        lineage = json.loads((self.root / "tacmind_lineage.json").read_text())
        self.episodes = lineage["episodes"]
        self.lengths = np.asarray([x["frames"] for x in self.episodes], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths[:-1], dtype=np.int64)))
        self.task_ids = [int(x["task_index"]) for x in self.episodes]
        self.num_steps, self.frameskip = int(num_steps), int(frameskip)
        self.span = self.num_steps * self.frameskip
        self.current_index = min(max(int(history_size), 1), self.num_steps) - 1
        self._keys = list(keys_to_load)
        self.clip_indices = [(episode, start) for episode, length in enumerate(self.lengths)
                             for start in range(int(length) - self.span + 1)]
        columns = []
        for path in sorted((self.root / "data").rglob("*.parquet")):
            table = pq.read_table(path, columns=["action"])
            columns.append(np.asarray(table.column("action").to_pylist(), dtype=np.float32))
        raw = np.concatenate(columns, axis=0)
        mean = raw.mean(axis=0, dtype=np.float64)
        std = np.maximum(raw.std(axis=0, dtype=np.float64), 1e-6)
        self.actions = np.clip((raw - mean) / std, -5, 5).astype(np.float32)
        self.transform = None
        # Build the file index once per training process; forked loader workers
        # inherit it instead of scanning thousands of Parquet footers each.
        self._dataset = LeRobotImages(self.root, capacity=3)

    def __getstate__(self):
        return {**self.__dict__, "_dataset": None}

    def _reader(self):
        if self._dataset is None:
            self._dataset = LeRobotImages(self.root, capacity=3)
        return self._dataset

    @property
    def column_names(self):
        return self._keys

    def __len__(self):
        return len(self.clip_indices)

    def get_col_data(self, col):
        if col == "action":
            return self.actions
        raise KeyError(f"No materialized LeRobot column: {col}")

    def get_dim(self, col):
        return int(self.get_col_data(col).shape[-1])

    def get_row_data(self, row_idx):
        return {"action": self.actions[row_idx]}

    def __getitem__(self, index):
        import cv2
        import torch
        from tacmind0.tactile.preprocessing import prepare_frame_pair

        episode, start = self.clip_indices[index]
        start_global = int(self.offsets[episode]) + start
        reader = self._reader()
        pixels = []
        for step in range(self.num_steps):
            frame = start + step * self.frameskip
            history = [int(self.offsets[episode]) + max(0, frame - 35 + 5 * i) for i in range(8)]
            pairs = [prepare_frame_pair(*reader.get(index, ("deformation_left", "deformation_right")))
                     for index in history]
            # Native pretraining consumes prepared BGR bytes. Use the same pixel
            # contract as policy training/inference, including prepared 42px input.
            pixels.append(torch.stack(pairs, dim=1).mul(255).round().to(torch.uint8))
        rgb = np.asarray(reader.get(start_global + self.current_index * self.frameskip, ("agentview",))[0])
        bgr = cv2.resize(rgb[..., ::-1], (224, 224), interpolation=cv2.INTER_AREA)
        batch = {"pixels": torch.stack(pixels),
                 "vision": torch.from_numpy(bgr.copy()).permute(2, 0, 1).unsqueeze(0),
                 "action": torch.from_numpy(self.actions[start_global:start_global + self.span:self.frameskip].copy()),
                 "source_id": torch.zeros(self.num_steps, dtype=torch.int64),
                 "has_action": torch.ones(self.num_steps, dtype=torch.int64),
                 "has_vision": torch.ones(self.num_steps, dtype=torch.int64)}
        batch = {key: batch[key] for key in self._keys}
        return self.transform(batch) if self.transform else batch
