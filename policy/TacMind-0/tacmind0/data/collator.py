"""Batch collation for training and norm-stat computation."""

from collections.abc import Sequence
from enum import Enum

import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F


class TrainingCollator:
    """Pad to a fixed length, or the longest sample when max_length is None."""

    def __init__(
        self,
        pad_token_id: int,
        max_length: int | None = 1024,
    ):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        padding = {
            "input_ids": self.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": 0,
        }
        sequences = {key: [] for key in padding}
        max_len = self.max_length
        if max_len is None:
            max_len = max(item["input_ids"].shape[1] for item in instances)
        for instance in instances:
            seq_len = instance["input_ids"].shape[1]
            if seq_len > max_len:
                logger.warning(
                    "Input sequence length {} exceeds max_length {}; truncating.",
                    seq_len,
                    max_len,
                )
            pad_len = max(0, max_len - seq_len)
            for key, pad_value in padding.items():
                value = instance[key][:, :max_len]
                if pad_len:
                    value = F.pad(value, (0, pad_len), value=pad_value)
                sequences[key].append(value)

        batch = {key: torch.cat(values, dim=0) for key, values in sequences.items()}
        for key in ("pixel_values", "action", "action_mask"):
            batch[key] = torch.cat([item[key] for item in instances], dim=0)

        tactile = ["tactile_pixel_values" in item for item in instances]
        if any(tactile):
            if not all(tactile):
                raise ValueError("Mixed tactile and vision-only samples in batch")
            batch["tactile_pixel_values"] = torch.cat(
                [item["tactile_pixel_values"] for item in instances], dim=0
            )
        return batch


class NormStatsCollator:
    def __call__(self, instances: Sequence[dict]) -> dict:
        grouped_instances: dict[str | None, list[dict]] = {}
        for instance in instances:
            robot_type = instance.get("meta_data", {}).get("robot_type")
            if isinstance(robot_type, Enum):
                robot_type = str(robot_type.value)
            elif robot_type is not None:
                robot_type = str(robot_type)
            grouped_instances.setdefault(robot_type, []).append(instance)

        if None in grouped_instances and len(grouped_instances) > 1:
            raise ValueError(
                "Cannot compute norm stats from a batch containing both typed "
                "and untyped robot samples"
            )

        robot_batches = {}
        for robot_type, robot_instances in grouped_instances.items():
            robot_batch = {}
            for key in ("state", "action"):
                presence = [key in instance for instance in robot_instances]
                if any(presence) and not all(presence):
                    raise ValueError(
                        f"Inconsistent {key!r} presence for robot_type {robot_type!r}"
                    )
                if not any(presence):
                    continue
                values = [instance[key] for instance in robot_instances]
                robot_batch[key] = (
                    np.stack(values, axis=0)
                    if key == "state"
                    else np.concatenate(values, axis=0)
                )
            if "action" not in robot_batch:
                raise ValueError(
                    f"Cannot compute norm stats without action for robot_type "
                    f"{robot_type!r}"
                )
            robot_batches[robot_type] = robot_batch
        return {"robot_batches": robot_batches}
