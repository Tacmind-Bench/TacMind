"""Compute normalization from the same action-transformed dataset used in training."""

from pathlib import Path

from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import tacmind0.data.normalize as normalize
from tacmind0.data.collator import NormStatsCollator


def compute_norm_stats(
    dataset: Dataset,
    output_path: str | Path,
    *,
    batch_size: int = 128,
    num_workers: int = 32,
    max_batches: int | None = None,
) -> None:
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "collate_fn": NormStatsCollator(),
    }
    if num_workers > 0:
        dataloader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 4,
            }
        )
    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs,
    )
    stats_by_robot: dict[
        str | None,
        dict[str, normalize.RunningStats],
    ] = {}
    keys_by_robot: dict[str | None, set[str]] = {}
    max_batches = max_batches or len(dataloader)
    total = min(len(dataloader), max_batches)
    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        total=total,
        desc="Computing norm stats",
    ):
        if batch_idx >= max_batches:
            break
        for robot_type, robot_batch in batch["robot_batches"].items():
            batch_keys = set(robot_batch)
            expected_keys = keys_by_robot.setdefault(robot_type, batch_keys)
            if batch_keys != expected_keys:
                raise ValueError(
                    f"Inconsistent norm fields for robot_type {robot_type!r}: "
                    f"expected {sorted(expected_keys)}, got {sorted(batch_keys)}"
                )
            robot_stats = stats_by_robot.setdefault(
                robot_type,
                {key: normalize.RunningStats() for key in sorted(expected_keys)},
            )
            for key, values in robot_batch.items():
                robot_stats[key].update(values.reshape(-1, values.shape[-1]))

    if not stats_by_robot:
        raise ValueError("Cannot compute norm stats from an empty dataset")
    if None in stats_by_robot and len(stats_by_robot) > 1:
        raise ValueError(
            "Cannot compute norm stats from both typed and untyped robot samples"
        )

    computed_stats_by_robot: dict[
        str | None,
        dict[str, normalize.NormStats],
    ] = {}
    for robot_type, robot_stats in stats_by_robot.items():
        computed_stats = {}
        for key, running_stats in robot_stats.items():
            result = running_stats.get_statistics()
            computed_stats[key] = result.model_copy(update={"min": None, "max": None})
        computed_stats_by_robot[robot_type] = computed_stats

    if None in computed_stats_by_robot:
        output = normalize.serialize_json(
            computed_stats_by_robot[None],
            exclude_none=True,
        )
    else:
        typed_stats = {
            robot_type: stats
            for robot_type, stats in computed_stats_by_robot.items()
            if robot_type is not None
        }
        if len(typed_stats) != 1:
            raise ValueError("TacMind-0 normalization requires one robot type")
        default_robot_type = next(iter(typed_stats))
        output = normalize.serialize_norm_stats_file(
            typed_stats,
            default_robot_type=default_robot_type,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output)
    logger.info(f"Saved norm stats to {output_path}")
