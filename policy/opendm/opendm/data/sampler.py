"""Dataset samplers used by OpenDM training recipes."""

import numpy as np
from torch.utils.data import Sampler


class WeightedMultiDatasetSampler(Sampler):
    """Sample a component dataset first, then a uniform local index.

    The RNG first selects a component according to ``weights``, then samples a
    local index uniformly. Distributed sharding remains the trainer / Accelerate
    dataloader wrapper's responsibility.
    """

    chunk_size = 1_000_000

    def __init__(
        self,
        datasets: list,
        weights: list[float] | None = None,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        self.dataset_sizes = np.asarray(
            [len(dataset) for dataset in datasets], dtype=np.int64
        )
        if np.any(self.dataset_sizes <= 0):
            raise ValueError("Weighted component datasets must be non-empty")
        if weights is None:
            weights = self.dataset_sizes.tolist()
        probabilities = np.asarray(weights, dtype=np.float64)
        if probabilities.shape != self.dataset_sizes.shape:
            raise ValueError("weights must match the component datasets")
        if np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("weights must be non-negative with a positive sum")

        self.weights = probabilities / probabilities.sum()
        self.cumulative_sizes = np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(self.dataset_sizes)]
        )
        if num_samples is None:
            if self.weights[0] == 0:
                raise ValueError(
                    "num_samples must be provided when the first dataset weight is zero"
                )
            self.total_samples = int(self.dataset_sizes[0] / self.weights[0])
        else:
            self.total_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        remaining = self.total_samples
        while remaining > 0:
            chunk_size = min(self.chunk_size, remaining)
            dataset_indices = rng.choice(
                len(self.dataset_sizes), size=chunk_size, p=self.weights
            )
            local_indices = np.empty(chunk_size, dtype=np.int64)
            for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
                positions = np.nonzero(dataset_indices == dataset_idx)[0]
                if positions.size:
                    local_indices[positions] = rng.integers(
                        0, int(dataset_size), size=positions.size
                    )
            yield from self.cumulative_sizes[dataset_indices] + local_indices
            remaining -= chunk_size

    def __len__(self) -> int:
        return self.total_samples
