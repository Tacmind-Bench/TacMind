"""Compact, ordered indexing of within-episode next-frame training pairs."""

from bisect import bisect_right
from collections.abc import Sequence
from operator import index


class EpisodeSampleIndex(Sequence):
    """Map a sample number to (episode key, frame) with O(episodes) storage.

    ``episodes`` contains (key, frame_count) pairs in sampling order. The last
    frame of each episode is excluded because it has no next-frame target.
    Indexing uses a binary search; iteration and slices retain list semantics.
    """

    def __init__(self, episodes):
        keys, stops = [], []
        total = 0
        for key, frame_count in episodes:
            total += max(0, index(frame_count) - 1)
            keys.append(key)
            stops.append(total)
        self.keys = tuple(keys)
        self.stops = tuple(stops)

    def __len__(self):
        return self.stops[-1] if self.stops else 0

    def __getitem__(self, sample):
        if isinstance(sample, slice):
            return [self[i] for i in range(*sample.indices(len(self)))]
        sample = index(sample)
        if sample < 0:
            sample += len(self)
        if not 0 <= sample < len(self):
            raise IndexError("episode sample index out of range")
        episode = bisect_right(self.stops, sample)
        start = self.stops[episode - 1] if episode else 0
        return self.keys[episode], sample - start

    def __iter__(self):
        start = 0
        for key, stop in zip(self.keys, self.stops):
            for frame in range(stop - start):
                yield key, frame
            start = stop
