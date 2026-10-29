"""Local episode data shared by policy training and normalization."""

import sys
from collections import Counter, OrderedDict
from pathlib import Path

import orjson
from torch.utils.data import Dataset

from .sample_index import EpisodeSampleIndex

_EPISODE_CACHE_BYTES = 16 * 1024 * 1024


class TacMind0Dataset(Dataset):
    def __init__(
        self,
        jsonl_dir: str,
        transforms=None,
        dataset_meta: dict | None = None,
    ):
        self.transforms = transforms
        self.dataset_meta = {} if dataset_meta is None else dataset_meta
        root = Path(jsonl_dir).resolve()
        files = sorted(root.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No episode JSONL files in {root}")
        # Scan the selected tree: copied caches may contain stale paths/counts.
        counts = []
        for path in files:
            with path.open() as handle:
                count = sum(bool(line.strip()) for line in handle)
            if count < 2:
                raise ValueError(f"Episode must contain at least two frames: {path}")
            counts.append(count)
        self.id_to_jsonl = dict(enumerate(map(str, files)))
        self.sample_index = EpisodeSampleIndex(enumerate(counts))
        self._episode_cache = OrderedDict()
        self._episode_cache_bytes = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_episode_cache=OrderedDict(), _episode_cache_bytes=0)
        return state

    def _read_episode(self, path: str) -> list[str]:
        stat = Path(path).stat()
        version = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._episode_cache.pop(path, None)
        if cached is not None:
            if cached[0] == version:
                self._episode_cache[path] = cached
                return list(cached[1])
            self._episode_cache_bytes -= cached[2]
        lines = _read_jsonl_lines(path)
        if stat.st_size > _EPISODE_CACHE_BYTES:
            return lines
        stored = tuple(lines)
        size = sys.getsizeof(stored) + sum(map(sys.getsizeof, stored))
        if size <= _EPISODE_CACHE_BYTES:
            while (
                self._episode_cache_bytes + size > _EPISODE_CACHE_BYTES
                or len(self._episode_cache) >= 32
            ):
                _, evicted = self._episode_cache.popitem(last=False)
                self._episode_cache_bytes -= evicted[2]
            self._episode_cache[path] = (version, stored, size)
            self._episode_cache_bytes += size
        return lines

    def __getitem__(self, idx: int) -> dict:
        file_index, frame_index = self.sample_index[idx]
        jsonl_file = self.id_to_jsonl[file_index]
        lines = self._read_episode(jsonl_file)
        return self._prepare_sample(lines, frame_index)

    def __getitems__(self, indices: list[int]) -> list[dict]:
        locations = [self.sample_index[idx] for idx in indices]
        remaining = Counter(file_index for file_index, _ in locations)
        episodes = {}
        samples = []
        for file_index, frame_index in locations:
            if file_index not in episodes:
                episodes[file_index] = self._read_episode(self.id_to_jsonl[file_index])
            lines = episodes[file_index]
            remaining[file_index] -= 1
            if remaining[file_index]:
                # Transforms may mutate raw_lines; each sample owns its list.
                lines = lines.copy()
            else:
                del episodes[file_index]
            samples.append(self._prepare_sample(lines, frame_index))
        return samples

    def _prepare_sample(self, lines: list[str], frame_index: int) -> dict:
        result = orjson.loads(lines[frame_index])
        result["raw_lines"] = lines
        result["meta_data"] = {
            **self.dataset_meta,
            "frame_index": frame_index,
        }
        if self.transforms is not None:
            result = self.transforms(result)
        return result

    def __len__(self) -> int:
        return len(self.sample_index)


def _read_jsonl_lines(file_path: str) -> list[str]:
    with open(file_path) as handle:
        return [line for line in handle if line.strip()]
