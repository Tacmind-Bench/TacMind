"""Read embedded LeRobot images with a bounded, process-local row-group cache."""
from bisect import bisect_right
from collections import OrderedDict
from io import BytesIO
import os
from pathlib import Path


class LeRobotImages:
    """Worker-local bounded Parquet row-group reader for embedded lossless images."""
    def __init__(self, root, capacity=2):
        import pyarrow.parquet as pq
        self.root = Path(root)
        self.capacity = capacity
        self.pid = os.getpid()
        self.groups = []
        self.ends = []
        offset = 0
        for path in sorted((self.root / "data").rglob("*.parquet")):
            metadata = pq.ParquetFile(path).metadata
            for group in range(metadata.num_row_groups):
                length = metadata.row_group(group).num_rows
                self.groups.append((path, group, offset))
                offset += length
                self.ends.append(offset)
        self.cache = OrderedDict()

    def get(self, index, keys):
        import pyarrow.parquet as pq
        from PIL import Image
        if self.pid != os.getpid():
            self.cache = OrderedDict()
            self.pid = os.getpid()
        group = bisect_right(self.ends, index)
        if index < 0 or group >= len(self.groups):
            raise IndexError(index)
        path, row_group, offset = self.groups[group]
        key = (group, tuple(keys))
        if key not in self.cache:
            columns = ["observation.images." + name for name in keys]
            self.cache[key] = pq.ParquetFile(path).read_row_group(row_group, columns=columns)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        table = self.cache[key]
        result = []
        for name in keys:
            item = table["observation.images." + name][index - offset].as_py()
            with Image.open(BytesIO(item["bytes"])) as image:
                result.append(image.convert("RGB"))
        return result
