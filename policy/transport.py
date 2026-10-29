"""Reusable shared array storage for a synchronous parent/worker exchange."""
import math
import mmap
from pathlib import Path

ACTION_KEY = "__action_chunk__"


class ArrayBuffer:
    """Map numeric arrays once; callers exchange ownership through pipe acknowledgements.

    Reads return independent snapshots so adapters may retain or mutate observations.
    The parent must wait for a response before writing the next observation.
    """

    def __init__(self, path, layout, *, create=False):
        import numpy as np

        self.layout, offsets, size = {}, {}, 0
        for name, spec in sorted(layout.items()):
            shape, dtype = spec.get("shape"), np.dtype(spec["dtype"])
            size = (size + 63) // 64 * 64
            offsets[name] = size
            size += math.prod(shape) * dtype.itemsize
            self.layout[name] = dict(shape=list(shape), dtype=dtype.str)
        self.path = Path(path)
        self._mapping = None
        self._arrays = {}
        with self.path.open("x+b" if create else "r+b") as file:
            if create:
                file.truncate(size)
            try:
                self._mapping = mmap.mmap(file.fileno(), size, access=mmap.ACCESS_WRITE)
                self._arrays = {name: np.ndarray(spec["shape"], dtype=spec["dtype"],
                                                buffer=self._mapping, offset=offsets[name])
                                for name, spec in self.layout.items()}
            except BaseException:
                self.close()
                raise

    @classmethod
    def create(cls, path, values):
        import numpy as np

        arrays = {key: np.asarray(value) for key, value in values.items()}
        buffer = cls(path, {key: dict(shape=array.shape, dtype=array.dtype.str)
                            for key, array in arrays.items()}, create=True)
        try:
            buffer.write(arrays)
        except BaseException:
            buffer.close()
            raise
        return buffer

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._arrays.clear()
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None

    def write(self, values):
        import numpy as np

        if self._mapping is None:
            raise RuntimeError("Array buffer is closed")
        arrays = {key: np.asarray(value) for key, value in values.items()}
        for name, value in arrays.items():
            np.copyto(self._arrays[name], value, casting="no")

    def read(self, names=None):
        if self._mapping is None:
            raise RuntimeError("Array buffer is closed")
        return {name: self._arrays[name].copy() for name in (self._arrays if names is None else names)}
