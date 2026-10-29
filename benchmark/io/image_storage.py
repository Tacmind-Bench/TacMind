"""Bounded RGB blocks in HDF5; lossless by default, with explicit compact RGB."""

from __future__ import annotations

from fractions import Fraction
from io import BytesIO
import operator

import h5py

import hdf5plugin  # Register recording filters at the data I/O boundary.
import numpy as np

RGB_FORMAT = 'tacmind.rgb.lossless-h264'
RGB_COMPACT_FORMAT = 'tacmind.rgb.h264'
RGB_BLOCK_FRAMES = 128
# Keep every RGB byte while avoiding the long searches in x264's veryslow
# preset. The preset changes compression cost/size, not lossless pixel values.
RGB_PRESET = 'medium'
RGB_STORAGE = dict(
    format=RGB_FORMAT, encoder='libx264rgb', crf=0, pixel_format='rgb24',
    lossless=True, preset=RGB_PRESET, block_frames=RGB_BLOCK_FRAMES,
    keyframe_interval=RGB_BLOCK_FRAMES,
)


def rgb_storage(crf=0):
    """Describe the actual image encoding without changing observation geometry."""
    return dict(RGB_STORAGE, format=RGB_COMPACT_FORMAT if crf else RGB_FORMAT,
                crf=crf, lossless=crf == 0,
                preset='veryslow' if crf else RGB_PRESET)


def _encode(frames, crf=0):
    import av

    preset = rgb_storage(crf)['preset']
    height, width, _ = frames.shape[1:]
    buffer = BytesIO()
    with av.open(buffer, 'w', format='h264') as output:
        stream = output.add_stream('libx264rgb', 20, options=dict(
            crf=str(crf), preset=preset, g=str(RGB_BLOCK_FRAMES), threads='2'))
        stream.width, stream.height, stream.pix_fmt = width, height, 'rgb24'
        for index, image in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, 20)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return buffer.getvalue()


def _decode(encoded):
    import av

    with av.open(BytesIO(encoded), 'r', format='h264') as container:
        container.streams.video[0].thread_count = 2
        return b''.join(frame.to_ndarray(format='rgb24').tobytes()
                        for frame in container.decode(video=0))


class RGBWriter:
    """Copy incoming uint8 frames; encode full blocks and explicitly flush the tail."""

    def __init__(self, parent, name, shape, *, crf=0):
        storage = rgb_storage(crf)
        shape = tuple(shape)
        self.group = parent.create_group(name)
        self.crf = crf
        self.group.attrs.update(format=storage['format'], frame_shape=shape, num_frames=0,
                                crf=crf, lossless=storage['lossless'])
        self.encoded = self.group.create_dataset(
            'encoded', shape=(0,), maxshape=(None,),
            chunks=(65536,), dtype=np.uint8,
        )
        # Index rows: first frame, frame count, byte offset, byte count.
        self.index = self.group.create_dataset(
            'index', shape=(0, 4), maxshape=(None, 4),
            chunks=(32, 4), dtype=np.int64,
        )
        self.buffer = np.empty((RGB_BLOCK_FRAMES, *shape), np.uint8)
        self.count = self.used = 0
        self.dtype = np.dtype(np.uint8)

    @property
    def shape(self):
        return (self.count, *self.buffer.shape[1:])

    def append(self, frames):
        frames = np.asarray(frames)
        start = 0
        while start < len(frames):
            length = min(len(frames) - start, RGB_BLOCK_FRAMES - self.used)
            self.buffer[self.used:self.used + length] = frames[start:start + length]
            self.count += length
            self.used += length
            start += length
            if self.used == RGB_BLOCK_FRAMES:
                self.flush()

    def flush(self):
        if not self.used:
            return
        encoded = _encode(self.buffer[:self.used], self.crf)
        offset, block = len(self.encoded), len(self.index)
        self.encoded.resize(offset + len(encoded), axis=0)
        self.encoded[offset:] = np.frombuffer(encoded, np.uint8)
        self.index.resize(block + 1, axis=0)
        self.index[block] = (self.count - self.used, self.used, offset, len(encoded))
        self.group.attrs['num_frames'] = self.count
        self.used = 0


class _RGBSequence:
    """Array-style reads with one decoded block cached per camera."""

    dtype = np.dtype(np.uint8)
    chunks = None

    def __init__(self, group):
        shape = tuple(group.attrs['frame_shape'])
        count = int(group.attrs['num_frames'])
        self.shape = (count, *shape)
        self.group = group
        self.index = group['index'][:]
        self.crf = group.attrs.get('crf', 0)
        self._cached_block = self._cached_frames = None

    def __len__(self):
        return self.shape[0]

    def _block(self, block):
        if block != self._cached_block:
            _, count, offset, size = map(int, self.index[block])
            encoded = self.group['encoded'][offset:offset + size].tobytes()
            raw = _decode(encoded)
            self._cached_frames = np.frombuffer(raw, np.uint8).reshape(count, *self.shape[1:])
            self._cached_block = block
        return self._cached_frames

    def __getitem__(self, selection):
        spatial = ()
        if isinstance(selection, tuple):
            if not selection:
                selection = slice(None)
            else:
                selection, *spatial = selection
        if selection is Ellipsis:
            selection = slice(None)
        scalar = not isinstance(selection, slice)
        if scalar:
            index = operator.index(selection)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError('RGB frame index out of range')
            indices = [index]
        else:
            indices = range(*selection.indices(len(self)))
        result = np.empty((len(indices), *self.shape[1:]), np.uint8)
        for row, index in enumerate(indices):
            block = int(np.searchsorted(self.index[:, 0], index, side='right') - 1)
            result[row] = self._block(block)[index - int(self.index[block, 0])]
        return result[(0 if scalar else slice(None), *spatial)]


def rgb_sequence(node):
    """Read a numeric HDF5 array or a block-encoded RGB sequence."""
    return _RGBSequence(node) if isinstance(node, h5py.Group) else node
