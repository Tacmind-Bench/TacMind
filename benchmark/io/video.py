"""Shared streaming RGB encoder and vision export settings."""

from contextlib import contextmanager, suppress
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryFile

VISION_VIDEO_CRF = 23
VISION_VIDEO_PRESET = "medium"


@contextmanager
def rgb_video(output_path: Path, *, width: int, height: int, fps: int = 20,
              encoder_threads: int = 4, buffered: bool = False,
              crf: int = 18, preset: str = "veryfast"):
    """Stream RGB frames, optionally overlapping drawing with one pending write.

    Buffered writes snapshot the frame before returning. Live capture keeps
    synchronous writes by default; both modes finish or raise before closing.
    """
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A pipe left unread while streaming can fill and block both processes.
    with TemporaryFile() as errors:
        process = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", preset,
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-threads", str(encoder_threads), "-filter_threads", str(encoder_threads),
            str(output_path),
        ], stdin=subprocess.PIPE, stderr=errors)
        count = 0

        def record_frame(frame):
            nonlocal count
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            if frame.shape != (height, width, 3):
                raise ValueError(f"Unexpected RGB frame shape: {frame.shape}; expected {(height, width, 3)}")
            process.stdin.write(memoryview(frame).cast("B"))
            count += 1
            return count

        writer = None
        try:
            if sys.platform == 'linux':
                import fcntl

                # Avoid splitting large RGB frames into many pipe chunks.
                # Restricted hosts retain their native pipe capacity.
                with suppress(OSError):
                    fcntl.fcntl(process.stdin, fcntl.F_SETPIPE_SZ, 1024*1024)
            if buffered:
                writer = ThreadPoolExecutor(max_workers=1)
                pending = None

                def enqueue(frame):
                    nonlocal pending
                    completed = 0 if pending is None else pending.result()
                    # The caller may immediately draw into this array again.
                    snapshot = np.array(frame, dtype=np.uint8, copy=True, order='C')
                    pending = writer.submit(record_frame, snapshot)
                    return completed+1

                yield enqueue
                if pending is not None:
                    pending.result()
            else:
                yield record_frame
            if not count:
                raise ValueError("No RGB frames to encode")
            process.stdin.close()
            status = process.wait()
        except BaseException as exc:
            process.kill()
            process.wait()
            output_path.unlink(missing_ok=True)
            if isinstance(exc, BrokenPipeError):
                errors.seek(0)
                error = errors.read().decode(errors="replace")
                raise RuntimeError(f"ffmpeg input pipe closed: {error}") from exc
            raise
        finally:
            # On interruption, kill/wait above must release a blocked pipe
            # writer before joining it or closing its BufferedWriter lock.
            if writer is not None:
                writer.shutdown(wait=True, cancel_futures=True)
            process.stdin.close()
        errors.seek(0)
        error = errors.read().decode(errors="replace")
    if status:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed ({status}): {error}")
    print(f"[VIDEO] {count} frames at {fps} fps: {output_path}")
