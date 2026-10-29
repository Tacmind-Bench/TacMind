"""Publish a checkpoint only after writing and validation finish successfully."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def staged_checkpoint(destination):
    """Yield a writable directory, then publish it beside its temporary parent."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".checkpoint-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "payload"
        staged.mkdir()
        yield staged
        if destination.exists():
            raise FileExistsError(destination)
        staged.rename(destination)
