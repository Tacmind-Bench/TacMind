"""Shared file output, recording image storage and video encoding."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_json(path: Path, value: dict) -> None:
    """Replace complete JSON atomically, using a private temporary file per writer."""
    contents = json.dumps(value, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            prefix=f".{path.name}-", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(contents)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
