"""Contract for an offline-prepared vision + Prefix MLP FP8 generation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PREPARED_FORMAT_VERSION = 1
DEFAULT_RELATIVE_DIR = Path(".opendm/fast_fp8")
READY_NAME = "prepared.json"
DEFAULT_PREFIX_BUCKETS = (576, 704, 768, 896, 1024)


class PreparedArtifactError(ValueError):
    """A ready record exists but is not safe for this deployment."""


@dataclass(frozen=True)
class PreparedFastArtifacts:
    root: Path
    vision_engine: Path
    prefix_mlp_bundle: Path
    prefix_seq_len_buckets: tuple[int, ...]


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(root: Path, path: Path) -> dict:
    root = root.resolve()
    if path.is_symlink():
        raise ValueError(f"Prepared artifact cannot be a symlink: {path}")
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Prepared artifact is outside output directory: {path}"
        ) from exc
    if not path.is_file():
        raise ValueError(f"Prepared artifact must be a regular file: {path}")
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _safe_artifact(root: Path, record: object, label: str) -> Path:
    if not isinstance(record, dict):
        raise PreparedArtifactError(f"prepared {label} record is missing")
    relative = record.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PreparedArtifactError(f"prepared {label} path is invalid")
    path = root / relative
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PreparedArtifactError(
            f"prepared {label} escapes its output directory"
        ) from exc
    if path.is_symlink() or not path.is_file():
        raise PreparedArtifactError(f"prepared {label} is missing or is a symlink")
    expected_bytes = record.get("bytes")
    expected_sha = record.get("sha256")
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise PreparedArtifactError(f"prepared {label} integrity mismatch")
    return path


def default_prepared_dir(checkpoint: str | Path) -> Path:
    return Path(checkpoint).expanduser() / DEFAULT_RELATIVE_DIR


def _effective_buckets(values: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    return (
        DEFAULT_PREFIX_BUCKETS
        if values is None
        else tuple(int(value) for value in values)
    )


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value"):
        return _json_value(value.value)
    return value


def preprocessing_contract(
    *,
    checkpoint: str | Path,
    norm_stats_path: str | Path,
    image_prompts,
    n_bins: int,
    add_state: bool,
    model_max_length: int,
    clip_to_bounds: bool,
    compose_eef_rot: bool,
    default_robot_type,
    default_state_desc,
    default_speed,
    default_control_mode,
) -> dict:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    norm_path = Path(norm_stats_path).expanduser().resolve()
    if not norm_path.is_file():
        raise FileNotFoundError(f"Normalization stats do not exist: {norm_path}")
    processor_names = {
        "added_tokens.json",
        "chat_template.jinja",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
    processor_files = []
    if checkpoint_path.is_dir():
        for path in sorted(checkpoint_path.rglob("*")):
            if path.is_file() and path.name in processor_names:
                processor_files.append(
                    {
                        "path": path.relative_to(checkpoint_path).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return {
        "norm_stats": {
            "bytes": norm_path.stat().st_size,
            "sha256": sha256_file(norm_path),
        },
        "processor_files": processor_files,
        "image_prompts": list(image_prompts),
        "n_bins": int(n_bins),
        "add_state": bool(add_state),
        "model_max_length": int(model_max_length),
        "clip_to_bounds": bool(clip_to_bounds),
        "compose_eef_rot": bool(compose_eef_rot),
        "default_robot_type": _json_value(default_robot_type),
        "default_state_desc": _json_value(default_state_desc),
        "default_speed": str(default_speed),
        "default_control_mode": _json_value(default_control_mode),
    }


def load_prepared_fast_artifacts(
    *,
    checkpoint: str | Path,
    prepared_dir: str | Path | None,
    num_images: int,
    chunk_size: int,
    diffusion_steps: int,
    prefix_capacity: int,
    prefix_seq_len_buckets: list[int] | tuple[int, ...] | None,
    is_history: bool,
    preprocessing: dict,
    verify_build_environment: bool = False,
) -> PreparedFastArtifacts:
    root = (
        Path(prepared_dir).expanduser()
        if prepared_dir is not None
        else default_prepared_dir(checkpoint)
    )
    ready_path = root / READY_NAME
    try:
        ready = json.loads(ready_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise PreparedArtifactError("prepared.json is not valid JSON") from exc
    if (
        not isinstance(ready, dict)
        or ready.get("format_version") != PREPARED_FORMAT_VERSION
    ):
        raise PreparedArtifactError("unsupported prepared.json format")
    expected = {
        "num_images": int(num_images),
        "chunk_size": int(chunk_size),
        "diffusion_steps": int(diffusion_steps),
        "prefix_capacity": int(prefix_capacity),
        "history": bool(is_history),
    }
    actual = ready.get("deployment")
    comparable = (
        {key: actual.get(key) for key in expected}
        if isinstance(actual, dict)
        else actual
    )
    if comparable != expected:
        raise PreparedArtifactError(
            f"deployment identity mismatch: expected {expected}, prepared {actual}"
        )
    actual_buckets = actual.get("prefix_seq_len_buckets")
    if (
        not isinstance(actual_buckets, list)
        or not actual_buckets
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in actual_buckets
        )
        or actual_buckets != sorted(set(actual_buckets))
        or any(not 0 < value <= prefix_capacity for value in actual_buckets)
    ):
        raise PreparedArtifactError("prepared prefix bucket identity is invalid")
    if prefix_seq_len_buckets is not None and actual.get(
        "prefix_seq_len_buckets"
    ) != list(_effective_buckets(prefix_seq_len_buckets)):
        raise PreparedArtifactError("explicit prefix bucket identity mismatch")
    if ready.get("preprocessing") != preprocessing:
        raise PreparedArtifactError("preprocessing identity mismatch")
    artifacts = ready.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PreparedArtifactError("prepared artifact records are missing")
    vision = _safe_artifact(root, artifacts.get("vision_engine"), "vision engine")
    prefix_manifest = _safe_artifact(
        root, artifacts.get("prefix_manifest"), "Prefix MLP manifest"
    )
    try:
        manifest = json.loads(prefix_manifest.read_text())
        from opendm.infer.prefix_mlp_fp8_contract import read_verified_file

        read_verified_file(
            prefix_manifest.parent / manifest["calibration"]["file"],
            manifest["calibration"],
        )
        engines = manifest["engines"]
        if not isinstance(engines, dict) or not engines:
            raise ValueError("FP8 bundle contains no layer engines")
        for record in engines.values():
            read_verified_file(prefix_manifest.parent / record["engine"], record)
    except (
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        raise PreparedArtifactError(
            f"prepared FP8 bundle is incomplete: {exc}"
        ) from exc
    if verify_build_environment:
        from opendm.infer.prefix_mlp_fp8_contract import (
            current_build_metadata,
            validate_build_metadata,
        )

        try:
            validate_build_metadata(
                manifest.get("build_metadata"),
                current_build_metadata(device=int(os.environ.get("LOCAL_RANK", "0"))),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise PreparedArtifactError(
                f"prepared FP8 build environment mismatch: {exc}"
            ) from exc
    bundle = prefix_manifest.parent
    return PreparedFastArtifacts(
        root=root,
        vision_engine=vision,
        prefix_mlp_bundle=bundle,
        prefix_seq_len_buckets=tuple(actual_buckets),
    )


def resolve_prepared_fast_artifacts(**kwargs) -> PreparedFastArtifacts | None:
    checkpoint = kwargs.get("checkpoint")
    prepared_dir = kwargs.get("prepared_dir")
    root = (
        Path(prepared_dir).expanduser()
        if prepared_dir is not None
        else default_prepared_dir(checkpoint)
    )
    if not (root / READY_NAME).exists():
        return None
    try:
        prepared = load_prepared_fast_artifacts(**kwargs)
    except PreparedArtifactError as exc:
        logger.warning(
            "Ignoring incompatible prepared FP8 artifacts at %s and using the normal "
            "BF16 fast path: %s",
            root,
            exc,
        )
        return None
    logger.info("Using offline-prepared fast artifacts from %s", root)
    return prepared
