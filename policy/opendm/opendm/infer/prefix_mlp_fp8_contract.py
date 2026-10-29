"""Identity and publication contract for optional DM05 FP8 MLP bundles.

Keep this module importable without CUDA, TensorRT or PyTorch so malformed
manifests can be rejected and publication behavior tested on a CPU host.
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PREFIX_MLP_FORMAT_VERSION = 2
SPEC = "fp8_e4m3_per_tensor_max"
MANIFEST_NAME = "prefix_mlp_quant_manifest.json"

logger = logging.getLogger(__name__)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc8_now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def prefix_mlp_model_contract(language_model) -> dict:
    """Hash the actual loaded MLP tensors, including their numerical dtype.

    Hashing happens once before replacing any MLP. Checkpoint path strings are
    provenance only: merged adapters and in-place weight changes alter identity.
    """
    import torch

    config = language_model.config
    hidden = _positive_int(getattr(config, "hidden_size", None), "hidden_size")
    intermediate = _positive_int(
        getattr(config, "intermediate_size", None), "intermediate_size"
    )
    count = _positive_int(getattr(config, "num_hidden_layers", None), "layer_count")
    if getattr(config, "model_type", None) != "gemma3_text":
        raise ValueError("FP8 Prefix MLP currently supports Gemma3 text MLPs only.")
    if getattr(config, "hidden_activation", None) != "gelu_pytorch_tanh":
        raise ValueError("FP8 Prefix MLP requires gelu_pytorch_tanh activation.")
    if len(language_model.layers) != count:
        raise ValueError("Prefix MLP layer count disagrees with the loaded config.")
    records = {}
    expected_shapes = {
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    for index, layer in enumerate(language_model.layers):
        mlp = layer.mlp
        if (type(mlp).__module__, type(mlp).__name__) != (
            "transformers.models.gemma3.modeling_gemma3",
            "Gemma3MLP",
        ):
            raise ValueError(f"Unsupported Prefix MLP implementation at layer {index}.")
        activation = mlp.act_fn
        tanh_gelu = (type(activation).__module__, type(activation).__name__) == (
            "transformers.activations",
            "PytorchGELUTanh",
        )
        if (type(activation).__module__, type(activation).__name__) == (
            "transformers.activations",
            "GELUTanh",
        ):
            # Transformers 5.3 renamed the module; its default implementation
            # delegates to the same native tanh GELU through a partial.
            implementation = activation.act
            tanh_gelu = (
                isinstance(implementation, functools.partial)
                and implementation.func is torch.nn.functional.gelu
                and not implementation.args
                and implementation.keywords == {"approximate": "tanh"}
            )
        tanh_gelu |= (
            type(activation) is torch.nn.GELU and activation.approximate == "tanh"
        )
        tanh_gelu |= (
            isinstance(activation, functools.partial)
            and activation.func is torch.nn.functional.gelu
            and not activation.args
            and activation.keywords == {"approximate": "tanh"}
        )
        if not tanh_gelu:
            raise ValueError(f"Layer {index} does not execute tanh-approximated GELU.")
        weights = {}
        for name, shape in expected_shapes.items():
            projection = getattr(mlp, name)
            if type(projection) is not torch.nn.Linear or projection.bias is not None:
                raise ValueError(f"Layer {index} {name} must be a bias-free Linear.")
            value = projection.weight.detach()
            if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
                raise ValueError(
                    f"Layer {index} {name} expected BF16 {shape}, "
                    f"got {value.dtype} {tuple(value.shape)}."
                )
            value = value.to(device="cpu").contiguous()
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Layer {index} {name} contains non-finite weights.")
            weights[name] = {
                "shape": list(shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(
                    value.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        records[str(index)] = weights
    contract = {
        "model_type": "gemma3_text",
        "activation": "gelu_pytorch_tanh",
        "bias": False,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "layer_count": count,
        "weights": records,
    }
    contract["sha256"] = hashlib.sha256(canonical_json(contract)).hexdigest()
    return contract


def prefix_source_model_sha256(vlm) -> str:
    """Bind calibration to all loaded upstream weights and semantic config.

    Attention implementation names, loader paths and dtype labels may change
    when attaching the fast runtime; tensor dtypes themselves remain hashed.
    """
    import torch

    def semantic_config(value):
        if isinstance(value, dict):
            return {
                str(key): semantic_config(item)
                for key, item in value.items()
                if not str(key).startswith("_")
                and key
                not in (
                    "transformers_version",
                    "dtype",
                    "torch_dtype",
                    "attn_implementation",
                )
            }
        if isinstance(value, (list, tuple)):
            return [semantic_config(item) for item in value]
        return value

    digest = hashlib.sha256(canonical_json(semantic_config(vlm.config.to_dict())))
    for name, tensor in sorted(vlm.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(canonical_json([name, list(value.shape), str(value.dtype)]))
        # memoryview avoids a second full-size allocation for large embeddings.
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def validate_model_contract(recorded: dict, actual: dict) -> None:
    if not isinstance(recorded, dict) or recorded != actual:
        raise ValueError(
            "FP8 Prefix MLP model contract mismatch: loaded weights, shapes, "
            "dtype or activation differ from the calibrated model. Recalibrate and rebuild."
        )


def validate_file_record(record: dict, *, filename_key: str) -> None:
    if not isinstance(record, dict):
        raise ValueError("Invalid FP8 bundle file record.")
    filename = record.get(filename_key)
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(
            "Bundle records must reference a plain filename in the bundle."
        )
    if filename in (".", ".."):
        raise ValueError("Invalid bundle filename.")
    _positive_int(record.get("bytes"), "file bytes")
    if not isinstance(record.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", record["sha256"]
    ):
        raise ValueError("Bundle record requires a SHA256 digest.")


def read_verified_file(path: str | Path, record: dict) -> bytes:
    """Read once, verify, then hand these exact bytes to the deserializer."""
    path = Path(path)
    validate_file_record(
        record, filename_key="engine" if "engine" in record else "file"
    )
    if path.is_symlink():
        raise ValueError(f"Bundle artifact cannot be a symlink: {path.name}.")
    payload = path.read_bytes()
    if (
        len(payload) != record["bytes"]
        or hashlib.sha256(payload).hexdigest() != record["sha256"]
    ):
        raise ValueError(f"FP8 bundle artifact integrity mismatch: {path.name}.")
    return payload


def read_activation_amax(calibration: dict, layer_idx: int, activation: str) -> float:
    try:
        stats = calibration["layers"][str(layer_idx)][activation]["visible_rows"]
        value = float(stats["max_abs"])
        _positive_int(stats["rows"], "calibration visible rows")
        _positive_int(stats["elements"], "calibration elements")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing valid calibration for layer {layer_idx} {activation}."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Calibration layer {layer_idx} {activation} range must be finite and positive."
        )
    return value


def validate_calibration(
    calibration: dict, model_contract: dict, *, prefix_source_sha256: str
) -> None:
    if (
        not isinstance(calibration, dict)
        or calibration.get("format_version") != PREFIX_MLP_FORMAT_VERSION
    ):
        raise ValueError("Unsupported Prefix MLP calibration format version.")
    if calibration.get("spec") != SPEC or calibration.get("batch_size") != 1:
        raise ValueError(
            "Prefix MLP calibration requires this FP8 recipe and batch size one."
        )
    validate_model_contract(calibration.get("model_contract"), model_contract)
    validate_file_record(calibration.get("vision_engine"), filename_key="file")
    if (
        _positive_int(calibration.get("distinct_sample_count"), "distinct_sample_count")
        < 2
    ):
        raise ValueError(
            "Calibration requires at least two distinct real input samples."
        )
    if _positive_int(calibration.get("sample_count"), "sample_count") < 2:
        raise ValueError(
            "Calibration requires at least two real transformed input samples."
        )
    if calibration.get("prefix_source_sha256") != prefix_source_sha256:
        raise ValueError("Calibration upstream VLM weight/config identity mismatch.")
    count = model_contract["layer_count"]
    if set(calibration.get("layers", {})) != {str(i) for i in range(count)}:
        raise ValueError("Calibration layer set does not match the model.")
    for index in range(count):
        for activation in ("mlp_input", "down_input"):
            read_activation_amax(calibration, index, activation)


def validate_bundle_manifest(
    manifest: dict, model_contract: dict, seq_lens=None, *, prefix_source_sha256: str
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != PREFIX_MLP_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported Prefix MLP bundle format version; rebuild with the current builder."
        )
    if manifest.get("spec") != SPEC:
        raise ValueError("Unsupported Prefix MLP quantization recipe.")
    validate_model_contract(manifest.get("model_contract"), model_contract)
    if manifest.get("prefix_source_sha256") != prefix_source_sha256:
        raise ValueError("FP8 Prefix MLP upstream VLM weight/config identity mismatch.")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Missing Prefix MLP engine shape contract.")
    lengths = tuple(
        _positive_int(contract.get(f"{key}_seq_len"), f"{key}_seq_len")
        for key in ("min", "opt", "max")
    )
    if lengths != tuple(sorted(lengths)):
        raise ValueError(
            "Prefix MLP engine requires min <= opt <= max sequence length."
        )
    count = model_contract["layer_count"]
    if (
        contract.get("batch_size") != 1
        or contract.get("hidden_size") != model_contract["hidden_size"]
        or contract.get("layer_indices") != list(range(count))
    ):
        raise ValueError(
            "Prefix MLP engine shape contract does not match the loaded model."
        )
    if seq_lens is not None:
        for length in seq_lens:
            _positive_int(length, "requested sequence length")
            if not lengths[0] <= length <= lengths[2]:
                raise ValueError(
                    f"Prefix bucket {length} is outside engine profile {lengths}."
                )
    records = manifest.get("engines")
    if not isinstance(records, dict) or set(records) != {str(i) for i in range(count)}:
        raise ValueError(
            "Prefix MLP bundle must contain every model layer exactly once."
        )
    filenames = set()
    for index in range(count):
        record = records[str(index)]
        validate_file_record(record, filename_key="engine")
        if record["engine"] in filenames:
            raise ValueError("Prefix MLP engine filenames must be unique.")
        filenames.add(record["engine"])
        for key in ("input_range", "down_range"):
            value = record.get(key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Prefix MLP {key} must be finite and positive.")
    validate_file_record(manifest.get("calibration"), filename_key="file")
    if not isinstance(manifest.get("build_metadata"), dict):
        raise ValueError("Missing FP8 build environment metadata.")


def current_build_metadata(device=None) -> dict:
    import torch

    from opendm.infer.dm05_trt_utils import load_tensorrt

    trt = load_tensorrt("FP8 Prefix MLP bundle identity verification")
    properties = torch.cuda.get_device_properties(device)
    return {
        "tensorrt": trt.__version__,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "gpu_multiprocessors": properties.multi_processor_count,
    }


def validate_build_metadata(recorded: dict, actual: dict) -> None:
    for key in (
        "tensorrt",
        "cuda",
        "gpu_name",
        "compute_capability",
        "gpu_multiprocessors",
    ):
        if key not in recorded or recorded[key] != actual.get(key):
            raise ValueError(
                f"FP8 Prefix MLP build/runtime {key} mismatch; rebuild for this environment."
            )


@contextmanager
def bundle_lock(engine_dir: str | Path, *, exclusive: bool = False):
    """Coordinate complete bundle publication with readers loading all layers."""
    directory = Path(engine_dir).expanduser()
    if directory.is_symlink():
        raise ValueError("FP8 bundle directory cannot be a symlink.")
    directory = directory.resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = directory.parent / f".{directory.name}.lock"
    try:
        handle = lock_path.open("r+b" if exclusive else "rb")
    except FileNotFoundError:
        try:
            handle = lock_path.open("a+b")
        except OSError as exc:
            raise OSError(
                exc.errno,
                "Missing FP8 bundle lock cannot be created; keep the bundle and "
                "its sibling lock together for read-only deployment.",
                str(lock_path),
            ) from exc
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cleanup_bundle_tree(path: str | Path, *, purpose: str) -> None:
    """Ignore ordinary filesystem cleanup errors without changing success."""
    path = Path(path)
    try:
        if not path.exists():
            return
        shutil.rmtree(path)
    except OSError as exc:
        logger.warning("Failed to remove %s %s: %s", purpose, path, exc)


def _recovery_path_exists(path: Path, primary: BaseException) -> bool | None:
    """Probe recovery state without replacing the operation's primary error."""
    try:
        return path.exists()
    except OSError as probe_error:
        logger.error("Failed to inspect bundle recovery path %s: %s", path, probe_error)
        if hasattr(primary, "add_note"):
            primary.add_note(
                f"Could not inspect bundle recovery path {path}: {probe_error!r}"
            )
        return None


def publish_bundle_generation(
    staging: str | Path,
    engine_dir: str | Path,
    backup: str | Path,
) -> None:
    """Publish a complete directory generation and preserve recoverability.

    A rename that moved ``staging`` into ``engine_dir`` is the commit point,
    even if Python observes an exception immediately after the syscall. Before
    that point, an existing generation is restored from ``backup``. Rollback
    failures retain the backup and annotate, but never replace, the publication
    exception.
    """
    staging = Path(staging)
    engine_dir = Path(engine_dir)
    backup = Path(backup)
    if backup.exists():
        raise FileExistsError(f"Bundle backup path already exists: {backup}")

    try:
        if engine_dir.exists():
            engine_dir.rename(backup)
        staging.rename(engine_dir)
    except BaseException as publish_error:
        # os.rename is atomic, but an injected exception or asynchronous
        # interruption can be observed just after it succeeds. The new target
        # plus an absent staging directory proves that publication committed.
        engine_exists = _recovery_path_exists(engine_dir, publish_error)
        staging_exists = _recovery_path_exists(staging, publish_error)
        if engine_exists is True and staging_exists is False:
            logger.warning(
                "Bundle rename reported %s after publication committed at %s",
                type(publish_error).__name__,
                engine_dir,
            )
            raise
        else:
            backup_exists = _recovery_path_exists(backup, publish_error)
            if backup_exists is True and engine_exists is False:
                try:
                    backup.rename(engine_dir)
                except BaseException as rollback_error:
                    logger.error(
                        "Failed to restore bundle %s from backup %s: %s",
                        engine_dir,
                        backup,
                        rollback_error,
                    )
                    if hasattr(publish_error, "add_note"):
                        publish_error.add_note(
                            "Bundle rollback reported an error; inspect both "
                            f"{engine_dir} and {backup}. Neither path is cleaned "
                            f"by this failure handler: {rollback_error!r}"
                        )
            raise

    # Once the commit point is reached, failure to retire the old generation
    # must not turn a usable newly published bundle into a reported failure.
    cleanup_bundle_tree(backup, purpose="retired bundle backup")
