"""Bind native checkpoints to the data and adapter settings used during optimization."""
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def atomic_checkpoint(path):
    """Publish a new checkpoint directory only after all its files are written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with TemporaryDirectory(dir=path.parent, prefix=f".{path.name}-") as temporary:
        staged = Path(temporary) / "checkpoint"
        staged.mkdir()
        yield staged
        staged.rename(path)


def update_best(checkpoint):
    """Atomically point best at a completed sibling checkpoint using a relative link."""
    checkpoint = Path(checkpoint)
    with TemporaryDirectory(dir=checkpoint.parent, prefix=".best-") as temporary:
        link = Path(temporary) / "best"
        link.symlink_to(checkpoint.name, target_is_directory=True)
        link.replace(checkpoint.parent / "best")


def bind_checkpoint(path, cfg, manifest):
    from benchmark.io import write_json
    path = Path(path)
    if cfg["policy_type"] == "tacmind0":
        from policy.tacmind0.representation import validate, write_contract
        validate(manifest, cfg["prepared_root"])
        write_contract(path)
    cfg = {**cfg, "prompt": cfg.get("prompt") or manifest["episodes"][0]["prompts"][0]}
    write_json(path / "train_config.json", {
        "policy": {"type": cfg["policy_type"]}, "dataset": cfg["dataset"],
        "training_provenance": manifest["training_provenance"],
    })
    write_json(path / "tacmind_policy.json", {"config": cfg})


def load_checkpoint(path):
    saved = json.loads((Path(path) / "tacmind_policy.json").read_text())
    if saved["config"]["policy_type"] == "tacmind0":
        from policy.tacmind0.representation import require_weights
        require_weights(path)
    return saved
