"""Train and connect tactile adaptation, VTLA policy and optional Tac-FRS stages."""

import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from policy.checkpoint import bind_checkpoint
from policy.tacmind0.data import export_jsonl
from policy.tacmind0.experiment import build_experiment
from policy.tacmind0.representation import require_weights, write_contract


def train_policy(cfg, manifest, data, tactile):
    from transformers import TrainerCallback

    require_weights(tactile)
    base = Path(cfg["pretrained_path"])
    base_config = base / "config.json"
    if base_config.is_file() and json.loads(base_config.read_text()).get("tactile_config") is not None:
        require_weights(base)
    resumes = [path for path in (Path(cfg["output_dir"]) / "policy").glob("checkpoint-*")
               if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()]
    if resumes:
        require_weights(max(resumes, key=lambda path: int(path.name.removeprefix("checkpoint-"))))
    exp = build_experiment(cfg, data=data, tactile=tactile)
    # Compute statistics using the same relative-action transform as the trainer.
    exp._prepare_training_contract()
    if not Path(exp.data_config.norm_file).is_file():
        exp.data_config.compute_norm_stats(
            cfg["chunk_size"], num_workers=cfg["num_workers"]
        )

    def bind(path):
        shutil.copy2(
            exp.data_config.norm_stats_path(cfg["chunk_size"]), path / "norm_stats.json"
        )
        exp.processor.save_pretrained(path)
        bind_checkpoint(path, cfg, manifest)

    class BindingCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                bind(Path(args.output_dir) / f"checkpoint-{state.global_step}")

    initialize = exp._initialize_train

    def initialize_bound():
        initialize()
        exp.trainer.add_callback(BindingCallback())

    exp._initialize_train = initialize_bound
    exp.train()
    result = Path(exp.trainer_config.output_dir)
    bind(result)
    return result


def train_tactile(cfg, data):
    """Adapt the world model on the prepared episodes."""
    from tacmind0.tactile.finetune import run

    model = cfg["model"]
    output = Path(cfg["output_dir"])
    require_weights(model["resources"])
    require_weights(model["world_weights"])

    args = SimpleNamespace(
        jsonl_dir=data,
        image_dir=cfg["prepared_root"],
        output_dir=output / "tactile",
        resources=Path(model["resources"]),
        weights=Path(model["world_weights"]),
        resume=None,
        steps=model["tactile_steps"],
        batch_size=model["tactile_batch_size"],
        workers=cfg["num_workers"],
        marker_cache_size=model["marker_cache_size"],
        lr=model["tactile_lr"],
        sigreg_weight=model["sigreg_weight"],
        train_scope=model["train_scope"],
        freeze_batchnorm=True,
        validation_fraction=model["validation_fraction"],
        validation_batches=model["validation_batches"],
        seed=cfg["seed"],
        device=cfg.get("device", "cuda:0"),
    )
    run(args)
    tactile = args.output_dir / f"checkpoint-{args.steps}"
    write_contract(tactile)
    return tactile


def train_frs(cfg, manifest, data, tactile, policy):
    """Train Tac-FRS against the selected policy and its matching world model."""
    from tacmind0.tac_frs import training

    require_weights(tactile)
    require_weights(policy)
    model = cfg["model"]
    output = Path(cfg["output_dir"])

    args = SimpleNamespace(
        policy=policy,
        world_model=tactile,
        jsonl_dir=data,
        image_dir=Path(cfg["prepared_root"]),
        output_dir=output / "frs",
        resume=None,
        steps=model["frs_steps"],
        batch_size=model["frs_batch_size"],
        diffusion_steps=model["diffusion_steps"],
        lr=model["frs_lr"],
        sigreg_weight=model["sigreg_weight"],
        seed=cfg["seed"],
        device=cfg.get("device", "cuda:0"),
        num_workers=cfg["num_workers"],
        checkpoint_steps=True,
        action_attention_backend=None,
        save_every=cfg["save_freq"],
        feature_cache_dir=None,
        precompute_tactile_cache=False,
    )
    # Bind each published checkpoint, including interrupted runs with a saved step.
    training.run(
        args, checkpoint_callback=lambda path: bind_checkpoint(path, cfg, manifest)
    )
    return args.output_dir / f"checkpoint-{args.steps}"


def train(cfg, manifest):
    """Execute the selected stages and record each completed checkpoint."""
    from transformers import set_seed

    model = cfg["model"]
    set_seed(cfg["seed"])
    os.environ.setdefault("WORLD_SIZE", "1")
    stage = model["stage"]
    data = export_jsonl(cfg, manifest)
    results = {}

    def record(name, path):
        results[name] = str(path)
        (Path(cfg["output_dir"]) / "stages.json").write_text(
            json.dumps(results, indent=2)
        )

    if stage in ("all", "tactile"):
        tactile = train_tactile(cfg, data)
        record("tactile", tactile)
    else:
        tactile = Path(model["tactile_checkpoint"])

    policy = None
    if stage in ("all", "policy"):
        policy = train_policy(cfg, manifest, data, tactile)
        record("policy", policy)
    if stage == "frs" or (stage == "all" and model["run_frs"]):
        policy = policy or Path(cfg["pretrained_path"])
        record("frs", train_frs(cfg, manifest, data, tactile, policy))
