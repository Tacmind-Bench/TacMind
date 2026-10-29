"""Configure datasets and delegate training to official LeRobot."""

import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

from lerobot.configs.train import TrainPipelineConfig
from policy.diffusion.ema import EMAConfig


@dataclass
class TacMindTrainPipelineConfig(TrainPipelineConfig):
    """Use the official checkpoint writer to persist the data actually trained on."""
    training_provenance: dict | None = None
    ema: EMAConfig | None = None

    def validate(self):
        super().validate()
        _prepare_training(self, validated=True)
        if self.policy.type == "diffusion":
            self.ema = self.ema or EMAConfig()


def _checkpoint_for_config(cfg):
    """Inspect the same checkpoint path the official validator will select."""
    from lerobot.configs import parser

    if path := parser.get_path_arg("policy"):
        return Path(path)
    if cfg.resume and (path := parser.parse_arg("config_path")):
        return Path(path).parent
    return getattr(cfg.policy, "pretrained_path", None)


def _prepare_training(cfg, *, validated=False):
    """Resolve dataset paths and model inputs after CLI/resume overrides."""
    from lerobot.configs import parser
    from benchmark.learning.config import ROOT, policy_input_features
    from benchmark.learning.provenance import training_provenance

    checkpoint = getattr(cfg.policy, "pretrained_path", None) if validated else _checkpoint_for_config(cfg)
    policy = cfg.policy
    if not validated and parser.get_path_arg("policy"):
        from lerobot.configs.policies import PreTrainedConfig

        policy = PreTrainedConfig.from_pretrained(
            checkpoint, cli_overrides=parser.get_cli_overrides("policy"))
    inputs = getattr(policy, "input_features", {}) or {}
    if root := getattr(cfg.dataset, "root", None):
        root = Path(root).expanduser()
        root = (root if root.is_absolute() else ROOT / root).resolve()
        cfg.dataset.root = str(root)
        metadata = root / "conversion_metadata.json"
        info = root / "meta/info.json"
        if info.is_file():
            features = json.loads(info.read_text())["features"]
            expected_inputs = policy_input_features(features)
            if not inputs:
                cfg.policy.input_features = expected_inputs
        if metadata.is_file():
            cfg.training_provenance = training_provenance(
                root, getattr(cfg.dataset, "episodes", None))
    return cfg


def episode_sampler(starts, ends, episode_indices_to_use=None, **options):
    """Use local frame offsets after LeRobot selects an episode subset."""
    from lerobot.datasets.sampler import EpisodeAwareSampler

    if episode_indices_to_use is None:
        return EpisodeAwareSampler(starts, ends, **options)
    selected = set(episode_indices_to_use)
    local_starts, local_ends, offset = [], [], 0
    for episode, (start, end) in enumerate(zip(starts, ends)):
        if episode in selected:
            local_starts.append(offset)
            offset += int(end - start)
            local_ends.append(offset)
    return EpisodeAwareSampler(local_starts, local_ends, **options)


def train(cfg):
    """Use the official loop, optimizer, processors and checkpoint implementation."""
    from lerobot.scripts import lerobot_train

    if is_dataclass(cfg) and not isinstance(cfg, TacMindTrainPipelineConfig):
        cfg = TacMindTrainPipelineConfig(**{
            field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init})
    if not isinstance(cfg, TacMindTrainPipelineConfig):
        cfg = _prepare_training(cfg)
    # LeRobot's CLI decorator accepts only an exact TrainPipelineConfig type;
    # our subclass is already parsed and must reach its unchanged training loop.
    trainer = getattr(lerobot_train.train, "__wrapped__", lerobot_train.train)
    from policy.diffusion.ema import training_ema
    original_sampler = lerobot_train.EpisodeAwareSampler
    lerobot_train.EpisodeAwareSampler = episode_sampler
    try:
        with training_ema(lerobot_train, cfg):
            return trainer(cfg)
    finally:
        lerobot_train.EpisodeAwareSampler = original_sampler


def main():
    import lerobot.policies.act  # noqa: F401
    import lerobot.policies.diffusion  # noqa: F401
    from lerobot.configs import parser
    @parser.wrap()
    def dispatch(cfg: TacMindTrainPipelineConfig):
        return train(cfg)

    dispatch()


if __name__ == "__main__":
    main()
