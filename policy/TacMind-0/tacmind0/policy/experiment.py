"""Assemble policy training and serving while restoring checkpoint contracts."""

from dataclasses import dataclass, field
from pathlib import Path

from tacmind0.constants.robot import ActionMode
from tacmind0.model.modeling import TacMind0Config
from tacmind0.runtime.experiment import BaseExperiment
from tacmind0.tac_frs.stream import StreamSessions

from .config import TacMind0DataConfig, TacMind0ModelConfig, TacMind0TrainerConfig
from .contracts import CONTRACT, validate_contract
from .serving import TacMind0InferenceConfig


@dataclass
class TacMind0Experiment(BaseExperiment):
    model_config: TacMind0ModelConfig = field(default_factory=TacMind0ModelConfig)
    data_config: TacMind0DataConfig = field(default_factory=TacMind0DataConfig)
    inference_config: TacMind0InferenceConfig = field(
        default_factory=TacMind0InferenceConfig
    )
    trainer_config: TacMind0TrainerConfig = field(default_factory=TacMind0TrainerConfig)

    def _restore_checkpoint_contract(self, path: str | Path) -> None:
        config = TacMind0Config.from_pretrained(path, local_files_only=True)
        if config.tactile_config is None:
            return
        contract = config.tactile_config
        validate_contract(contract)
        norm = Path(path) / "norm_stats.json"
        if not norm.is_file():
            raise FileNotFoundError("TacMind-0 checkpoint must contain norm_stats.json")
        self.model_config.chunk_size = config.chunk_size
        self.data_config.action_mode = ActionMode(contract["action_mode"])
        self.data_config.norm_file = str(norm)
        self.data_config.add_state = contract["add_state"]
        self.data_config.n_bins = contract["n_bins"]
        self.data_config.state_desc = list(contract["state_desc"])
        self.data_config.image_prompts = list(contract["image_prompts"])
        augmentation = contract.get(
            "visual_augmentation", CONTRACT["visual_augmentation"]
        )
        self.data_config.agentview_augmentation = augmentation["agentview"]
        self.data_config.wrist_augmentation = augmentation["wrist"]
        self.data_config.augmentation_probability = augmentation["probability"]
        self.inference_config.output_action_dim = contract["output_action_dim"]
        self.inference_config.image_prompts = list(contract["image_prompts"])

    def _prepare_training_contract(self) -> dict:
        from transformers.trainer_utils import get_last_checkpoint

        output = Path(self.trainer_config.output_dir)
        latest = get_last_checkpoint(str(output)) if output.is_dir() else None
        if latest is not None:
            # The parent Trainer resumes this same checkpoint. Build the model and
            # dataset from its contract before reading data or loading any weights.
            self.model_config.model_name_or_path = latest
            # Resume the encoder saved with the optimizer, not the external
            # initialization selected for the start of the original run.
            self.model_config.replace_tactile_encoder = False
        self._restore_checkpoint_contract(self.model_config.model_name_or_path)
        contract = {
            "action_mode": self.data_config.action_mode.value,
            "add_state": self.data_config.add_state,
            "n_bins": self.data_config.n_bins,
            "state_desc": list(self.data_config.state_desc),
            "image_prompts": list(self.data_config.image_prompts),
            "visual_augmentation": self.data_config.visual_augmentation_contract(),
        }
        validate_contract({**CONTRACT, **contract})
        return contract

    def _initialize_train(self):
        contract = self._prepare_training_contract()
        super()._initialize_train()
        self.model.config.tactile_config.update(contract)

    def _initialize_inference_runtime(self):
        self._restore_checkpoint_contract(self.model_config.model_name_or_path)
        super()._initialize_inference_runtime()
        path = Path(self.model_config.model_name_or_path)
        if (path / "tac_frs.json").is_file():
            from tacmind0.tac_frs.model import TacFRS

            self.inference_config.tac_frs_runtime = TacFRS.load(
                path, self.inference_config.device
            )
            self.inference_config.tac_frs_sessions = StreamSessions()
