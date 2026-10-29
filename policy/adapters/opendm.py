"""Explicit target JSONL conversion and native OpenDM relative-action training."""
import json
import os
from pathlib import Path
import shutil
import sys
from collections import OrderedDict
from functools import partial

import numpy as np

from policy.adapters import temporary_attribute
from policy.checkpoint import bind_checkpoint


class CachedJsonlDataset:
    """Bound episode text caching while retaining the native sample pipeline."""
    CACHE_BYTES = 64 * 1024 * 1024

    def __init__(self, *args, **kwargs):
        from opendm.data.dataset import JsonlDataset
        self.dataset = JsonlDataset(*args, **kwargs)
        self._lines = OrderedDict()
        self._bytes = 0

    def __len__(self):
        return len(self.dataset)

    def __getstate__(self):
        # Spawn workers with empty caches instead of copying a warmed parent cache.
        return {**self.__dict__, "_lines": OrderedDict(), "_bytes": 0}

    def _read_lines(self, path):
        if path in self._lines:
            self._lines.move_to_end(path)
            return list(self._lines[path][0])
        # TacMind exports immutable, local JSONL files for this training run.
        with open(path) as stream:
            lines = tuple(stream.readlines())
        size = sys.getsizeof(lines) + sum(map(sys.getsizeof, lines))
        if size <= self.CACHE_BYTES:
            while self._bytes + size > self.CACHE_BYTES:
                _, (_, removed_size) = self._lines.popitem(last=False)
                self._bytes -= removed_size
            self._lines[path] = (lines, size)
            self._bytes += size
        return list(lines)

    def __getitem__(self, index):
        from opendm.data import dataset as native
        # Scope the override to this worker's sample. This also works under spawn;
        # the native recursive tail sampling and every transform stay intact.
        with temporary_attribute(native, "_read_jsonl_lines", self._read_lines):
            return self.dataset[index]


def cached_jsonl():
    from opendm.exp import dm05_exp
    return temporary_attribute(dm05_exp, "JsonlDataset", CachedJsonlDataset)


def native_state(state):
    """Represent the parallel gripper by measured aperture (sum of both fingers)."""
    state = np.asarray(state)
    return np.concatenate((state[..., :7], state[..., 7:9].sum(axis=-1, keepdims=True)), axis=-1)


def export_jsonl(cfg, manifest):
    root = Path(cfg["prepared_root"])
    destination = Path(cfg["output_dir"]) / "opendm_data"
    destination.mkdir()
    for row in manifest["episodes"]:
        episode = row["index"]
        with np.load(root / f"episode_{episode:06d}.npz", allow_pickle=False) as data:
            with (destination / f"episode_{episode:06d}.jsonl").open("w") as stream:
                for frame in range(row["length"]):
                    record = dict(state=native_state(data["state"][frame]).tolist(), action=data["action"][frame].tolist(),
                                  prompt=row["prompts"][frame], is_robot=True)
                    for number, camera in enumerate(("agentview", "wristview"), 1):
                        record[f"images_{number}"] = dict(type="image", url=str(
                            root / "images" / f"{episode:06d}" / camera / f"{frame:06d}.png"))
                    stream.write(json.dumps(record) + "\n")
    return destination


def experiment(cfg, checkpoint=None):
    from opendm.constants.robot import RobotStateDesc
    from opendm.dataset.register import register_dataset
    from opendm.exp.dm05_exp import DM05Exp
    desc = [RobotStateDesc.JOINT] * 7 + [RobotStateDesc.GRIPPER]
    register_dataset({"tacmind": dict(jsonl_dir=str(Path(cfg["output_dir"]) / "opendm_data"),
                                    image_dir=cfg["prepared_root"], image_keys=["images_1", "images_2"],
                                    image_prompts=["Agent view", "Wrist view"], state_desc=desc,
                                    robot_type="TacMind Panda")})
    exp = DM05Exp()
    exp.model_config.model_name_or_path = str(checkpoint or cfg["pretrained_path"])
    exp.model_config.chunk_size = cfg["chunk_size"]
    exp.use_lora = cfg["model"].get("use_lora", False)
    exp.data_config.dataset_name = "tacmind"
    exp.data_config.norm_stats_root = str(Path(cfg["output_dir"]) / "normalization")
    exp.inference_config.output_action_dim = 8
    exp.inference_config.image_prompts = ["Agent view", "Wrist view"]
    exp.inference_config.backend = "default"
    exp.optimizer_config.base_lr = cfg["learning_rate"]
    exp.trainer_config.output_dir = str(Path(cfg["output_dir"]) / "checkpoints")
    exp.trainer_config.num_train_steps = cfg["steps"]
    exp.trainer_config.seed = cfg["seed"]
    exp.trainer_config.run_name = Path(cfg["output_dir"]).name
    exp.trainer_config.per_device_train_batch_size = cfg["batch_size"]
    exp.trainer_config.save_steps = cfg["save_freq"]
    exp.trainer_config.dataloader_num_workers = cfg["num_workers"]
    exp.trainer_config.dataloader_persistent_workers = cfg["num_workers"] > 0
    exp.trainer_config.dataloader_prefetch_factor = 2 if cfg["num_workers"] else None
    exp.trainer_config.fsdp1 = False
    return exp


def train(cfg, manifest):
    from transformers import TrainerCallback, set_seed
    set_seed(cfg["seed"])
    os.environ.setdefault("WORLD_SIZE", "1")
    export_jsonl(cfg, manifest)
    exp = experiment(cfg)
    exp.data_config.compute_norm_stats = partial(
        exp.data_config.compute_norm_stats, num_workers=cfg["num_workers"])

    def bind(path):
        shutil.copy2(exp.data_config.norm_stats_path(cfg["chunk_size"]), path / "norm_stats.json")
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

    with temporary_attribute(exp, "_initialize_train", initialize_bound), cached_jsonl():
        exp.train()
    bind(Path(exp.trainer_config.output_dir))


class Inference:
    def __init__(self, cfg, checkpoint):
        self.exp = experiment(cfg, checkpoint)
        self.exp._initialize_inference_runtime()
        self.exp.inference_config.model.eval()

    def infer(self, sample):
        from PIL import Image
        from opendm.constants.robot import RobotStateDesc
        data = dict(images=[Image.fromarray(sample[view]) for view in ("agentview", "wristview")],
                    history_images=[], prompt=sample["prompt"], state=native_state(sample["state"]),
                    meta_data=dict(robot_type="TacMind Panda", speed="0.5", control_mode=None,
                                   state_desc=[RobotStateDesc.JOINT] * 7 + [RobotStateDesc.GRIPPER]))
        return self.exp.inference_config._predict(data)


def load(cfg, checkpoint):
    return Inference(cfg, checkpoint)
