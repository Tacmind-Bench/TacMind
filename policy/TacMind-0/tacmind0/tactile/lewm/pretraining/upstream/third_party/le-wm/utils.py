from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent


def get_artifacts_root() -> Path:
    root = get_repo_root() / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_repo_checkpoints_root() -> Path:
    root = get_artifacts_root() / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_repo_reports_root() -> Path:
    root = get_artifacts_root() / "reports"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_checkpoint_run_dir(run_name: str, run_id: str | None = None) -> Path:
    run_dir = get_repo_checkpoints_root() / run_name
    if run_id:
        run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_report_dir(*parts: str) -> Path:
    out_dir = get_repo_reports_root().joinpath(*parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (get_repo_root() / path).resolve()


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path

    repo_relative = resolve_repo_path(path)
    if repo_relative.exists():
        return repo_relative

    checkpoint_relative = (get_repo_checkpoints_root() / path).resolve()
    return checkpoint_relative


def load_pretrained_from_repo(name: str | Path, extra_args=None):
    name = Path(name) if not isinstance(name, Path) else name
    if name.is_absolute():
        return swm.wm.utils.load_pretrained(str(name), extra_args=extra_args)

    repo_candidate = resolve_repo_path(name)
    if repo_candidate.exists():
        return swm.wm.utils.load_pretrained(str(repo_candidate), extra_args=extra_args)

    return swm.wm.utils.load_pretrained(
        str(name),
        cache_dir=str(get_artifacts_root()),
        extra_args=extra_args,
    )


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1, cache_dir: str | None = None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.cache_dir = cache_dir or str(get_artifacts_root())

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
            cache_dir=self.cache_dir,
        )


def _resolve_dataset_name(name: str) -> str:
    if "://" in name:
        return name
    path = Path(name)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def _normalize_dataset_names(dataset_cfg: dict) -> list[str]:
    dataset_name = dataset_cfg.pop("name", None)
    dataset_names = dataset_cfg.pop("names", None)
    suite_names = dataset_cfg.pop("suites", None)

    if suite_names is not None:
        if dataset_name is None:
            raise ValueError("`dataset.suites` requires a dataset root in `dataset.name`.")
        dataset_root = Path(_resolve_dataset_name(str(dataset_name)))
        return [str((dataset_root / suite_name).resolve()) for suite_name in suite_names]

    if dataset_names is not None:
        return [_resolve_dataset_name(str(name)) for name in dataset_names]

    if dataset_name is None:
        raise ValueError("Dataset config must define either `name`, `names`, or `suites`.")

    return [_resolve_dataset_name(str(dataset_name))]


def build_dataset_from_cfg(dataset_cfg, cache_dir=None):
    dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    dataset_cfg = dict(dataset_cfg)

    dataset_format = dataset_cfg.pop("format", None)
    dataset_names = _normalize_dataset_names(dataset_cfg)

    datasets = [
        swm.data.load_dataset(
            name=name,
            cache_dir=cache_dir,
            format=dataset_format,
            transform=None,
            **dataset_cfg,
        )
        for name in dataset_names
    ]

    if len(datasets) == 1:
        return datasets[0]
    return swm.data.ConcatDataset(datasets)


def get_dataset_tag_from_cfg(dataset_cfg) -> str:
    dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    dataset_cfg = dict(dataset_cfg)

    suite_names = dataset_cfg.get("suites")
    dataset_names = _normalize_dataset_names(dataset_cfg)

    if suite_names is not None:
        root_name = Path(dataset_names[0]).parent.name
        return f"{root_name}_{len(dataset_names)}_suites"

    if len(dataset_names) == 1:
        return dataset_names[0].replace("lerobot://", "").replace("/", "_").replace(".", "_")

    root_name = Path(dataset_names[0]).parent.name
    return f"{root_name}_{len(dataset_names)}_datasets"
