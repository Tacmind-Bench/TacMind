"""Export LeRobot v3 episodes without depending on each upstream's LeRobot version."""
import json
from bisect import bisect_right
from itertools import accumulate
import operator
from pathlib import Path
from tempfile import TemporaryDirectory

from policy.config import cameras, resolve


def _preparation_settings(cfg):
    dataset = cfg["dataset"]
    settings = dict(policy_type=cfg["policy_type"], prompt=cfg.get("prompt") or None,
                dataset=dict(root=str(resolve(dataset["root"])), repo_id=dataset["repo_id"],
                             episodes=dataset.get("episodes")))
    if cfg["policy_type"] == "tacmind0":
        from policy.tacmind0.representation import TACTILE_INPUT
        settings["tactile_input"] = TACTILE_INPUT
    return settings


def load_prepared(cfg):
    """Require the exported data to match the requested training inputs."""
    path = Path(cfg["prepared_root"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    if cfg["policy_type"] == "tacmind0":
        from policy.tacmind0.representation import validate
        validate(manifest, path)
    expected = _preparation_settings(cfg)
    changed = [key for key in ("policy_type", "prompt")
               if key not in manifest or manifest[key] != expected[key]]
    saved = manifest.get("dataset", {})
    changed.extend(f"dataset.{key}" for key, value in expected["dataset"].items()
                   if key not in saved or saved[key] != value)
    if changed:
        raise ValueError(f"Prepared data does not match {', '.join(changed)}: {path}. "
                         "Set a new prepared_root and prepare the data again.")
    if not manifest.get("episodes"):
        raise ValueError(f"Prepared data contains no episodes: {path}")
    return manifest


def _samples_for_preparation(dataset, views):
    """Decode consecutive frames together using LeRobot's timestamp checks."""
    import torch

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset._ensure_hf_dataset_loaded()
        video_keys = [f"observation.images.{view}" for view in views
                      if f"observation.images.{view}" in dataset.meta.video_keys]
        pending = []

        def decode(rows):
            episode = int(rows[0]["episode_index"])
            timestamps = [row["timestamp"].item() for row in rows]
            videos = dataset._query_videos({key: timestamps for key in video_keys}, episode)
            for index, row in enumerate(rows):
                images = {key: value if len(rows) == 1 else value[index] for key, value in videos.items()}
                task = dataset.meta.tasks.iloc[int(row["task_index"])].name
                yield {**images, **row, "task": task}

        for index in range(len(dataset)):
            row = dataset.hf_dataset[index]
            if pending and (int(row["episode_index"]) != int(pending[0]["episode_index"])
                            or len(pending) == 8):
                yield from decode(pending)
                pending = []
            pending.append(row)
        if pending:
            yield from decode(pending)
    finally:
        torch.set_num_threads(previous_threads)


def prepare(cfg):
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from benchmark.io import write_json
    from benchmark.learning.provenance import training_provenance

    root = Path(cfg["dataset"]["root"])
    if cfg["policy_type"] == "tacmind0":
        from policy.tacmind0.representation import require_dataset
        require_dataset(root)
    target = Path(cfg["prepared_root"])
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise FileExistsError(f"Preparation requires an empty output directory: {target}")
    provenance = training_provenance(root, cfg["dataset"].get("episodes"))
    views = cameras(cfg["policy_type"])
    dataset = LeRobotDataset(cfg["dataset"]["repo_id"], root=root,
                             episodes=provenance["dataset"]["episodes"], video_backend="pyav")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as tmp:
        staged = Path(tmp) / "dataset"
        staged.mkdir()
        records, states, actions, prompts = [], [], [], []
        current = None

        def finish():
            if current is None:
                return
            np.savez(staged / f"episode_{current:06d}.npz", state=np.stack(states), action=np.stack(actions))
            records.append(dict(index=current, length=len(states), prompts=list(prompts)))

        for sample in _samples_for_preparation(dataset, views):
            episode = int(sample["episode_index"])
            if episode != current:
                finish()
                current = episode
                states, actions, prompts = [], [], []
                image_folders = {}
            state = np.asarray(sample["observation.state"], dtype=np.float32)
            action = np.asarray(sample["action"], dtype=np.float32)
            frame = len(states)
            images = {}
            for camera in views:
                pixels = np.asarray(sample[f"observation.images.{camera}"])
                pixels = np.rint(np.clip(pixels.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                images[camera] = np.ascontiguousarray(pixels)
            if cfg["policy_type"] == "tacmind0":
                for side in ("left", "right"):
                    pixels = np.asarray(sample[f"observation.images.deformation_{side}"])
                    images[f"deformation_{side}"] = np.rint(
                        np.clip(pixels.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            for camera, pixels in images.items():
                if camera not in image_folders:
                    folder = staged / "images" / f"{episode:06d}" / camera
                    folder.mkdir(parents=True, exist_ok=True)
                    image_folders[camera] = folder
                Image.fromarray(pixels).save(image_folders[camera] / f"{frame:06d}.png", compress_level=1)
            states.append(state)
            actions.append(action)
            prompt = cfg.get("prompt") or sample.get("task")
            prompts.append(prompt)
        finish()
        manifest = dict(cameras=views, episodes=records, training_provenance=provenance,
                        **_preparation_settings(cfg))
        write_json(staged / "manifest.json", manifest)
        staged.rename(target)
    return target


class EpisodeDataset:
    """Raw absolute targets with clamped chunks and optional episode-zero tactile baselines."""
    BASELINE_CACHE_BYTES = 32 * 1024 * 1024

    def __init__(self, root, horizon, views, *, tactile_baselines=False):
        from collections import OrderedDict
        import numpy as np
        self.root, self.horizon, self.views = Path(root), horizon, views
        self.tactile_baselines = tactile_baselines
        self._baselines, self._baseline_bytes = OrderedDict(), 0
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.arrays = {}
        self.offsets = list(accumulate((row["length"] for row in self.manifest["episodes"]), initial=0))
        self._action_offsets = np.arange(horizon)
        for row in self.manifest["episodes"]:
            with np.load(self.root / f"episode_{row['index']:06d}.npz", allow_pickle=False) as data:
                self.arrays[row["index"]] = {key: data[key] for key in ("state", "action")}

    def __len__(self):
        return self.offsets[-1]

    def locate(self, index):
        """Resolve a frame without storing a Python object for every dataset row."""
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.offsets, index) - 1
        return self.manifest["episodes"][episode], index - self.offsets[episode]

    def _baseline(self, folder, first_frame=None):
        import numpy as np
        from PIL import Image
        if folder in self._baselines:
            self._baselines.move_to_end(folder)
            return self._baselines[folder].copy()
        if first_frame is None:
            with Image.open(folder / "000000.png") as image:
                first_frame = np.array(image)
        if first_frame.nbytes <= self.BASELINE_CACHE_BYTES:
            while self._baseline_bytes + first_frame.nbytes > self.BASELINE_CACHE_BYTES:
                _, removed = self._baselines.popitem(last=False)
                self._baseline_bytes -= removed.nbytes
            self._baselines[folder] = first_frame.copy()
            self._baseline_bytes += first_frame.nbytes
        return first_frame.copy()

    def __getitem__(self, index):
        import numpy as np
        from PIL import Image
        row, frame = self.locate(index)
        episode = row["index"]
        source = self.arrays[episode]["action"][frame:frame + self.horizon]
        if len(source) == self.horizon:
            actions = source.copy()
        else:
            actions = np.empty((self.horizon, *source.shape[1:]), dtype=source.dtype)
            actions[:len(source)] = source
            actions[len(source):] = source[-1]
        data = dict(state=self.arrays[episode]["state"][frame].copy(),
                    actions=actions,
                    is_pad=self._action_offsets >= len(source),
                    prompt=row["prompts"][frame])
        for view in self.views:
            folder = self.root / "images" / f"{episode:06d}" / view
            with Image.open(folder / f"{frame:06d}.png") as image:
                data[view] = np.array(image)
            if self.tactile_baselines and view.startswith("gsmini_"):
                data[view + ".baseline"] = self._baseline(folder, data[view] if frame == 0 else None)
        return data

