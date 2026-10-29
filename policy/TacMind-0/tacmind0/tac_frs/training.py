"""Second-stage training. Example: python -m tacmind0.tac_frs.training --help."""

import argparse
import fcntl
import json
import os
import random
import shutil
import time
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoProcessor

from tacmind0.constants.robot import ActionMode
from tacmind0.data.episodes import read_episode_rows, split_episodes
from tacmind0.data.normalize import load_norm_stats_file
from tacmind0.data.transforms import LoadImages, PixelTransform
from tacmind0.policy.checkpoints import strict_policy_load
from tacmind0.policy.config import TacMind0DataConfig
from tacmind0.policy.contracts import CONTRACT, validate_visual_augmentation
from tacmind0.runtime.checkpointing import staged_checkpoint
from tacmind0.runtime.distributed import average_gradients
from tacmind0.tactile.lewm.losses import SIGReg
from tacmind0.tactile.preprocessing import MarkerHistory, prepare_frame_pair


from .model import TacFRS, masked_mse
from .random_state import capture_rng, preserve_rng, restore_rng


def initialize(policy_path, world_path, device):
    policy = strict_policy_load(policy_path).to(device).eval()
    if policy.config.tactile_config is None:
        raise ValueError("Tac-FRS requires a trained tactile policy")
    saved = torch.load(world_path / "world_model.pt", map_location="cpu", weights_only=True)
    finetuning = json.loads((world_path / "finetuning.json").read_text())
    spec = json.loads((world_path / "world_model_config.json").read_text())
    contract = policy.config.tactile_config
    stats = load_norm_stats_file(policy_path / "norm_stats.json").select("Franka")["action"]
    bridge = dict(
        normalization="std",
        q01=np.asarray(stats.q01).tolist(),
        q99=np.asarray(stats.q99).tolist(),
        mean=finetuning["action_statistics"]["mean"],
        std=finetuning["action_statistics"]["std"],
        relative_mask=[
            float(contract["action_mode"] == "relative" and x != "gripper")
            for x in contract["state_desc"]
        ],
    )
    config = dict(
        world_model=spec,
        backbone=json.loads((world_path / "backbone_config.json").read_text()),
        action_bridge=bridge,
        layers=len(policy.model.action_expert.layers),
        hidden_size=policy.model.action_expert.config.hidden_size,
        source_policy=str(policy_path.resolve()),
        chunk_size=policy.config.chunk_size,
        vision_residual=False,
    )
    model = TacFRS(config).to(device)
    model.load_world(saved)
    policy.requires_grad_(False)
    return policy, model


def _stack_stream_states(states):
    first = states[0]
    values = dict(
        patch_keys=[
            torch.cat([state.patch_keys[i] for state in states], dim=0)
            for i in range(len(first.patch_keys))
        ],
        patch_values=[
            torch.cat([state.patch_values[i] for state in states], dim=0)
            for i in range(len(first.patch_values))
        ],
        cls_state=torch.cat([state.cls_state for state in states], dim=0),
    )
    for name in ("next_slot", "next_time", "history_size", "patches_per_slot"):
        if hasattr(first, name):
            expected = getattr(first, name)
            if any(getattr(state, name) != expected for state in states):
                raise ValueError(f"stream state {name} is not aligned")
            values[name] = expected
    return type(first)(**values)


def _split_stream_state(state, index):
    values = dict(
        patch_keys=[value[index : index + 1] for value in state.patch_keys],
        patch_values=[value[index : index + 1] for value in state.patch_values],
        cls_state=state.cls_state[index : index + 1],
    )
    for name in ("next_slot", "next_time", "history_size", "patches_per_slot"):
        if hasattr(state, name):
            values[name] = getattr(state, name)
    return type(state)(**values)


class _SequentialVideo:
    """Decode one marker video in increasing frame order without repeated seeks."""

    def __init__(self, path):
        import av
        import megfile

        self.path = path
        self._handle = megfile.smart_open(path, mode="rb")
        self._container = av.open(self._handle)
        self._stream = self._container.streams.video[0]
        self._stream.codec_context.thread_count = 1
        if self._stream.average_rate is None or self._stream.time_base is None:
            self.close()
            raise RuntimeError("Video must expose average_rate and time_base")
        self._fps = float(self._stream.average_rate)
        self._time_base = float(self._stream.time_base)
        self._frames = iter(self._container.decode(self._stream))
        self._next_index = 0

    def frame(self, frame_index):
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("video frame_idx must be a nonnegative integer")
        if frame_index < self._next_index:
            self.close()
            self.__init__(self.path)
        while True:
            try:
                frame = next(self._frames)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Failed to decode frame_idx={frame_index} from {self.path}"
                ) from exc
            if frame.pts is None:
                continue
            current = int(frame.pts * self._time_base * self._fps + 0.5)
            self._next_index = current + 1
            if current == frame_index:
                from PIL import Image

                return Image.fromarray(frame.to_ndarray(format="rgb24")).convert("RGB")
            if current > frame_index:
                self.close()
                self.__init__(self.path)
                raise RuntimeError(
                    f"Video frame order skipped frame_idx={frame_index} in {self.path}"
                )

    def close(self):
        container = getattr(self, "_container", None)
        handle = getattr(self, "_handle", None)
        if container is not None:
            container.close()
        if handle is not None:
            handle.close()
        self._container = None
        self._handle = None


class _SequentialEpisodeLoader:
    """Load an episode's marker frames while keeping each MP4 decoder open."""

    def __init__(self, image_dir):
        self.image_dir = str(image_dir)
        self.image_loaders = {
            key: LoadImages([key], self.image_dir, require_rgb=True)
            for key in ("marker_left", "marker_right")
        }
        self.readers = {}

    def reset(self):
        self.close()
        self.readers = {}

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def __call__(self, row):
        import os

        images = []
        for key in ("marker_left", "marker_right"):
            spec = row[key]
            if spec["type"] != "video":
                images.extend(self.image_loaders[key]({key: spec})["images"])
                continue
            path = os.path.join(self.image_dir, spec["url"])
            reader = self.readers.get(path)
            if reader is None:
                reader = _SequentialVideo(path)
                self.readers[path] = reader
            images.append(reader.frame(spec["frame_idx"]))
        return {"images": images}


class EpisodeFeatures:
    """Frozen Tac-LeWM latents with a shared, atomic per-episode cache.

    A stream state is needed only while an episode is being encoded.  The
    completed latent sequence is stored on CPU and optionally persisted so
    four distributed ranks never recompute the same episode.
    """

    def __init__(self, model, image_dir, cache_dir=None, *, max_episodes=32):
        self.model = model
        self.loader = _SequentialEpisodeLoader(image_dir)
        self.entries = OrderedDict()
        self.max_episodes = max_episodes
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, path):
        if self.cache_dir is None:
            return None
        # The caller supplies a fresh directory shared only by this run's ranks.
        source = Path(path).resolve()
        result = self.cache_dir.joinpath(*source.parts[1:]).with_suffix(".pt")
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    def _encode_episode(self, path, rows):
        if not rows:
            raise ValueError("Cannot encode an empty episode")
        encoder = self.model.encoder.encoder
        parameter = next(encoder.parameters())
        marker_history = MarkerHistory()
        phases = []
        values = None
        self.loader.reset()
        try:
            # Decode only the next five phases; retain latents, not episode pixels.
            for start in range(0, len(rows), 5):
                pixels = []
                for index in range(start, min(start + 5, len(rows))):
                    pair = self.loader(dict(rows[index]))["images"]
                    if start == 0:
                        marker_history.append(*pair, frame_index=index)
                        pixels.append(marker_history.tensor())
                    else:
                        pixels.append(prepare_frame_pair(*pair))
                batch = torch.stack(pixels).to(parameter.device, parameter.dtype)
                if start == 0:
                    cls, state = encoder.stream_init(batch)
                else:
                    state = _stack_stream_states(phases[: len(pixels)])
                    cls, state = encoder.stream_step(
                        batch, state, corrected_window=True
                    )
                projected = self.model.projector(cls.float()).detach().cpu()
                if values is None:
                    values = projected.new_empty((len(rows), 1, projected.shape[-1]))
                values[start : start + len(pixels), 0] = projected
                phases = [_split_stream_state(state, i) for i in range(len(pixels))]
        finally:
            self.loader.close()
        return values

    def _load_or_encode(self, path, rows):
        cache_path = self._cache_path(path)
        if cache_path is None:
            return self._encode_episode(path, rows)
        lock_path = cache_path.with_suffix(".lock")
        with lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache_path.exists():
                return torch.load(cache_path, map_location="cpu", weights_only=True)
            values = self._encode_episode(path, rows)
            if tuple(values.shape) != (len(rows), 1, 384):
                raise RuntimeError(f"Unexpected latent shape for {path}: {tuple(values.shape)}")
            temp = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            torch.save(values, temp)
            os.replace(temp, cache_path)
            return values

    def precompute(self, paths, *, rank=0, world_size=1):
        paths = list(paths)
        assigned = paths[rank::world_size]
        for offset, path in enumerate(assigned, 1):
            rows = read_episode_rows(path)
            self._load_or_encode(path, rows)
            if rank == 0 and (offset == 1 or offset % 25 == 0 or offset == len(assigned)):
                print(
                    f"tactile latent cache {offset}/{len(assigned)} episodes",
                    flush=True,
                )

    def through(self, path, end):
        if path in self.entries:
            self.entries.move_to_end(path)
            rows, values = self.entries[path]
        else:
            rows = read_episode_rows(path)
            values = self._load_or_encode(path, rows)
            self.entries[path] = (rows, values)
            while len(self.entries) > self.max_episodes:
                self.entries.popitem(last=False)
        if end < 0:
            raise ValueError("episode end must be nonnegative")
        # Action chunks at an episode tail intentionally request padded future
        # rows; all latent rows are already cached, so no upper-bound error is
        # needed here.
        return rows, values


def world_loss(model, current, actions, targets, valid, regularizer, sigreg_weight):
    prediction = model.rollout(current, actions)
    mse = masked_mse(prediction, targets.detach(), valid)
    # Pool only valid future predictions across batch/time. SIGReg reduces its
    # sample axis (-3 after projection); (1,N,D) makes N that sample axis.
    selected = prediction[valid.squeeze(-1).bool()].float()
    sigreg = regularizer(selected.unsqueeze(0)) if len(selected) else selected.sum()
    return mse + sigreg_weight * sigreg, mse, sigreg


def save_checkpoint(
    path, policy, model, processor, policy_path, optimizer, step, manifest, rank_rngs=None
):
    with staged_checkpoint(path) as temp:
        policy.save_pretrained(temp, safe_serialization=True)
        processor.save_pretrained(temp)
        shutil.copyfile(policy_path / "norm_stats.json", temp / "norm_stats.json")
        model.save(temp)
        torch.save(
            dict(
                optimizer=optimizer.state_dict(),
                step=step,
                rank_rngs=rank_rngs,
                **capture_rng(next(model.parameters()).device),
            ),
            temp / "tac_frs_trainer.pt",
        )
        (temp / "tac_frs_training.json").write_text(json.dumps(manifest, indent=2))


def augmentation_contract(policy):
    value = policy.config.tactile_config.get("visual_augmentation", CONTRACT["visual_augmentation"])
    validate_visual_augmentation(value)
    return dict(value)


def make_data(policy, policy_path, processor, args, *, training=False):
    contract = policy.config.tactile_config
    augmentation = augmentation_contract(policy)
    config = TacMind0DataConfig(
        jsonl_dir=str(args.jsonl_dir),
        image_dir=str(args.image_dir),
        norm_file=str(policy_path / "norm_stats.json"),
        action_mode=ActionMode(contract["action_mode"]),
        state_desc=contract["state_desc"],
        image_prompts=contract["image_prompts"],
        add_state=contract["add_state"],
        n_bins=contract["n_bins"],
        agentview_augmentation=augmentation["agentview"] if training else "none",
        wrist_augmentation=augmentation["wrist"] if training else "none",
        augmentation_probability=augmentation["probability"],
    )
    return config.build_dataset(processor, policy.config.chunk_size)


def seeded_sample(dataset, index, seed):
    # Albumentations owns RNGs independent of numpy/torch; seed each sample/view
    # explicitly so cache warm-up and checkpoint reconstruction cannot alter it.
    if seed is not None:
        for transform in dataset.transforms.transforms:
            if isinstance(transform, PixelTransform):
                for view, pipeline in enumerate(transform.image_pipelines):
                    pipeline.compose.set_random_seed((seed + view) % (2**32))
    return dataset[index]


class PreparedSamples(torch.utils.data.Dataset):
    """CPU-only decoding; spawn workers never own models or CUDA state."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, request):
        index, seed = request
        return index, seeded_sample(self.dataset, index, seed)


class PreparedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __call__(self, samples):
        indices, values = zip(*samples, strict=False)
        return list(indices), self.collator(list(values))


def sample_batches(indices, *, steps, start, batch_size, seed, rank):
    # Worker scheduling must not change sample selection or augmentation.
    for step in range(start + 1, start + steps + 1):
        rng = random.Random(seed + 1000003 * rank + step)
        selected = rng.choices(indices, k=batch_size)
        yield [
            (index, seed + 1000003 * rank + step * 2 * batch_size + 2 * b)
            for b, index in enumerate(selected)
        ]


def prepared_loader(dataset, collator, indices, args, *, start, rank):
    workers = args.num_workers
    kwargs = dict(
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        batch_sampler=sample_batches(
            indices,
            steps=args.steps,
            start=start,
            batch_size=args.batch_size,
            seed=args.seed,
            rank=rank,
        ),
        collate_fn=PreparedCollator(collator),
        generator=torch.Generator().manual_seed(args.seed + rank),
    )
    if workers:
        kwargs.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
            prefetch_factor=2,
        )
    return torch.utils.data.DataLoader(PreparedSamples(dataset), **kwargs)


def make_batch(
    dataset,
    collator,
    indices,
    features,
    model,
    device,
    horizon,
    *,
    augmentation_seed=None,
    prepared=None,
):
    samples = (
        [
            seeded_sample(
                dataset, i, None if augmentation_seed is None else augmentation_seed + 2 * b
            )
            for b, i in enumerate(indices)
        ]
        if prepared is None
        else None
    )
    cpu_batch = collator(samples) if prepared is None else prepared
    batch = {k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()}
    states, latent, actions, targets, valid = [], [], [], [], []
    for index in indices:
        file_id, t = dataset.sample_index[index]
        rows, values = features.through(dataset.id_to_jsonl[file_id], t + horizon)
        states.append(rows[t]["state"])
        latent.append(values[t][0])
        actions.append([rows[min(t + j, len(rows) - 1)]["action"] for j in range(horizon)])
        targets.append(
            torch.stack([values[min(t + j + 1, len(rows) - 1)][0] for j in range(horizon)])
        )
        valid.append([t + j + 1 < len(rows) for j in range(horizon)])
    states = torch.tensor(states, device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.float32)
    actions = model.bridge.normalize_physical(actions)
    targets = torch.stack(targets).to(device)
    valid = torch.tensor(valid, device=device).unsqueeze(-1)
    return batch, states, torch.stack(latent).to(device), actions, targets, valid


def configure_action_attention(policy, requested=None):
    """Preserve checkpoint runtime unless an explicit backend override is given."""
    expert = policy.model.action_expert
    if requested is not None:
        if requested not in {"eager", "sdpa"}:
            raise ValueError("Tac-FRS action attention must be eager or sdpa")
        expert.set_action_attention_backend(requested)
    policy.config.tac_frs_action_attention_backend = expert._suffix_attn_backend
    return expert._suffix_attn_backend


def run(args, *, checkpoint_callback=None):
    if args.num_workers < 0 or args.save_every < 0:
        raise ValueError("num_workers and save_every must be nonnegative")
    if args.steps < 1 or args.batch_size < 1 or args.diffusion_steps < 1:
        raise ValueError("steps, batch_size and diffusion_steps must be positive")
    if args.resume is None and (args.policy is None or args.world_model is None):
        raise ValueError("Initial training requires --policy and --world-model")
    if not np.isfinite(args.sigreg_weight) or args.sigreg_weight <= 0:
        raise ValueError("sigreg_weight must be positive and finite")
    if not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be positive and finite")
    if args.output_dir.exists():
        raise FileExistsError("Choose a new output directory")
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        rank = 0
        device = torch.device(args.device)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    # Identical initialization is essential before averaging rank gradients.
    torch.manual_seed(args.seed)
    source = args.resume or args.policy
    if args.resume:
        policy = strict_policy_load(source).to(device).eval()
        policy.requires_grad_(False)
        model = TacFRS.load(source, device)
    else:
        policy, model = initialize(source, args.world_model, device)
    if distributed:
        for value in model.state_dict().values():
            dist.broadcast(value, src=0)
    action_backend = configure_action_attention(policy, args.action_attention_backend)
    if rank == 0:
        print(json.dumps({"runtime": {"action_attention_backend": action_backend}}), flush=True)
    torch.manual_seed(args.seed + rank)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    train, val = split_episodes(args.jsonl_dir, 0.1, args.seed)
    manifest = dict(
        train_episodes=train,
        validation_episodes=val,
        image_root=str(args.image_dir.resolve()),
        batch_size=args.batch_size,
        diffusion_steps=args.diffusion_steps,
        lr=args.lr,
        seed=args.seed,
        action_weight=1.0,
        world_weight=1.0,
        sigreg_weight=args.sigreg_weight,
        sigreg_target="valid_predicted_future_latents_pooled_batch_time",
        history_stride=5,
        history_size=8,
        visual_augmentation=augmentation_contract(policy),
        world_size=dist.get_world_size() if distributed else 1,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, fused=device.type == "cuda")
    start = 0
    if args.resume:
        state = torch.load(source / "tac_frs_trainer.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        rank_rngs = state.get("rank_rngs")
        if rank_rngs is None:
            restore_rng(state, device)
        elif rank < len(rank_rngs):
            restore_rng(rank_rngs[rank], device)
    dataset, collator = make_data(policy, source, processor, args, training=True)
    validation_dataset, validation_collator = make_data(policy, source, processor, args)
    train_indices = [
        i for i, (f, _) in enumerate(dataset.sample_index) if dataset.id_to_jsonl[f] in train
    ]
    val_indices = [
        i for i, (f, _) in enumerate(dataset.sample_index) if dataset.id_to_jsonl[f] in val
    ]
    if rank == 0:
        args.output_dir.mkdir(parents=True)
    if distributed:
        dist.barrier(device_ids=[local_rank])
    cache_root = args.feature_cache_dir or args.output_dir / "tactile_latent_cache"
    cache_directory = [None]
    if rank == 0:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_directory[0] = tempfile.mkdtemp(prefix="run-", dir=cache_root)
    if distributed:
        dist.broadcast_object_list(cache_directory, src=0)
    features = EpisodeFeatures(model, str(args.image_dir), cache_directory[0])
    if args.precompute_tactile_cache:
        if rank == 0:
            print(
                f"precomputing tactile latents for {len(train) + len(val)} episodes",
                flush=True,
            )
        features.precompute(
            train + val,
            rank=rank,
            world_size=dist.get_world_size() if distributed else 1,
        )
        if distributed:
            dist.barrier(device_ids=[local_rank])
        if rank == 0:
            print("tactile latent cache ready", flush=True)
    regularizer = SIGReg().to(device)

    def collect_rank_rngs():
        state = capture_rng(device)
        if not distributed:
            return [state]
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, state)
        return states

    loader = iter(prepared_loader(dataset, collator, train_indices, args, start=start, rank=rank))
    records = []

    for step in range(start + 1, start + args.steps + 1):
        started = time.perf_counter()
        indices, prepared = next(loader)
        batch, states, latent, actions, targets, valid = make_batch(
            dataset,
            collator,
            indices,
            features,
            model,
            device,
            policy.config.chunk_size,
            prepared=prepared,
        )
        data_done = time.perf_counter()
        inputs = {k: v for k, v in batch.items() if k != "action"}
        inputs["diffusion_steps"] = args.diffusion_steps
        optimizer.zero_grad(set_to_none=True)
        generated, _ = model.generate(
            policy, inputs, latent, states, checkpoint_steps=args.checkpoint_steps
        )
        # The current action exists even at the terminal row. Later actions
        # exist exactly when the preceding predicted observation is valid.
        action_valid = torch.cat((torch.ones_like(valid[:, :1]), valid[:, :-1]), dim=1)
        action_loss = masked_mse(
            generated, batch["action"], action_valid & batch["action_mask"].bool()
        )
        wm_loss, future_mse, sigreg = world_loss(
            model, latent, actions, targets, valid, regularizer, args.sigreg_weight
        )
        loss = action_loss + wm_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite Tac-FRS loss")
        loss.backward()
        # Explicit reduction preserves access to predictor internals and state keys.
        if distributed:
            average_gradients(trainable)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        record = dict(
            sigreg=float(sigreg.detach()),
            future_latent_mse=float(future_mse.detach()),
            step=step,
            loss=float(loss.detach()),
            action_loss=float(action_loss.detach()),
            world_loss=float(wm_loss.detach()),
            gradient_norm=float(norm),
        )
        finished = time.perf_counter()
        record.update(step_seconds=finished - started, data_seconds=data_done - started)
        records.append(record)
        if rank == 0:
            print(json.dumps(record), flush=True)
        if args.save_every and step % args.save_every == 0 and step != start + args.steps:
            rank_rngs = collect_rank_rngs()
            if distributed:
                dist.barrier(device_ids=[local_rank])
            if rank == 0:
                save_checkpoint(
                    args.output_dir / f"checkpoint-{step}",
                    policy,
                    model,
                    processor,
                    source,
                    optimizer,
                    step,
                    manifest,
                    rank_rngs,
                )
                if checkpoint_callback is not None:
                    checkpoint_callback(args.output_dir / f"checkpoint-{step}")
            if distributed:
                dist.barrier(device_ids=[local_rank])
    if distributed:
        dist.barrier(device_ids=[local_rank])
    with preserve_rng(device), torch.no_grad():
        batch, states, latent, actions, targets, valid = make_batch(
            validation_dataset,
            validation_collator,
            val_indices[: args.batch_size],
            features,
            model,
            device,
            policy.config.chunk_size,
        )
        validation, validation_mse, validation_sigreg = world_loss(
            model, latent, actions, targets, valid, regularizer, args.sigreg_weight
        )
    rank_rngs = collect_rank_rngs()
    if rank == 0:
        checkpoint = args.output_dir / f"checkpoint-{start + args.steps}"
        save_checkpoint(
            checkpoint,
            policy,
            model,
            processor,
            source,
            optimizer,
            start + args.steps,
            manifest,
            rank_rngs,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(checkpoint)
    report = dict(
        performance=dict(
            action_attention_backend=action_backend,
            num_workers=args.num_workers,
            checkpoint_steps=args.checkpoint_steps,
            save_every=args.save_every,
        ),
        training=records,
        validation_world_loss=float(validation),
        validation_future_latent_mse=float(validation_mse),
        validation_sigreg=float(validation_sigreg),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    )
    if rank == 0:
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    if distributed:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--world-model", type=Path)
    parser.add_argument("--jsonl-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--action-attention-backend", choices=("eager", "sdpa"), default=None,
        help="Override action attention; otherwise restore the checkpoint backend.",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument(
        "--precompute-tactile-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
