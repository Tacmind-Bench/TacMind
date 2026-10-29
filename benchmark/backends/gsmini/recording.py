"""Episode recording and successful HDF5 publication for expert collectors."""

from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

from benchmark.contracts import VISION_CAMERAS, camera_image

EPISODE_COMPRESSION = "zstd"
_VISION_CRF = ContextVar("vision_crf", default=0)


@contextmanager
def vision_compression(crf):
    """Scope visual compression to this collector; tactile streams remain lossless."""
    token = _VISION_CRF.set(crf)
    try:
        yield
    finally:
        _VISION_CRF.reset(token)


class _AsyncDatasetWriter:
    """Overlap recording writes with simulation, keeping at most one queued chunk."""

    def __init__(self, handler):
        from concurrent.futures import ThreadPoolExecutor

        self._handler = handler
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="episode-writer")
        self._pending = None
        self._closed = False

    def _wait(self):
        if self._pending is not None:
            try:
                self._pending.result()
            finally:
                self._pending = None

    def append_data(self, episode):
        import copy
        import torch

        self._wait()
        # Transfer on the producing stream; the worker owns a stacked CPU chunk.
        streams = {}

        def to_cpu(value):
            if isinstance(value, dict):
                return {key: to_cpu(item) for key, item in value.items()}
            if value.is_cuda:
                streams[value.device] = torch.cuda.current_stream(value.device)
                return torch.empty_like(value, device="cpu", pin_memory=True).copy_(
                    value.detach(), non_blocking=True)
            return value.detach().to("cpu", non_blocking=True)

        chunk = copy.copy(episode)
        chunk.data = to_cpu(episode.data)
        # Let collection continue while the disk worker waits for these copies.
        events = []
        for stream in streams.values():
            event = torch.cuda.Event()
            event.record(stream)
            events.append(event)

        def write():
            for event in events:
                event.synchronize()
            self._handler.append_data(chunk)

        self._pending = self._executor.submit(write)

    def end_episode(self, success):
        self._wait()
        self._handler.end_episode(success=success)

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._wait()
        finally:
            self._executor.shutdown(wait=True)
            self._handler.close()


def _start_episode(resources, env, output: Path, suite: str, task_id: int, seed: int,
                   extra_env_args: dict | None = None, *, task: dict | None = None):
    import tempfile
    from benchmark.catalog import get_task
    from benchmark.backends.gsmini.environment import environment_contract
    from isaaclab.utils.datasets import EpisodeData
    from benchmark.backends.gsmini.streaming_hdf5_handler import StreamingHDF5DatasetFileHandler
    from benchmark.io.image_storage import RGB_STORAGE, RGBWriter, rgb_storage

    vision_crf = _VISION_CRF.get()

    class RGBDatasetHandler(StreamingHDF5DatasetFileHandler):
        """Encode visual and optical tactile RGB; keep the vendored numeric writer."""

        def __init__(self, **options):
            self._rgb_streams = {}
            super().__init__(**options)

        def _append_to_dataset(self, group, key, value, datasets_cache):
            if key in (*VISION_CAMERAS, 'gsmini_left', 'gsmini_right') and group.name.endswith('/obs'):
                values = self._leaf_to_numpy(group, key, value)
                path = group.name + '/' + key
                if path not in self._rgb_streams:
                    self._rgb_streams[path] = RGBWriter(group, key, values.shape[1:],
                                                       crf=vision_crf if key in VISION_CAMERAS else 0)
                self._rgb_streams[path].append(values)
            else:
                super()._append_to_dataset(group,key,value,datasets_cache)

        def end_episode(self, success=None, episode_index=None):
            for stream in self._rgb_streams.values():
                stream.flush()
            self._rgb_streams.clear()
            return super().end_episode(success=success, episode_index=episode_index)

    task = task if task is not None else get_task(suite, task_id)
    suite_name = suite.removeprefix("tac_")
    directory = resources.enter_context(tempfile.TemporaryDirectory(
        prefix=f"{suite_name}{task_id}-", dir=output.parent))
    staged = Path(directory) / "episode.hdf5"
    compression_level = 3
    handler = RGBDatasetHandler(compression_level=compression_level)
    writer = _AsyncDatasetWriter(handler)
    resources.callback(writer.close)
    handler.create(str(staged), env_name=f"TacMind-Tac{suite_name.title()}-{task_id}-GelSightMini")
    env_args = {
        **environment_contract(env, task, suite=suite, backend="gsmini"),
        "observations": "agentview,wristview,gsmini_left,gsmini_right,joint_pos,joint_vel",
    }
    extras = extra_env_args or {}
    env_args.update(extras)
    if "gsmini_left" in env.scene.keys():
        marker_cfg = env.scene["gsmini_left"].cfg.marker_motion_sim_cfg
        if marker_cfg is not None:
            env_args["gsmini_marker_motion"] = {
                "simulator": "FOTS",
                "model_version": "fots-centroid-v2",
                "coordinate_order": "x,y pixels",
                "frame_order": "initial,current",
                "marker_grid": [
                    marker_cfg.marker_params.num_markers_row,
                    marker_cfg.marker_params.num_markers_col,
                ],
                "units": "pixels",
            }
            if getattr(marker_cfg, "motion_source", None) == "target_pose":
                env_args["gsmini_marker_motion"].update({
                    "model_version": "fots-relative-pose-v2",
                    "motion_source": "contact-local tool pose relative to image-aligned gel camera",
                    "pixels_per_mm": marker_cfg.mm_to_pixel,
                    "contact_gate": "Taxim indentation and deformed-gel contact mask",
                    "normal_reference": "positive displacement from Taxim's unloaded curved gel",
                    "calibration": "FOTS approximate deformation; no hardware shear/force calibration",
                    "rgb": "raw Taxim unchanged; marker coordinates are a separate observation",
                })
    if "gsmini_left_contact" in env.scene.keys():
        env_args["gsmini_contact"] = {
            "units": "N", "frame": "world", "components": "normal XYZ and tangent XYZ",
            "tangent_field": "obs/gsmini_gel_target_tangent_force_w",
            "tangent_source": "native gel surface-contact friction forces; configured target filters",
            "tangent_sampling": "current pre-action control boundary; no tangent history",
            "side_order": ["left", "right"],
            "filter_paths": list(env.scene["gsmini_left_contact"].cfg.filter_prim_paths_expr),
            "history_length": env.cfg.decimation, "physics_dt": env.cfg.sim.dt,
            "history_order": "oldest to newest",
            "time_reference": "environment control clock; pre-action",
            "required_for_grasp": False,
        }
    env_args["storage"] = {"compression": EPISODE_COMPRESSION, "compression_level": compression_level,
                           "lossless": vision_crf == 0,
                           "rgb_chunk_shape": [8, 60, 60, 3], "vision": rgb_storage(vision_crf),
                           "tactile_rgb": dict(RGB_STORAGE)}
    handler.add_env_args(env_args)
    env.handler = writer
    env.episode = EpisodeData()
    env.episode.seed = seed
    return staged


def _prepare_output_path(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)


def _record_tactile_observations(env, episode, record_tactile=None):
    """Use a backend recorder or the shared raw GelSight RGB observations."""
    if record_tactile is not None:
        record_tactile(episode)
    else:
        for side in ("left", "right"):
            episode.add(f"obs/gsmini_{side}", camera_image(env, f"gsmini_{side}"))
            sensor = env.scene[f"gsmini_{side}"]
            if sensor.cfg.marker_motion_sim_cfg is None:
                continue
            marker_motion = sensor.data.output["marker_motion"]
            episode.add(
                f"obs/gsmini_{side}_marker_motion",
                marker_motion[0].detach(),
            )


def _record_step(env, action):
    """Store configured pre-action observations and the command actually sent."""
    episode = env.episode
    robot = env.scene["robot"]
    # EpisodeData clones samples; the async writer transfers each chunk on flush.
    episode.add("actions", action[0].detach())
    episode.add("obs/joint_pos", robot.data.joint_pos[0].detach())
    episode.add("obs/joint_vel", robot.data.joint_vel[0].detach())
    for name in VISION_CAMERAS:
        episode.add(f"obs/{name}", camera_image(env, name))
    _record_tactile_observations(env, episode, getattr(env, "record_tactile", None))
    _record_contact_observations(env, episode)


def _record_contact_observations(env, episode):
    """Keep native physical-step histories, without summing candidate filters."""
    import torch

    if "gsmini_left_contact" not in env.scene.keys():
        return
    samples = {}
    for part in ("gel", "case"):
        sensors = [env.scene[f"gsmini_{side}_contact" if part == "gel" else
                             f"gsmini_{side}_case_contact"].data
                   for side in ("left", "right")]
        values = {
            "target_normal_force_w": torch.stack([data.force_matrix_w[0].sum(dim=0)
                                                   for data in sensors]),
            "net_normal_force_w": torch.stack([data.net_forces_w[0].sum(dim=0)
                                                for data in sensors]),
            "target_normal_force_w_history": torch.stack([
                data.force_matrix_w_history[0].sum(dim=1) for data in sensors], dim=1).flip(0),
            "net_normal_force_w_history": torch.stack([
                data.net_forces_w_history[0].sum(dim=1) for data in sensors], dim=1).flip(0),
        }
        for name, value in values.items():
            samples[(part, name)] = value
    surfaces = [env.scene[f"gsmini_{side}_surface_contact"].data for side in ("left", "right")]
    tangent = torch.stack([data.friction_forces_w[0].sum(dim=0) for data in surfaces])
    samples[("gel", "target_tangent_force_w")] = tangent
    for (part, name), value in samples.items():
        episode.add(f"obs/gsmini_{part}_{name}", value.detach())
    length = values["net_normal_force_w_history"].shape[0]
    times = (torch.arange(1 - length, 1, dtype=torch.float64)
             * env.cfg.sim.dt + env.common_step_counter * env.step_dt)
    episode.add("obs/gsmini_contact_time", times)


def _record_step_and_flush(env, action):
    """Record before execution and stream each 64-frame chunk."""
    _record_step(env, action)
    if len(env.episode.data["actions"]) >= 64:
        _flush_episode(env)


def _prepare_recording_step(env, action):
    """Record the command and observations before simulation advances."""
    if env.episode is not None:
        _record_step_and_flush(env, action)


def _flush_episode(env):
    """Bound image-buffer memory while keeping one continuous demonstration."""
    episode = env.episode
    if episode.is_empty():
        return
    episode.pre_export()
    env.handler.append_data(episode)
    episode.data = {}


def _publish_dataset(output, staged, handler):
    """Expose the staged file only after the complete episode succeeds."""
    import os

    handler.end_episode(success=True)
    handler.close()
    os.link(staged, output)


def _finish_episode(env, output, staged):
    """Publish the HDF5 after task success; the public collector then exports tactile data."""
    _flush_episode(env)
    _publish_dataset(output, staged, env.handler)
