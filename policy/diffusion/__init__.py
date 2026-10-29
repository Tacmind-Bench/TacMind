"""Diffusion Policy rollout adapters and the shared training entry."""
from policy.processed import ProcessedPolicy


class ChunkDiffusionPolicy(ProcessedPolicy):
    """Execute a native DP prefix using its saved action processors."""

    @property
    def needs_observation(self):
        """Whether this tick belongs to the next prediction's observation window."""
        from lerobot.utils.constants import ACTION

        return len(self.policy._queues[ACTION]) < self.policy.config.n_obs_steps

    def _select_action(self, batch):
        from lerobot.utils.constants import ACTION

        action = self.policy._queues[ACTION].popleft() if batch is None else self.policy.select_action(batch)
        return self.postprocessor(action)


class TemporalDiffusionPolicy(ProcessedPolicy):
    """Replan every tick and aggregate absolute targets across overlapping DP chunks."""

    def __init__(self, policy, preprocessor, postprocessor, *, coefficient: float = 0.01):
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

        # DP chunks start at the oldest observation; actions from the current tick onward remain.
        self.start = policy.config.n_obs_steps - 1
        self.coefficient = coefficient
        self.ensemble = ACTTemporalEnsembler(coefficient, policy.config.horizon - self.start)
        super().__init__(policy, preprocessor, postprocessor)

    def reset(self):
        super().reset()
        self.ensemble.reset()

    def _select_action(self, batch):
        import torch
        from lerobot.policies.utils import populate_queues
        from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

        if self.policy.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.policy.config.image_features], dim=-4)
        self.policy._queues = populate_queues(self.policy._queues, batch, exclude_keys=[ACTION])
        queued_batch = {
            key: torch.stack(list(self.policy._queues[key]), dim=1)
            for key in (OBS_STATE, OBS_ENV_STATE, OBS_IMAGES)
            if key in self.policy._queues and len(self.policy._queues[key]) > 0
        }
        global_cond = self.policy.diffusion._prepare_global_conditioning(queued_batch)
        chunk = self.policy.diffusion.conditional_sample(batch_size=1, global_cond=global_cond)
        return self.ensemble.update(self.postprocessor(chunk[:, self.start:]))
