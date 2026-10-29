"""Shared inference lifecycle for policies using LeRobot processors."""


class ProcessedPolicy:
    needs_observation = True

    def __init__(self, policy, preprocessor, postprocessor):
        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor
        self.reset()

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def select_action(self, observation):
        import torch

        with torch.inference_mode():
            batch = None
            if self.needs_observation:
                if getattr(self.policy, "training", True):
                    self.policy.eval()
                batch = self.preprocessor(observation)
            return self._select_action(batch)

    def _select_action(self, batch):
        """Return absolute targets; batch is None when consuming a cached action."""
        raise NotImplementedError
