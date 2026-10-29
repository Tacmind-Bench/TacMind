"""Official ACT training and temporal evaluation of absolute joint targets."""
from policy.processed import ProcessedPolicy


class TemporalACTPolicy(ProcessedPolicy):
    """Postprocess each chunk to absolute targets before native ACT ensembling."""


    def _select_action(self, batch):
        chunk = self.policy.predict_action_chunk(batch)
        return self.policy.temporal_ensembler.update(self.postprocessor(chunk))
