"""Maintain per-tick tactile history and invoke the native VTLA inference service."""

from policy.tacmind0.data import native_state
from policy.tacmind0.experiment import build_experiment


class TacMind0Inference:
    def __init__(self, cfg, checkpoint):
        from policy.tacmind0.representation import require_weights
        require_weights(checkpoint)
        self.exp = build_experiment(cfg, checkpoint)
        self.exp._initialize_inference_runtime()
        self.service = self.exp.inference_config
        self.service.model.eval()
        self.reset()

    def reset(self):
        from tacmind0.tactile.preprocessing import MarkerHistory
        from tacmind0.tac_frs.stream import PhaseHistory

        self.frs = getattr(self.service, "tac_frs_runtime", None)
        self.phase = PhaseHistory() if self.frs is not None else None
        self.history = self.phase.history if self.phase is not None else MarkerHistory()
        self.step = 0

    def observe(self, sample):
        left, right = sample["deformation_left"], sample["deformation_right"]
        if self.phase is not None:
            self.phase.append(self.frs, left, right, self.step)
        else:
            self.history.append(left, right, frame_index=self.step)
        self.step += 1

    def infer(self, sample):
        from PIL import Image
        from tacmind0.constants.robot import RobotStateDesc

        data = dict(
            images=[
                Image.fromarray(sample[view]) for view in ("agentview", "wristview")
            ],
            history_images=[],
            prompt=sample["prompt"],
            state=native_state(sample["state"]),
            tactile_pixel_values=self.history.tensor().unsqueeze(0),
            meta_data=dict(
                robot_type="Franka",
                speed="0.5",
                control_mode=None,
                state_desc=[RobotStateDesc.JOINT] * 7 + [RobotStateDesc.GRIPPER],
            ),
        )
        if self.phase is not None:
            data.update(sampling_mode="tac_frs", tac_frs_latent=self.phase.latent)
        return self.service._predict(data)
