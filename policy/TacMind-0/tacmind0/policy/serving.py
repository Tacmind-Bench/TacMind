"""TacMind-0 inference recipe and Tac-FRS service integration."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from tacmind0.constants.robot import RobotType
from tacmind0.runtime.serving import BaseInferenceConfig
from tacmind0.tac_frs.serving import TacFRSServingMixin
from tacmind0.tac_frs.stream import StreamSessions

from .contracts import CONTRACT
from .protocol import observation_state


@dataclass
class TacMind0InferenceConfig(TacFRSServingMixin, BaseInferenceConfig):
    cuda_graphs: bool = True
    action_backend: Literal["torch", "tensorrt"] = "tensorrt"
    output_action_dim: int = field(default=CONTRACT["output_action_dim"], init=False)
    image_prompts: list[str] = field(
        default_factory=lambda: list(CONTRACT["image_prompts"]), init=False
    )
    port: int = 7897

    def __post_init__(self):
        self.tac_frs_sessions = StreamSessions()

    def _initialize(self, model, *args, **kwargs):
        if self.action_backend not in ("torch", "tensorrt"):
            raise ValueError("action_backend must be torch or tensorrt")
        if type(self.cuda_graphs) is not bool:
            raise ValueError("cuda_graphs must be boolean")
        super()._initialize(model, *args, **kwargs)
        from tacmind0.model.inference_graphs import ActionFlowGraphs

        if self.action_backend == "tensorrt":
            from tacmind0.model.tensorrt_suffix import require_tensorrt

            if (
                self.device.type != "cuda"
                or model.model.action_expert._suffix_attn_backend != "sdpa"
            ):
                raise ValueError(
                    "TensorRT action inference requires CUDA and SDPA; "
                    "set action_backend=torch for CPU or other attention backends"
                )
            require_tensorrt()

        previous = getattr(model, "_action_flow_graphs", None)
        if previous is not None:
            previous.clear()
        model._action_flow_graphs = (
            ActionFlowGraphs(
                model, backend=self.action_backend, cuda_graphs=self.cuda_graphs
            )
            if (self.cuda_graphs or self.action_backend == "tensorrt")
            and self.device.type == "cuda"
            else None
        )

    def _parse_state(self, states) -> np.ndarray:
        return observation_state(
            super()._parse_state(states), field="observation.state"
        )

    def _resolve_state_desc(self, robot_type: str | None):
        if robot_type is not None and robot_type != RobotType.FRANKA.value:
            raise ValueError(
                "TacMind-0 requires the checkpoint's Franka robot contract"
            )
        return super()._resolve_state_desc(robot_type)
