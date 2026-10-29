"""FOTS marker-motion configuration adapted from UniVTAC's TacEx package.

Source: https://github.com/univtac/UniVTAC/tree/0dafa10262e22f486f160a55d6f11aeab12d8e7b/third_party/TacEx
License: Apache-2.0 (see benchmark/backends/gsmini/tacex/LICENSE.Tabero).
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .fots_marker_sim import FOTSMarkerSimulator


@configclass
class FOTSMarkerSimulatorCfg(GelSightSimulatorCfg):
    """Estimate marker motion from Taxim's depth-based gel deformation."""

    simulation_approach_class: type = FOTSMarkerSimulator
    tactile_img_res: tuple[int, int] = (320, 240)
    mm_to_pixel: float = 19.58
    lamb: list[float] = [0.00125, 0.00021, 0.00038]
    motion_source: Literal["contact_centroid", "target_pose"] = "contact_centroid"
    """Pose mode requires an image-aligned source frame and a contact-local target."""

    @configclass
    class MarkerParams:
        num_markers_col: int = 11
        num_markers_row: int = 9
        num_markers: int = 99
        x0: float = 15.0
        y0: float = 26.0

    marker_params: MarkerParams = MarkerParams()
    frame_transformer_cfg: FrameTransformerCfg = MISSING
