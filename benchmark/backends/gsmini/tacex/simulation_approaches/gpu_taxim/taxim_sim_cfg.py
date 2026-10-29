from dataclasses import MISSING

from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .taxim_sim import TaximSimulator
from .sim.calibration import CALIB_GELSIGHT_MINI

# Configuration for a tactile RGB simulation with Taxim.


@configclass
class TaximSimulatorCfg(GelSightSimulatorCfg):
    simulation_approach_class: type = TaximSimulator

    calib_folder_path: str = str(CALIB_GELSIGHT_MINI)
    """Calibration folder; other sensor models must supply their own calibration."""

    backend: str = "auto"
    """Backend selection for Taxim; the Isaac adapter requires torch."""

    with_shadow: bool = False

    tactile_img_res: tuple = (320, 240)
    """Resolution of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is up/down sampled.
    """

    gelpad_height: float = MISSING
    """Used for computing indentation depth from height map"""

    # Asset Data
    gelpad_to_camera_min_distance: float = MISSING
    """Min distance of camera to the gelpad.
    Used for computing the indentation depth out of the
    camera height map.
    """
