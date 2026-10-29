from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import torch
import torchvision.transforms.functional as F

from ..gelsight_simulator import GelSightSimulator
from .sim import Taxim
from .sim.taxim_torch import TaximRGBRenderer

if TYPE_CHECKING:
    from .taxim_sim_cfg import TaximSimulatorCfg


class TaximSimulator(GelSightSimulator):
    """Adapt Taxim's optical simulation to batched GelSight sensors in Isaac Sim."""

    cfg: TaximSimulatorCfg

    def _initialize_impl(self):
        calib_folder = Path(self.cfg.calib_folder_path)

        if self.cfg.device is None:
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        # Indentation depth in mm; used to shift the Taxim height map.
        self._indentation_depth = torch.zeros(self.sensor._num_envs, device=self.sensor._device)

        # This simulator uses torch tensors and torchvision operations downstream.
        self._taxim: Taxim = Taxim(
            calib_folder=calib_folder,
            device=self._device,
            backend=self.cfg.backend,
        )
        if self._taxim.backend_name != "torch":
            raise ValueError(
                f"TaximSimulator only supports the torch backend, but got backend='{self._taxim.backend_name}'. "
                "Set TaximSimulatorCfg.backend='torch' (or 'auto' with torch installed)."
            )
        self._rgb_renderer = TaximRGBRenderer(self._taxim, with_shadow=self.cfg.with_shadow)
        # Resize the unloaded CHW background, then convert it to Isaac's HWC layout.
        self.background_img = self._taxim.background_img
        if self.background_img.shape != (3, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            self.background_img = F.resize(
                self.background_img, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0])
            )
        self.background_img = self.background_img.movedim(0, 2)
        self.tactile_rgb_img = (self.background_img * 255).to(torch.uint8).repeat(
            self.sensor._num_envs, 1, 1, 1)

    def optical_simulation(self, env_ids=slice(None)):
        """Return the selected environments' tactile images as NHWC uint8."""
        height_map = self.sensor._data.output["height_map"][env_ids]

        # up/downscale height map if camera res different than tactile img res
        if (height_map.shape[1], height_map.shape[2]) != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()

        self.tactile_rgb_img[env_ids] = self._rgb_renderer(height_map, self._indentation_depth[env_ids])

        return self.tactile_rgb_img[env_ids]

    def compute_indentation_depth(self, env_ids=slice(None)):
        height_map = self.sensor._data.output["height_map"][env_ids] / 1000  # convert height map from mm to meter
        min_distance_obj = height_map.amin((1, 2))
        # smallest distance between object and sensor case
        dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance

        dist_obj_sensor_case = dist_obj_sensor_case.clamp_min(0)

        self._indentation_depth[env_ids] = torch.where(
            dist_obj_sensor_case <= self.cfg.gelpad_height, (self.cfg.gelpad_height - dist_obj_sensor_case) * 1000, 0
        )

        return self._indentation_depth[env_ids]

    def reset(self, env_ids=slice(None)):
        self._indentation_depth[env_ids] = 0
        self.tactile_rgb_img[env_ids] = (self.background_img * 255).to(torch.uint8)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create windows used by the sensor's debug_tactile_rgb toggle."""
        if debug_vis and not hasattr(self, "_debug_windows"):
            self._debug_windows = {}
            self._debug_img_providers = {}

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return
        import omni.ui

        # Update the GUI windows
        for i, prim in enumerate(self.sensor.prim_view.prims):
            if "tactile_rgb" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_tactile_rgb").Get()
                if show_img:
                    if str(i) not in self._debug_windows:
                        # create a window
                        window = omni.ui.Window(
                            self.sensor._prim_view.prim_paths[i] + "/taxim_rgb",
                            height=self.cfg.tactile_img_res[1],
                            width=self.cfg.tactile_img_res[0],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self.sensor.data.output["tactile_rgb"][i].cpu().numpy()

                    # update image of the window
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    height, width = frame.shape[:2]

                    with self._debug_windows[str(i)].frame:
                        self._debug_img_providers[str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._debug_img_providers[str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._debug_windows:
                    # remove window/img_provider from dictionary and destroy them
                    self._debug_windows.pop(str(i)).destroy()
                    self._debug_img_providers.pop(str(i)).destroy()
