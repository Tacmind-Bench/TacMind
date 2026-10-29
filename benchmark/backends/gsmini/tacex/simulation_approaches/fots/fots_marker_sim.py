"""Taxim-coupled FOTS marker-motion simulator adapted from UniVTAC TacEx.

Source: https://github.com/univtac/UniVTAC/tree/0dafa10262e22f486f160a55d6f11aeab12d8e7b/third_party/TacEx
License: Apache-2.0 (see benchmark/backends/gsmini/tacex/LICENSE.Tabero).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as torch_functional

from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import axis_angle_from_quat, euler_xyz_from_quat, quat_conjugate, quat_mul

from ..gelsight_simulator import GelSightSimulator
from ..gpu_taxim import TaximSimulator
from .marker_motion import MarkerMotion

if TYPE_CHECKING:
    from .fots_marker_sim_cfg import FOTSMarkerSimulatorCfg


class FOTSMarkerSimulator(GelSightSimulator):
    """Estimate marker fields using Taxim gel deformation and object motion."""

    cfg: FOTSMarkerSimulatorCfg

    def __init__(self, sensor, cfg):
        super().__init__(sensor=sensor, cfg=cfg)
        self.frame_transformer = FrameTransformer(cfg.frame_transformer_cfg)

    def _initialize_impl(self):
        self._num_envs = self.sensor._num_envs
        self._device = self.cfg.device or self.sensor.device
        if not isinstance(self.sensor.optical_simulator, TaximSimulator):
            raise RuntimeError("FOTS marker motion requires the Taxim optical simulator.")

        self._taxim = self.sensor.optical_simulator._taxim
        self._unloaded_deformation = None
        if self.cfg.motion_source == "target_pose":
            width, height = self.cfg.tactile_img_res
            self._unloaded_deformation, _ = self._taxim._TaximTorch__compute_gel_pad_deformation(
                torch.zeros((1, height, width), device=self._taxim.device))
        params = self.cfg.marker_params
        if params.num_markers != params.num_markers_col * params.num_markers_row:
            raise ValueError("FOTS marker count must equal marker rows times columns.")
        width, height = self.cfg.tactile_img_res
        self.marker_model = MarkerMotion(
            lamb=self.cfg.lamb,
            mm2pix=self.cfg.mm_to_pixel,
            num_markers_col=params.num_markers_col,
            num_markers_row=params.num_markers_row,
            tactile_img_width=width,
            tactile_img_height=height,
            x0=params.x0,
            y0=params.y0,
        )
        initial = np.stack(
            (self.marker_model.init_marker_x_pos, self.marker_model.init_marker_y_pos),
            axis=-1,
        ).reshape(-1, 2)
        self.marker_data = torch.zeros(
            (self._num_envs, 2, params.num_markers, 2),
            device=self._device,
            dtype=torch.float32,
        )
        self.marker_data[:, 0] = torch.as_tensor(initial, device=self._device)
        self.marker_data[:, 1] = self.marker_data[:, 0]
        self.trajectories = [[] for _ in range(self._num_envs)]
        self._contact_reference = [None for _ in range(self._num_envs)]

        self.frame_transformer._initialize_impl()
        self.frame_transformer._is_initialized = True

    def _env_indices(self, env_ids):
        if env_ids is None:
            return list(range(self._num_envs))
        if isinstance(env_ids, slice):
            return list(range(*env_ids.indices(self._num_envs)))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.flatten().tolist()
        if isinstance(env_ids, int):
            return [env_ids]
        return list(env_ids)

    def marker_motion_simulation(self, env_ids=slice(None)):
        width, height = self.cfg.tactile_img_res
        height_map = self.sensor._data.output["height_map"]
        if (height_map.shape[-1], height_map.shape[-2]) != (width, height):
            height_map = torch_functional.interpolate(
                height_map.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        indentation = self.sensor._indentation_depth
        shifted = self._taxim._TaximTorch__get_shifted_height_map(indentation, height_map)
        deformation, contact_mask = self._taxim._TaximTorch__compute_gel_pad_deformation(shifted)
        if self.cfg.motion_source == "target_pose":
            # Strain is relative to the unloaded curved gel, not its highest
            # surface point. Absolute curvature creates false marker jumps
            # when a sparse contact sample crosses the binary mask boundary.
            deformation = (self._unloaded_deformation - deformation).clamp_min(0)
        else:
            deformation = deformation.amax(dim=(-2, -1), keepdim=True) - deformation
        indices = self._env_indices(env_ids)
        if any(indentation[index].item() > 0 and contact_mask[index].any().item()
               for index in indices):
            self.frame_transformer.update(dt=self.sensor.cfg.update_period, force_recompute=True)

        for env_id in indices:
            mask = contact_mask[env_id]
            if indentation[env_id].item() <= 0 or not mask.any().item():
                self.trajectories[env_id] = []
                self._contact_reference[env_id] = None
                self.marker_data[env_id, 1] = self.marker_data[env_id, 0]
                continue

            points = torch.argwhere(mask)
            center = points.float().mean(dim=0).cpu().numpy()
            center_y = (center[0] - height / 2) / self.marker_model.mm2pix
            center_x = (center[1] - width / 2) / self.marker_model.mm2pix
            if self.cfg.motion_source == "target_pose":
                self.trajectories[env_id] = self._relative_trajectory(env_id, np.array([center_x, center_y]))
            else:
                quaternion = self.frame_transformer.data.target_quat_source[env_id]
                _, _, yaw = euler_xyz_from_quat(quaternion)
                self.trajectories[env_id].append(
                    [center_x, center_y, yaw.reshape(-1)[0].item()])
                # FOTS uses only the first and current contact samples.
                if len(self.trajectories[env_id]) > 2:
                    self.trajectories[env_id] = [self.trajectories[env_id][0], self.trajectories[env_id][-1]]

            marker_x, marker_y = self.marker_model.marker_sim(
                deformation[env_id].detach().cpu().numpy(),
                ((deformation[env_id] > 0) if self.cfg.motion_source == "target_pose" else mask).detach().cpu().numpy(),
                self.trajectories[env_id],
            )
            marker_positions = np.stack((marker_x, marker_y), axis=-1).reshape(-1, 2)
            self.marker_data[env_id, 1] = torch.as_tensor(
                marker_positions, device=self._device, dtype=self.marker_data.dtype)

        return self.marker_data[env_ids]

    def _relative_trajectory(self, env_id, center_mm):
        """Retain subpixel tool translation and twist in the gel camera frame.

        The silhouette of a broad flat handle may stay fixed during shear.
        Track its contact-local material frame instead of that silhouette.
        No force diagnostic or task phase enters this observation model.
        """
        data = self.frame_transformer.data
        position = data.target_pos_source[env_id, 0]
        quaternion = data.target_quat_source[env_id, 0]
        if self._contact_reference[env_id] is None:
            self._contact_reference[env_id] = (position.clone(), quaternion.clone(), center_mm.copy())
        initial_position, initial_quaternion, center = self._contact_reference[env_id]
        shift_mm = ((position - initial_position)[:2] * 1000).detach().cpu().numpy()
        rotation = quat_mul(quaternion, quat_conjugate(initial_quaternion))
        twist = float(axis_angle_from_quat(rotation)[2])
        return [[*center, 0.0], [*(center + shift_mm), twist]]

    def reset(self, env_ids=slice(None)):
        indices = self._env_indices(env_ids)
        if not indices:
            return
        self.frame_transformer.reset(indices)
        for env_id in indices:
            self.trajectories[env_id] = []
            self._contact_reference[env_id] = None
            self.marker_data[env_id, 1] = self.marker_data[env_id, 0]
