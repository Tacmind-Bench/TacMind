"""FOTS marker-motion model adapted from UniVTAC's TacEx package.

Based on Zhao et al., “FOTS: A Fast Optical Tactile Simulator for Sim2Real
Learning of Tactile-motor Robot Manipulation Skills,” RA-L, 2024.
Source: https://github.com/univtac/UniVTAC/tree/0dafa10262e22f486f160a55d6f11aeab12d8e7b/third_party/TacEx
License: Apache-2.0 (see benchmark/backends/gsmini/tacex/LICENSE.Tabero).
"""

import math

import numpy as np


class MarkerMotion:
    """Approximate GelSight marker displacement from contact and object motion."""

    def __init__(self, *, lamb, mm2pix, num_markers_col, num_markers_row,
                 tactile_img_width, tactile_img_height, x0, y0):
        self.lamb = lamb
        self.mm2pix = mm2pix
        self.num_markers_col = num_markers_col
        self.num_markers_row = num_markers_row
        self.tactile_img_width = tactile_img_width
        self.tactile_img_height = tactile_img_height
        self.contact = []

        x = np.arange(tactile_img_width, dtype=np.float64)
        y = np.arange(tactile_img_height, dtype=np.float64)
        self.xx, self.yy = np.meshgrid(x, y)
        marker_x = np.linspace(x0, tactile_img_width - x0, num_markers_col, dtype=int)
        marker_y = np.linspace(y0, tactile_img_height - y0, num_markers_row, dtype=int)
        marker_x, marker_y = np.meshgrid(marker_x, marker_y)
        self.init_marker_x_pos = marker_x.astype(np.float64)
        self.init_marker_y_pos = marker_y.astype(np.float64)

    def _shear(self, center_x, center_y, lamb, shear_x, shear_y, xx, yy, shear_max=10):
        weight = np.exp(-lamb * ((xx - center_x) ** 2 + (yy - center_y) ** 2))
        shear_x = np.clip(shear_x, -shear_max, shear_max)
        shear_y = np.clip(shear_y, -shear_max, shear_max)
        return shear_x * weight, shear_y * weight

    def _twist(self, center_x, center_y, lamb, theta, xx, yy, theta_max_deg=60):
        theta = np.clip(theta, -theta_max_deg / 180.0 * math.pi,
                        theta_max_deg / 180.0 * math.pi)
        offset_x = xx - center_x
        offset_y = yy - center_y
        weight = np.exp(-lamb * (offset_x**2 + offset_y**2))
        rot_x = offset_x * (np.cos(theta) - 1) - offset_y * np.sin(theta)
        rot_y = offset_x * np.sin(theta) + offset_y * (np.cos(theta) - 1)
        return rot_x * weight, rot_y * weight

    def _dilate(self, lamb, xx, yy):
        dx, dy = 0.0, 0.0
        for center_y, center_x, depth in self.contact:
            weight = np.exp(-lamb * ((xx - center_x) ** 2 + (yy - center_y) ** 2))
            dx += depth * (xx - center_x) * weight
            dy += depth * (yy - center_y) * weight
        return dx, dy

    def _motion_callback(self, depth_map, contact_mask, trajectory):
        depth_map = depth_map - depth_map.min()
        depth_map = depth_map / 10.0

        for row in range(self.num_markers_row):
            for col in range(self.num_markers_col):
                y = int(self.init_marker_y_pos[row, col])
                x = int(self.init_marker_x_pos[row, col])
                if contact_mask[y, x]:
                    self.contact.append([y, x, depth_map[y, x]])

        if not self.contact:
            return self.init_marker_x_pos, self.init_marker_y_pos

        x_disp, y_disp = self._dilate(
            self.lamb[0], self.init_marker_x_pos, self.init_marker_y_pos)
        marker_x = self.init_marker_x_pos + x_disp
        marker_y = self.init_marker_y_pos + y_disp

        if len(trajectory) >= 2:
            center_x = trajectory[0][0] * self.mm2pix + self.tactile_img_width / 2
            center_y = trajectory[0][1] * self.mm2pix + self.tactile_img_height / 2
            shear_x = (trajectory[-1][0] - trajectory[0][0]) * self.mm2pix
            shear_y = (trajectory[-1][1] - trajectory[0][1]) * self.mm2pix
            dx, dy = self._shear(
                center_x, center_y, self.lamb[1], shear_x, shear_y,
                self.init_marker_x_pos, self.init_marker_y_pos)
            marker_x += dx
            marker_y += dy

            theta = trajectory[-1][2] - trajectory[0][2]
            theta = math.atan2(math.sin(theta), math.cos(theta))
            center_x = trajectory[-1][0] * self.mm2pix + self.tactile_img_width / 2
            center_y = trajectory[-1][1] * self.mm2pix + self.tactile_img_height / 2
            dx, dy = self._twist(
                center_x, center_y, self.lamb[2], theta,
                self.init_marker_x_pos, self.init_marker_y_pos)
            marker_x += dx
            marker_y += dy

        return marker_x, marker_y

    def marker_sim(self, depth_map, contact_mask, trajectory):
        try:
            return self._motion_callback(depth_map, contact_mask, trajectory)
        finally:
            self.contact = []
