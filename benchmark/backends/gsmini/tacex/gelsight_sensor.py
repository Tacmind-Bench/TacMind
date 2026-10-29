from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from isaaclab.sensors import SensorBase, TiledCamera, TiledCameraCfg
from isaaclab.sim import get_current_stage_id
from isaaclab.utils.math import convert_camera_frame_orientation_convention, quat_from_matrix
from isaacsim.core.prims import XFormPrim
from pxr import Sdf, UsdGeom

from .gelsight_sensor_data import GelSightSensorData
from .simulation_approaches.gelsight_simulator import GelSightSimulator

if TYPE_CHECKING:
    from .gelsight_sensor_cfg import GelSightSensorCfg


class MountedTiledCamera(TiledCamera):
    """Read camera poses without writing into the renderer's Fabric transforms.

    Isaac Lab's XformPrimView initializes Fabric world matrices from USD on its
    first pose read. USD can be stale after PhysX has moved the parent body.
    Compose the live parent-body pose with the authored camera mount instead.
    """

    def _update_poses(self, env_ids):
        import usdrt

        fabric_stage = usdrt.Usd.Stage.Attach(get_current_stage_id())
        matrices = []
        for index in env_ids:
            prim = self._sensor_prims[int(index)].GetPrim()
            parent = fabric_stage.GetPrimAtPath(str(prim.GetParent().GetPath()))
            parent_world = parent.GetAttribute("omni:fabric:worldMatrix").Get()
            local = UsdGeom.Xformable(prim).GetLocalTransformation()
            matrices.append(np.asarray(local) @ np.asarray(parent_world))
        matrices = torch.as_tensor(np.asarray(matrices), dtype=torch.float32, device=self.device)
        self._data.pos_w[env_ids] = matrices[:, 3, :3]
        orientations = quat_from_matrix(matrices[:, :3, :3].transpose(1, 2))
        self._data.quat_w_world[env_ids] = convert_camera_frame_orientation_convention(
            orientations, origin="opengl", target="world",
        )


class GelSightSensor(SensorBase):
    cfg: GelSightSensorCfg

    def __init__(self, cfg: GelSightSensorCfg, gelpad_obj=None):
        # initialize base class
        super().__init__(cfg)

        self._prim_view = None

        # sensor camera
        self.camera = None
        self.gelpad_obj = gelpad_obj

        self._indentation_depth: torch.tensor = None

        # simulation approaches for simulating GelSight sensor output
        self.optical_simulator: GelSightSimulator = None
        self.marker_motion_simulator: GelSightSimulator = None
        self.compute_indentation_depth_func = None

        # Create empty variables for storing output data
        self._data = GelSightSensorData()
        self._data.output = dict.fromkeys(self.cfg.data_types, None)

        # initialize classes for GelSight simulation approaches for simulating GelSight sensor output
        if self.cfg.optical_sim_cfg is not None:
            # initialize class we set in the cfg class of the sim approach
            self.optical_simulator = self.cfg.optical_sim_cfg.simulation_approach_class(
                sensor=self,
                cfg=self.cfg.optical_sim_cfg,
            )

        if self.cfg.marker_motion_sim_cfg is not None:
            if (self.optical_simulator is not None) and (
                self.cfg.optical_sim_cfg.simulation_approach_class
                == self.cfg.marker_motion_sim_cfg.simulation_approach_class
            ):
                # if same class for optical and marker sim, then use same obj
                self.marker_motion_simulator = self.optical_simulator
            else:
                self.marker_motion_simulator = self.cfg.marker_motion_sim_cfg.simulation_approach_class(
                    sensor=self, cfg=self.cfg.marker_motion_sim_cfg
                )

        self._debug_vis_is_initialized = False

    # Properties

    @property
    def data(self) -> GelSightSensorData:
        """Data related to Camera sensor."""
        # update sensors if needed
        self._update_outdated_buffers()
        return self._data

    @property
    def frame(self) -> torch.tensor:
        """Frame number when the measurement took place."""
        return self._frame

    @property
    def tactile_image_shape(self) -> tuple[int, int, int]:
        """Shape of the simulated tactile RGB image in height, width, channels order."""
        return (self.cfg.optical_sim_cfg.tactile_img_res[1],
                self.cfg.optical_sim_cfg.tactile_img_res[0], 3)

    @property
    def camera_resolution(self) -> tuple[int, int]:
        """The resolution (width x height) of the camera used by this sensor."""
        return self.cfg.sensor_camera_cfg.resolution[0], self.cfg.sensor_camera_cfg.resolution[1]  # type: ignore

    @property
    def indentation_depth(self):
        """How deep objects are inside the gel pad of the sensor.

        Units: [mm]
        """
        return self._indentation_depth

    @property
    def prim_view(self):
        return self._prim_view

    # Operations

    # MARK: reset
    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        # note: cannot do smart indexing here since we do a for loop over data.
        if env_ids is None:
            env_ids = self._ALL_INDICES  # type: ignore
        if len(env_ids) == 0:
            return

        # reset camera
        if self.camera is not None:
            self.camera.reset(env_ids)

        self._indentation_depth[env_ids] = 0

        # reset height map
        self._data.output["height_map"][env_ids] = 0
        if "camera_depth" in self._data.output:
            self._data.output["camera_depth"][env_ids] = 0

        # simulate optical/marker output, but without indentation
        if (self.optical_simulator is not None) and ("tactile_rgb" in self._data.output):
            self.optical_simulator.reset(env_ids)
            self._data.output["tactile_rgb"][env_ids] = self.optical_simulator.optical_simulation(env_ids)

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self._data.output):
            self.marker_motion_simulator.reset(env_ids)
            self._data.output["marker_motion"][env_ids] = self.marker_motion_simulator.marker_motion_simulation(env_ids)
            # (yy_init_pos, xx_init_pos), i.e. along height x width of tactile img
            self._data.output["init_marker_pos"] = ([0], [0])

        # Reset the frame count
        self._frame[env_ids] = 0

    ####
    # Implementation of abstract methods of base sensor class
    ####
    # MARK: _init_impl
    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers."""
        print(f"Initializing GelSight Sensor `{self.cfg.prim_path}`...")

        # Initialize parent class
        super()._initialize_impl()

        self._prim_view = XFormPrim(prim_paths_expr=self.cfg.prim_path, name=f"{self.cfg.prim_path}", usd=False)
        self._prim_view.initialize()
        # Check that sizes are correct
        if self._prim_view.count != self._num_envs:
            raise RuntimeError(
                f"Number of sensor prims in the view ({self._prim_view.count}) does not match"
                f" the number of environments ({self._num_envs})."
            )

        # set device, if specified (per default the same as the simulation)
        if self.cfg.device is not None:
            self._device = self.cfg.device

        # Create all env_ids buffer
        self._ALL_INDICES = torch.arange(self._num_envs, device=self._device, dtype=torch.long)
        # Create frame count buffer
        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)

        if self.cfg.sensor_camera_cfg is not None:
            self.camera_cfg: TiledCameraCfg = TiledCameraCfg(
                prim_path=self.cfg.prim_path + self.cfg.sensor_camera_cfg.prim_path_appendix,
                update_period=self.cfg.sensor_camera_cfg.update_period,
                height=self.cfg.sensor_camera_cfg.resolution[1],
                width=self.cfg.sensor_camera_cfg.resolution[0],
                data_types=self.cfg.sensor_camera_cfg.data_types,
                update_latest_camera_pose=self.cfg.sensor_camera_cfg.update_latest_camera_pose,
                spawn=None,  # use camera which is part of the GelSight Mini Asset
            )
            self.camera = MountedTiledCamera(cfg=self.camera_cfg)

            # need to initialize the camera manually, since its not part of the scene cfg
            self.camera._initialize_impl()
            self.camera._is_initialized = True
            for camera_prim in self.camera._sensor_prims:
                camera_prim.GetClippingRangeAttr().Set(self.cfg.sensor_camera_cfg.clipping_range)

        self._data.output["height_map"] = torch.zeros(
            (self._num_envs, self.camera_cfg.height, self.camera_cfg.width), device=self.cfg.device
        )

        # initialize sim approaches
        if self.optical_simulator is not None:
            self.optical_simulator._initialize_impl()

        if self.marker_motion_simulator is not None:
            self.marker_motion_simulator._initialize_impl()

        # create buffers for output
        if "camera_depth" in self.cfg.data_types:
            self._data.output["camera_depth"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 1),
                device=self.cfg.device, dtype=torch.uint8,
            )
        if "camera_rgb" in self.cfg.data_types:
            self._data.output["camera_rgb"] = torch.zeros(
                (self._num_envs, self.camera_resolution[1], self.camera_resolution[0], 3), device=self.cfg.device
            )
        if "tactile_rgb" in self.cfg.data_types:
            self._data.output["tactile_rgb"] = torch.zeros(
                (
                    self._num_envs,
                    self.cfg.optical_sim_cfg.tactile_img_res[1],
                    self.cfg.optical_sim_cfg.tactile_img_res[0],
                    3,
                ),
                device=self.cfg.device,
                dtype=torch.uint8,
            )
        if "marker_motion" in self.cfg.data_types:
            self._data.output["marker_motion"] = torch.zeros(
                (
                    self._num_envs,
                    2,  # two, initial and current
                    self.cfg.marker_motion_sim_cfg.marker_params.num_markers,
                    2,  # two, because each marker at (row,col) has position value (y,x)
                ),
                device=self.cfg.device,
            )

        if "markers_rgb" in self.cfg.data_types:
            self._data.output["markers_rgb"] = torch.zeros(
                (
                    self._num_envs,
                    self.cfg.marker_motion_sim_cfg.tactile_img_res[1],
                    self.cfg.marker_motion_sim_cfg.tactile_img_res[0],
                    3,
                ),
                device=self.cfg.device,
                dtype=torch.uint8,
            )

        # set how the indentation depth should be computed
        if (self.cfg.compute_indentation_depth_class) == "optical_sim" and (self.optical_simulator is not None):
            self.compute_indentation_depth_func = self.optical_simulator.compute_indentation_depth
        elif (self.cfg.compute_indentation_depth_class == "marker_motion_sim") and (
            self.marker_motion_simulator is not None
        ):
            self.compute_indentation_depth_func = self.marker_motion_simulator.compute_indentation_depth
        else:
            self.compute_indentation_depth_func = None

        # reset internal buffers
        self.reset()

        # create debug visualization
        self._initialize_debug_vis(self._initialize_debug_vis_flag)

    # MARK: _update_buffers_impl
    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Capture camera data and update the selected tactile outputs."""
        self._frame[env_ids] += 1

        # -- update camera buffer
        if self.camera is not None:
            # The parent sensor schedules captures; keep child clocks independent.
            self.camera._timestamp.copy_(self._timestamp)
            self.camera._is_outdated[env_ids] = True
            self.camera._update_outdated_buffers()

        if self.compute_indentation_depth_func is not None:
            # -- height_map
            self._get_height_map(env_ids)
            # -- pressing depth
            self._indentation_depth[env_ids] = self.compute_indentation_depth_func(env_ids)

        if "camera_depth" in self._data.output:
            self._get_camera_depth(env_ids)

        if "camera_rgb" in self._data.output:
            self._data.output["camera_rgb"][env_ids] = self.camera.data.output["rgb"][env_ids].to(
                self._data.output["camera_rgb"].dtype
            )

        if (self.optical_simulator is not None) and ("tactile_rgb" in self.cfg.data_types):
            self._data.output["tactile_rgb"][env_ids] = self.optical_simulator.optical_simulation(env_ids)

        if (self.marker_motion_simulator is not None) and ("marker_motion" in self.cfg.data_types):
            self._data.output["marker_motion"][env_ids] = self.marker_motion_simulator.marker_motion_simulation(env_ids)

        if (self.marker_motion_simulator is not None) and ("markers_rgb" in self.cfg.data_types):
            self._data.output["markers_rgb"][env_ids] = self.marker_motion_simulator.marker_motion_rendering(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # we actually set the debug_vis in _initialize_impl, since we need the _prim_view, which
        # is only correctly initialized after _initialize_impl method (in ManagerBased workflow, in Direct workflow you can control it yourself)
        self._initialize_debug_vis_flag: bool = debug_vis

    def _initialize_debug_vis(self, debug_vis: bool):
        """Creates an USD attribute for the sensor assets, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.

        > Method has to be called after the prim_view was initialized.
        """
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # need to create the attribute for the debug_vis here since it depends on self._prim_view
            for prim in self._prim_view.prims:
                # creates USD attributes for each data type, which can be found in the Isaac GUI under "Raw Usd Properties -> "Extra Properties"
                if "camera_depth" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_camera_depth", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "camera_rgb" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_camera_rgb", Sdf.ValueTypeNames.Bool)
                    attr.Set(False)
                if "tactile_rgb" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_tactile_rgb", Sdf.ValueTypeNames.Bool)
                    attr.Set(True)
                if "marker_motion" in self.cfg.data_types:
                    attr = prim.CreateAttribute("debug_marker_motion", Sdf.ValueTypeNames.Bool)
                    attr.Set(True)

            if not hasattr(self, "_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._windows = {}
                self._img_providers = {}
                # todo check if there is a more efficient implementation than dict of dicts
                if "camera_depth" in self.cfg.data_types:
                    self._windows["camera_depth"] = {}
                    self._img_providers["camera_depth"] = {}
                if "camera_rgb" in self.cfg.data_types:
                    self._windows["camera_rgb"] = {}
                    self._img_providers["camera_rgb"] = {}

            if "tactile_rgb" in self.cfg.data_types:
                self.optical_simulator._set_debug_vis_impl(debug_vis)

            if "marker_motion" in self.cfg.data_types:
                self.marker_motion_simulator._set_debug_vis_impl(debug_vis)

            self._debug_vis_is_initialized = True

    def _debug_vis_callback(self, event):
        if not self._debug_vis_is_initialized:
            return
        import omni.ui

        # Update the GUI windows
        for i, prim in enumerate(
            self._prim_view.prims
        ):  # note: bad that we iterate through all prims multiple times (once per sim data type)
            if "camera_rgb" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_camera_rgb").Get()
                if show_img:
                    if str(i) not in self._windows["camera_rgb"]:
                        # create a window
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/camera_rgb",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["camera_rgb"][str(i)] = window
                        # create image provider
                        self._img_providers["camera_rgb"][
                            str(i)
                        ] = omni.ui.ByteImageProvider()  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self._data.output["camera_rgb"][i].cpu().numpy()

                    # update image of the window
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width = frame.shape[:2]
                    with self._windows["camera_rgb"][str(i)].frame:
                        self._img_providers["camera_rgb"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._img_providers["camera_rgb"][str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._windows["camera_rgb"]:
                    # remove window/img_provider from dictionary and destroy them
                    self._windows["camera_rgb"].pop(str(i)).destroy()
                    self._img_providers["camera_rgb"].pop(str(i)).destroy()

            if "camera_depth" in self.cfg.data_types:
                show_img = prim.GetAttribute("debug_camera_depth").Get()
                if show_img:
                    if str(i) not in self._windows["camera_depth"]:
                        # create a window
                        window = omni.ui.Window(
                            self._prim_view.prim_paths[i] + "/camera_depth",
                            height=self.cfg.sensor_camera_cfg.resolution[1],
                            width=self.cfg.sensor_camera_cfg.resolution[0],
                        )
                        self._windows["camera_depth"][str(i)] = window
                        # create image provider
                        self._img_providers["camera_depth"][
                            str(i)
                        ] = omni.ui.ByteImageProvider()  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self._data.output["camera_depth"][i].cpu().numpy()
                    # convert to 3 channel image, to later turn it into 4 channel RGBA for Isaac Widget
                    frame = np.dstack((frame, frame, frame)).astype(np.uint8)
                    frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # update image of the window
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                    height, width = frame.shape[:2]
                    with self._windows["camera_depth"][str(i)].frame:
                        self._img_providers["camera_depth"][str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._img_providers["camera_depth"][str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._windows["camera_depth"]:
                    # remove window/img_provider from dictionary and destroy them
                    self._windows["camera_depth"].pop(str(i)).destroy()
                    self._img_providers["camera_depth"].pop(str(i)).destroy()

        if "tactile_rgb" in self.cfg.data_types:
            self.optical_simulator._debug_vis_callback(event)

        if "marker_motion" in self.cfg.data_types:
            self.marker_motion_simulator._debug_vis_callback(event)

    # Private Helper methods
    def _get_camera_depth(self, env_ids=slice(None)):
        if self.camera is not None:
            # Keep NHWC layout and leave the shared camera measurements in meters.
            depth = self.camera.data.output["depth"][env_ids]
            near, far = self.cfg.sensor_camera_cfg.clipping_range
            normalized = torch.nan_to_num(depth, nan=far, posinf=far, neginf=near)
            normalized = ((normalized - near) / (far - near)).clamp(0, 1)
            self._data.output["camera_depth"][env_ids] = (normalized * 255).round().to(torch.uint8)

        return self._data.output["camera_depth"]

    def _get_height_map(self, env_ids=slice(None)):
        if self.camera is not None:
            depth = self.camera.data.output["depth"][env_ids, :, :, 0]
            self._data.output["height_map"][env_ids] = torch.where(
                torch.isinf(depth), self.cfg.sensor_camera_cfg.clipping_range[1], depth
            ) * 1000  # Optical simulation uses millimeters.
            return self._data.output["height_map"]

    # Internal simulation callbacks.

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._prim_view = None

        if self.camera is not None:
            self.camera._invalidate_initialize_callback(event)
            self.camera.__del__()

        if hasattr(self, "_windows"):
            self._windows = None
            self._img_providers = None
