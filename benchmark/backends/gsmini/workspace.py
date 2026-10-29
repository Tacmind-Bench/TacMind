# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-aware tabletop sampling shared by lift collection, training and evaluation."""

import math
import weakref
from copy import deepcopy

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from isaaclab.managers import ManagerTermBase


class _CollisionWorldCache(dict):
    """Weak-referenceable state for the native USD notice subscription."""


def _collision_body_data(env):
    """Expose measured link frames as well as independently simulated bodies."""
    from types import SimpleNamespace

    result = {obj.root_physx_view.prim_paths[0]: obj.data
              for obj in env.scene.rigid_objects.values()}
    for obj in getattr(env.scene, "articulations", {}).values():
        positions, rotations = obj.data.body_pos_w, obj.data.body_quat_w
        for index, path in enumerate(obj.root_physx_view.link_paths[0]):
            result[path] = SimpleNamespace(root_pos_w=positions[:, index],
                                           root_quat_w=rotations[:, index])
    return result


def read_collision_world(env, obstacle_names=("Table",), *,
                         environment_root="/World/envs/env_0",
                         robot_link_path="/World/envs/env_0/Robot/panda_link0",
                         obstacle_paths=None):
    """Cache geometry while refreshing measured rigid-body poses on every read."""
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom

    source = env.sim.stage
    state = getattr(env, "_collision_world_cache", None)
    if state is None or state["stage"] != source:
        if state is not None:
            state["notice"].Revoke()
        state = _CollisionWorldCache(stage=source, entries={}, revision=0, token=object())
        cache_ref = weakref.ref(state)

        def invalidate(notice, sender):
            # Resetting a drive, a joint frame or the clock does not change a
            # collider's shape. Body poses are refreshed from PhysX on every read.
            # Keep all other authored edits conservative, including composition.
            if not notice.GetResyncedPaths():
                for path in notice.GetChangedInfoOnlyPaths():
                    if (path == Sdf.Path.absoluteRootPath and set(notice.GetChangedFields(path))
                            <= {"timeCodesPerSecond", "framesPerSecond"}):
                        continue
                    if path.IsPropertyPath() and path.name in {
                            "physics:localPos0", "drive:linear:physics:targetVelocity",
                            "physxScene:timeStepsPerSecond"}:
                        continue
                    break
                else:
                    return
            cache_state = cache_ref()
            if cache_state is not None:
                cache_state["entries"].clear()
                cache_state["revision"] += 1

        state["notice"] = Tf.Notice.Register(Usd.Notice.ObjectsChanged, invalidate, source)
        state["callback"] = invalidate  # Tf keeps only a weak reference to Python callbacks.
        weakref.finalize(state, state["notice"].Revoke)
        env._collision_world_cache = state
    paths = (tuple(obstacle_paths) if obstacle_paths is not None else
             tuple(f"{environment_root}/{name}" for name in obstacle_names))
    key = (paths, robot_link_path)
    rigid_data = _collision_body_data(env)
    base = UsdGeom.XformCache().GetLocalToWorldTransform(source.GetPrimAtPath(robot_link_path))
    if key not in state["entries"]:
        world = _read_collision_geometry(env, obstacle_names, environment_root=environment_root,
                                         robot_link_path=robot_link_path, obstacle_paths=paths)
        offsets = {}
        for obstacle in world.objects:
            parents = [path for path in rigid_data
                       if obstacle.name == path or obstacle.name.startswith(path + "/")]
            if parents:
                body = max(parents, key=len)
                data = rigid_data[body]
                body_pose = _collision_pose_matrix([*data.root_pos_w[0].tolist(),
                                                    *data.root_quat_w[0].tolist()])
                offsets[obstacle.name] = (body, _collision_pose_matrix(obstacle.pose)
                                         * base * body_pose.GetInverse())
        state["entries"][key] = (world, offsets)
    template, offsets = state["entries"][key]
    world = deepcopy(template)
    inverse_base = base.GetInverse()
    for obstacle in world.objects:
        if obstacle.name not in offsets:
            continue
        body, offset = offsets[obstacle.name]
        data = rigid_data[body]
        transform = Gf.Transform(offset * _collision_pose_matrix([
            *data.root_pos_w[0].tolist(), *data.root_quat_w[0].tolist()]) * inverse_base)
        quaternion = transform.GetRotation().GetQuat()
        obstacle.pose = [*transform.GetTranslation(), quaternion.GetReal(), *quaternion.GetImaginary()]
    world._geometry_version = (state["token"], state["revision"])
    return world


def _collision_pose_matrix(pose):
    from pxr import Gf

    transform = Gf.Transform()
    transform.SetTranslation(Gf.Vec3d(*pose[:3]))
    transform.SetRotation(Gf.Rotation(Gf.Quatd(pose[3], Gf.Vec3d(*pose[4:]))))
    return transform.GetMatrix()


def update_collision_world(planner, world):
    """Refresh native GPU obstacle caches and invalidate the planner's graph paths."""
    from curobo.types import Pose

    version = getattr(world, "_geometry_version", None)
    loaded = getattr(planner, "_collision_geometry_version", None)
    checker = getattr(planner, "scene_collision_checker", None)
    names = {obstacle.name for obstacle in world.objects} if version is not None else set()
    if (version is None or loaded != version or checker is None
            or not names.issubset(checker.get_obstacle_names())):
        if checker is not None and loaded is not None and loaded != version:
            # load_collision_model retains same-name Warp meshes by default.
            # A changed USD revision may include new vertices or inherited scale.
            if checker.data.meshes is not None:
                checker.data.meshes.clear(clear_warp_cache=True)
        planner.update_world(world)
        planner._collision_geometry_version = version
        return
    for name in checker.get_obstacle_names():
        checker.enable_obstacle(name, enable=name in names)
    for obstacle in world.objects:
        checker.update_obstacle_pose(obstacle.name, Pose.from_list(obstacle.pose, planner.device_cfg))
    checker.scene_model = world
    if planner.graph_planner is not None:
        planner.graph_planner.reset_buffer()


def _read_collision_geometry(env, obstacle_names=("Table",), *,
                             environment_root="/World/envs/env_0",
                             robot_link_path="/World/envs/env_0/Robot/panda_link0",
                             obstacle_paths=None):
    """Read fixture collisions in robot coordinates without modifying simulation USD."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    from isaaclab.sim import make_uninstanceable
    # cuRobo currently exposes its USD reader only through this module.
    from curobo._src.util.usd_scene_parser import UsdSceneParser

    paths = (tuple(obstacle_paths) if obstacle_paths is not None else
             tuple(f"{environment_root}/{name}" for name in obstacle_names))
    source = env.sim.stage
    mask = Usd.StagePopulationMask([*paths, robot_link_path]).GetIntersection(
        source.GetPopulationMask())
    # Compose only the requested subtrees before flattening. Retain the live
    # stage's composition state; all pose/instance edits stay in the clone.
    selected = Usd.Stage.OpenMasked(
        source.GetRootLayer(), source.GetSessionLayer(), source.GetPathResolverContext(),
        mask, Usd.Stage.LoadNone,
    )
    selected.MuteAndUnmuteLayers(source.GetMutedLayers(), [])
    selected.SetLoadRules(source.GetLoadRules())
    planning_stage = Usd.Stage.Open(selected.Flatten())
    rigid_data = _collision_body_data(env)
    collision_paths = []
    for path in paths:
        if not planning_stage.GetPrimAtPath(path):
            raise ValueError(f"Missing collision obstacle: {path}")
        make_uninstanceable(path, stage=planning_stage)
        for body in Usd.PrimRange(planning_stage.GetPrimAtPath(path)):
            if str(body.GetPath()) not in rigid_data:
                continue
            data = rigid_data[str(body.GetPath())]
            position = data.root_pos_w[0].tolist()
            w, x, y, z = data.root_quat_w[0].tolist()
            # PhysX/Fabric poses can differ from the authored USD after settling.
            # Update only the planning clone, retaining the mesh's world scale.
            transform = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(body))
            transform.SetTranslation(Gf.Vec3d(*position))
            transform.SetRotation(Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))))
            xform = UsdGeom.Xformable(body)
            xform.MakeMatrixXform().Set(transform.GetMatrix())
            xform.SetResetXformStack(True)
        collision_paths.extend(
            str(prim.GetPath()) for prim in Usd.PrimRange(planning_stage.GetPrimAtPath(path))
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        )
    enabled_colliders = set(collision_paths)
    for prim in planning_stage.Traverse():
        # cuRobo filters path prefixes, not CollisionAPI; preserve child transforms
        # while hiding non-colliding geometry from its reader in the clone only.
        if prim.IsA(UsdGeom.Gprim) and str(prim.GetPath()) not in enabled_colliders:
            prim.SetTypeName("Xform")
        elif prim.IsA(UsdGeom.Cube) and prim.GetAttribute("xformOp:scale").Get() is None:
            # The reader requires this attribute even for default/matrix transforms.
            # Leave it out of xformOpOrder so the actual transform is unchanged.
            prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(1))
        elif prim.IsA(UsdGeom.Mesh) and any(n > 4 for n in UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()):
            from benchmark.scene.collision_geometry import triangulate_planning_mesh
            triangulate_planning_mesh(prim)
    reader = UsdSceneParser()
    reader.load_stage(planning_stage)
    world = reader.get_obstacles_from_stage(
        only_paths=collision_paths,
        reference_prim_path=robot_link_path,
    )
    # The cuRobo USD reader uses only a Cube's local scale for its dimensions.
    # Include inherited scale so planning agrees with the physical collider.
    cache = UsdGeom.XformCache()
    for cuboid in world.cuboid or ():
        prim = planning_stage.GetPrimAtPath(cuboid.name)
        scale = Gf.Transform(cache.GetLocalToWorldTransform(prim)).GetScale()
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        cuboid.dims = [abs(float(value)) * size for value in scale]
    if not world.mesh and not world.cuboid:
        raise RuntimeError("cuRobo could not import the fixture collision geometry")

    return world


def grasp_poses(env, positions, env_ids=None, quaternion=None):
    """Convert world-space TCP targets to hand goals; default to a downward grasp."""
    from curobo.types import DeviceCfg, Pose

    robot = env.scene["robot"]
    device = DeviceCfg(device=env.device)
    ids = slice(None) if env_ids is None else env_ids
    base = Pose(robot.data.root_pos_w[ids].contiguous(), robot.data.root_quat_w[ids].contiguous())
    quaternion = ([0.0, 1.0, 0.0, 0.0] if quaternion is None else quaternion)
    quaternion = device.to_device(quaternion).reshape(-1, 4).expand(len(positions), -1).contiguous()
    tcp = Pose(positions.contiguous(), quaternion)
    offset = env.cfg.scene.ee_frame.target_frames[0].offset
    hand_to_tcp = Pose.from_list([*offset.pos, *offset.rot], device)
    return base.inverse().multiply(tcp).multiply(hand_to_tcp.inverse())


def interior_mask(reachable, spacing, margin):
    """Erode sampled reachability, including unknown grid borders and cell jitter."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Workspace margin must be finite and non-negative")
    distances = distance_transform_edt(np.pad(reachable, 1), sampling=spacing)[1:-1, 1:-1]
    # Account for the unknown boundary between samples and the farthest cell corner.
    return distances > margin + math.sqrt(2) * spacing


class ResetReachableObject(ManagerTermBase):
    """Cache a tabletop IK map; sample continuous positions inside its safe cells.

    Valid only for the shared static table/robot layout. IK checks grasp, approach
    and lift endpoints; the execution planner still checks connecting trajectories.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ik = None
        self.spacing = 0.025
        self._margin = None

    def _reachable(self, env, positions):
        from curobo.types import GoalToolPose

        # Keep solver randomness separate from Isaac Lab's seeded reset sampling.
        with torch.random.fork_rng(devices=[torch.device(env.device)]):
            self.ik.reset_seed()
            success = torch.ones(len(positions), dtype=torch.bool, device=env.device)
            batch_size = self.ik.config.max_batch_size
            for height in (0.0, 0.10, 0.20):
                targets = positions.clone()
                targets[:, 2] += height
                for chunk, valid in zip(targets.split(batch_size), success.split(batch_size)):
                    poses = grasp_poses(env, chunk, env_ids=slice(0, 1))
                    result = self.ik.solve_pose(GoalToolPose.from_poses({"panda_hand": poses}))
                    valid &= result.success.reshape(len(chunk), -1).any(-1)
        return success

    def _build(self, env, robot_config):
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import DeviceCfg
        from pxr import Usd, UsdGeom, UsdPhysics

        # SeattleLabTable includes a tall rear frame in its visual bounds.
        # Its tabletop is the hidden box collider, not that visual AABB.
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], False, True)
        table_prim = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Table")
        surfaces = [bounds.ComputeWorldBound(prim).ComputeAlignedRange()
                    for prim in Usd.PrimRange(table_prim, Usd.TraverseInstanceProxies())
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
        if len(surfaces) != 1:
            raise RuntimeError("Workspace sampling expects the shared table's single box collider")
        table = surfaces[0]
        obj = bounds.ComputeWorldBound(env.sim.stage.GetPrimAtPath("/World/envs/env_0/Object")).ComputeAlignedRange()
        half_size = np.asarray(obj.GetSize())[:2] / 2
        low = np.asarray(table.GetMin())[:2] + half_size
        high = np.asarray(table.GetMax())[:2] - half_size
        axes = [np.arange(a, b, self.spacing) for a, b in zip(low, high)]
        xy = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        grid_shape = xy.shape[:2]
        points = torch.empty((xy.shape[0] * xy.shape[1], 3), device=env.device)
        points[:, :2] = torch.as_tensor(xy.reshape(-1, 2), device=env.device)
        # Query at the settled grasp height; spawn 2 mm above it to avoid overlap.
        points[:, 2] = (env.scene["object"].data.default_root_state[0, 2]
                        + env.scene.env_origins[0, 2] - float(obj.GetMin()[2]) + float(table.GetMax()[2]))
        with torch.random.fork_rng(devices=[torch.device(env.device)]):
            self.ik = InverseKinematics(InverseKinematicsCfg.create(
                robot="franka.yml" if robot_config is None else robot_config(env),
                scene_model=read_collision_world(env), device_cfg=DeviceCfg(device=env.device),
                random_seed=0, max_batch_size=128,
            ))
        cleanup = weakref.finalize(self, self.ik.destroy)
        try:
            self._reachable_grid = self._reachable(env, points).cpu().numpy().reshape(grid_shape)
            self._points = points
        except BaseException:
            self.ik = None
            cleanup()
            raise

    def __call__(self, env, env_ids, margin=0.05, robot_config=None):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        if len(env_ids) == 0:
            return
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("Workspace margin must be finite and non-negative")
        if self.ik is None:
            self._build(env, robot_config)
        if margin != self._margin:
            safe = interior_mask(self._reachable_grid, self.spacing, margin)
            centers = self._points[torch.as_tensor(safe.ravel(), device=env.device)]
            if not len(centers):
                raise RuntimeError(f"No reachable tabletop interior with margin={margin:g} m")
            self.centers = centers
            self._margin = margin
            print(f"[WORKSPACE] {safe.sum()}/{safe.size} safe cells; spacing={self.spacing:g}m, margin={margin:g}m", flush=True)
        positions = torch.empty((len(env_ids), 3), device=env.device)
        pending = torch.arange(len(env_ids), device=env.device)
        for _ in range(32):
            candidates = self.centers[torch.randint(len(self.centers), (len(pending),), device=env.device)].clone()
            candidates[:, :2] += (torch.rand((len(pending), 2), device=env.device) - 0.5) * self.spacing
            valid = self._reachable(env, candidates)
            positions[pending[valid]] = candidates[valid]
            pending = pending[~valid]
            if not len(pending):
                break
        else:
            raise RuntimeError("Unable to sample collision-free reachable object positions")
        obj = env.scene["object"]
        state = obj.data.default_root_state[env_ids].clone()
        state[:, :3] = positions - env.scene.env_origins[0] + env.scene.env_origins[env_ids]
        state[:, 2] += 0.002
        obj.write_root_state_to_sim(state, env_ids=env_ids)
