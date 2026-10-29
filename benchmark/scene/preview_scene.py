"""Open a nominal Tac task layout in Isaac Sim for visual inspection.

Tabletop previews reuse seeded reset layouts without running task physics or
success checks.
"""

from __future__ import annotations


import argparse
import math
import tempfile
from contextlib import nullcontext
from pathlib import Path

from benchmark.scene.cameras import CAMERAS, add_preview_cameras
from benchmark.catalog import is_tabletop_task, ASSETS, COUNTS, ROBOT_ASSET, get_task, object_asset, scene_asset
from benchmark.scene.reset import robot_joint_defaults
from benchmark.scene.visuals import visual_material_cfg


def prepare_dynamic_preview(stage, task: dict, drop_clearance_m: float = 0.0,
                           *, reset: dict | None = None) -> None:
    """Show the task asset resting on its physical conveyor platform."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, Usd, UsdGeom, UsdShade
    from benchmark.tasks.dynamic.scene import configure_moving_asset, dynamic_asset_scale, dynamic_source_payload_position, configure_source_holder_overrides, configure_conveyor_platform, platform_reset_pose, sample_dynamic_reset

    episode = task["episode_config"]
    reset = sample_dynamic_reset(task, seed=0) if reset is None else reset
    platform_pos, platform_size = platform_reset_pose(
        stage, task, reset, drop_clearance_m,
    )
    platform_top_z = platform_pos[2] + platform_size[2] / 2
    jitter = reset["moving_asset_xy_jitter_m"]
    asset_x = (platform_pos[0] + float(episode.get("moving_asset_x_offset_m", 0.0))
               + jitter[0])
    asset_y = platform_pos[1] + jitter[1]

    moving_fixture = episode["moving_asset"] == "target_fixture"
    asset_type = episode["fixture_asset_type" if moving_fixture else "payload_asset_type"]
    asset = object_asset(asset_type)

    def asset_min_z(path):
        asset_stage = Usd.Stage.Open(str(path))
        bounds = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        return bounds.ComputeWorldBound(
            asset_stage.GetDefaultPrim(),
        ).ComputeAlignedRange().GetMin()[2]

    scale = dynamic_asset_scale(task, "fixture" if moving_fixture else "object")
    clearance = (episode["fixture_mount_clearance_m"] if moving_fixture
                 else 0.002)
    holder_type = episode["source_holder_type"] if not moving_fixture else None
    if holder_type:
        holder_asset = object_asset(holder_type)
        holder_min_z = asset_min_z(holder_asset)
        holder_scale = float(episode["source_holder_scale"])
        holder_yaw = float(episode.get("base_yaw_deg", 0.0))
        holder_pos = (asset_x, asset_y,
                      platform_top_z - holder_min_z * holder_scale + clearance)
        asset_pos = dynamic_source_payload_position(task, holder_pos)
    else:
        min_z = asset_min_z(asset)
        asset_pos = (asset_x, asset_y, platform_top_z - min_z * scale[2] + clearance)

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        configure_conveyor_platform(stage, task, reset, drop_clearance_m)
        if holder_type:
            holder_xform = UsdGeom.Xform.Define(stage, "/World/PreviewSourceHolder")
            holder_xform.AddTranslateOp().Set(Gf.Vec3d(*holder_pos))
            holder_xform.AddRotateZOp().Set(holder_yaw)
            holder_xform.AddScaleOp().Set(Gf.Vec3d(holder_scale, holder_scale, holder_scale))
            holder_asset_prim = UsdGeom.Xform.Define(stage, "/World/PreviewSourceHolder/Asset")
            holder_asset_prim.GetPrim().GetReferences().AddReference(str(holder_asset.resolve()))
            configure_moving_asset(
                stage, "/World/PreviewSourceHolder", episode.get("source_holder_mass_kg"))
            configure_source_holder_overrides(
                stage, "/World/PreviewSourceHolder", task,
            )
        xform = UsdGeom.Xform.Define(stage, "/World/PreviewMovingAsset")
        xform.AddTranslateOp().Set(Gf.Vec3d(*asset_pos))
        yaw = episode.get("fixture_yaw_deg" if moving_fixture else "base_yaw_deg", 0.0)
        xform.AddRotateZOp().Set(yaw)
        xform.AddScaleOp().Set(Gf.Vec3d(*scale))
        asset_prim = UsdGeom.Xform.Define(stage, "/World/PreviewMovingAsset/Asset")
        asset_prim.GetPrim().GetReferences().AddReference(str(asset.resolve()))
        moving_mass = (episode.get("fixture_mass_kg") if moving_fixture
                       else episode.get("payload_mass_kg"))
        configure_moving_asset(stage, "/World/PreviewMovingAsset", moving_mass)

        instances = episode["fixture_instances" if moving_fixture else "payload_instances"]
        appearance = instances[0]["material"]
        cfg = visual_material_cfg(task, {"type": asset_type}, appearance=appearance)
        with sim_utils.use_stage(stage):
            cfg.func("/World/PreviewMovingAssetLook", cfg)
        binding = UsdShade.MaterialBindingAPI.Apply(xform.GetPrim())
        material = UsdShade.Material(stage.GetPrimAtPath("/World/PreviewMovingAssetLook"))
        binding.Bind(material, UsdShade.Tokens.strongerThanDescendants)


def build_tabletop_preview(suite: str, task_id: int, output: Path,
                           seed: int | None = None) -> Path:
    """Compose source assets without changing their authored USD layers."""
    from pxr import Gf, Usd, UsdGeom
    from benchmark.scene.visuals import author_task_lighting
    from benchmark.scene.reset import sample_reset

    task = get_task(suite, task_id)
    sampled = sample_reset(task, 0 if seed is None else seed, allow_in_hand=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    def place(name, asset, position, scale, quaternion=None, euler=None):
        xform = UsdGeom.Xform.Define(stage, f"/World/{name}")
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        if quaternion is not None:
            w, x, y, z = quaternion
            xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        elif euler is not None:
            xform.AddRotateXYZOp().Set(Gf.Vec3d(*(math.degrees(value) for value in euler)))
        xform.AddScaleOp().Set(Gf.Vec3d(*scale))
        reference = UsdGeom.Xform.Define(stage, f"/World/{name}/Asset")
        reference.GetPrim().GetReferences().AddReference(str(asset.resolve()))

    for name, fixture in task["fixtures"].items():
        place(name, object_asset(fixture["type"]), fixture["initial_pose"],
              fixture["scale"], quaternion=fixture["initial_quat"])

    for name, spec in task["objects"].items():
        pose = sampled[name]
        place(name, object_asset(spec["type"]), pose["position"], spec["scale"],
              quaternion=pose.get("quaternion_wxyz"), euler=pose.get("euler_xyz_rad"))

    place("Robot", ASSETS / ROBOT_ASSET,
          task["robot_base_pos"], (1.0, 1.0, 1.0), quaternion=task["robot_base_ori"])

    author_task_lighting(stage, task, persistent=True)
    stage.GetRootLayer().Save()
    return output


def pose_preview_robot(stage, robot_path: str = "/World/Robot/Asset",
                       task: dict | None = None) -> None:
    """Show the task's downward GelSight pose in the session layer."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from yourdfpy import URDF

    source = Usd.Stage.Open(str((ASSETS / ROBOT_ASSET).resolve()))
    urdf = URDF.load(str((ASSETS / 'robots/Franka_gsmini/panda_arm_hand/panda_arm_hand.urdf').resolve()))
    cache = UsdGeom.XformCache()
    zero = {name: np.array(cache.GetLocalToWorldTransform(
        source.GetPrimAtPath(f"/panda/{name}"))).T for name in urdf.link_map}
    urdf.update_cfg(robot_joint_defaults(task))

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name in (*urdf.link_map, "gelsight_mini_case_left", "gelsight_mini_case_right",
                     "gelpad_left", "gelpad_right"):
            if name in urdf.link_map:
                transform = urdf.get_transform(name)
            else:
                finger = "panda_leftfinger" if name.endswith("left") else "panda_rightfinger"
                sensor_zero = np.array(cache.GetLocalToWorldTransform(
                    source.GetPrimAtPath(f"/panda/{name}"))).T
                transform = (urdf.get_transform(finger) @ np.linalg.inv(zero[finger])
                             @ sensor_zero)
            prim = stage.GetPrimAtPath(f"{robot_path}/{name}")
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp(opSuffix="previewHome").Set(
                Gf.Matrix4d(*transform.T.reshape(-1).tolist()))


def pose_preview_held_object(stage, task: dict, robot_path: str) -> None:
    """Align a held object's authored grasp point with the GelSight pad midpoint."""
    from pxr import Gf, Usd, UsdGeom

    dynamic = bool(task.get("scene_layout"))
    episode = task.get("episode_config") or {}
    contract = episode.get("in_hand_reset") if dynamic else task.get("reset_contract")
    if not contract or contract.get("mode") != "in_hand":
        return
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    if dynamic:
        asset = object_asset(episode["payload_asset_type"])
        object_path = "/World/PreviewHeldAsset"
        grasp = Gf.Vec3d(*episode["in_hand_grasp_local_m"])
        from benchmark.tasks.dynamic.scene import dynamic_asset_scale

        scale = dynamic_asset_scale(task, "object")
    else:
        object_name = contract["object"]
        asset = object_asset(task["objects"][object_name]["type"])
        object_path = f"/World/{object_name}"
        scale = tuple(task["objects"][object_name]["scale"])
        if "grasp_point_local" in contract:
            grasp = Gf.Vec3d(*contract["grasp_point_local"])
        else:
            source = Usd.Stage.Open(str(asset))
            grasp = bounds.ComputeWorldBound(source.GetDefaultPrim()).ComputeCentroid()
            if "grasp_center_local_xy" in contract:
                grasp = Gf.Vec3d(*contract["grasp_center_local_xy"], grasp[2])

    pads = [bounds.ComputeWorldBound(stage.GetPrimAtPath(
        f"{robot_path}/gelpad_{side}/mesh")).ComputeCentroid() for side in ("left", "right")]
    center = (pads[0] + pads[1]) / 2
    hand = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(f"{robot_path}/panda_hand"))
    local_quat = episode.get("in_hand_object_local_quat_wxyz", (0, 0, 1, 0))
    basis = Gf.Rotation(Gf.Quatd(local_quat[0], Gf.Vec3d(*local_quat[1:])))
    rotation = (hand.ExtractRotation() * basis
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(
                    float(contract.get("grasp_yaw_rad", 0.0)))))
    position = center - rotation.TransformDir(Gf.Vec3d(*(grasp[i] * scale[i] for i in range(3))))
    quat = rotation.GetQuat()

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        xform = (UsdGeom.Xform.Define(stage, object_path) if dynamic else
                 UsdGeom.Xform(stage.GetPrimAtPath(object_path)))
        if dynamic:
            xform.GetPrim().GetReferences().AddReference(str(asset.resolve()))
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(position)
        xform.AddOrientOp().Set(Gf.Quatf(quat.GetReal(), Gf.Vec3f(quat.GetImaginary())))
        xform.AddScaleOp().Set(Gf.Vec3d(*scale))
        if dynamic:
            from benchmark.scene.visuals import apply_dynamic_material

            apply_dynamic_material(stage, object_path, task, "object")


def apply_task_materials(task: dict) -> None:
    """Bind task visual materials with Isaac Lab; leave physical materials intact."""
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    stage = sim_utils.get_current_stage()
    UsdGeom.Scope.Define(stage, "/World/Looks")
    for name, spec in task["objects"].items():
        cfg = visual_material_cfg(task, spec)
        if cfg is None:
            continue
        material_path = f"/World/Looks/{name}"
        cfg.func(material_path, cfg)
        sim_utils.bind_visual_material(f"/World/{name}/Asset", material_path)


def _run_extra_preview(env, app, frames):
    """Advance the native simulation until the window closes or frames are exhausted."""
    count = 0
    while app.is_running() and (frames == 0 or count < frames):
        env.env.step(env.action)
        cutting = getattr(env, "cutting", None)
        if cutting is not None:
            cutting.update()
        count += 1


def main() -> None:
    import warp  # noqa: F401  -- use the installed Warp before Kit loads its bundle
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=COUNTS)
    parser.add_argument("task_id", type=int)
    parser.add_argument("--frames", type=int, default=0,
                        help="Exit after this many updates; 0 keeps the GUI window open")
    parser.add_argument("--seed", type=int, help="Tabletop reset seed (default: 0)")
    parser.add_argument("--camera", choices=CAMERAS, default="agentview",
                        help="Camera shown in the interactive viewport")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    task = get_task(args.suite, args.task_id)
    if args.suite == "tac_extra" and not is_tabletop_task(task):
        args.enable_cameras = True
    if args.headless and args.frames == 0:
        args.frames = 1
    if args.suite == "tac_dynamic":
        from benchmark.tasks.dynamic.scene import resolve_gsmini_task

        task = resolve_gsmini_task(task)
    if args.seed is not None and args.suite == "tac_dynamic":
        parser.error("Conveyor reset requires platform motion control")

    temporary_stage = (tempfile.TemporaryDirectory(prefix="tacmind_preview_")
                       if args.suite != "tac_dynamic" else nullcontext(None))
    with temporary_stage as temporary:
        app = AppLauncher(args).app
        try:
            from isaacsim.core.utils.stage import is_stage_loading
            if args.suite == "tac_extra" and not is_tabletop_task(task):
                from benchmark.tasks.extra.environment import ExtraEnvironment

                env = ExtraEnvironment(args.task_id, args.seed or 0, device=args.device)
                try:
                    if not args.headless:
                        from omni.kit.viewport.utility import get_active_viewport
                        camera = ("agentview" if args.camera == "agentview" else "panda_hand/wristview")
                        get_active_viewport().set_active_camera(f"/World/envs/env_0/Robot/{camera}")
                    _run_extra_preview(env, app, args.frames)
                finally:
                    env.close()
                return
            import isaaclab.sim as sim_utils
            from benchmark.scene.visuals import apply_task_render_settings, author_task_lighting

            apply_task_render_settings()

            path = (scene_asset(task) if args.suite == "tac_dynamic" else
                    build_tabletop_preview(args.suite, args.task_id,
                                           Path(temporary) / f"{args.suite}_{args.task_id}.usda",
                                           seed=args.seed))
            if not sim_utils.open_stage(str(path)):
                raise RuntimeError(f"Could not open {path}")
            stage = sim_utils.get_current_stage()
            robot_path = ("/World/FrankaArm" if args.suite == "tac_dynamic" else
                          "/World/Robot/Asset")
            if args.suite == "tac_dynamic":
                while app.is_running() and is_stage_loading():
                    app.update()
                if not app.is_running():
                    return
                stage = sim_utils.get_current_stage()
                from benchmark.tasks.dynamic.scene import _prepare_dynamic_scene, sample_dynamic_reset

                _prepare_dynamic_scene(
                    stage, task, sample_dynamic_reset(task, seed=0),
                    include_held_payload=False, drop_clearance_m=0.0,
                )
            else:
                apply_task_materials(task)
            pose_preview_robot(stage, robot_path, task)
            pose_preview_held_object(stage, task, robot_path)
            add_preview_cameras(stage, robot_path, suite=args.suite)
            author_task_lighting(stage, task)
            if not args.headless:
                from omni.kit.viewport.utility import get_active_viewport

                camera_path = (f"{robot_path}/agentview" if args.camera == "agentview" else
                               f"{robot_path}/panda_hand/wristview")
                get_active_viewport().set_active_camera(camera_path)
            print(f"[PREVIEW] {args.suite}:{args.task_id} {task['task_name']} | {path}", flush=True)
            count = 0
            while app.is_running() and (args.frames == 0 or count < args.frames):
                app.update()
                count += 1
        finally:
            app.close()


if __name__ == "__main__":
    main()
