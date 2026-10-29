"""Shared scene lighting, rendering and visual material profiles."""

from __future__ import annotations

from pathlib import Path

from benchmark.catalog import ASSETS


DOME_INTENSITY = 500.0
WORKSPACE_KEY_INTENSITY = 5000.0
WORKSPACE_KEY_RADIUS = 0.5
WORKSPACE_KEY_OFFSET = (0.0, -0.6, 0.7)

TASK_QUALITY_SETTINGS = {
    "/rtx/post/aa/op": 4,  # DLAA at native resolution.
    "/rtx/translucency/enabled": True,
    "/rtx/ambientOcclusion/enabled": True,
    "/rtx/ambientOcclusion/denoiserMode": 0,
    "/rtx/reflections/enabled": True,
    "/rtx/reflections/denoiser/enabled": True,
    "/rtx/directLighting/sampledLighting/enabled": True,
    "/rtx/sceneDb/ambientLightIntensity": 1.0,
    "/rtx/shadows/enabled": True,
    "/rtx/indirectDiffuse/enabled": True,
    "/rtx/indirectDiffuse/denoiser/enabled": True,
    "/rtx/raytracing/subpixel/mode": 1,
    "/rtx/raytracing/cached/enabled": True,
    "/rtx-transient/dlssg/enabled": False,
    "/rtx-transient/dldenoiser/enabled": True,
    "/rtx/pathtracing/maxSamplesPerLaunch": 1000000,
    "/rtx/viewTile/limit": 1000000,
}

TASK_EXPOSURE_SETTINGS = {
    "/rtx/post/histogram/enabled": True,
    "/rtx/post/histogram/tau": 3.5,
    "/rtx/post/histogram/filterType": "median",
    "/rtx/post/histogram/whiteScale": 10.0,
    "/rtx/post/histogram/useExposureClamping": True,
    "/rtx/post/histogram/minEV": 50.0,
    "/rtx/post/histogram/maxEV": 100000.0,
    "/rtx/post/tonemap/enabled": True,
    "/rtx/post/tonemap/op": 4,
    "/rtx/post/tonemap/filmIso": 200.0,
    "/rtx/post/tonemap/whitepoint": 6500.0,
}


def task_background_path(task: dict) -> Path:
    """Resolve a tabletop task's HDR background."""
    background = task.get("background")
    if not background:
        raise ValueError(f"Task {task.get('task_id')} has no HDR background")
    path = ASSETS / "backgrounds" / background
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def workspace_key_position(task: dict, surface_z: float | None = None) -> tuple[float, float, float]:
    if surface_z is None:
        offset = task.get("workspace_offset")
        if not offset:
            raise ValueError(f"Task {task.get('task_id')} requires a workspace surface height")
        surface_z = float(offset[2])
    return (WORKSPACE_KEY_OFFSET[0], WORKSPACE_KEY_OFFSET[1],
            float(surface_z) + WORKSPACE_KEY_OFFSET[2])


def stage_surface_z(stage, prim_path: str | None = None) -> float:
    """Measure a tabletop's world height from its USD bounds."""
    from pxr import Usd, UsdGeom

    paths = (prim_path,) if prim_path else ("/World/BigTable", "/World/Table")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim:
            return float(bounds.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2])
    raise ValueError(f"No tabletop prim found for paths: {paths}")


def configure_task_lighting(scene_cfg, task: dict, *, surface_z: float | None = None) -> None:
    """Light tabletop scenes; retain an authored Dynamic scene's own lights."""
    if task.get("scene_layout"):
        scene_cfg.light = None
        scene_cfg.workspace_key = None
        return
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/TaskVisuals/Background",
        spawn=sim_utils.DomeLightCfg(
            texture_file=str(task_background_path(task)),
            texture_format="latlong",
            visible_in_primary_ray=True,
            intensity=DOME_INTENSITY,
        ),
    )
    scene_cfg.workspace_key = AssetBaseCfg(
        prim_path="/World/TaskVisuals/WorkspaceKey",
        init_state=AssetBaseCfg.InitialStateCfg(pos=workspace_key_position(task, surface_z)),
        spawn=sim_utils.SphereLightCfg(
            intensity=WORKSPACE_KEY_INTENSITY,
            radius=WORKSPACE_KEY_RADIUS,
        ),
    )


def expand_robot_visual_instances(stage, robot_path: str) -> None:
    """Avoid missing Hydra prototypes without changing mesh or physics properties."""
    import isaaclab.sim as sim_utils
    from pxr import Usd

    robot = stage.GetPrimAtPath(robot_path)
    if not robot:
        raise ValueError(f"Missing robot prim: {robot_path}")
    paths = [str(prim.GetPath()) for prim in Usd.PrimRange(robot)
             if prim.GetName() == "visuals"]
    # Keep assets and collision instances intact; author only session metadata.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path in paths:
            sim_utils.make_uninstanceable(path, stage=stage)


def hide_environment_visuals(stage, *, persistent: bool = False) -> None:
    """Expose the task HDR while retaining ground/warehouse collision geometry."""
    from pxr import Usd, UsdGeom

    with Usd.EditContext(stage, stage.GetRootLayer() if persistent else stage.GetSessionLayer()):
        for path in ("/World/GroundPlane/Environment", "/World/Background"):
            prim = stage.GetPrimAtPath(path)
            if prim:
                UsdGeom.Imageable(prim).MakeInvisible()


def author_task_lighting(stage, task: dict, *, surface_z: float | None = None,
                         persistent: bool = False) -> None:
    """Author tabletop lights without overriding an authored Dynamic scene."""
    if task.get("scene_layout"):
        return
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    if surface_z is None and not task.get("workspace_offset"):
        surface_z = stage_surface_z(stage)

    with Usd.EditContext(stage, stage.GetRootLayer() if persistent else stage.GetSessionLayer()):
        replace_existing_lights(stage, persistent=persistent)
        hide_environment_visuals(stage, persistent=persistent)
        UsdGeom.Scope.Define(stage, "/World/TaskVisuals")
        dome = UsdLux.DomeLight.Define(stage, "/World/TaskVisuals/Background")
        dome.CreateTextureFileAttr(Sdf.AssetPath(str(task_background_path(task))))
        dome.CreateTextureFormatAttr("latlong")
        dome.CreateIntensityAttr(DOME_INTENSITY)
        dome.GetPrim().CreateAttribute(
            "visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, custom=False,
        ).Set(True)

        key = UsdLux.SphereLight.Define(stage, "/World/TaskVisuals/WorkspaceKey")
        key.CreateIntensityAttr(WORKSPACE_KEY_INTENSITY)
        key.CreateRadiusAttr(WORKSPACE_KEY_RADIUS)
        translate = next(
            (op for op in key.GetOrderedXformOps()
             if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if translate is None:
            translate = key.AddTranslateOp()
        translate.Set(Gf.Vec3d(*workspace_key_position(task, surface_z)))


def replace_existing_lights(stage, *, persistent: bool = False) -> None:
    """Mute authored USD lights in the session layer, preserving task-profile lights."""
    from pxr import Sdf, Usd, UsdGeom, UsdLux

    with Usd.EditContext(stage, stage.GetRootLayer() if persistent else stage.GetSessionLayer()):
        profile_root = Sdf.Path("/World/TaskVisuals")
        light_types = {
            "DomeLight", "RectLight", "DiskLight", "SphereLight",
            "DistantLight", "CylinderLight",
        }
        for prim in stage.Traverse():
            if prim.GetTypeName() not in light_types or prim.GetPath().HasPrefix(profile_root):
                continue
            light = UsdLux.LightAPI(prim)
            intensity = light.GetIntensityAttr()
            if intensity:
                intensity.Set(0.0)
            UsdGeom.Imageable(prim).MakeInvisible()


def _render_settings_interface(settings=None):
    if settings is None:
        try:
            import carb.settings
        except ImportError as exc:
            raise RuntimeError(f"task rendering requires an Isaac Sim runtime: {exc}") from exc
        settings = carb.settings.get_settings()
    return settings


def _apply_render_settings(settings, values: dict[str, int | float | str | bool]) -> None:
    for path, value in values.items():
        settings.set(path, value)


def apply_task_exposure_settings(settings=None) -> dict[str, int | float | str | bool]:
    """Apply task-specific exposure after Isaac Lab loads the Quality preset."""
    settings = _render_settings_interface(settings)
    _apply_render_settings(settings, TASK_EXPOSURE_SETTINGS)
    return dict(TASK_EXPOSURE_SETTINGS)


def apply_task_render_settings(settings=None) -> dict[str, int | float | str | bool]:
    """Apply Quality rendering with DLAA for previews without an environment cfg."""
    settings = _render_settings_interface(settings)
    _apply_render_settings(settings, TASK_QUALITY_SETTINGS)
    _apply_render_settings(settings, TASK_EXPOSURE_SETTINGS)
    return {**TASK_QUALITY_SETTINGS, **TASK_EXPOSURE_SETTINGS}


# TacBenchHub's ManiFeel runtime appearance profiles. The source task JSON
# omits them, so keep the visual-only settings here for the precision family.
PRECISION_VISUAL_MATERIALS = {
    "USB_plug": ((0.10, 0.12, 0.15), 0.22, 0.46),
    "USB_socket": ((0.08, 0.10, 0.16), 0.26, 0.72),
    "ball_class_golf": ((0.86, 0.86, 0.86), 0.28, 0.04),
    "ball_class_tabletennis": ((0.86, 0.86, 0.86), 0.28, 0.04),
    "ball_class_blank": ((0.86, 0.86, 0.86), 0.28, 0.04),
    "ball_class_base": ((0.08, 0.10, 0.15), 0.30, 0.70),
}

STEEL_MDL_PATH = (
    'assets/materials/Steel_A/Steel_A.mdl'
)
ROUGH_BLACK_PLASTIC_MDL_PATH = (
    'assets/materials/Plastic_Rough_Black_A/Plastic_Rough_Black_A.mdl'
)
PLASTIC_MDL_PATH = (
    'assets/materials/Plastic/Plastic.mdl'
)
# Reuse bundled PBR plastics for simple precision-family housings.
PRECISION_MDL_MATERIALS = {
    "box_battery": {
        "path": ROUGH_BLACK_PLASTIC_MDL_PATH,
        "project_uvw": True,
    },
    "power_plug": {
        "path": ROUGH_BLACK_PLASTIC_MDL_PATH,
        "project_uvw": True,
    },
    "tactile_bulb": {
        "path": PLASTIC_MDL_PATH,
        "project_uvw": True,
        "albedo_brightness": 0.94,
    },
    "power_socket": {
        "path": PLASTIC_MDL_PATH,
        "project_uvw": True,
        "albedo_brightness": 0.76,
    },
    "tactile_bulb_socket": {
        "path": PLASTIC_MDL_PATH,
        "project_uvw": True,
        "albedo_brightness": 0.82,
    },
}
# Recog Task 0 gives the tray and both loose nuts the same black appearance.
BLIND_MDL_MATERIALS = {
    "blind_ver2_tray_station": ROUGH_BLACK_PLASTIC_MDL_PATH,
    "factory_nut_m16_loose": ROUGH_BLACK_PLASTIC_MDL_PATH,
    "factory_nut_m8_loose": ROUGH_BLACK_PLASTIC_MDL_PATH,
}
PRECISION_STEEL_TYPES = frozenset({
    "gear_base", "gear_medium",
    "box_rectangular_peg_16mm",
})

DYNAMIC_APPEARANCE_MATERIALS = {
    "steel": ((0.46, 0.50, 0.54), 0.24, 0.88),
    "green": ((0.08, 0.52, 0.20), 0.12, 0.36),
    "white": ((0.92, 0.92, 0.88), 0.31, 0.02),
    "black": ((0.025, 0.025, 0.025), 0.42, 0.05),
}


def visual_material_for_task(task: dict, spec: dict) -> dict | None:
    if task.get("source_group") == "blind_verified":
        path = BLIND_MDL_MATERIALS.get(spec["type"])
        if path is not None:
            return {"kind": "mdl", "path": path, "project_uvw": True}
    if spec.get("visual_material"):
        return spec["visual_material"]
    if task.get("source_group") != "precision_family":
        return None
    if spec["type"] in PRECISION_STEEL_TYPES:
        return {"kind": "mdl", "path": STEEL_MDL_PATH, "project_uvw": True}
    mdl_material = PRECISION_MDL_MATERIALS.get(spec["type"])
    if mdl_material is not None:
        return {"kind": "mdl", **mdl_material}
    profile = PRECISION_VISUAL_MATERIALS.get(spec["type"])
    if profile is None:
        return None
    color, roughness, metallic = profile
    return {"kind": "preview", "color": color,
            "roughness": roughness, "metallic": metallic}


def visual_material_cfg(task: dict, spec: dict, *, appearance: str | None = None):
    """Resolve task or dynamic appearance through Isaac Lab material configs."""
    import isaaclab.sim as sim_utils
    from benchmark.catalog import ROOT

    if appearance is None:
        material = visual_material_for_task(task, spec)
    else:
        try:
            color, roughness, metallic = DYNAMIC_APPEARANCE_MATERIALS[appearance]
        except KeyError:
            raise ValueError(f"Unknown dynamic appearance: {appearance}") from None
        material = {"kind": "preview", "color": color,
                    "roughness": roughness, "metallic": metallic}
    if not material:
        return None
    if material["kind"] == "mdl":
        return sim_utils.MdlFileCfg(
            mdl_path=str(ROOT / material["path"]),
            project_uvw=material.get("project_uvw"),
            albedo_brightness=material.get("albedo_brightness"),
            texture_scale=material.get("texture_scale"),
        )
    if material["kind"] == "preview":
        return sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(material["color"]),
            roughness=material["roughness"], metallic=material["metallic"],
        )
    raise ValueError(f"Unsupported visual material: {material['kind']}")


def apply_dynamic_material(stage, prim_path: str, task: dict, role: str, *,
                           asset_type: str | None = None, appearance: str | None = None) -> None:
    """Apply the role's visual appearance without changing physical materials."""
    import isaaclab.sim as sim_utils
    from pxr import UsdShade

    episode = task["episode_config"]
    asset_type = asset_type or episode[
        "payload_asset_type" if role == "object" else "fixture_asset_type"]
    if appearance is None:
        instances = episode.get("payload_instances" if role == "object" else "fixture_instances", ())
        appearance = instances[0].get("material") if instances else None
    cfg = visual_material_cfg(task, {"type": asset_type}, appearance=appearance)
    if cfg is not None:
        material_path = f"{prim_path}/TacMindMaterial"
        with sim_utils.use_stage(stage):
            cfg.func(material_path, cfg)
        binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prim_path))
        binding.Bind(UsdShade.Material(stage.GetPrimAtPath(material_path)),
                     UsdShade.Tokens.strongerThanDescendants)
