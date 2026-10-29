"""Apply opt-in evaluation camera and lighting settings to runtime scenes."""
from contextlib import contextmanager
from benchmark.scene.perturbations import backlight_key_position, horizontal_fov_rad, orbit_agentview


# Calibrate the conveyor's authored lights separately from tabletop presets.
# Night needs one additional stop; cool needs less neutral fill and a bluer tint.
# Keep camera auto-exposure unchanged.
AUTHORED_NIGHT_LIGHT_SCALE = 0.5
AUTHORED_COOL_COLOR = (0.25, 0.5, 1.0)
AUTHORED_COOL_AMBIENT_SCALE = 0.2
AMBIENT_LIGHT_INTENSITY = '/rtx/sceneDb/ambientLightIntensity'


@contextmanager
def evaluation_lighting(stage, look, *, authored_lights, camera_position, look_at, excluded_roots=()):
    """Scope global environment fill to the lifetime of an evaluation scene."""
    preset = look.lighting_look()
    settings = None
    if authored_lights and preset is not None and preset.name in ('night', 'cool'):
        from carb import settings as carb_settings
        settings = carb_settings.get_settings()
        original_ambient = settings.get(AMBIENT_LIGHT_INTENSITY)
        ambient_scale = (preset.dome_intensity_scale * 2 ** preset.exposure_ev
                         if preset.name == 'night' else AUTHORED_COOL_AMBIENT_SCALE)
        ambient = original_ambient * ambient_scale
    try:
        if settings is not None:
            settings.set(AMBIENT_LIGHT_INTENSITY, ambient)
        report = apply_eval_lighting(stage, look, authored_lights=authored_lights,
            camera_position=camera_position, look_at=look_at, excluded_roots=excluded_roots)
        if settings is not None:
            report['profile'] = f'dynamic_{preset.name}_v2'
            report['ambient_light'] = {'original_intensity': original_ambient, 'intensity': ambient}
        yield report
    finally:
        if settings is not None:
            settings.set(AMBIENT_LIGHT_INTENSITY, original_ambient)


def configure_eval_camera(scene, task, suite, look, *, look_at_z=None):
    """Apply an orbit from the authored robot-local baseline; never accumulate."""
    camera = scene.agentview
    baseline = getattr(camera, '_eval_camera_baseline', None)
    if baseline is None:
        if not look.camera:
            return None
        baseline = (tuple(camera.offset.pos), tuple(camera.offset.rot), tuple(camera.spawn.clipping_range))
        camera._eval_camera_baseline = baseline
    camera.offset.pos, camera.offset.rot, camera.spawn.clipping_range = baseline
    if not look.camera:
        return None
    if look_at_z is None:
        look_at_z = (.1 if suite == 'tac_dynamic' else
                     task['workspace_offset'][2] - task['robot_base_pos'][2])
    result = orbit_agentview(baseline[0], baseline[1], look_at_z=look_at_z,
        yaw_rad=look.signed_camera_yaw_rad, pitch_rad=look.signed_camera_pitch_rad,
        distance_scale=look.camera_distance_scale,
        horizontal_fov_rad=horizontal_fov_rad(camera.spawn.focal_length, camera.spawn.horizontal_aperture),
        clipping_range=baseline[2], pull_back=suite != 'tac_dynamic')
    camera.offset.pos = result.position
    camera.offset.rot = result.rotation_wxyz
    camera.spawn.clipping_range = result.clipping_range
    return result.to_dict()




def apply_eval_lighting(stage, look, *, authored_lights, camera_position, look_at, excluded_roots=()):
    """Apply each USD light look from its original values; identity restores them."""
    from pxr import Gf, Usd, UsdGeom, UsdLux

    preset = look.lighting_look()
    rows = []
    baseline_key = 'tacmind:evalLightBaseline'
    backlight_path = '/World/EvalBacklight'
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        # A synthetic source belongs to one look. Never let it enter the next
        # look's baseline or survive a switch back to authored lighting.
        if stage.GetPrimAtPath(backlight_path):
            stage.RemovePrim(backlight_path)
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            kind = prim.GetTypeName()
            if kind not in ('DomeLight', 'DistantLight', 'SphereLight', 'RectLight', 'DiskLight', 'CylinderLight'):
                continue
            if 'manifeel_white_bulb_light' in path.lower() or 'gsmini' in path.lower():
                continue
            if any(path == root or path.startswith(root+'/') for root in excluded_roots):
                continue
            if not authored_lights and not path.startswith('/World/TaskVisuals/'):
                continue
            light = UsdLux.LightAPI(prim)
            original = prim.GetCustomDataByKey(baseline_key)
            if not original:
                if preset is None:
                    continue
                original = {'intensity': float(light.GetIntensityAttr().Get()),
                            'exposure': float(light.GetExposureAttr().Get()),
                            'color': light.GetColorAttr().Get(),
                            'enable_temperature': light.GetEnableColorTemperatureAttr().Get(),
                            'temperature': float(light.GetColorTemperatureAttr().Get())}
                prim.SetCustomDataByKey(baseline_key, original)
            dome = kind == 'DomeLight'
            color, scale = original['color'], 1.0
            if preset is not None:
                color = preset.dome_color if dome else preset.color
                scale = preset.dome_intensity_scale if dome else preset.intensity_scale
                if authored_lights and preset.name == 'night':
                    scale *= AUTHORED_NIGHT_LIGHT_SCALE
                if authored_lights and preset.name == 'cool':
                    # The factory's neutral fill otherwise washes out this tint.
                    color = AUTHORED_COOL_COLOR
            light.CreateColorAttr(Gf.Vec3f(*color))
            light.CreateEnableColorTemperatureAttr(original['enable_temperature'] if preset is None else False)
            light.CreateColorTemperatureAttr(original['temperature'])
            light.CreateIntensityAttr(original['intensity'] * scale)
            light.CreateExposureAttr(original['exposure'] + (preset.exposure_ev if preset else 0.0))
            rows.append({'path': path, 'kind': kind, 'intensity': light.GetIntensityAttr().Get(),
                         'color': list(light.GetColorAttr().Get()), 'exposure': light.GetExposureAttr().Get()})
        if preset is not None and preset.key_placement == 'behind_subject':
            position = backlight_key_position(camera_position, look_at)
            direct_intensities = []
            for row in rows:
                if row['kind'] == 'DomeLight':
                    continue
                original = stage.GetPrimAtPath(row['path']).GetCustomDataByKey(baseline_key)
                direct_intensities.append(original['intensity'])
                # DistantLight has no local position to relocate and remains
                # active, matching the authored-scene perturbation contract.
                if row['kind'] != 'DistantLight':
                    UsdLux.LightAPI(stage.GetPrimAtPath(row['path'])).CreateIntensityAttr(0.0)
                    row['intensity'] = 0.0
            light = UsdLux.SphereLight.Define(stage, backlight_path)
            UsdGeom.XformCommonAPI(light).SetTranslate(Gf.Vec3d(*position))
            light.CreateColorAttr(Gf.Vec3f(*preset.color))
            light.CreateIntensityAttr(max(direct_intensities, default=5000.0) * preset.intensity_scale)
            light.CreateRadiusAttr(preset.key_radius)
            rows.append({'path': str(light.GetPath()), 'position': list(position),
                         'intensity': light.GetIntensityAttr().Get()})
    if preset is None:
        return None
    if not rows:
        raise RuntimeError('Lighting perturbation found no eligible scene lights')
    return {'preset': preset.name, 'authored_lights': authored_lights, 'lights': rows}

