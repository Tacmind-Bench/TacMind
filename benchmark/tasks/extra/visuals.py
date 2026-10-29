"""Shared Extra presentation, framing and native line-of-sight checks."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import json

from benchmark.catalog import ASSETS


@lru_cache(maxsize=1)
def presentation_profiles():
    return json.loads((ASSETS / "scenes/extra/visuals.json").read_text())


def configure_extra_presentation(task):
    """Resolve one presentation for catalog, preview, reset and collection."""
    from copy import deepcopy

    config = presentation_profiles()
    profile = config["tasks"][str(task["task_id"])]
    task["background"] = profile["background"]
    task["visual_profile"] = deepcopy(profile)
    task["view_contract"] = deepcopy(config["view_contract"])
    task["reset_contract"].update(deepcopy(profile.get("reset_overrides", {})))
    for name, spec in task["objects"].items():
        spec.update(deepcopy(profile.get("object_poses", {}).get(name, {})))
        spec["view_bounds_local_m"] = config["asset_bounds"][spec["type"]]["bounds_m"]


def bounds_corners(bounds):
    return list(product(*zip(*bounds)))


def contact_sweep_points(task, poses):
    """Conservative payload bounds along the nominal contact task paths.

    This bounds piecewise linear payload transport, not an arbitrary planner
    trajectory or a physically executed demonstration.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    kind = task["episode_config"]["task_type"]
    if kind in {"wipe_bottle", "wipe_whiteboard"}:
        from benchmark.tasks.extra.wiping import surface_cells, surface_frame_field
        spec = task["objects"]["mechanism"]
        local = np.asarray(bounds_corners(spec["view_bounds_local_m"]))
        q = poses["surface"]["quaternion_wxyz"]
        frame = Rotation.from_quat([*q[1:], q[0]])
        cells = surface_cells(task)
        # Exact authored tool bounds across the raster, including withdrawal.
        tool_frame = surface_frame_field(task)
        points = np.concatenate([cell+tool_frame(cell).apply(local) for cell in cells])
        points = np.concatenate((points,points+[0.,0.,.05]))
        sweep = frame.apply(points)+poses["surface"]["position"]
        q = poses["mechanism"]["quaternion_wxyz"]
        pickup = Rotation.from_quat([*q[1:],q[0]]).apply(local)+poses["mechanism"]["position"]
        return np.concatenate((sweep,pickup,pickup+[0.,0.,.12]))
    if kind not in ("push", "roll", "hang", "slip", "stand", "ram"):
        return np.empty((0, 3))
    spec, goal = task["objects"]["mechanism"], task["goals"][0]
    bounds = np.asarray(spec["view_bounds_local_m"])*spec["scale"]
    q = poses["mechanism"]["quaternion_wxyz"]
    rotation = Rotation.from_quat((*q[1:], q[0]))
    origin = np.asarray(poses["mechanism"]["position"])
    if kind == "ram":
        corners = rotation.apply(bounds_corners(bounds))+origin
        return np.concatenate((corners, corners+[0., 0., .04]))
    if kind == "stand":
        grasp = np.asarray(spec['grasp_point_local_m'])
        radius = float(np.linalg.norm(np.asarray(bounds_corners(bounds))-grasp,axis=1).max())
        corners = np.asarray(bounds_corners([[-radius]*3, [radius]*3]))
        pickup = origin+rotation.apply(grasp)
        axis = rotation.apply([0.,0.,1.])
        yaw = np.arctan2(axis[1],axis[0])
        target = origin+Rotation.from_euler('z',yaw).apply(goal['placement_offset_local_m'])
        target[2] = task['objects']['cradle']['initial_position'][2]-.0002+abs(bounds[:,2]).max()+grasp[2]
        # The target tolerance is included in the complete reset-range bound.
        target_corners = corners.copy()
        target_corners[:,:2] += np.sign(target_corners[:,:2])*goal['placement_tolerance_m']
        return np.concatenate((corners+pickup, corners+pickup+[0.,0.,.10],
                               target_corners+target, target_corners+target+[0.,0.,.08]))
    if kind == "hang":
        from benchmark.tasks.extra.contact_assets import asset_specs
        specs = asset_specs()
        hook = poses["hook"]
        hq = hook["quaternion_wxyz"]
        hr = Rotation.from_quat([*hq[1:], hq[0]])
        seat = np.asarray(hook["position"])+hr.apply(specs["extra_mug_hook"]["seat_local_m"])
        tip = np.asarray(hook["position"])+hr.apply(specs["extra_mug_hook"]["tip_local_m"])
        mug = specs["extra_hanging_mug"]
        handle_local = np.asarray(mug["handle_center_local_m"])
        grasp_local = handle_local+np.array([mug["handle_ellipse_radii_m"][0], 0., 0.])
        direction = rotation.apply([0., 1., 0.])
        entry = origin+rotation.apply(grasp_local)+[0., 0., .10]
        raised = origin+rotation.apply(grasp_local)+[0., 0., .11]
        cross_center = np.asarray([tip[0], tip[1], tip[2]+.017])
        prethread_center = cross_center-direction*.10
        lower_center = np.asarray([tip[0], tip[1],
            seat[2]+mug["handle_inner_ellipse_radii_xz_m"][1]+.0002])
        roots = [origin, entry-rotation.apply(grasp_local), raised-rotation.apply(grasp_local),
                 prethread_center-rotation.apply(handle_local),
                 cross_center-rotation.apply(handle_local),
                 lower_center-rotation.apply(handle_local)]
        vertices = rotation.apply(bounds_corners(bounds))
        return np.concatenate([vertices+position for position in roots])
    if kind == "slip":
        extent = np.max(abs(bounds), axis=0)
        # Axisymmetric yaw is unrestricted; the axis can tilt by two degrees.
        radial = np.linalg.norm(extent[:2])
        tilt = goal["max_axis_tilt_rad"]
        xy = radial+extent[2]*np.sin(tilt)+goal["max_hand_translation_m"]
        z = extent[2]+radial*np.sin(tilt)
        local = np.asarray(bounds_corners([[-xy, -xy, -z], [xy, xy, z]]))
        return np.concatenate((rotation.apply(local)+origin,
                               rotation.apply(local)+origin+[0., 0., .1]))
    if kind == "push":
        # The free block can yaw while being pushed. Its XY circumradius
        # bounds every orientation; the rectangle bounds the entire S curve.
        radius = np.linalg.norm(np.max(np.abs(bounds[:, :2]), axis=0))
        tolerance = goal["cross_track_m"]
        bounds[:, :2] = [[-radius-tolerance, -radius-goal["amplitude_m"]-tolerance],
                         [radius+goal["length_m"]+tolerance, radius+goal["amplitude_m"]+tolerance]]
        bounds[1, 2] += goal["max_lift_m"]
    else:
        # Rotate the entire authored box through the half-turn. The box's XZ
        # circumradius bounds all roll angles; the norm perturbation also
        # bounds every permitted off-axis rotation, including axle tilt.
        extent = np.max(abs(bounds), axis=0)
        radial = np.linalg.norm(extent[[0, 2]])
        margin = 2*np.linalg.norm(extent)*np.sin(goal["max_off_axis_rad"]/2)
        bounds = np.array([[-radial-margin, -extent[1]-margin, -radial-margin],
                           [radial+margin, extent[1]+margin, radial+margin]])
        bounds[0, 0] -= goal["translation_tolerance_m"]
        bounds[1, 0] += (goal["radius_m"]*(goal["target_rad"]+goal["tolerance_rad"])
                         + goal["translation_tolerance_m"])
    return rotation.apply(bounds_corners(bounds))+origin


def layout_frame_report(task, poses):
    """Conservative complete-asset bounds, including in-hand tools, before spawn."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from benchmark.scene.cameras import camera_frame_report

    rows = {}
    for name, spec in task["objects"].items():
        q = poses[name]["quaternion_wxyz"]
        rotation = Rotation.from_quat((*q[1:], q[0]))
        local = np.asarray(bounds_corners(spec["view_bounds_local_m"]))*spec["scale"]
        points = rotation.apply(local)+poses[name]["position"]
        if name == "mechanism":
            points = np.concatenate((points, contact_sweep_points(task, poses)))
        rows[name] = camera_frame_report(points, base_position=task["robot_base_pos"],
            base_rotation=task["robot_base_ori"], border=task["view_contract"]["border_fraction"])
    presentation_task = task.get("scene_template", {}).get("task_id", task.get("task_id", -1))
    full_range = reset_range_frame_report(task) if presentation_task >= 10 else None
    return {"passed": all(r["passed"] for r in rows.values())
                      and (full_range is None or full_range["passed"]),
            "objects": rows, "full_reset_range": full_range}


def reset_range_frame_report(task):
    """Bound the entire common SE(2) reset interval, without rejection sampling.

    Camera-plane extrema include interior yaw angles and all translation
    corners. This is a geometry/framing check, not a reachability/occlusion
    certificate; hand-relative tools require live robot forward kinematics.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from benchmark.scene.cameras import camera_frame_report

    anchor = np.asarray(task["objects"][task["obj_of_interest"][0]]["initial_position"])
    widths = np.asarray(task["reset_contract"]["xy_jitter_m"], dtype=float)
    yaw = float(task["reset_contract"]["yaw_jitter_rad"])
    if widths.shape != (2,) or not np.isfinite([*widths, yaw]).all() or min(*widths, yaw) < 0:
        raise ValueError("Extra reset jitter must be finite and nonnegative")
    poses = {}
    for name, spec in task["objects"].items():
        q = spec.get("initial_quat")
        if q is None:
            angle = spec.get("initial_yaw_rad", 0.)/2
            q = [np.cos(angle), 0., 0., np.sin(angle)]
        poses[name] = {"position": spec["initial_position"], "quaternion_wxyz": q}
    rows, unchecked = {}, []
    for name, spec in task["objects"].items():
        if spec.get("reset_frame") == "robot_hand":
            unchecked.append(name)
            continue
        q = poses[name]["quaternion_wxyz"]
        rotation = Rotation.from_quat([*q[1:], q[0]])
        local = np.asarray(bounds_corners(spec["view_bounds_local_m"]))*spec["scale"]
        points = rotation.apply(local)+poses[name]["position"]
        if name == "mechanism":
            points = np.concatenate((points, contact_sweep_points(task, poses)))
        start = Rotation.from_euler("z", -yaw).apply(points-anchor)+anchor
        extrema = sweep_vertices(start, {"axis": np.array([0., 0., 1.]),
            "pivot_m": anchor, "angle_rad": 2*yaw}, task)
        points = np.concatenate([extrema+[dx, dy, 0.]
            for dx, dy in product((-widths[0], widths[0]), (-widths[1], widths[1]))])
        rows[name] = camera_frame_report(points, base_position=task["robot_base_pos"],
            base_rotation=task["robot_base_ori"], border=task["view_contract"]["border_fraction"])
    return {"passed": bool(rows) and not unchecked and all(row["passed"] for row in rows.values()),
            "objects": rows, "unchecked_hand_relative_objects": unchecked,
            "xy_jitter_m": widths.tolist(), "yaw_jitter_rad": yaw,
            "scope": "Full common reset interval and authored bounds; push/roll/hang/slip/stand nominal payload swept bounds included. "
                     "Other task motion, joint reset ranges, visibility and reachability require separate checks."}


def hang_affordance_view_report(task, poses, surfaces):
    """Inspect the actual aperture and hook surfaces from the front AgentView.

    The 0.35 view cosine and four-pixel aperture are readability requirements,
    independent of physical clearance. Rays through the hole stop at its plane
    so background behind an open aperture does not count as an occluder.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from benchmark.scene.cameras import CAMERAS, camera_frame_report
    from benchmark.tasks.extra.contact_assets import asset_specs

    specs = asset_specs()
    mug, hook = specs["extra_hanging_mug"], specs["extra_mug_hook"]
    def transform(name, points):
        pose = poses[name]
        q = pose["quaternion_wxyz"]
        return Rotation.from_quat([*q[1:], q[0]]).apply(points)+pose["position"]
    center = np.asarray(mug["handle_center_local_m"])
    rx, rz = mug["handle_inner_ellipse_radii_xz_m"]
    theta = np.arange(64)*2*np.pi/64
    rim = center+np.column_stack((rx*np.cos(theta), np.zeros(64), rz*np.sin(theta)))
    frame = camera_frame_report(transform("mechanism", rim),
        base_position=task["robot_base_pos"], base_rotation=task["robot_base_ori"],
        border=task["view_contract"]["border_fraction"])
    origin, world_center = np.asarray(frame["camera_world_position_m"]), transform("mechanism", center)
    normal = transform("mechanism", np.asarray(mug["handle_plane_normal_local"]))-transform("mechanism", [0., 0., 0.])
    view = origin-world_center
    cosine = float(abs(normal@view)/np.linalg.norm(view))
    pixels = np.asarray(frame["uv"])*CAMERAS["agentview"]["resolution"]
    diameter = float(np.linalg.norm(pixels[:32]-pixels[32:], axis=1).min())
    area = float(abs(pixels[:, 0]@np.roll(pixels[:, 1], 1)-pixels[:, 1]@np.roll(pixels[:, 0], 1))/2)
    hole_rays = []
    samples = center+np.array([[0., 0., 0.], [rx/2, 0., 0.], [-rx/2, 0., 0.],
                              [0., 0., rz/2], [0., 0., -rz/2]])
    for target in transform("mechanism", samples):
        vector = target-origin
        distance = float(np.linalg.norm(vector))
        first = min(visible_surface_hits(surfaces, origin, vector/distance, distance-1e-5), default=(None, ""))
        hole_rays.append({"first_distance_m": first[0], "first_surface": first[1], "visible": first[0] is None})
    hook_rays = []
    # The authored seat is on the shaft's upper surface. Aim at its centerline
    # to avoid classifying an exactly tangential surface ray as a clear view.
    for label, point in (("tip", hook["tip_local_m"]),
            ("seat", np.asarray(hook["seat_local_m"])-[0., 0., hook["shaft_radius_m"]])):
        target = transform("hook", point)
        vector = target-origin
        distance = float(np.linalg.norm(vector))
        first = min(visible_surface_hits(surfaces, origin, vector/distance, distance+.003), default=(None, ""))
        expected = [f"/World/envs/env_0/hook/Hook/{part}"
                    for part in (("Upturn",) if label == "tip" else ("Shaft", "Upturn"))]
        hook_rays.append({"feature": label, "target_world_m": target.tolist(),
            "first_distance_m": first[0], "first_surface": first[1],
            "visible": any(first[1] == path or first[1].startswith(path+"/") for path in expected)})
    readable = bool(frame["passed"] and np.isfinite([cosine, diameter, area]).all()
                    and cosine >= .35 and diameter >= 4.)
    unobstructed = all(ray["visible"] for ray in hole_rays)
    return {"passed": readable and unobstructed and all(ray["visible"] for ray in hook_rays),
            "camera_world_position_m": origin.tolist(), "hole_center_world_m": world_center.tolist(),
            "hole_view_cosine": cosine, "minimum_view_cosine": .35,
            "hole_projected_area_px2": area, "hole_minimum_diameter_px": diameter,
            "minimum_diameter_px": 4., "hole_readable": readable, "hole_unobstructed": unobstructed,
            "hole_rays": hole_rays, "hook_rays": hook_rays,
            "scope": "Settled reset aperture readability and hook LOS; continuous execution visibility is separate."}


def apply_source_normal_map(prim, normal_map, strength):
    """Overlay only OmniPBR normal inputs in the session, retaining source UVs."""
    from pxr import Sdf, Usd, UsdShade

    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    shader = material.ComputeSurfaceSource("mdl")[0]
    stage = prim.GetStage()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(normal_map)))
        shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(float(strength))


def apply_extra_materials(env):
    """Bind mapped visual MDLs; explicit physical material bindings stay intact."""
    from fnmatch import fnmatchcase
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdShade
    from benchmark.catalog import ASSETS, ROOT

    config, stage = presentation_profiles(), env.sim.stage
    profile = env.task["visual_profile"]
    with Usd.EditContext(stage, stage.GetSessionLayer()), sim_utils.use_stage(stage):
        materials = {}
        used = {selected for rule in profile["materials"].values()
                for selected in [rule.get("default"), *rule.get("overrides", {}).values()] if selected}
        for name in sorted(used):
            material = config["materials"][name]
            cfg = sim_utils.MdlFileCfg(mdl_path=str(ROOT / material["path"]), project_uvw=True,
                                      texture_scale=tuple(float(x) for x in material.get("texture_scale", [1, 1])))
            path = f"/World/TaskVisuals/Materials/{name}"
            cfg.func(path, cfg)
            materials[name] = UsdShade.Material(stage.GetPrimAtPath(path))
        for name, rule in profile["materials"].items():
            root = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                    continue
                relative = str(prim.GetPath().MakeRelativePath(root.GetPath()))
                material = rule.get("default")
                for pattern, selected in rule.get("overrides", {}).items():
                    if fnmatchcase(relative, pattern):
                        material = selected
                if material:
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[material],
                        UsdShade.Tokens.strongerThanDescendants)
                elif rule.get("source_normal_map"):
                    apply_source_normal_map(prim, ASSETS/rule["source_normal_map"],
                                            rule.get("source_normal_strength", .12))


def articulation_sweep(env, root):
    """Read passive joint frames for the rail stroke or door's task arc."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from benchmark.tasks.extra.interaction import _rotation

    kind = env.task["episode_config"]["task_type"]
    if kind not in {"drawer", "door"}:
        return set(), None
    asset = env.scene["mechanism"]
    name = env.task["episode_config"]["target_joint"]
    joints = [UsdPhysics.Joint(p) for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.Joint)]
    joint = next(j for j in joints if j.GetPrim().GetName() == name)
    moving = {str(p) for p in joint.GetBody1Rel().GetTargets()}
    while True:
        previous = len(moving)
        for child in joints:
            if child.GetPrim().IsA(UsdPhysics.FixedJoint) and any(
                    str(p) in moving for p in child.GetBody0Rel().GetTargets()):
                moving.update(str(p) for p in child.GetBody1Rel().GetTargets())
        if len(moving) == previous:
            break
    body = joint.GetBody0Rel().GetTargets()[0].name
    body_index = asset.body_names.index(body)
    body_rotation = _rotation(asset.data.body_quat_w[0, body_index].cpu().tolist())
    local = joint.GetLocalRot0Attr().Get()
    rotation = body_rotation*_rotation([local.GetReal(), *local.GetImaginary()])
    typed = UsdPhysics.PrismaticJoint if kind == "drawer" else UsdPhysics.RevoluteJoint
    axis = rotation.apply(np.eye(3)["XYZ".index(typed(joint.GetPrim()).GetAxisAttr().Get())])
    index = asset.joint_names.index(name)
    if kind == "drawer":
        upper = float(asset.root_physx_view.get_dof_limits()[0,index,1])
        remaining = max(0., upper-float(asset.data.joint_pos[0,index]))
        return moving, {"translation_m":axis*remaining,
            "scope":"Full native prismatic stroke including fixed handle descendants"}
    import math
    body_prim = root.GetStage().GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    scale = np.asarray(Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(body_prim)).GetScale())
    pivot = asset.data.body_pos_w[0,body_index].cpu().numpy()+body_rotation.apply(
        np.asarray(joint.GetLocalPos0Attr().Get())*scale)
    angle = env.task["episode_config"]["joint_direction"]*(env.task["goals"][0]["minimum_rad"]+math.radians(6))
    return moving, {"axis":axis,"pivot_m":pivot,"angle_rad":angle,
        "scope":"Door task arc through the 60-degree goal plus 6-degree motion allowance; exact camera half-space extrema"}


def sweep_vertices(vertices, sweep, task):
    """Bound a rigid linear/hinged path without changing simulation state."""
    import numpy as np
    from benchmark.scene.cameras import CAMERAS
    from benchmark.tasks.extra.interaction import _rotation

    if "translation_m" in sweep:
        return vertices+sweep["translation_m"]
    camera = CAMERAS["agentview"]
    rotation = _rotation(task["robot_base_ori"])*_rotation(camera["rotation"])
    sx = camera["horizontal_aperture"]/camera["focal_length"]*(.5-task["view_contract"]["border_fraction"])
    sy = sx*camera["resolution"][1]/camera["resolution"][0]
    normals = rotation.apply([[1,0,sx],[-1,0,sx],[0,1,sy],[0,-1,sy],[0,0,1]])
    axis, pivot, end = sweep["axis"], sweep["pivot_m"], sweep["angle_rad"]
    offsets = vertices-pivot
    parallel = np.outer(offsets@axis,axis)
    a, b = offsets-parallel, np.cross(axis,offsets)
    critical = np.arctan2(b@normals.T,a@normals.T)
    angles = np.concatenate([np.zeros((len(vertices),1)),np.full((len(vertices),1),end),
                             critical,critical-np.pi,critical+np.pi],axis=1)
    valid = (angles>=min(0,end)) & (angles<=max(0,end))
    points = (pivot+parallel[:,None,:]+a[:,None,:]*np.cos(angles)[...,None]
              +b[:,None,:]*np.sin(angles)[...,None])
    return points[valid]


def native_view_report(env):
    """Check settled body bounds and native rays to the operation surface."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from benchmark.scene.cameras import camera_frame_report
    from benchmark.tasks.extra.interaction import _rotation, grasp_point

    stage = env.sim.stage
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=False)
    transforms, points, geometry, object_frames = UsdGeom.XformCache(), [], {}, {}
    stroke_points, sweep_config = [], None
    for name in env.task["objects"]:
        asset = env.scene[name]
        root = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
        moving, motion = articulation_sweep(env, root) if name == "mechanism" else (set(), None)
        vertices = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                continue
            box = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if box.IsEmpty():
                continue  # Empty/purpose-filtered prims have no rendered surface.
            matrix = transforms.GetLocalToWorldTransform(prim)*transforms.GetLocalToWorldTransform(body).GetInverse()
            local = np.asarray([matrix.Transform(Gf.Vec3d(*p)) for p in bounds_corners([list(box.GetMin()), list(box.GetMax())])])
            i = asset.body_names.index(body.GetName())
            p, q = asset.data.body_pos_w[0, i].cpu().numpy(), asset.data.body_quat_w[0, i].cpu().tolist()
            world = _rotation(q).apply(local)+p
            vertices.extend(world.tolist())
            if str(body.GetPath()) in moving:
                stroke_points.extend(sweep_vertices(world,motion,env.task).tolist())
                sweep_config = motion
        geometry[name] = len(vertices)
        object_frames[name] = camera_frame_report(vertices, base_position=env.task["robot_base_pos"],
            base_rotation=env.task["robot_base_ori"], border=env.task["view_contract"]["border_fraction"])
        points.extend(vertices)
    if env.task["episode_config"]["task_type"] in ("push", "roll", "hang", "slip", "stand", "ram", "wipe_bottle", "wipe_whiteboard"):
        poses = {name: {"position": env.scene[name].data.root_pos_w[0].cpu().numpy(),
                       "quaternion_wxyz": env.scene[name].data.root_quat_w[0].cpu().tolist()}
                 for name in env.task["objects"]}
        stroke_points.extend(contact_sweep_points(env.task, poses).tolist())
        sweep_config = {"kind": env.task["episode_config"]["task_type"],
                        "scope": "Nominal piecewise linear payload transport and conservative rotated geometry; "
                                 "planner/executed trajectory visibility requires separate validation"}
    report = camera_frame_report(points+stroke_points, base_position=env.task["robot_base_pos"],
        base_rotation=env.task["robot_base_ori"], border=env.task["view_contract"]["border_fraction"])
    sweep = (camera_frame_report(stroke_points, base_position=env.task["robot_base_pos"],
        base_rotation=env.task["robot_base_ori"], border=env.task["view_contract"]["border_fraction"])
        if stroke_points else None)
    if sweep is not None:
        sweep.update({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in sweep_config.items()})
    origin = np.asarray(report["camera_world_position_m"])
    # The point is an affordance centroid, often inside its collider. Stop rays
    # at their first surface and accept only the intended operation body.
    target = (env.scene["mechanism"].data.root_pos_w[0].cpu().numpy()
              if env.cutting is not None else grasp_point(env))
    rays = []
    spec = env.task["objects"]["mechanism"]
    body_names = spec.get("operation_contact_bodies", spec["contact_bodies"])
    allowed = [f"/World/envs/env_0/mechanism/{b}/" for b in body_names]
    if env.cutting is not None:
        allowed = [f"/World/envs/env_0/{name}/" for name in env.cutting.names]
    surfaces = visual_surfaces(env)
    targets = [(path, triangles) for path,triangles in surfaces
               if any((path+"/").startswith(prefix) for prefix in allowed)]
    for offset in ((0,0,0), (0,.003,0), (0,-.003,0), (0,0,.003), (0,0,-.003)):
        vector = target+offset-origin
        distance = float(np.linalg.norm(vector))
        hits = visible_surface_hits(surfaces, origin, vector/distance, distance+.01)
        target_hit = bool(visible_surface_hits(targets, origin, vector/distance, distance+.01))
        first = min(hits, default=(None, ""))
        rays.append({"first_distance_m": first[0], "first_collider": first[1],
                     "intersects_affordance": target_hit,
                     "visible": any((first[1]+"/").startswith(path) for path in allowed)})
    sampled = [r for r in rays if r["intersects_affordance"]]
    fraction = sum(r["visible"] for r in sampled)/len(sampled) if sampled else 0.0
    framed = report["passed"] and all(geometry.values())
    features = None
    if env.task["episode_config"]["task_type"] in {"wipe_bottle", "wipe_whiteboard"}:
        from benchmark.tasks.extra.wiping import operation_view_report
        features = operation_view_report(env, surfaces, origin)
    if env.task["episode_config"]["task_type"] == "surface_inspection":
        from benchmark.tasks.extra.inspection import front_window_report
        features = front_window_report(env, surfaces, origin)
    if env.task["episode_config"]["task_type"] == "hang":
        poses = {name: {"position": env.scene[name].data.root_pos_w[0].cpu().numpy(),
                       "quaternion_wxyz": env.scene[name].data.root_quat_w[0].cpu().tolist()}
                 for name in ("mechanism", "hook")}
        features = hang_affordance_view_report(env.task, poses, surfaces)
    return {**report, "geometry_vertex_counts": geometry, "object_frames": object_frames, "framing_passed": framed,
            "operation_features": features,
            "interaction_sweep": sweep,
            "ray_backend": "Rendered triangles transformed by native body poses; avoids stale CPU queries after GPU root resets",
            "affordance_rays": rays, "affordance_visible_fraction": fraction,
            "affordance_samples": len(sampled),
            "passed": framed and (features is None or features["passed"]) and rays[0]["visible"] and len(sampled)>=3
                      and fraction>=env.task["view_contract"]["minimum_visible_fraction"]}


def visual_surfaces(env):
    """Rendered triangles at current native poses, cached in rigid-body frames."""
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics
    from benchmark.scene.collision_geometry import mesh_geometry
    from benchmark.tasks.extra.interaction import _rotation

    if not hasattr(env, "_visual_surface_cache"):
        rows = []
        transforms = UsdGeom.XformCache()
        for name in [*env.task["objects"], "robot", "table"]:
            path = "Robot" if name == "robot" else "Table" if name == "table" else name
            root = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{path}")
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                imageable = UsdGeom.Imageable(prim)
                if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() not in {"default", "render"}:
                    continue
                if prim.IsA(UsdGeom.Mesh):
                    mesh = mesh_geometry(prim)
                elif prim.IsA(UsdGeom.Cube):
                    mesh = trimesh.creation.box(extents=[UsdGeom.Cube(prim).GetSizeAttr().Get()]*3)
                elif prim.IsA(UsdGeom.Cylinder):
                    shape = UsdGeom.Cylinder(prim)
                    mesh = trimesh.creation.cylinder(radius=shape.GetRadiusAttr().Get(), height=shape.GetHeightAttr().Get(), sections=64)
                    axis = shape.GetAxisAttr().Get()
                    if axis != "Z":
                        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0] if axis=="X" else [1,0,0]))
                elif prim.IsA(UsdGeom.Sphere):
                    mesh = trimesh.creation.icosphere(subdivisions=3, radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get())
                else:
                    raise ValueError(f"Uninspected rendered primitive: {prim.GetPath()} ({prim.GetTypeName()})")
                if not len(mesh.faces):
                    continue
                body = prim
                while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = body.GetParent()
                matrix = transforms.GetLocalToWorldTransform(prim)
                index = None
                if body and name != "table":
                    matrix *= transforms.GetLocalToWorldTransform(body).GetInverse()
                    index = env.scene[name].body_names.index(body.GetName())
                mesh.apply_transform(np.asarray(matrix).T)
                rows.append((str(prim.GetPath()), name, index, mesh.triangles.copy()))
        env._visual_surface_cache = rows
    result = []
    for path, name, index, triangles in env._visual_surface_cache:
        if index is not None:
            asset = env.scene[name]
            q = asset.data.body_quat_w[0,index].cpu().tolist()
            p = asset.data.body_pos_w[0,index].cpu().numpy()
            triangles = _rotation(q).apply(triangles.reshape(-1,3)).reshape(-1,3,3)+p
        result.append((path, triangles))
    return result


def visible_surface_hits(surfaces, origin, direction, maximum):
    """Broad-phase AABB followed by exact visual triangle intersections."""
    import numpy as np
    from types import SimpleNamespace
    from benchmark.scene.collision_geometry import _ray_distances

    hits = []
    for path, triangles in surfaces:
        low, high = triangles.min(axis=(0,1)), triangles.max(axis=(0,1))
        inverse = np.divide(1., direction, out=np.full(3, np.inf), where=abs(direction)>1e-12)
        a, b = (low-origin)*inverse, (high-origin)*inverse
        if max(float(np.minimum(a,b).max()), 0)>min(float(np.maximum(a,b).min()), maximum):
            continue
        values = _ray_distances(SimpleNamespace(triangles=triangles), origin, direction)
        if values and values[0]<=maximum:
            hits.append((values[0], path))
    return hits
