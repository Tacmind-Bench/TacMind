"""Prepared contact-asset specifications and runtime collision support."""

from __future__ import annotations

from copy import deepcopy
import math

BALSA_DATASHEET_URL = "https://ochromagroup.com/wp-content/uploads/2025/01/SpecSheets/Ochroma-balsa-wood-specs-v2.pdf"
SPECS = {
    "extra_push_block": {"bodies": ["Block"], "mass_kg": 0.014720302298225406,
        "density_kg_m3": 150., "material": "balsa wood", "elastic_modulus_pa": 3.71e9,
        "density_basis": "Manufacturer average dried density; fixed representative engineering specimen, not measured hardware.",
        "material_source_url": BALSA_DATASHEET_URL,
        "size_m": [0.05, 0.05, 0.05], "radius_m": .025, "radial_segments": 128,
        "grasp_point_local_m": [-0.025, 0, 0.025],
        "bounds_m": [[-0.025, -0.025, 0], [0.025, 0.025, 0.05]],
        "model": "Independent 50 mm diameter, 50 mm tall balsa push block with 128 radial facets; the round pushing face allows the gel normal to follow the S curve. Actual closed mesh volume at manufacturer average dried density 150 kg/m3 determines mass and uniform-density inertia. Rigid approximation for low contact loads; no hardware calibration."},
    "extra_roll_can": {"bodies": ["Can"], "mass_kg": 0.2,
        "radius_m": 0.03384, "length_m": 0.101855, "axis": "Y",
        "bounds_m": [[-0.03384, -0.0509275, -0.03384], [0.03384, 0.0509275, 0.03384]],
        "collision_geometry": "UsdGeom.Cylinder",
        "collision_reference_url": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html",
        "angular_damping_s_inv": 2.0,
        "angular_damping_basis": "Rolling resistance of a hard can on a tabletop, Crr=0.0035; PhysX default 0.05 s^-1 is effectively frictionless rolling and leaves the reset settle rolling indefinitely.",
        "model": "Immutable NVIDIA can visuals and fitted independent cylinder collision; solid-cylinder inertia approximation and authored rolling resistance."},
    "extra_gear_transmission": {"bodies": ["Base", "Small", "Medium", "Large"],
        "mass_kg": {"Base": 0.05, "Small": 0.05, "Medium": 0.07160123857910965, "Large": 0.05},
        "input_knob": {"material": "PA12", "density_kg_m3": 1010.,
            "outer_across_flats_m": .03, "bore_across_flats_m": .01,
            "bottom_m": .03, "height_m": .03,
            "collar_inner_radius_m": .015, "collar_outer_radius_m": .017,
            "collar_bottom_m": .027, "collar_height_m": .003,
            "original_medium_mass_kg": .05},
        "grasp_point_local_m": [.02025, 0., .0495],
        "joints": ["small_turn", "medium_turn", "large_turn"],
        "tooth_counts": [20, 40, 60], "centers_x_m": [0.05075, 0.02025, -0.03025],
        "assembly_phase_degrees": [9., 0., 3.],
        "bearing_z_m": 0.0175, "bearing_damping_nm_s": 0.001,
        "output_load_damping_nm_s": 0.002,
        "bounds_m": [[-0.075, -0.0375, 0], [0.075, 0.0375, 0.06]],
        "model": "Unscaled NVIDIA Factory gears with their original 50 g masses and passive bearings. Independently engineered hollow PA12 hex input knob and socket collar add volume-derived mass/inertia to Medium; no additional joint, attachment or output drive. The elevated grasp clears output gears during the yaw sweep. Small/Large retain 9/3 degree assembly phases. Tooth contacts alone transmit torque; material density is a manufacturer typical value, not hardware calibration."},
    "extra_hanging_mug": {"bodies": ["Mug"], "mass_kg": 0.005569826903209591, "body_radius_m": 0.03,
        "body_center_x_m": -0.017, "height_m": 0.04, "wall_thickness_m": 0.0002,
        "handle_center_m": [-0.017, 0.0, 0.0698], "handle_center_local_m": [-0.017, 0.0, 0.0698],
        "handle_plane_normal_local": [0, 1, 0], "handle_ellipse_radii_m": [0.0175, 0.026],
        "handle_tube_radius_m": 0.0038, "handle_inner_ellipse_radii_xz_m": [0.0137, 0.0222],
        "handle_grip_center_local_m": [-0.017, 0.0, 0.1348],
        "handle_grip_size_m": [0.018, 0.0175, 0.018], "grasp_point_local_m": [-0.017, 0.0, 0.1348],
        "bounds_m": [[-0.047, -0.03, 0.0], [0.013, 0.03, 0.1438]],
        "collision_regions": {"body": "Mug/Body", "handle": "Mug/Handle", "grip": "Mug/Grip"},
        "model": "Thin-wall PP cup with raised hollow bail, hollow upper grip and four integral contact rails. Actual material union at 900 kg/m3 determines mass, COM and inertia. Open bowl and handle tunnel remain open; no GSmini changes or calibrated hardware claim.",
        "grip_shell_size_m": [0.018, 0.0135, 0.018], "handle_wall_thickness_m": 0.0004,
        "grip_wall_thickness_m": 0.0005, "body_top_m": 0.04075, "density_kg_m3": 900.0,
        "elastic_modulus_pa": 1600000000.0,
        "material": "Moplen HP640J polypropylene engineering selection",
        "material_source_url": "https://www.lyondellbasell.com/en/polymers/p/Moplen-HP640J/f2e8cae3-2ee3-4464-bf3b-153fdf2913f5"},
    "extra_mug_hook": {"bodies": ["Hook"], "mass_kg": 0.2, "fixed_mount": True,
        "shaft_radius_m": 0.002, "hook_height_m": 0.12,
        "tip_local_m": [0, -0.005, 0.138], "seat_local_m": [0, 0, 0.122],
        "bend_local_m": [0, -0.005, 0.12], "insertion_normal_local": [0, 1, 0],
        "post_bounds_m": [[-0.005, 0.075, 0.008], [0.005, 0.085, 0.12]],
        "base_bounds_m": [[-0.03, 0.06, 0], [0.03, 0.1, 0.008]],
        "bounds_m": [[-0.03, -0.007, 0], [0.03, 0.1, 0.14]],
        "model": "Independent fixed steel-like hook, post and base; compound engineering inertia."},
    "extra_slip_rod": {"bodies": ["Rod"], "mass_kg": .006048,
        "density_kg_m3": 150., "material": "balsa wood", "elastic_modulus_pa": 3.71e9,
        "material_source_url": BALSA_DATASHEET_URL,
        "axis": "Z", "length_m": .120,
        "core_size_xy_m": [.020, .014], "contact_strip_size_xy_m": [.004, .0035],
        "contact_strip_centers_xy_m": [[x, y] for x in (-.006, .006) for y in (-.00875, .00875)],
        "grasp_point_local_m": [0., 0., .020],
        "bounds_m": [[-.010, -.0105, -.060], [.010, .0105, .060]],
        "model": "Solid 120 mm balsa dowel: 20x14 mm core with two 4x3.5 mm longitudinal contact strips on each closing face. Constant axial section without stops. Five disjoint convex material solids preserve the channels; the visible closed union determines mass, COM and inertia at 150 kg/m3. No GSmini changes or hardware calibration."},
}


def asset_specs():
    """Simulator-independent geometry/interface contract; no readiness implied."""
    return deepcopy(SPECS)


def supported_roll_reset_poses(env, poses):
    """Place the reset cylinder on the native tabletop without an initial fall."""
    import omni.physx
    from scipy.spatial.transform import Rotation
    from benchmark.tasks.extra.contact import cylinder_minimum_z

    if not env._resetting:
        raise RuntimeError("Rolling support placement is allowed only during reset")
    result = deepcopy(poses)
    pose = result["mechanism"]
    heights = []
    def hit(item):
        if "/Table/" in item.collision:
            heights.append(float(item.position[2]))
        return True
    omni.physx.get_physx_scene_query_interface().raycast_all(
        (float(pose["position"][0]),float(pose["position"][1]),1.), (0.,0.,-1.), 1., hit)
    if not heights:
        raise RuntimeError("No native tabletop beneath the rolling cylinder")
    q = pose["quaternion_wxyz"]
    axis_z = Rotation.from_quat([*q[1:],q[0]]).apply([0.,1.,0.])[2]
    radius = env.task["goals"][0]["radius_m"]
    extent = -cylinder_minimum_z(0.,float(axis_z),radius,SPECS["extra_roll_can"]["length_m"])
    pose["position"][2] = max(heights)+extent
    return result


def roll_collision_support(prim, matrix, rotation, position, scale):
    """Check the actual can collider; native primitives have no mesh vertices."""
    import numpy as np
    from pxr import Gf, UsdGeom
    from benchmark.tasks.extra.contact import cylinder_minimum_z

    geometry = SPECS["extra_roll_can"]
    radius, length, axis = geometry["radius_m"], geometry["length_m"], "Y"
    matches = np.allclose(np.asarray(matrix), np.eye(4), atol=1e-7, rtol=0.) and scale[0] == scale[2]
    primitive = prim.IsA(UsdGeom.Cylinder)
    if primitive:
        shape = UsdGeom.Cylinder(prim)
        radius, length, axis = shape.GetRadiusAttr().Get(), shape.GetHeightAttr().Get(), shape.GetAxisAttr().Get()
        matches &= (axis == "Y" and math.isclose(radius, geometry["radius_m"], abs_tol=1e-8)
                    and math.isclose(length, geometry["length_m"], abs_tol=1e-8))
    elif not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Unsupported rolling collider: {prim.GetPath()} ({prim.GetTypeName()})")
    axis_z = float(rotation.apply(np.eye(3)[("X", "Y", "Z").index(axis)])[2])
    minimum = cylinder_minimum_z(float(position[2]), axis_z, radius*scale[0], length*scale[1])
    report = {"support_method": "finite_cylinder", "collision_primitive": prim.GetTypeName()}
    if not primitive:
        points = [matrix.Transform(Gf.Vec3d(*point)) for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
        vertex_minimum = float((rotation.apply(np.asarray(points))+position)[:, 2].min())
        difference = abs(minimum-vertex_minimum)
        report.update(collision_vertex_minimum_z_m=vertex_minimum, analytic_vertex_difference_m=difference)
        matches &= difference <= .0001
    report["matches_declared_geometry"] = bool(matches)
    return minimum, report


