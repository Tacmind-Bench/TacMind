"""Prepared asset identities and numerical specifications used at runtime."""

SPECS = {
    'length_m': .13335, 'height_m': .03125,
    'notch_center_x_m': .07225-.13335/2,
    'notch_width_m': .0015, 'notch_depth_m': .0038,
    'pcb_thickness_m': .0014, 'spreader_thickness_m': .0088,
    'spreader_bottom_m': .012, 'spreader_length_m': .127,
    'spreader_height_m': .044, 'spreader_wall_m': .0006,
    'slot_width_m': .0018, 'socket_outer_width_m': .008,
    'socket_height_m': .008, 'floor_height_m': .002,
    'mount_height_m': .035,
    'end_clearance_m': .00025, 'key_width_m': .0011, 'key_height_m': .0035,
    'contact_offset_m': .00005, 'rest_offset_m': 0.,
    'grasp_point_module_m': [0., 0., -.005],
    'grip_tape_thickness_m': .0008, 'grip_tape_length_m': .060,
    'grip_tape_bottom_m': .030, 'grip_tape_height_m': .014,
    'grip_tape_areal_mass_kg_m2': .440, 'grip_tape_friction': 1.1,
    'grip_extension_length_m': .040, 'grip_extension_bottom_m': -.015,
    'grip_extension_top_m': .018, 'grip_extension_inner_y_m': .006,
    'grip_extension_projection_m': .0018, 'grip_extension_bridge_bottom_m': .014,
}
