"""Passive press-and-slide fixture; task assets and native history only.

This is an independently authored low-load engineering mechanism, not a
replica of the cited commercial latch or a calibrated hardware model.
"""

from copy import deepcopy
import math
from numbers import Real

import numpy as np

from benchmark.tasks.extra.contact_runtime import ContactRuntime, rotation


FIELDS = ('left_gel_force_n', 'right_gel_force_n', 'slide_m', 'depth_m',
          'joint_speed', 'keeper_clearance_m', 'keeper_force_n',
          'housing_contact_load_n', 'other_robot_force_n', 'robot_force_n',
          'penetration_m', 'attachment_error_m', 'attachment_angle_rad',
          'gel_compression_m', 'gel_thickness_m',
          'release_clearance_m', 'object_speed_m_s', 'angular_speed_rad_s')
SIGNED_FIELDS = frozenset(('slide_m', 'depth_m', 'keeper_clearance_m'))


class LatchProgress:
    """Depress at the start, traverse below a physical keeper, then release."""

    def __init__(self, task):
        self.task, self.kind = task, 'press_slide'
        self.goal, self.config = task['goals'][0], task['success_config']
        self.reset()

    def reset(self):
        self.samples = self.motion_samples = 0
        self.last_sample = None
        self.previous_slide = None
        self.elapsed = self.press_time = self.stable_time = self.gap = 0.
        self.pressed = self.crossed = self.contact_seen = False
        self.invalid_reason = None
        self.result = {'success': False, 'reason': 'press and slide sequence not observed'}

    def update(self, sample, dt, *, sample_id=None, advanced=True):
        if (isinstance(dt, bool) or not math.isfinite(dt)
                or not 0 < dt <= self.task['episode_config']['physics_dt']+1e-12):
            raise ValueError('Latch history requires positive physics-rate samples')
        if not advanced:
            return deepcopy(self.result)
        if type(sample_id) is not int or sample_id < 1:
            raise ValueError('Latch physics sample IDs must be positive integers')
        if sample_id == self.last_sample:
            return deepcopy(self.result)
        if sample_id != self.samples+1:
            raise ValueError('Latch physics sample IDs must be consecutive')
        for key in FIELDS:
            value = sample.get(key)
            if (not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
                    or (key not in SIGNED_FIELDS and value < 0)):
                raise ValueError(f'Missing or invalid latch metric: {key}')
        self.last_sample = sample_id
        self.samples += 1
        self.elapsed += dt
        g = self.goal
        two = min(sample['left_gel_force_n'], sample['right_gel_force_n']) >= self.config['contact_threshold_n']
        self.contact_seen |= two
        x, depth = sample['slide_m'], sample['depth_m']
        if self.previous_slide is not None and abs(x-self.previous_slide) > g['max_progress_step_m']+1e-12:
            self.invalid_reason = self.invalid_reason or 'native trajectory skipped the keeper passage'
        self.previous_slide = x
        if (sample['housing_contact_load_n'] > g['max_forbidden_load_n']
                or sample['other_robot_force_n'] > g['max_forbidden_load_n']
                or sample['penetration_m'] > g['max_penetration_m']
                or sample['gel_compression_m'] >= sample['gel_thickness_m']-g['gel_backing_margin_m']
                or sample['attachment_error_m'] > g['max_attachment_error_m']
                or sample['attachment_angle_rad'] > g['max_attachment_angle_rad']):
            self.invalid_reason = self.invalid_reason or 'forbidden contact, penetration or broken assembly'
        if self.elapsed > self.task['episode_config']['timeout_s']:
            self.invalid_reason = self.invalid_reason or 'latch episode timed out'
        if not -.001 <= x <= .051 or not -.001 <= depth <= .009:
            self.invalid_reason = self.invalid_reason or 'native joint outside physical limits'
        if not self.pressed:
            if x > g['start_tolerance_m']:
                self.invalid_reason = self.invalid_reason or 'slider moved before the press hold'
            pressing = two and depth >= g['press_depth_m'] and abs(x) <= g['start_tolerance_m']
            self.press_time = self.press_time+dt if pressing else 0.
            self.pressed = self.press_time+1e-12 >= g['press_duration_s']
        if self.pressed:
            self.motion_samples += int(not self.crossed)
            if not self.crossed:
                self.gap = 0. if two else self.gap+dt
                if self.gap > g['max_contact_gap_s']+1e-12:
                    self.invalid_reason = self.invalid_reason or 'gel grasp lost before keeper exit'
                if g['gate_entry_m'] <= x <= g['gate_exit_m']:
                    if depth < g['press_depth_m'] or sample['keeper_clearance_m'] < g['minimum_keeper_clearance_m']:
                        self.invalid_reason = self.invalid_reason or 'keeper crossed without physical clearance'
                self.crossed = x > g['gate_exit_m'] and two
            if self.crossed and x < g['gate_exit_m']:
                self.invalid_reason = self.invalid_reason or 'slider returned through the keeper'
        quiet = (sample['joint_speed'] <= self.config['max_joint_speed']
                 and sample['object_speed_m_s'] <= self.config['max_object_speed_m_s']
                 and sample['angular_speed_rad_s'] <= self.config['max_angular_speed_rad_s'])
        clear = (sample['robot_force_n'] < g['release_force_n']
                 and max(sample['left_gel_force_n'], sample['right_gel_force_n']) < self.config['contact_threshold_n'])
        ready = (self.crossed and x >= g['minimum_slide_m'] and depth <= g['release_depth_m']
                 and quiet and clear and sample['release_clearance_m'] >= g['release_clearance_m']
                 and self.invalid_reason is None)
        self.stable_time = self.stable_time+dt if ready else 0.
        self.result = {'success': bool(ready and self.stable_time+1e-12 >= self.config['stable_duration_s']),
                      'contact_seen': self.contact_seen, 'stable_duration_s': self.stable_time,
                      'metrics': deepcopy(sample), 'history': {'physics_sample_count': self.samples,
                      'pressed': self.pressed, 'crossed': self.crossed, 'invalid_reason': self.invalid_reason}}
        if not self.result['success']:
            self.result['reason'] = self.invalid_reason or 'press, keeper passage or released hold incomplete'
        return deepcopy(self.result)


def configure_contacts(cfg):
    """Add task-mechanism audits to the existing native contact adapter."""
    from isaaclab.sensors import ContactSensorCfg
    robot_filters = cfg.scene.object_robot_contact.filter_prim_paths_expr
    sensor_type = cfg.scene.object_robot_contact.class_type
    for name, body, filters in (
        ('latch_carriage_robot', 'Carriage', robot_filters),
        ('latch_base_robot', 'Base', robot_filters),
        ('latch_keeper', 'Plunger', ['{ENV_REGEX_NS}/mechanism/Base']),
    ):
        setattr(cfg.scene, name, ContactSensorCfg(class_type=sensor_type,
            prim_path='{ENV_REGEX_NS}/mechanism/'+body, update_period=0., history_length=cfg.decimation,
            filter_prim_paths_expr=list(filters), track_friction_forces=True,
            track_contact_points=False, max_contact_data_count_per_prim=256))


class LatchRuntime(ContactRuntime):
    """Observe the two passive DOFs; never drive or teleport mechanism links."""

    def __init__(self, env):
        self.env, self.kind = env, 'press_slide'
        self.spec, self.goal = deepcopy(env.task['objects']['mechanism']), env.task['goals'][0]
        self.spec['view_bounds_local_m'] = [[-.01,-.012,-.077], [.018,.012,.009]]
        self.geometry = {}
        self.initialized = False
        self.last_sample = {}
        self._install_observer()

    def _sample(self, dt):
        state = super()._sample(dt)
        asset = self.env.scene['mechanism']
        q = dict(zip(asset.joint_names, asset.data.joint_pos[0].cpu().tolist()))
        speed = float(asset.data.joint_vel[0].abs().max())
        positions = asset.data.body_pos_w[0].cpu().numpy()
        rotations = [rotation(q) for q in asset.data.body_quat_w[0].cpu().numpy()]
        base, carriage, plunger = [asset.body_names.index(k) for k in ('Base', 'Carriage', 'Plunger')]
        p, r = positions[plunger], rotations[plunger]
        local = rotations[base].inv().apply(p-positions[base])
        residual = max(float(np.linalg.norm(local-[q['slide'], 0., .100+q['press']])),
                       float(np.linalg.norm(rotations[base].inv().apply(positions[carriage]-positions[base])
                                            -[q['slide'], 0., .012])),
                       float(np.linalg.norm(positions[base]-np.array(self.env.reset_poses['mechanism']['position']))))
        angle = max(float((rotations[base].inv()*rr).magnitude()) for rr in rotations)
        angle = max(angle, float((self.layout.inv()*rotations[base]).magnitude()))
        housing = 0.
        for side in ('left', 'right'):
            sensor = self.env.scene[f'gsmini_{side}_case_contact']
            housing += contact_load(sensor.contact_physx_view,
                range(len(sensor.cfg.filter_prim_paths_expr)), self.env.cfg.sim.dt)
        allowed = {'gelpad_left', 'gelpad_right', 'gelsight_mini_case_left', 'gelsight_mini_case_right'}
        other = 0.
        for name in ('object_robot_contact', 'latch_carriage_robot', 'latch_base_robot'):
            sensor = self.env.scene[name]
            indices = [i for i, path in enumerate(sensor.cfg.filter_prim_paths_expr)
                       if name != 'object_robot_contact' or path.rsplit('/', 1)[-1] not in allowed]
            other += contact_load(sensor.contact_physx_view, indices, self.env.cfg.sim.dt)
        sensor = self.env.scene['object_robot_contact']
        gel_indices = [i for i,path in enumerate(sensor.cfg.filter_prim_paths_expr)
                       if path.rsplit('/',1)[-1] in {'gelpad_left','gelpad_right'}]
        rigid_indices = [i for i in range(len(sensor.cfg.filter_prim_paths_expr)) if i not in gel_indices]
        penetration = contact_penetration(sensor.contact_physx_view, rigid_indices, self.env.cfg.sim.dt)
        compression = contact_penetration(sensor.contact_physx_view, gel_indices, self.env.cfg.sim.dt)
        for name in ('latch_keeper', 'latch_carriage_robot', 'latch_base_robot'):
            sensor = self.env.scene[name]
            penetration = max(penetration, contact_penetration(sensor.contact_physx_view,
                range(len(sensor.cfg.filter_prim_paths_expr)), self.env.cfg.sim.dt))
        if not hasattr(self, '_gel_thickness'):
            robot = self.env.scene['robot']
            pads = {robot.body_names.index('gelpad_'+side) for side in ('left','right')}
            self._gel_thickness = min(float(np.ptp(corners, axis=0).min())
                for index,corners in self._release_geometry() if index in pads)
        normal, tangent = self.force('latch_keeper')
        # Lug upper plane is z=.043+press; keeper lower plane is z=.038.
        clearance = .038-(float(local[2])-.057)
        state.update(slide_m=float(q['slide']), depth_m=-float(q['press']), joint_speed=speed,
                     keeper_clearance_m=clearance, keeper_force_n=float(np.linalg.norm(normal+tangent, axis=-1).sum()),
                     housing_contact_load_n=housing, other_robot_force_n=other,
                     robot_force_n=state['robot_force_n']+housing+other, penetration_m=penetration,
                     gel_compression_m=compression, gel_thickness_m=self._gel_thickness,
                     attachment_error_m=residual, attachment_angle_rad=angle,
                     release_clearance_m=self._robot_clearance(p, r),
                     object_speed_m_s=float(np.linalg.norm(asset.data.body_lin_vel_w[0,plunger].cpu().numpy())),
                     angular_speed_rad_s=float(np.linalg.norm(asset.data.body_ang_vel_w[0,plunger].cpu().numpy())))
        return state

    def penetration_report(self):
        rows = []
        for name in ('support_contact', 'object_robot_contact', 'latch_keeper',
                     'latch_carriage_robot', 'latch_base_robot'):
            _, distances = self.contact_points(name)
            rows.append({'sensor': name, 'contact_count': len(distances),
                         'minimum_separation_m': float(distances.min()) if len(distances) else None})
        return {'passed': all(row['minimum_separation_m'] is None or row['minimum_separation_m'] >= -.001
                              for row in rows), 'status': 'measured', 'tolerance_m': .001, 'pairs': rows,
                'scope': 'Native populated keeper, table and robot contact pairs. Passive joint bearings and the world-fixed mount support the assembly; no free-body weight-balance claim.'}


def contact_load(view, indices, dt):
    """Sum raw patch magnitudes in selected filters without cancelling loads."""
    import torch
    selected = list(indices)
    if not selected:
        return 0.
    def total(values, counts, starts):
        counts, starts = counts.reshape(-1)[selected], starts.reshape(-1)[selected]
        slots = torch.arange(values.shape[0], device=values.device)
        populated = ((slots[None] >= starts[:, None])
                     & (slots[None] < (starts+counts)[:, None])).any(dim=0)
        vectors = values.reshape(values.shape[0], math.prod(values.shape[1:]))
        return torch.linalg.vector_norm(torch.where(populated[:, None], vectors, 0.), dim=-1).sum()
    normal, _, _, _, counts, starts = view.get_contact_data(dt=dt)
    # Reduce before querying friction, which may reuse the native index buffers.
    load = total(normal, counts, starts)
    friction, _, counts, starts = view.get_friction_data(dt=dt)
    return float(load+total(friction, counts, starts))


def contact_penetration(view, indices, dt):
    """Separate rigid overlap from intentional compression of the delivered gel."""
    import torch
    selected = list(indices)
    if not selected:
        return 0.
    _, _, _, distances, counts, starts = view.get_contact_data(dt=dt)
    if not distances.numel():
        return 0.
    counts, starts = counts.reshape(-1)[selected], starts.reshape(-1)[selected]
    slots = torch.arange(distances.numel(), device=distances.device)
    populated = ((slots[None] >= starts[:,None]) & (slots[None] < (starts+counts)[:,None])).any(dim=0)
    minimum = float(torch.where(populated, distances.reshape(-1), 0.).min())
    if not math.isfinite(minimum):
        raise RuntimeError('Nonfinite native latch contact separation')
    return max(0., -minimum)


def supported_reset_poses(env, poses):
    """Base bottom is the authored collision plane z=0, above the native table."""
    import omni.physx
    result = deepcopy(poses)
    p = result['mechanism']['position']
    heights = []
    def hit(item):
        if '/Table/' in item.collision:
            heights.append(float(item.position[2]))
        return True
    omni.physx.get_physx_scene_query_interface().raycast_all((p[0], p[1], 1.), (0., 0., -1.), 1., hit)
    if not heights:
        raise RuntimeError('No native tabletop beneath latch base')
    p[2] = max(heights)+.0002
    return result


def interaction_path(env):
    from benchmark.tasks.extra.interaction import grasp_point
    p = grasp_point(env)
    r = rotation(env.reset_poses['mechanism']['quaternion_wxyz'])
    yaw = math.atan2(r.as_matrix()[1,0], r.as_matrix()[0,0])
    # Delivered GSmini gel end faces are 12.66 mm below the TCP in this
    # downward hand pose. The button top is 9 mm above its grasp centroid.
    p = p+[0., 0., .009+.01266+.001]
    down = p+[0., 0., -.007]
    slide = down+r.apply([.046, 0., 0.])
    return [(v, yaw) for v in (p+[0.,0.,.06], p, down, slide, slide+[0.,0.,.03], slide+[0.,0.,.09])]


def execute_latch(env, motion):
    """Press, slide under the keeper and release using robot actions only."""
    from benchmark.tasks.extra.gsmini_contact import _move, _hold, _state, _settle_success
    from benchmark.tasks.extra.interaction import path_quaternions
    path = interaction_path(env)
    quaternions = path_quaternions(env, path)
    env.action[:, -1] = 1.
    env.expert_phase = 0
    _move(env, motion, *path[0], duration=2., quaternion=quaternions[0], converge=True, planned=True)
    # Native scans reject the old side pinch: housing contact precedes useful
    # preload. Open a measured gap around the cap and press with both gel ends.
    env.action[:, -1] = -.73
    _hold(env, motion, .3)
    _move(env, motion, *path[1], duration=2., quaternion=quaternions[1], converge=True)
    env.expert_phase = 1
    position, yaw = path[1][0].copy(), path[1][1]
    depth = env.task['goals'][0]['press_depth_m']+.0004
    for _ in range(65):
        if _state(env)['depth_m'] >= depth:
            _hold(env, motion, .2, grasp=True)
            if not env.progress.pressed:
                raise RuntimeError('Robot did not establish the physical press hold')
            break
        position[2] -= .0002
        _move(env, motion, position, yaw, duration=.1, quaternion=quaternions[1])
    else:
        raise RuntimeError('Direct gel press exhausted its bounded stroke')
    env.expert_phase = 2
    layout = rotation(env.reset_poses['mechanism']['quaternion_wxyz'])
    slide = _slide_under_keeper(env, motion, position, yaw, quaternions[1], layout)
    env.expert_phase = 3
    _move(env, motion, slide+[0.,0.,.03], yaw, duration=2., quaternion=quaternions[1], converge=True)
    env.action[:, -1] = 1.
    _hold(env, motion, .3)
    _move(env, motion, slide+[0.,0.,.09], yaw, duration=2., quaternion=quaternions[1], converge=True)
    _settle_success(env, motion)


def _slide_under_keeper(env, motion, position, yaw, quaternion, layout):
    """Maintain real press depth and at most 1 mm lead over the carriage."""
    from benchmark.tasks.extra.gsmini_contact import _state, _move
    goal = env.task['goals'][0]
    position, base = np.asarray(position).copy(), np.asarray(position).copy()
    travel = 0.
    for _ in range(round(25./env.step_dt)):
        sample = _state(env)
        if sample['slide_m'] >= goal['minimum_slide_m']+.0015:
            break
        depth = sample['depth_m']
        position[2] -= float(np.clip((goal['press_depth_m']+.0005-depth)*.5,-.00005,.00005))
        if depth >= goal['press_depth_m']+.0004:
            travel = min(.046, travel+.0002, sample['slide_m']+.001)
        position[:2] = (base+layout.apply([travel,0.,0.]))[:2]
        _move(env, motion, position, yaw, duration=env.step_dt, quaternion=quaternion, grasp=True)
    else:
        raise RuntimeError('Measured press-slide servo stalled')
    if not env.progress.crossed or _state(env)['slide_m'] < goal['minimum_slide_m']:
        raise RuntimeError('Robot did not physically traverse the keeper before release')
    return position


def geometry():
    """Closed collision solids in their rigid-body frames; SI engineering masses."""
    return {
        'Base': (.2, [0,0,0], [
            ('Plate', [.125,.064,.006], [.0225,0,.003]),
            ('KeeperL', [.008,.018,.012], [.024,-.015,.044]),
            ('KeeperR', [.008,.018,.012], [.024,.015,.044]),
            ('PostL', [.008,.008,.032], [.024,-.020,.022]),
            ('PostR', [.008,.008,.032], [.024,.020,.022]),
            ('RailL', [.112,.006,.016], [.022, -.027,.014]),
            ('RailR', [.112,.006,.016], [.022, .027,.014])]),
        'Carriage': (.025, [0,0,.012], [
            ('Sled', [.026,.014,.008], [0,0,0]),
            ('GuideX0', [.004,.018,.014], [-.007,0,.011]),
            ('GuideX1', [.004,.018,.014], [.007,0,.011]),
            ('GuideY0', [.010,.004,.014], [0,-.007,.011]),
            ('GuideY1', [.010,.004,.014], [0,.007,.011])]),
        'Plunger': (.01409152, [0,0,.100], [
            ('Grip', [.020,.024,.018], [0,0,0]),
            ('Stem', [.008,.008,.066], [0,0,-.042]),
            ('Lug', [.020,.016,.004], [.008,0,-.059])]),
    }


