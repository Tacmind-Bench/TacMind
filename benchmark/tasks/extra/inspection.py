"""Touch-and-lift inspection of rigid bench-mounted surface coupons.

Engineering adaptation of visuotactile inspection, not micron metrology or a
soft-tissue model. Native loaded points are diagnostic samples, not gel area.
"""

from copy import deepcopy
import math
from numbers import Real

import numpy as np

from benchmark.tasks.extra.contact_runtime import ContactRuntime, rotation


NAMES = ('mechanism', 'reference_a', 'reference_b')
FIELDS = ('defect_force_n', 'reference_a_force_n', 'reference_b_force_n',
          'contact_y_m', 'contact_z_m', 'target_error_m', 'gel_clearance_m',
          'housing_contact_load_n', 'other_robot_force_n', 'penetration_m',
          'gel_compression_m', 'gel_thickness_m', 'attachment_error_m',
          'attachment_angle_rad', 'object_speed_m_s', 'angular_speed_rad_s', 'gel_force_n',
          'gel_speed_m_s', 'non_inspection_gel_force_n', 'robot_force_n')
SIGNED_FIELDS = frozenset(('contact_y_m', 'contact_z_m'))
FORCES = FIELDS[:3]


class InspectionProgress:
    """Three short probes with liftoff, then an irrevocable terminal answer."""

    def __init__(self, task):
        self.task, self.kind = task, 'surface_inspection'
        self.goal, self.config = task['goals'][0], task['success_config']
        self.reset()

    def reset(self):
        self.samples = self.motion_samples = 0
        self.last_sample = self.active = self.origin = None
        self.elapsed = self.dwell = self.liftoff = self.stable_time = 0.
        self.visited = set()
        self.armed = self.answer = self.contact_seen = False
        self.releasing = False
        self.invalid_reason = None
        self.result = {'success': False, 'reason': 'inspection sequence not observed'}

    def update(self, sample, dt, *, sample_id=None, advanced=True):
        if (isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt)
                or not 0 < dt <= self.task['episode_config']['physics_dt']+1e-12):
            raise ValueError('Inspection history requires positive physics-rate samples')
        if not advanced:
            return deepcopy(self.result)
        if type(sample_id) is not int or sample_id < 1:
            raise ValueError('Inspection sample IDs must be positive integers')
        if sample_id == self.last_sample:
            return deepcopy(self.result)
        if sample_id != self.samples+1:
            raise ValueError('Inspection requires consecutive native sample IDs')
        for key in FIELDS:
            value = sample.get(key)
            if (not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
                    or (key not in SIGNED_FIELDS and value < 0)):
                raise ValueError(f'Missing or invalid inspection metric: {key}')
        self.samples += 1
        self.last_sample = sample_id
        self.elapsed += dt
        g = self.goal
        contacts = [i for i,key in enumerate(FORCES) if sample[key] >= self.config['contact_threshold_n']]
        self.contact_seen |= bool(contacts)
        forbidden = (sample['housing_contact_load_n'] > g['max_forbidden_load_n']
                     or sample['non_inspection_gel_force_n'] > g['max_forbidden_load_n']
                     or sample['other_robot_force_n'] > g['max_forbidden_load_n']
                     or sample['penetration_m'] > g['max_penetration_m']
                     or sample['gel_compression_m'] >= sample['gel_thickness_m']-g['gel_backing_margin_m']
                     or sample['attachment_error_m'] > .001 or sample['attachment_angle_rad'] > .01)
        if forbidden or len(contacts) > 1 or self.elapsed > self.task['episode_config']['timeout_s']:
            self.invalid_reason = self.invalid_reason or 'forbidden contact, broken fixture or timeout'
        if contacts:
            index = contacts[0]
            if self.releasing:
                # A threshold crossing is not a measured liftoff. Restart the
                # continuous dwell; never sum separate above-threshold impacts.
                self.dwell = 0.
                self.releasing = False
            self.motion_samples += 1
            self.liftoff = 0.
            point = np.array([sample['contact_y_m'], sample['contact_z_m']])
            if self.active is None:
                if not self.armed:
                    self.invalid_reason = self.invalid_reason or 'probe started without measured liftoff'
                self.active, self.origin, self.dwell = index, point, 0.
                self.armed = False
            if index != self.active or np.linalg.norm(point-self.origin) > g['max_probe_excursion_m']:
                self.invalid_reason = self.invalid_reason or 'sliding is not a touch-and-lift probe'
            self.dwell += dt
            if self.dwell > g['max_search_dwell_s']+1e-12 and not self.answer:
                self.answer = True
                if len(self.visited) != 3 or index != 0:
                    self.invalid_reason = self.invalid_reason or 'premature or incorrect terminal answer'
            if self.answer:
                correct = (index == 0 and sample['target_error_m'] <= g['localization_tolerance_m']
                           and sample['gel_speed_m_s'] <= self.config['max_gel_speed_m_s']
                           and sample['object_speed_m_s'] <= self.config['max_object_speed_m_s']
                           and sample['angular_speed_rad_s'] <= self.config['max_angular_speed_rad_s'])
                if not correct:
                    self.invalid_reason = self.invalid_reason or 'terminal answer left the defect'
                self.stable_time = self.stable_time+dt if correct else 0.
        else:
            self.releasing = self.active is not None
            if self.answer:
                self.invalid_reason = self.invalid_reason or 'terminal answer released'
            if sample['gel_force_n'] <= g['release_force_n'] and sample['gel_clearance_m'] >= g['liftoff_m']:
                self.liftoff += dt
                if self.liftoff+1e-12 >= g['liftoff_duration_s']:
                    if self.active is not None and self.dwell+1e-12 >= g['minimum_probe_dwell_s']:
                        self.visited.add(self.active)
                    self.active = self.origin = None
                    self.releasing = False
                    self.dwell = 0.
                    self.armed = True
            else:
                self.liftoff = 0.
        success = self.answer and self.stable_time+1e-12 >= self.config['stable_duration_s'] and self.invalid_reason is None
        self.result = {'success': bool(success), 'contact_seen': self.contact_seen,
                      'stable_duration_s': self.stable_time, 'metrics': deepcopy(sample),
                      'history': {'physics_sample_count': self.samples, 'probed_count': len(self.visited),
                                  'answer_committed': self.answer, 'invalid_reason': self.invalid_reason}}
        if not success:
            self.result['reason'] = self.invalid_reason or 'short probes or terminal answer incomplete'
        return deepcopy(self.result)


def sample_slots(poses, rng):
    """Permute the three physical coupons within the common assembly frame."""
    slots = [deepcopy(poses[name]) for name in NAMES]
    rng.shuffle(slots)
    for name, pose in zip(NAMES, slots):
        poses[name] = pose


def configure_contacts(cfg):
    from isaaclab.sensors import ContactSensorCfg
    source = cfg.scene.object_robot_contact
    for name in NAMES[1:]:
        setattr(cfg.scene, name+'_robot', ContactSensorCfg(class_type=source.class_type,
            prim_path='{ENV_REGEX_NS}/'+name+'/Coupon', update_period=0., history_length=cfg.decimation,
            filter_prim_paths_expr=list(source.filter_prim_paths_expr), track_friction_forces=True,
            track_contact_points=False, max_contact_data_count_per_prim=256))


def loaded_front_points(view, dt, transforms):
    """Loaded native points on the small front inspection windows, by filter.

    Reject side/back/post contacts. These points identify loaded windows;
    localization uses the native pad geometry, not this sparse contact manifold.
    """
    force, points, _, _, counts, starts = view.get_contact_data(dt=dt)
    force = force.detach().cpu().numpy().reshape(-1)
    points = points.detach().cpu().numpy()
    counts, starts = [x.detach().cpu().numpy().reshape(-1) for x in (counts, starts)]
    result = []
    for count,start,(position,orientation) in zip(counts, starts, transforms):
        weights = force[start:start+count]
        world = points[start:start+count]
        valid = weights > 1e-7
        if not np.isfinite(weights).all() or not np.isfinite(world[valid]).all():
            raise RuntimeError('Nonfinite populated native inspection contact')
        local = orientation.inv().apply(world[valid]-position)
        front = ((local[:,0] <= -.0035) & (local[:,0] >= -.008)
                 & (abs(local[:,1]) <= .0183) & (abs(local[:,2]-.09) <= .0153))
        result.append((float(weights[valid][front].sum()), local[front],
                       float(weights[valid][~front].sum())))
    if len(result) != len(transforms):
        raise RuntimeError('Native inspection filters do not match physical coupons')
    return result


def window_center(corners, position, orientation):
    """Native collision-window geometric center, independent of contact sampling."""
    corners = np.asarray(corners)
    return orientation.apply((corners.min(axis=0)+corners.max(axis=0))/2)+position


class InspectionRuntime(ContactRuntime):
    def __init__(self, env):
        self.env, self.kind = env, 'surface_inspection'
        self.spec, self.goal = env.task['objects']['mechanism'], env.task['goals'][0]
        self.geometry = {}
        self.initialized, self.last_sample = False, {}
        self._install_observer()

    def _sample(self, dt):
        from benchmark.tasks.extra.latch import contact_load, contact_penetration
        state = super()._sample(dt)
        # These mounted specimens have no free-body weight-support contract.
        for key in ('gel_up_force_n','housing_up_force_n','gel_positive_up_force_n',
                    'housing_net_up_force_n','table_force_n'):
            state.pop(key,None)
        transforms = [self.pose(name) for name in NAMES]
        forces, points = np.zeros(3), [[], [], []]
        gel_speed = total_load = 0.
        robot = self.env.scene['robot']
        pads = {robot.body_names.index('gelpad_'+s) for s in ('left','right')}
        geometry = self._release_geometry()
        pad_geometry = {i:np.concatenate([c for index,c in geometry if index == i]) for i in pads}
        windows = []
        for part in ('gel','housing'):
            state[part+'_force_n'] = state['all_'+part+'_force_n'] = state[part+'_total_force_n'] = 0.
        for side in ('left', 'right'):
            sensor = self.env.scene[f'gsmini_{side}_contact']
            for part,suffix in (('gel','contact'),('housing','case_contact')):
                normal,tangent = self.force(f'gsmini_{side}_{suffix}')
                # Every coupon is an intended inspection surface in this task.
                # Retain all their native wrenches, including reference probes.
                state[f'{side}_{part}_normal_force_w_n'] = normal.sum(axis=0).tolist()
                state[f'{side}_{part}_tangent_force_w_n'] = tangent.sum(axis=0).tolist()
                load = float(np.linalg.norm(normal,axis=-1).sum())
                state[part+'_force_n'] += load
                state['all_'+part+'_force_n'] += load
                state[part+'_total_force_n'] += float(np.linalg.norm(normal+tangent,axis=-1).sum())
                if part == 'gel':
                    state[side+'_gel_force_n'] = load
            measurements = loaded_front_points(sensor.contact_physx_view, self.env.cfg.sim.dt, transforms)
            total_load += sum(load+outside for load,_,outside in measurements)
            if any(load > 1e-7 for load,_,_ in measurements):
                index = robot.body_names.index('gelpad_'+side)
                radius = float(np.linalg.norm(pad_geometry[index],axis=-1).max())
                gel_speed = max(gel_speed,float(robot.data.body_lin_vel_w[0,index].norm())
                                +radius*float(robot.data.body_ang_vel_w[0,index].norm()))
                world = window_center(pad_geometry[index],robot.data.body_pos_w[0,index].cpu().numpy(),
                                      rotation(robot.data.body_quat_w[0,index].cpu().numpy()))
                windows.append(([load for load,_,_ in measurements],world))
            for i,(load,local,_) in enumerate(measurements):
                forces[i] += load
                points[i].extend(local.tolist())
        selected = int(np.argmax(forces))
        local = np.asarray(points[selected])
        # Sparse native points can move while a stationary pad indents. They
        # are telemetry only, never a pad-motion or contact-footprint estimate.
        center = local.mean(axis=0) if len(local) else np.array([0.,0.,.09])
        active = [(loads[selected],world) for loads,world in windows if loads[selected] > 1e-7]
        p,r = transforms[selected]
        probe = r.inv().apply(max(active,key=lambda pair:pair[0])[1]-p) if active else np.array([0.,0.,.09])
        p,r = transforms[0]
        error = max((float(np.linalg.norm(r.inv().apply(world-p)[1:]-[0.,.09]))
                     for loads,world in windows if loads[0] > 1e-7),default=1.)
        housing = sum(contact_load(self.env.scene[f'gsmini_{s}_case_contact'].contact_physx_view,
                      range(3), self.env.cfg.sim.dt) for s in ('left','right'))
        other = penetration = compression = robot_load = 0.
        allowed = {'gelpad_left','gelpad_right','gelsight_mini_case_left','gelsight_mini_case_right'}
        for name in ('object_robot_contact','reference_a_robot','reference_b_robot'):
            sensor = self.env.scene[name]
            filters = sensor.cfg.filter_prim_paths_expr
            robot_load += contact_load(sensor.contact_physx_view,range(len(filters)),self.env.cfg.sim.dt)
            gel = [i for i,path in enumerate(filters) if path.rsplit('/',1)[-1] in {'gelpad_left','gelpad_right'}]
            rigid = [i for i in range(len(filters)) if i not in gel]
            other += contact_load(sensor.contact_physx_view,
                [i for i,path in enumerate(filters) if path.rsplit('/',1)[-1] not in allowed], self.env.cfg.sim.dt)
            penetration = max(penetration, contact_penetration(sensor.contact_physx_view, rigid, self.env.cfg.sim.dt))
            compression = max(compression, contact_penetration(sensor.contact_physx_view, gel, self.env.cfg.sim.dt))
        clearances, thickness = [], []
        for index,corners in geometry:
            if index not in pads:
                continue
            thickness.append(float(np.ptp(corners,axis=0).min()))
            world = rotation(robot.data.body_quat_w[0,index].cpu().numpy()).apply(corners)+robot.data.body_pos_w[0,index].cpu().numpy()
            for name,(p,r) in zip(NAMES, transforms):
                local_pad = r.inv().apply(world-p)
                bounds = np.array(self.env.task['objects'][name]['view_bounds_local_m'])
                gap = np.maximum(0., np.maximum(bounds[0]-local_pad.max(axis=0), local_pad.min(axis=0)-bounds[1]))
                clearances.append(float(np.linalg.norm(gap)))
        residual = angle = speed = omega = 0.
        for name,(p,r) in zip(NAMES,transforms):
            pose = self.env.reset_poses[name]
            residual = max(residual,float(np.linalg.norm(p-pose['position'])))
            angle = max(angle,float((rotation(pose['quaternion_wxyz']).inv()*r).magnitude()))
            data = self.env.scene[name].data
            speed = max(speed,float(data.root_lin_vel_w[0].norm()))
            omega = max(omega,float(data.root_ang_vel_w[0].norm()))
        state.update(dict(zip(FORCES,forces.tolist())))
        state.update(contact_y_m=float(probe[1]), contact_z_m=float(probe[2]), target_error_m=error,
                     native_contact_centroid_local_m=center.tolist(), native_contact_sample_count=len(local),
                     gel_clearance_m=min(clearances), housing_contact_load_n=housing, other_robot_force_n=other,
                     penetration_m=penetration, gel_compression_m=compression, gel_thickness_m=min(thickness),
                     attachment_error_m=residual, attachment_angle_rad=angle,
                     object_speed_m_s=speed, angular_speed_rad_s=omega,
                     gel_speed_m_s=gel_speed, gel_force_n=total_load,
                     robot_force_n=robot_load,
                     non_inspection_gel_force_n=max(0.,total_load-float(forces.sum())))
        return state

    def penetration_report(self):
        rows = []
        for name in ('support_contact','object_robot_contact','reference_a_robot','reference_b_robot'):
            _, distances = self.contact_points(name)
            rows.append({'sensor':name, 'minimum_separation_m':float(distances.min()) if len(distances) else None})
        return {'passed':all(x['minimum_separation_m'] is None or x['minimum_separation_m'] >= -.001 for x in rows),
                'status':'measured', 'pairs':rows, 'scope':'Bench-mounted kinematic fixtures, not free-body weight support.'}


def supported_reset_poses(env, poses):
    from benchmark.tasks.extra.latch import supported_reset_poses as support
    result = deepcopy(poses)
    for name in NAMES:
        # Reuse the native tabletop ray for every physical mount.
        shifted = support(env, {'mechanism':deepcopy(poses[name])})
        result[name]['position'][2] = shifted['mechanism']['position'][2]
    return result


def front_window_report(env, surfaces, origin):
    """Require unobstructed rendered front windows on all three coupons."""
    from benchmark.tasks.extra.visuals import visible_surface_hits
    rows = []
    for name in NAMES:
        p,r = env.contact_runtime.pose(name)
        prefix = '/World/envs/env_0/'+name+'/Coupon/Plate'
        allowed = [prefix]
        if name == 'mechanism':
            allowed.append('/World/envs/env_0/mechanism/Coupon/RaisedDefect')
        rays = []
        for y,z in ((0.,0.),(.006,0.),(-.006,0.),(0.,.006),(0.,-.006)):
            target = p+r.apply([-.004,y,.09+z])
            direction = target-origin
            distance = float(np.linalg.norm(direction))
            hits = visible_surface_hits(surfaces,origin,direction/distance,distance+.003)
            first = min(hits,default=(None,''))
            rays.append({'visible':any(first[1] == path for path in allowed),'first_surface':first[1]})
        rows.append({'object':name,'rays':rays,'passed':all(row['visible'] for row in rays)})
    return {'passed':all(row['passed'] for row in rows),'front_windows':rows}


def interaction_path(env):
    from benchmark.tasks.extra.interaction import single_pad_tcp_path
    poses = sorted((env.reset_poses[name] for name in NAMES), key=lambda x:x['position'][1])
    poses.append(env.reset_poses['mechanism'])
    r = rotation(env.reset_poses['mechanism']['quaternion_wxyz'])
    normal = r.apply([1.,0.,0.])
    yaw = math.atan2(normal[1],normal[0])+math.pi/2
    surface = [(np.array(p['position'])+r.apply([-.004,0.,.09]),yaw) for p in poses]
    contact = single_pad_tcp_path(env,surface,normal)
    path = []
    for p,yaw in contact:
        path.extend(((p-normal*.025,yaw),(p,yaw),(p-normal*.025,yaw)))
    return path


def execute_inspection(env, motion):
    """Diagnostic robot-only probes; GT path is never a policy observation."""
    from benchmark.tasks.extra.gsmini_contact import _move, _hold
    from benchmark.tasks.extra.interaction import path_quaternions
    path = interaction_path(env)
    quaternions = path_quaternions(env,path)
    env.action[:,-1] = 1.
    normal = rotation(env.reset_poses['mechanism']['quaternion_wxyz']).apply([1.,0.,0.])
    for probe in range(4):
        i = probe*3
        env.expert_phase = probe
        p,yaw = path[i+1]
        q = quaternions[i+1]
        _move(env,motion,*path[i],quaternion=q,duration=1.5,planned=probe == 0,converge=True)
        _hold(env,motion,.12)
        _move(env,motion,p-normal*.004,yaw,quaternion=q,duration=1.,converge=True)
        # Bounded normal search stops at measured contact. It does not change
        # gel properties or keep closing until an arbitrary preload is reached.
        for offset in np.linspace(-.004,.001,21):
            _move(env,motion,p+normal*offset,yaw,quaternion=q,duration=.05)
            state = env.read_task_state()
            if max(state[key] for key in FORCES) >= env.task['success_config']['contact_threshold_n']:
                _hold(env,motion,.05)
                state = env.read_task_state()
                if (max(state[key] for key in FORCES) >= env.task['success_config']['contact_threshold_n']
                        and env.progress.dwell >= env.task['goals'][0]['minimum_probe_dwell_s']):
                    break
        else:
            raise RuntimeError('No stable native front-window gel contact inside the bounded inspection stroke')
        if probe == 3:
            _hold(env,motion,.8)
            if not env.progress.result.get('success',False):
                raise RuntimeError(f'Inspection expert did not complete the physical success contract: {env.progress.result}')
            return
        _move(env,motion,p-normal*.006,yaw,quaternion=q,duration=.2)
        _move(env,motion,*path[i+2],quaternion=q,duration=.4)
        _hold(env,motion,.12)


