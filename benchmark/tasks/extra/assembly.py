"""RAM native diagnostics, Extra success history and shared robot controls."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from benchmark.tasks.extra.pc_assembly import SPECS
from benchmark.tasks.extra.contact_runtime import ContactRuntime, rotation

RAM_METRICS = ('slot_rotation_rad', 'left_finger_force_n', 'right_finger_force_n',
               'housing_force_n', 'other_robot_force_n', 'table_force_n', 'penetration_m',
               'socket_up_force_n', 'object_weight_n', 'object_speed_m_s',
               'angular_speed_rad_s', 'release_clearance_m')


def seating_goal():
    return {'relationship': 'reseat_keyed_ram', 'ref_obj': 'socket', 'minimum_withdrawal_m': .02,
            'seat_height_m': SPECS['floor_height_m'], 'seat_tolerance_m': .0002,
            'lateral_tolerance_m': [min(SPECS['end_clearance_m'],
                                       (SPECS['notch_width_m']-SPECS['key_width_m'])/2),
                                    (SPECS['slot_width_m']-SPECS['pcb_thickness_m'])/2],
            'rotation_tolerance_rad': .005, 'max_penetration_m': .0002,
            'max_forbidden_load_n': .01, 'release_clearance_m': .002,
            'max_gel_indentation_m': .004, 'min_gel_contact_n': .005}


def _seated_pose(position, angle, goal):
    # A seated board can rest anywhere inside its real X/Y interface clearances.
    return (bool(np.all(np.abs(position[:2]) <= goal['lateral_tolerance_m']))
            and abs(position[2]-goal['seat_height_m']) <= goal['seat_tolerance_m']
            and angle <= goal['rotation_tolerance_rad'])


def slot_pose(position, quaternion, socket_position, socket_quaternion):
    frame = rotation(socket_quaternion)
    return (frame.inv().apply(np.asarray(position)-socket_position),
            float((frame.inv()*rotation(quaternion)).magnitude()))


def grasp_hold_report(samples, dt):
    """Reject creeping grasps even before the shared tabletop loss limit is crossed."""
    errors = []
    if (not math.isfinite(dt) or dt <= 0 or len(samples)*dt < 1.-1e-9):
        return {'passed': False, 'errors': ['Missing one-second grasp hold']}
    try:
        times = np.asarray([sample['time_s'] for sample in samples])
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), dt, atol=1e-9, rtol=0.):
            raise ValueError('Missing or noncontiguous hold samples')
        positions, orientations = [], []
        for sample in samples:
            frame = rotation(sample['tcp_quaternion_wxyz'])
            positions.append(frame.inv().apply(np.asarray(sample['object_position_w_m'])
                                               -sample['tcp_position_w_m']))
            orientations.append(frame.inv()*rotation(sample['object_quaternion_wxyz']))
        positions = np.asarray(positions)
        drift = float(np.linalg.norm(positions-positions[0], axis=1).max())
        angle = max(float((orientations[0].inv()*value).magnitude()) for value in orientations)
        if not np.isfinite(positions).all() or drift > .001 or angle > .02:
            errors.append('TCP-relative hold drift exceeds 1 mm or 0.02 rad')
        return {'passed': not errors, 'errors': errors, 'duration_s': len(samples)*dt,
                'max_position_drift_m': drift, 'max_rotation_drift_rad': angle,
                'sampling_hz': 1/dt, 'scope': 'Control-rate suspended grasp stability; contact/penetration has separate native gates.'}
    except (KeyError, TypeError, ValueError) as exc:
        return {'passed': False, 'errors': [str(exc)]}


def closure_commands():
    contact = SPECS['spreader_thickness_m']/.04-1
    return np.r_[np.linspace(1., contact, 20), np.linspace(contact, -1., 20)]


def settled_grasp_samples(env, motion, *, timeout=6.):
    """Wait for a real one-second TCP-relative window; never reset body state."""
    from benchmark.backends.gsmini.motion import _check_grasp_motion, _tcp_object_pose
    from benchmark.tasks.success import PoseStabilityWindow

    window = PoseStabilityWindow(_tcp_object_pose(env), step_dt=env.step_dt, duration_s=1.,
                                 max_position_drift=.001, max_orientation_drift=.02)
    samples = []
    for index in range(math.ceil(timeout/env.step_dt)):
        motion.step()
        state = env.state()
        if (state['object_position_w_m'][2]-state['socket_position_w_m'][2] < .025
                or min(state['left_finger_force_n'], state['right_finger_force_n']) < .05
                or state['socket_force_n'] > .01 or state['housing_force_n'] > .01):
            raise RuntimeError(f'RAM lost its suspended carrying load path: {state}')
        _check_grasp_motion(env)
        samples.append({'time_s': env.extra_step_counter*env.step_dt, **state})
        del samples[:-window.steps]
        window.observe(_tcp_object_pose(env))
        if window.ready and grasp_hold_report(samples, env.step_dt)['passed']:
            return samples, (index+1)*env.step_dt
    raise RuntimeError(f'RAM did not establish a stable carrying grasp within {timeout:g} s')


def interaction_path(env):
    obj = env.scene['mechanism'].data
    frame = rotation(obj.root_quat_w[0].cpu().numpy())
    point = obj.root_pos_w[0].cpu().numpy()+frame.apply(SPECS['grasp_point_module_m'])
    yaw = math.atan2(frame.as_matrix()[1, 0], frame.as_matrix()[0, 0])
    return [(point+[0., 0., .10], yaw), (point, yaw),
            (point+[0., 0., .035], yaw), (point, yaw), (point+[0., 0., .10], yaw)]


def configure_contacts(cfg):
    """Observe existing colliders; never change robot contact mechanics."""
    from isaaclab.sensors import ContactSensorCfg

    targets = {'ram_socket_contact': ('mechanism/Body', 'socket/Body'),
               **{f'panda_{side}_contact': (f'Robot/panda_{side}finger', 'mechanism/Body')
                  for side in ('left', 'right')}}
    for name, (body, target) in targets.items():
        setattr(cfg.scene, name, ContactSensorCfg(
            class_type=cfg.scene.support_contact.class_type,
            prim_path='{ENV_REGEX_NS}/'+body, update_period=0., history_length=cfg.decimation,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/'+target],
            track_friction_forces=True, max_contact_data_count_per_prim=128))


def contact_buffer_metrics(normal, distances, counts, starts, friction, friction_counts,
                           friction_starts, *, load_indices=None, penetration_indices=None):
    """Reduce populated native patches on device, retaining opposing loads."""
    import torch

    def mask(values, count, start, indices=None):
        count, start = count.reshape(-1), start.reshape(-1)
        if indices is not None:
            count, start = count[indices], start[indices]
        slots = torch.arange(values.shape[0], device=values.device)
        return ((slots[None] >= start[:, None]) & (slots[None] < (start+count)[:, None])).any(dim=0)

    def load(values, populated):
        if not values.numel():
            return values.new_zeros(())
        vectors = values.reshape(values.shape[0], -1)
        return torch.linalg.vector_norm(torch.where(populated[:, None], vectors, 0.), dim=-1).sum()

    normal_load = load(normal, mask(normal, counts, starts, load_indices))
    tangent_load = load(friction, mask(friction, friction_counts, friction_starts, load_indices))
    penetration = distances.new_zeros(())
    if distances.numel():
        populated = mask(normal, counts, starts, penetration_indices)
        penetration = (-torch.where(populated, distances.reshape(-1), 0.).min()).clamp_min(0.)
    return normal_load+tangent_load, penetration


def native_contact_metrics(sensor, dt, *, load_indices=None, penetration_indices=None):
    view = sensor.contact_physx_view
    normal, _, _, distances, counts, starts = view.get_contact_data(dt=dt)
    # PhysX reuses these two buffers for the friction query. Preserve the
    # normal ranges before that query, including calls at the same timestamp.
    counts, starts = counts.clone(), starts.clone()
    friction, _, fc, fs = view.get_friction_data(dt=dt)
    return contact_buffer_metrics(normal, distances, counts, starts, friction, fc, fs,
                                  load_indices=load_indices, penetration_indices=penetration_indices)


class RamRuntime(ContactRuntime):
    """Reuse Extra's native observer and gripper clearance measurements."""

    def __init__(self, env):
        self.env, self.kind = env, 'ram'
        self.spec, self.goal = env.task['objects']['mechanism'], env.task['goals'][0]
        self.geometry = {}
        self.initialized, self.last_sample = False, {}
        self._install_observer()

    def _extra_snapshot_tensors(self):
        import torch

        scene = self.env.scene
        tensors = {'socket/root': scene['socket'].data.root_state_w[0],
                   "robot/body_positions": scene['robot'].data.body_pos_w[0],
                   "robot/body_quaternions": scene['robot'].data.body_quat_w[0]}
        for name in ('ram_socket_contact', 'panda_left_contact', 'panda_right_contact'):
            data = scene[name].data
            tensors[name+'/normal'] = data.force_matrix_w[0]
            tensors[name+'/tangent'] = data.friction_forces_w[0]
        if not hasattr(self, '_other_contact_indices'):
            allowed = {'panda_leftfinger', 'panda_rightfinger', 'gelpad_left', 'gelpad_right',
                       'gelsight_mini_case_left', 'gelsight_mini_case_right'}
            filters = scene['object_robot_contact'].cfg.filter_prim_paths_expr
            self._other_contact_indices = torch.tensor(
                [i for i, path in enumerate(filters) if path.rsplit('/', 1)[-1] not in allowed],
                device=self.env.device, dtype=torch.long)
            self._rigid_contact_indices = torch.tensor(
                [i for i, path in enumerate(filters)
                 if path.rsplit('/', 1)[-1] not in {'gelpad_left', 'gelpad_right'}],
                device=self.env.device, dtype=torch.long)

        def metrics(name, indices=None):
            return native_contact_metrics(scene[name], self.env.cfg.sim.dt, load_indices=indices)

        other, penetration = native_contact_metrics(
            scene['object_robot_contact'], self.env.cfg.sim.dt,
            load_indices=self._other_contact_indices, penetration_indices=self._rigid_contact_indices)
        tensors['ram/other_load'] = other
        penetrations = [penetration, metrics('support_contact')[1], metrics('ram_socket_contact')[1]]
        housing = []
        for side in ('left', 'right'):
            load, penetration = metrics(f'gsmini_{side}_case_contact')
            housing.append(load)
            penetrations.append(penetration)
        tensors['ram/housing_load'] = sum(housing)
        tensors['ram/penetration'] = torch.stack(penetrations).max()
        # Native compliant gel overlap is elastic indentation, bounded
        # separately from the unchanged rigid-body penetration criterion.
        tensors['ram/gel_indentation'] = torch.stack([
            metrics(f'gsmini_{side}_contact')[1] for side in ('left', 'right')]).max()
        return tensors

    def _sample(self, dt):
        state = super()._sample(dt)
        position, orientation = self.pose()
        socket_position, socket_orientation = self.pose('socket')
        local = socket_orientation.inv().apply(position-socket_position)
        angle = float((socket_orientation.inv()*orientation).magnitude())
        state.update(slot_position_m=local.tolist(), slot_x_m=float(local[0]),
                     slot_y_m=float(local[1]), slot_z_m=float(local[2]), slot_rotation_rad=angle,
                     object_weight_n=self.spec['mass_kg']*9.81,
                     release_clearance_m=self._robot_clearance(position, orientation))
        normal, tangent = self.force('ram_socket_contact')
        state['socket_up_force_n'] = max(0., float((normal+tangent).sum(axis=0)[2]))
        state['socket_force_n'] = float(np.linalg.norm(normal+tangent, axis=-1).sum())
        for side in ('left', 'right'):
            normal, _ = self.force(f'panda_{side}_contact')
            state[f'{side}_finger_force_n'] = float(np.linalg.norm(normal, axis=-1).sum())
        state['other_robot_force_n'] = float(self._cpu_cache['ram/other_load'])
        state['housing_force_n'] = float(self._cpu_cache['ram/housing_load'])
        state['penetration_m'] = float(self._cpu_cache['ram/penetration'])
        state['gel_indentation_m'] = float(self._cpu_cache['ram/gel_indentation'])
        return state

    def penetration_report(self):
        state = self.sample(0.)
        supported = .9*state['object_weight_n'] <= state['socket_up_force_n'] <= 1.1*state['object_weight_n']
        return {'passed': state['penetration_m'] <= self.goal['max_penetration_m'] and supported,
                'status': 'measured', 'penetration_m': state['penetration_m'],
                'socket_support_n': state['socket_up_force_n'],
                'scope': 'Native RAM/table, RAM/robot and RAM/socket pairs; real socket weight support.'}


class RamProgress:
    """Native history for withdrawal, real seating and released weight support."""

    def __init__(self, task):
        self.task, self.goal, self.config = task, task['goals'][0], task['success_config']
        self.kind = 'ram'
        self.withdrawn = False
        self.seated = False
        self.lift_time = self.seat_time = self.stable_time = self.contact_gap = 0.
        self.invalid_reason = None
        self.last_sample = None
        self.samples = 0
        self.result = {'success': False, 'reason': 'withdrawal and seating history incomplete'}

    def update(self, sample, dt, *, sample_id=None, advanced=True):
        from copy import deepcopy

        if (isinstance(dt, bool) or not math.isfinite(dt)
                or not 0 < dt <= self.task['episode_config']['physics_dt']+1e-12):
            raise ValueError('RAM history requires positive physics-rate samples')
        if not advanced:
            return deepcopy(self.result)
        if type(sample_id) is not int or sample_id < 1:
            raise ValueError('RAM physics sample IDs must be positive integers')
        if sample_id == self.last_sample:
            return deepcopy(self.result)
        if sample_id != self.samples+1:
            raise ValueError('RAM physics sample IDs must be consecutive')
        position = np.asarray(sample['slot_position_m'])
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Invalid native RAM slot_position_m')
        for key in RAM_METRICS:
            value = sample.get(key)
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0.):
                raise ValueError(f'Missing or invalid native RAM metric: {key}')
        if sample['object_weight_n'] <= 0.:
            raise ValueError('RAM object_weight_n must be positive')
        tactile = 'max_gel_indentation_m' in self.goal
        if tactile:
            for key in ('gel_indentation_m', 'left_gel_force_n', 'right_gel_force_n'):
                value = sample.get(key)
                if (not isinstance(value, Real) or isinstance(value, bool)
                        or not math.isfinite(value) or value < 0.):
                    raise ValueError(f'Missing or invalid native RAM metric: {key}')
        self.last_sample = sample_id
        self.samples += 1
        for key in ('housing_force_n', 'other_robot_force_n', 'table_force_n'):
            if sample[key] > self.goal['max_forbidden_load_n']:
                self.invalid_reason = self.invalid_reason or f'forbidden load: {key}'
        if sample['penetration_m'] > self.goal['max_penetration_m']:
            self.invalid_reason = self.invalid_reason or 'excessive native penetration'
        if tactile and sample['gel_indentation_m'] > self.goal['max_gel_indentation_m']:
            self.invalid_reason = self.invalid_reason or 'gel compressed beyond its physical thickness'
        pads = min(sample['left_finger_force_n'], sample['right_finger_force_n'])
        clear_lift = (position[2]-self.goal['seat_height_m'] >= self.goal['minimum_withdrawal_m']
                      and pads >= self.config['contact_threshold_n']
                      and sample['socket_up_force_n'] <= self.goal['max_forbidden_load_n'])
        if tactile:
            clear_lift &= min(sample['left_gel_force_n'], sample['right_gel_force_n']) >= self.goal['min_gel_contact_n']
        self.lift_time = self.lift_time+dt if clear_lift else 0.
        self.withdrawn |= self.lift_time+1e-9 >= .2
        weight = sample['object_weight_n']
        supported = weight > 0 and .9*weight <= sample['socket_up_force_n'] <= 1.1*weight
        aligned = _seated_pose(position, sample['slot_rotation_rad'], self.goal)
        quiet = (sample['object_speed_m_s'] <= self.config['max_object_speed_m_s']
                 and sample['angular_speed_rad_s'] <= self.config['max_angular_speed_rad_s'])
        if self.withdrawn and not self.seated:
            two = pads >= self.config['contact_threshold_n']
            self.contact_gap = 0. if two else self.contact_gap+dt
            if self.contact_gap > .05+1e-12:
                self.invalid_reason = self.invalid_reason or 'grasp lost before controlled seating'
            # Clamped contacts may vibrate inside the keyed pose bounds.
            # Quiet motion is required after release, when support is unassisted.
            seating = aligned and two and sample['socket_up_force_n'] >= .5*weight
            self.seat_time = self.seat_time+dt if seating else 0.
            self.seated = self.seat_time+1e-12 >= .1
        released = (max(sample['left_finger_force_n'], sample['right_finger_force_n']) <= .01
                    and sample['release_clearance_m'] >= self.goal['release_clearance_m'])
        ready = self.seated and aligned and supported and quiet and released and not self.invalid_reason
        self.stable_time = self.stable_time+dt if ready else 0.
        self.result = {'success': self.stable_time+1e-9 >= self.config['stable_duration_s'],
                       'contact_seen': self.withdrawn, 'stable_duration_s': self.stable_time,
                       'metrics': deepcopy(sample),
                       'history': {'withdrawn': self.withdrawn, 'seated': self.seated,
                                   'physics_sample_count': self.samples,
                                   'invalid_reason': self.invalid_reason}}
        if not self.result['success']:
            self.result['reason'] = self.invalid_reason or 'withdrawal and released seating incomplete'
        return deepcopy(self.result)


def spawn_socket(path, cfg, translation=None, orientation=None, **kwargs):
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom, UsdPhysics

    prim = sim_utils.spawn_from_usd(path, cfg, translation=translation, orientation=orientation, **kwargs)
    # A world mount must be moved with the specimen at spawn/reset.
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetChild('Body'))
    joint = UsdPhysics.FixedJoint(prim.GetChild('Mount'))
    joint.CreateLocalPos0Attr(Gf.Vec3f(*matrix.ExtractTranslation()))
    q = matrix.ExtractRotationQuat()
    joint.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(), Gf.Vec3f(*q.GetImaginary())))
    return prim


def reseat_and_release(env, motion):
    """Use the shared object-pose controller and the measured rigid grasp."""
    from curobo.types import Pose
    from benchmark.backends.gsmini.motion import _check_grasp_motion, _hand_pose_joint_target, _object_pose_hand_goal

    obj, socket = env.scene['object'], env.scene['socket']
    criteria = seating_goal()
    start = obj.data.root_pos_w.clone()
    goal = socket.data.root_pos_w.clone()
    goal[:, 2] += SPECS['floor_height_m']
    orientation = socket.data.root_quat_w.clone()
    frame = rotation(orientation[0].cpu().numpy())
    yaw = math.atan2(frame.as_matrix()[1, 0], frame.as_matrix()[0, 0])
    env.phase = 'reseat'
    steps = round(4./env.step_dt)
    progress = getattr(env, 'progress', None)
    minimum = steps+round(1./env.step_dt)
    budget = steps+round(6./env.step_dt) if progress is not None else minimum
    for index in range(budget):
        fraction = min(1., (index+1)/steps)
        fraction = fraction**3*(10.-15.*fraction+6.*fraction**2)
        desired = Pose(start+(goal-start)*fraction, orientation)
        hand = _object_pose_hand_goal(env, desired)
        arm = env.action_manager.get_term('arm_action')
        target = _hand_pose_joint_target(env, motion.controller, hand,
                                        center_joints=False, max_joint_step=.02)
        env.action[:, :arm.action_dim] = arm.position_actions(target)
        motion.step()
        _check_grasp_motion(env)
        if env.state()['housing_force_n'] > .01:
            raise RuntimeError('Sensor housing contacted RAM or socket during reseating')
        if index+1 >= minimum and (progress is None or progress.seated):
            break
    if progress is not None and not progress.seated:
        raise RuntimeError(f'RAM controlled seating was not confirmed before release: {progress.result}')
    env.phase = 'release'
    env.action[:, -1] = 1.
    for _ in range(round(.5/env.step_dt)):
        motion.step()
    point = env.scene['ee_frame'].data.target_pos_w[0, 0].cpu().numpy()
    env.phase = 'retreat'
    motion.move(point+[0., 0., .10], yaw, duration=2., converge=True, interpolate=True)
    env.phase = 'released_hold'
    for _ in range(round(.5/env.step_dt)):
        motion.step()
        state = env.state()
        q, sq = state['object_quaternion_wxyz'], state['socket_quaternion_wxyz']
        delta, angle = slot_pose(state['object_position_w_m'], q, state['socket_position_w_m'], sq)
        if (not _seated_pose(delta, angle, criteria) or state['object_speed_m_s'] > .005
                or state['angular_speed_rad_s'] > .05
                or max(state['left_finger_force_n'], state['right_finger_force_n'],
                       state['housing_force_n'], state['left_gel_force_n'], state['right_gel_force_n']) > .01
                or not .9*state['object_weight_n'] <= state['socket_up_force_n'] <= 1.1*state['object_weight_n']):
            raise RuntimeError(f'Released RAM did not retain aligned socket support: {state}')


class _RamScene:
    def __init__(self, scene):
        self.scene = scene

    def __getitem__(self, name):
        return self.scene['mechanism' if name == 'object' else name]

    def __getattr__(self, name):
        return getattr(self.scene, name)


class _RamGraspView:
    """Let the shared tabletop object-pose helpers read Extra's mechanism role."""

    def __init__(self, env):
        self.env, self.scene = env, _RamScene(env.scene)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        self.env.expert_phase = (0 if value == 'approach' else 1 if value in {'close', 'preload'}
                                 else 3 if value in {'release', 'retreat'}
                                 else 4 if value == 'released_hold' else 2)

    def state(self):
        socket = self.scene['socket'].data
        return {**self.env.read_task_state(),
                'socket_position_w_m': socket.root_pos_w[0].cpu().tolist(),
                'socket_quaternion_wxyz': socket.root_quat_w[0].cpu().tolist(),
                'tcp_quaternion_wxyz': self.scene['ee_frame'].data.target_quat_w[0, 0].cpu().tolist(),
                'tcp_position_w_m': self.scene['ee_frame'].data.target_pos_w[0, 0].cpu().tolist()}


def execute_ram(env, motion):
    """Withdraw, reinsert and release with robot actions and shared controllers."""
    from benchmark.backends.gsmini.motion import _tcp_object_pose

    view = _RamGraspView(env)
    path = interaction_path(env)
    point, yaw = path[1]
    view.phase = 'approach'
    motion.move(path[0][0], yaw, duration=2., converge=True, interpolate=True)
    motion.move(point, yaw, duration=2., converge=True, interpolate=True)
    view.phase = 'close'
    for command in closure_commands():
        env.action[:, -1] = float(command)
        motion.step()
    view.phase = 'preload'
    for _ in range(round(.5/env.step_dt)):
        motion.step()
    state = view.state()
    if min(state['left_finger_force_n'], state['right_finger_force_n']) < .8:
        raise RuntimeError(f'RAM carrying preload was not established: {state}')
    view.grasp_reference = _tcp_object_pose(view)
    view.phase = 'lift'
    motion.move(path[2][0], yaw, duration=3., converge=True, interpolate=True)
    view.phase = 'hold'
    settled_grasp_samples(view, motion)
    view.grasp_reference = _tcp_object_pose(view)
    if not env.progress.withdrawn or env.progress.invalid_reason:
        raise RuntimeError(f'RAM native withdrawal failed: {env.progress.result}')
    reseat_and_release(view, motion)
    if not env.progress.result['success']:
        raise RuntimeError(f'RAM native seating/release failed: {env.progress.result}')
