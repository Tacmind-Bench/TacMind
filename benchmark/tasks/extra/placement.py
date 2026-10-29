"""Extra standing placement: native history, cradle geometry and robot expert."""

from collections import deque
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from benchmark.tasks.extra.contact_runtime import ContactRuntime
from benchmark.tasks.extra.latch import contact_load


FIELDS = ('left_gel_force_n', 'right_gel_force_n', 'gel_up_force_n', 'cradle_force_n',
          'housing_contact_load_n', 'other_robot_force_n', 'robot_force_n',
          'table_force_n', 'table_up_force_n', 'lift_clearance_m',
          'rod_axis_tilt_rad', 'end_cap_support', 'release_clearance_m', 'placement_xy_error_m',
          'penetration_m', 'object_speed_m_s', 'angular_speed_rad_s')


class StandProgress:
    """Require an unsupported grasp, controlled seating and a clear release."""

    def __init__(self, task):
        self.task, self.kind = task, 'stand'
        self.goal, self.config = task['goals'][0], task['success_config']
        self.reset()

    def reset(self):
        self.samples = self.motion_samples = 0
        self.elapsed = 0.
        self.last_sample = None
        self.grasp_time = self.seat_time = self.stable_time = self.gap = 0.
        self.armed = self.seated = self.released = self.contact_seen = False
        self.lift_started = False
        self.invalid_reason = None
        self.weight_window = deque()
        self.weight_time = self.weight_impulse = 0.
        self.result = {'success': False, 'reason': 'standing transfer not observed'}

    def update(self, sample, dt, *, sample_id=None, advanced=True):
        if isinstance(dt, bool) or not math.isfinite(dt) or not 0 < dt <= self.task['episode_config']['physics_dt']+1e-12:
            raise ValueError('Standing history requires positive physics-rate samples')
        if not advanced:
            return deepcopy(self.result)
        if type(sample_id) is not int or sample_id < 1:
            raise ValueError('Standing physics sample IDs must be positive integers')
        if sample_id == self.last_sample:
            return deepcopy(self.result)
        if sample_id != self.samples+1:
            raise ValueError('Standing physics sample IDs must be consecutive')
        for field in FIELDS:
            value = sample.get(field)
            valid = (type(value) is bool if field == 'end_cap_support' else
                     isinstance(value,Real) and not isinstance(value,bool)
                     and math.isfinite(value) and (field == 'lift_clearance_m' or value >= 0.))
            if not valid:
                raise ValueError(f'Missing or invalid standing metric: {field}')
        self.last_sample = sample_id
        self.samples += 1
        self.elapsed += dt
        goal, limit = self.goal, self.config['contact_threshold_n']
        if self.elapsed > self.task['episode_config']['timeout_s']:
            self.invalid_reason = self.invalid_reason or 'standing episode timed out'
        weight = self.task['objects']['mechanism']['mass_kg']*9.81
        two = min(sample['left_gel_force_n'], sample['right_gel_force_n']) >= limit
        self.contact_seen |= two
        if (sample['housing_contact_load_n'] > goal['max_forbidden_load_n']
                or sample['other_robot_force_n'] > goal['max_forbidden_load_n']
                or sample['penetration_m'] > goal['max_penetration_m']):
            self.invalid_reason = self.invalid_reason or 'forbidden robot support or penetration'
        # Use the same vertical, weight-derived support floor as seating;
        # the gel-presence threshold measures a different contact.
        support_limit = min(limit, .8*weight)
        supported = sample['table_up_force_n'] >= support_limit
        upright = sample['rod_axis_tilt_rad'] <= goal['tilt_max_rad']
        quiet = (sample['object_speed_m_s'] <= self.config['max_object_speed_m_s']
                 and sample['angular_speed_rad_s'] <= self.config['max_angular_speed_rad_s'])
        if not self.armed:
            self.lift_started |= (two and not supported and sample['lift_clearance_m'] >= .005
                                  and sample['cradle_force_n'] < goal['max_forbidden_load_n'])
            if self.lift_started:
                self.gap = 0. if two else self.gap+dt
                if self.gap > goal['max_contact_gap_s']+1e-12:
                    self.invalid_reason = self.invalid_reason or 'rod was dropped during initial pickup'
            held = (two and not supported and sample['cradle_force_n'] < goal['max_forbidden_load_n']
                    and sample['lift_clearance_m'] >= goal['minimum_lift_m']
                    and sample['gel_up_force_n'] >= .8*weight and quiet
                    and sample['rod_axis_tilt_rad'] >= goal['minimum_initial_tilt_rad'])
            self.grasp_time = self.grasp_time+dt if held else 0.
            self.armed = self.grasp_time+1e-12 >= goal['grasp_duration_s']
        ready = False
        if self.armed:
            if sample['cradle_force_n'] >= goal['max_forbidden_load_n']:
                self.invalid_reason = self.invalid_reason or 'rod returned to cradle support'
            self.motion_samples += int(not self.seated)
            cap = (supported and upright and sample['end_cap_support']
                   and sample['placement_xy_error_m'] <= goal['placement_tolerance_m'])
            if supported and not cap:
                self.invalid_reason = self.invalid_reason or 'rod touched down on its side or tilted end'
            if not self.seated:
                self.gap = 0. if two else self.gap+dt
                if self.gap > goal['max_contact_gap_s']+1e-12:
                    self.invalid_reason = self.invalid_reason or 'rod was released before controlled seating'
                seating = cap and two and quiet and sample['table_up_force_n'] >= .8*weight
                self.seat_time = self.seat_time+dt if seating else 0.
                self.seated = self.seat_time+1e-12 >= goal['seating_duration_s']
            else:
                if not cap:
                    self.invalid_reason = self.invalid_reason or 'rod lost its seated support'
                clear = (sample['robot_force_n'] < goal['release_force_n']
                         and max(sample['left_gel_force_n'], sample['right_gel_force_n']) < limit)
                if self.released and not clear:
                    self.invalid_reason = self.invalid_reason or 'rod was regrasped after release'
                self.released |= clear
                ready = (clear and cap and quiet
                         and sample['release_clearance_m'] >= goal['release_clearance_m'])
        ready &= self.invalid_reason is None
        if ready:
            self.stable_time += dt
            self.weight_window.append((dt, sample['table_up_force_n']))
            self.weight_time += dt
            self.weight_impulse += dt*sample['table_up_force_n']
            while self.weight_time-self.weight_window[0][0] >= self.config['stable_duration_s']:
                old_dt, force = self.weight_window.popleft()
                self.weight_time -= old_dt
                self.weight_impulse -= old_dt*force
        else:
            self.stable_time = self.weight_time = self.weight_impulse = 0.
            self.weight_window.clear()
        fraction = self.weight_impulse/max(self.weight_time*weight, 1e-12)
        success = (ready and self.stable_time+1e-12 >= self.config['stable_duration_s']
                   and .8 <= fraction <= 1.2)
        self.result = {'success': bool(success), 'contact_seen': self.contact_seen,
                      'stable_duration_s': self.stable_time, 'metrics': deepcopy(sample),
                      'history': {'physics_sample_count': self.samples, 'armed': self.armed,
                                  'seated': self.seated, 'released': self.released,
                                  'table_weight_fraction': fraction, 'invalid_reason': self.invalid_reason}}
        if not success:
            self.result['reason'] = self.invalid_reason or 'standing transfer or release hold incomplete'
        return deepcopy(self.result)


def end_cap_support(points, position, orientation, length, tolerance):
    points = np.asarray(points)
    if not points.size:
        return False
    local = orientation.inv().apply(points-np.asarray(position))
    bottom = -length/2 if orientation.apply([0.,0.,1.])[2] >= 0 else length/2
    return bool(np.isfinite(local).all() and np.all(abs(local[:,2]-bottom) <= tolerance))


class StandRuntime(ContactRuntime):
    """Reuse the existing native observer; add standing-task measurements only."""

    def _sample(self, dt):
        state = super()._sample(dt)
        p, r = self.pose()
        normal, tangent = self.force('support_contact')
        _, distances = self.contact_points('support_contact')
        loaded_points, _ = self.contact_points('support_contact', loaded_only=True)
        support = normal+tangent
        cradle_force = float(np.linalg.norm(support[-1]))
        support = support[:-1]
        housing = 0.
        for side in ('left', 'right'):
            view = self.env.scene[f'gsmini_{side}_case_contact'].contact_physx_view
            housing += contact_load(view, [0], self.env.cfg.sim.dt)
        filters = self.env.scene['object_robot_contact'].cfg.filter_prim_paths_expr
        allowed = {'gelpad_left','gelpad_right','gelsight_mini_case_left','gelsight_mini_case_right'}
        other = contact_load(self.env.scene['object_robot_contact'].contact_physx_view,
            [i for i,path in enumerate(filters) if path.rsplit('/',1)[-1] not in allowed],self.env.cfg.sim.dt)
        state.update(table_up_force_n=max(0., float(support.sum(axis=0)[2])),
                     table_force_n=float(np.linalg.norm(support,axis=-1).sum()), cradle_force_n=cradle_force,
                     lift_clearance_m=self.lowest_point(p,r)-self.table_height,
                     rod_axis_tilt_rad=math.acos(float(np.clip(abs(r.apply([0,0,1])[2]),0,1))),
                     end_cap_support=end_cap_support(loaded_points,p,r,self.geometry['length_m'],self.goal['cap_contact_tolerance_m']),
                     release_clearance_m=self._robot_clearance(p,r),
                     placement_xy_error_m=float(np.linalg.norm((p-standing_target(self.env))[:2])),
                     penetration_m=max(0., -float(distances.min())) if len(distances) else 0.,
                     housing_contact_load_n=housing, other_robot_force_n=float(other))
        return state


def supported_reset_poses(env, poses):
    """Position the unchanged rod on the actual native table support plane."""
    import omni.physx
    from pxr import Usd, UsdGeom, UsdPhysics
    from benchmark.scene.collision_geometry import mesh_geometry

    poses = deepcopy(poses)
    pose = poses['mechanism']
    heights = []
    def hit(item):
        if '/Table/' in item.collision:
            heights.append(float(item.position[2]))
        return True
    omni.physx.get_physx_scene_query_interface().raycast_all(
        (float(pose['position'][0]),float(pose['position'][1]),1.), (0.,0.,-1.), 1., hit)
    if not heights:
        raise RuntimeError('No native tabletop beneath the standing rod')
    root = env.sim.stage.GetPrimAtPath('/World/envs/env_0/mechanism')
    cache = UsdGeom.XformCache()
    vertices = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            mesh = mesh_geometry(prim)
            matrix, _ = cache.ComputeRelativeTransform(prim,root)
            mesh.apply_transform(np.asarray(matrix).T)
            vertices.append(mesh.vertices)
    if not vertices:
        raise RuntimeError('Standing rod has no explicit collision support geometry')
    poses['cradle']['position'][2] = float(max(heights)+.0002)
    from benchmark.tasks.extra.interaction import _rotation
    relative = _rotation(poses['cradle']['quaternion_wxyz']).inv()*_rotation(pose['quaternion_wxyz'])
    vertices = relative.apply(np.concatenate(vertices))
    # Actual rod vertices must clear both 30 degree V faces, including rails.
    clearance = float(np.max(-vertices[:,2]+abs(vertices[:,1])*math.tan(math.pi/6)))
    pose['position'][2] = poses['cradle']['position'][2]+.022+clearance+.0004
    return poses


def standing_target(env):
    """A table placement point clear of the reset cradle, in its seeded frame."""
    from scipy.spatial.transform import Rotation
    pose = env.reset_poses['mechanism']
    q = pose['quaternion_wxyz']
    axis = Rotation.from_quat([*q[1:],q[0]]).apply([0.,0.,1.])
    yaw = math.atan2(axis[1],axis[0])
    point = np.asarray(pose['position'])+Rotation.from_euler('z',yaw).apply(
        env.task['goals'][0]['placement_offset_local_m'])
    point[2] = env.contact_runtime.table_height+env.contact_runtime.geometry['length_m']/2
    return point


def interaction_path(env):
    """Pickup, suspended reorientation, controlled seating, release retreat."""
    from benchmark.tasks.extra.interaction import grasp_point
    point = grasp_point(env)
    axis = env.contact_runtime.orientation.apply([0.,0.,1.])
    yaw = math.atan2(axis[1],axis[0])
    high = point+np.array([0.,0.,.10])
    place = standing_target(env)+[0.,0.,.005+env.task['objects']['mechanism']['grasp_point_local_m'][2]]
    return [(p,yaw) for p in (high,point,high,high.copy(),place,place+[0.,0.,.08])]


def pickup_preload(env, motion):
    """Require sustained frictional carrying capacity inside the safe jaw window."""
    from scipy.spatial.transform import Rotation
    from benchmark.tasks.extra.interaction import grasp_point
    from benchmark.tasks.extra.gsmini_contact import _balance_grasp, _carrying_preload, _grasp_load, _hold, _housing_load, _move, _state, _step
    # Follow the settled rod's measured grasp point and configured offset to
    # avoid loading the cradle or bringing the case rim into contact.
    env.action[:,-1] = 1.
    robot = env.scene['robot']
    quaternion = robot.data.body_quat_w[0,robot.body_names.index('panda_hand')].cpu().tolist()
    frame = Rotation.from_quat([*quaternion[1:],quaternion[0]])
    _move(env,motion,grasp_point(env),
          frame.as_euler('xyz')[2],duration=.4,quaternion=quaternion,converge=True)
    # The checked jaw floor remains authoritative even if that load is unreachable.
    target = _carrying_preload(env)
    floor = -.76
    command = float(env.action[0,-1])
    if command < floor-1e-6:
        raise RuntimeError('Standing pickup started beyond its checked closure window')
    hold, steady = 0, 0
    required_steady = math.ceil(.6/env.step_dt)
    for step in range(280):
        state = _state(env)
        _housing_load(state)
        force = min(state['left_gel_force_n'],state['right_gel_force_n'])
        # Accept only a sustained load: a coarse step can spike above the
        # target for one sample and relax below it immediately afterwards.
        steady = steady+1 if force >= target else 0
        if steady >= required_steady:
            break
        if (max(state['left_gel_force_n'],state['right_gel_force_n']) >= .04
                and abs(state['left_gel_force_n']-state['right_gel_force_n'])
                > .15*max(state['left_gel_force_n'],state['right_gel_force_n'])):
            # Centre before the unequal compression pushes the rod into a case.
            _balance_grasp(env,motion,limit=.001,step=.0002,relative=.05)
            state = _state(env)
            _housing_load(state)
            force = min(state['left_gel_force_n'],state['right_gel_force_n'])
            # A second read at this iteration is not another settled interval.
            if force < target:
                steady = 0
        if command > floor+1e-6:
            # Let gel load settle between closures; use fine increments near
            # carrying preload to preserve case clearance.
            if step % 2 == 0:
                command = max(floor,command-(.025 if force < .7*target else .004))
                env.action[:,-1] = command
        else:
            hold += 1
            if hold > 40:
                raise RuntimeError(f'Standing preload target was not reached at the checked '
                                   f'closure floor (min gel {force:.3f} N, '
                                   f'housing {state["housing_total_force_n"]:.3f} N)')
        _step(env,motion)
    else:
        raise RuntimeError(f'Standing preload target was not reached '
                           f'(min gel {force:.3f} N, housing {state["housing_total_force_n"]:.3f} N)')
    state = _state(env)
    if (abs(state['left_gel_force_n']-state['right_gel_force_n'])
            > .15*max(state['left_gel_force_n'],state['right_gel_force_n'])):
        _balance_grasp(env,motion,limit=.001,step=.0002,relative=.05)
    _grasp_load(env,_state(env))
    _hold(env,motion,.4,grasp=True)
    state = _state(env)
    if min(state['left_gel_force_n'],state['right_gel_force_n']) < target:
        raise RuntimeError('Standing preload relaxed below its carrying target before lift')


def _rotate_balanced(env, motion, position, yaw, quaternion):
    """Reorient through a slow collision-checked path for the held rod."""
    from benchmark.tasks.extra.gsmini_contact import _balance_grasp, _grasp_load, _move, _state

    _move(env, motion, position, yaw, duration=6., quaternion=quaternion,
          converge=True, grasp=True, planned=True, payload=True, repeat_factor=1)
    _balance_grasp(env, motion, limit=.002, step=.0002, relative=.05)
    _grasp_load(env, _state(env))


def _align_standing(env, motion, quaternions):
    """Trim the measured rod axis, then check the resulting seating wrist pose."""
    from scipy.spatial.transform import Rotation
    from benchmark.tasks.extra.interaction import _rotation
    from benchmark.tasks.extra.gsmini_contact import _hold, _move, _state

    for _ in range(6):
        state = _state(env)
        if state['rod_axis_tilt_rad'] <= math.radians(1.):
            _hold(env, motion, .25, grasp=True)
            state = _state(env)
            if state['rod_axis_tilt_rad'] <= math.radians(1.5):
                break
        axis = _rotation(state['object_quaternion_wxyz']).apply([0.,0.,1.])
        up = np.array([0.,0.,1. if axis[2] >= 0 else -1.])
        vector = np.cross(axis,up)
        sine = float(np.linalg.norm(vector))
        if sine < 1e-9:
            continue
        angle = math.atan2(sine,float(np.dot(axis,up)))
        if angle > .2:
            raise RuntimeError('Rod did not follow the planned standing rotation')
        correction = Rotation.from_rotvec(vector/sine*min(angle,.1))
        frame = env.scene['ee_frame'].data
        current = _rotation(frame.target_quat_w[0,0].cpu().tolist())
        q = (correction*current).as_quat()[[3,0,1,2]].tolist()
        point = frame.target_pos_w[0,0].cpu().numpy().copy()
        _move(env, motion, point, 0., duration=1., quaternion=q,
              converge=True, rotation_tolerance=.005, grasp=True)
    else:
        raise RuntimeError('Measured rod axis did not settle inside the standing alignment margin')
    q = env.scene['ee_frame'].data.target_quat_w[0,0].cpu().tolist()
    quaternions[4] = quaternions[5] = q
    from benchmark.backends.gsmini.motion import _create_motion_planner, _planner_joint_state
    from benchmark.backends.gsmini.sensors import planner_robot
    from benchmark.backends.gsmini.workspace import grasp_poses, read_collision_world
    from benchmark.tasks.extra.interaction import contact_obstacle_paths
    from curobo.types import GoalToolPose
    import torch

    paths = contact_obstacle_paths(env,free_motion=True,payload=True)
    world = read_collision_world(env,obstacle_paths=tuple(paths))
    planner = _create_motion_planner(env,env.seed_value,world,robot_config=planner_robot)
    try:
        final = standing_target(env)+[0.,0.,env.task['objects']['mechanism']['grasp_point_local_m'][2]]
        goal = grasp_poses(env,torch.tensor(final[None],device=env.device,dtype=torch.float32),quaternion=q)
        result = planner.ik_solver.solve_pose(GoalToolPose.from_poses({'panda_hand':goal}),
                                             _planner_joint_state(env.scene['robot'],planner))
        if not bool(result.success.any().item()):
            raise RuntimeError('Corrected standing wrist collides at the seating height')
    finally:
        planner.destroy()


def _release_standing(env, motion, position, yaw, quaternion):
    """Unload the jaws while retaining the last commanded seating pressure."""
    from benchmark.tasks.extra.gsmini_contact import _housing_load, _move, _state

    point = np.asarray(position).copy()
    origin = point.copy()
    weight = env.task['objects']['mechanism']['mass_kg']*9.81
    support_limit = min(env.task['success_config']['contact_threshold_n'], .8*weight)
    quiet = 0
    for _ in range(32):
        state = _state(env)
        _housing_load(state)
        quiet = quiet+1 if state['table_up_force_n'] >= support_limit else 0
        if quiet >= 3:
            break
        if state['table_up_force_n'] < support_limit:
            point[2] -= .000025
        _move(env, motion, point, yaw, duration=env.step_dt,
              quaternion=quaternion, grasp=True)
    else:
        raise RuntimeError('No sustained downward seating margin inside 0.8 mm TCP stroke')
    # Once the end cap is independently supporting the rod, open fully in one
    # command. Incremental jaw motion briefly unloaded the pads below the
    # release gate, then elastic rebound restored contact before the next
    # 20 Hz command could advance the aperture.
    env.action[:, -1] = 1.
    point[2] -= .0001*env.step_dt
    _move(env, motion, point, yaw, duration=env.step_dt, quaternion=quaternion)
    for _ in range(math.ceil(30./env.step_dt)):
        state = _state(env)
        _housing_load(state)
        released = state['robot_force_n'] < env.task['goals'][0]['release_force_n']
        if released and state['table_up_force_n'] >= support_limit:
            return
        if state['table_up_force_n'] < support_limit:
            point[2] -= .000025
        else:
            # Preserve the verified end-cap support while the open jaws clear.
            point[2] -= .0001*env.step_dt
        if origin[2]-point[2] > .002+1e-9:
            raise RuntimeError('Release exceeded its bounded 2 mm downward TCP stroke')
        _move(env, motion, point, yaw, duration=env.step_dt, quaternion=quaternion)
    raise RuntimeError('Supported standing release exceeded its 30 second budget')


def execute_stand(env, motion):
    """Diagnostic robot attempt; deliberately not registered as a ready expert."""
    from benchmark.tasks.extra.gsmini_contact import _move, _hold, _step, _settle_success
    from benchmark.tasks.extra.interaction import path_quaternions
    path = interaction_path(env)
    quaternions = path_quaternions(env,path)
    env.expert_phase = 0
    env.action[:,-1] = 1.
    for i in (0,1):
        _move(env,motion,*path[i],duration=2.,quaternion=quaternions[i],converge=True,planned=i==0)
    env.expert_phase = 1
    pickup_preload(env,motion)
    _move(env,motion,*path[2],duration=3.,quaternion=quaternions[2],converge=True,grasp=True)
    _hold(env,motion,.25,grasp=True)
    if not env.progress.armed:
        raise RuntimeError('Standing probe did not establish a suspended gel grasp')
    _rotate_balanced(env, motion, path[3][0], path[3][1], quaternions[3])
    _hold(env,motion,.25,grasp=True)
    env.expert_phase = 2
    _move(env,motion,*path[4],duration=3.,quaternion=quaternions[4],converge=True,grasp=True,
          planned=True,payload=True,repeat_factor=1)
    _align_standing(env,motion,quaternions)
    p = env.scene['ee_frame'].data.target_pos_w[0,0].detach().cpu().numpy().copy()
    for _ in range(240):
        if env.progress.seated:
            break
        # Retain the pending descent; use the original 1 mm/s near contact.
        # The measured bottom clearance bounds the faster free approach.
        state=env.read_task_state()
        p[2] -= .0003 if state['lift_clearance_m']>.003 else .00005
        _move(env,motion,p,path[4][1],duration=env.step_dt,quaternion=quaternions[4],grasp=True)
    if not env.progress.seated:
        raise RuntimeError('Standing probe did not seat the rod end cap')
    env.expert_phase = 3
    _release_standing(env, motion, p, path[4][1], quaternions[4])
    for _ in range(10):
        _step(env,motion)
    _move(env,motion,*path[5],duration=2.,quaternion=quaternions[5],converge=True)
    _settle_success(env,motion)
