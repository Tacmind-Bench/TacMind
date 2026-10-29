"""Native contact wiping of a fixed bottle or tabletop whiteboard.

Only reset writes body poses. Ink is a nonphysical surface label: cleaning
requires a loaded cloth/felt contact moving over each cell, never a commanded pose.
"""

from copy import deepcopy
import math

import numpy as np

from benchmark.tasks.extra.contact_runtime import ContactRuntime, rotation


def compressive_forces(scalars, normals, outward):
    """Native scalar signs follow pair ordering; project actual world forces."""
    return np.maximum(np.asarray(scalars)*(np.asarray(normals)@outward),0.)


def snapshot_contact_view(sensor, dt, *, reuse_friction=False, cpu=False):
    """Freeze per-pair ranges: native normal/friction queries share indices."""
    from types import SimpleNamespace
    view=sensor.contact_physx_view
    retained=getattr(sensor,'_extra_contact_friction_data',None) if reuse_friction else None
    if retained is not None:
        values,counts,starts=retained
        friction=(values,None,counts.clone(),starts.clone())
    normal=view.get_contact_data(dt=dt)
    normal=(*normal[:4],normal[4].clone(),normal[5].clone())
    if retained is None:
        values,points,counts,starts=view.get_friction_data(dt=dt)
        friction=(values,points,counts.clone(),starts.clone())
    if cpu:
        # These small audits otherwise launch many CUDA reductions and wait
        # separately for each scalar. Preserve all raw entries and integer
        # ranges while copying the complete view to the host in one transfer.
        import torch
        entries=normal+friction
        packed=torch.cat([value.reshape(-1) for value in entries if value is not None]).detach().cpu()
        copied=[]
        offset=0
        for value in entries:
            if value is None:
                copied.append(None)
            else:
                copied.append(packed[offset:offset+value.numel()].reshape(value.shape).to(value.dtype))
                offset+=value.numel()
        normal,friction=tuple(copied[:6]),tuple(copied[6:])
    return SimpleNamespace(filter_count=view.filter_count,
        get_contact_data=lambda **kwargs:normal,
        get_friction_data=lambda **kwargs:friction)


def wiping_contact_mask(task, local_points):
    """Only the authored soft wiping face can remove surface labels."""
    spec = task['objects']['mechanism']
    half = np.asarray(spec['wiping_half_extents_m'])+.0005
    return ((np.abs(local_points[:,:2]) <= half).all(axis=1)
            & (local_points[:,2] <= spec['wiping_height_m']+.0005))


def surface_cells(task):
    config=task['episode_config']
    if config['task_type']=='wipe_whiteboard':
        if 'ink_text' in config:
            from benchmark.tasks.extra.whiteboard import ink_geometry
            return ink_geometry()['cells']
        mask=np.array([list(row) for row in config['ink_bitmap']])=='#'
        rows,columns=np.nonzero(mask)
        pitch=config['ink_pitch_m']
        return np.column_stack(((rows+.5-mask.shape[0]/2)*pitch,
                                (columns+.5-mask.shape[1]/2)*pitch,
                                np.full(len(rows),config['ink_surface_z_m'])))
    return np.asarray(config['surface_cells_local_m'], dtype=float)


def surface_frame_field(task):
    """Interpolate tangent frames on the measured surface, in fixture space."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.spatial.transform import Rotation
    if task['episode_config']['task_type']=='wipe_whiteboard':
        return lambda point: Rotation.identity()
    cells=surface_cells(task)
    xs,ys=np.unique(cells[:,0]),np.unique(cells[:,1])
    dx,dy=np.gradient(cells[:,2].reshape(len(xs),len(ys)),xs,ys,edge_order=2)
    gradient=RegularGridInterpolator((xs,ys),np.stack((dx,dy),axis=-1),
                                    bounds_error=False,fill_value=None)
    def frame(point):
        gx,gy=gradient(np.clip(np.asarray(point)[:2],[xs[0],ys[0]],[xs[-1],ys[-1]]))[0]
        normal=np.array([-gx,-gy,1.]);normal/=np.linalg.norm(normal)
        x=np.array([1.,0.,gx]);x/=np.linalg.norm(x)
        return Rotation.from_matrix(np.column_stack((x,np.cross(normal,x),normal)))
    return frame


def loaded_patch_cells(cells, points, radius):
    """Distance to the convex loaded contact patch, including line contacts."""
    from scipy.spatial import ConvexHull, QhullError
    import trimesh
    if not len(points):
        return []
    distance = np.linalg.norm(cells[:,None,:]-points[None,:,:],axis=-1).min(axis=1)
    if len(points) >= 3:
        try:
            polygon = points[ConvexHull(points[:,:2]).vertices]
            for i in range(1,len(polygon)-1):
                triangles = np.tile(polygon[[0,i,i+1]],(len(cells),1,1))
                nearest = trimesh.triangles.closest_point(triangles,cells)
                distance = np.minimum(distance,np.linalg.norm(nearest-cells,axis=1))
            return np.flatnonzero(distance <= radius).tolist()
        except QhullError:
            pass  # A native line contact has zero planar area.
    # A line patch is its extreme segment; enumerating every pair adds no
    # coverage and makes dense native SDF contacts quadratic Python work.
    if len(points)>1:
        lengths=np.sum((points[:,None,:]-points[None,:,:])**2,axis=-1)
        i,j=np.unravel_index(lengths.argmax(),lengths.shape)
        a,b=points[i],points[j]
        delta=b-a
        if np.dot(delta,delta)>1e-16:
            u=np.clip((cells-a)@delta/np.dot(delta,delta),0.,1.)
            distance=np.minimum(distance,np.linalg.norm(cells-a-u[:,None]*delta,axis=1))
    return np.flatnonzero(distance <= radius).tolist()


def wiping_normal_step(error, force, target=.10, *, residual=0., loaded=False):
    """Use the geometric gap only in free space, including on curved objects."""
    if not loaded and force <= 1e-5 and error < -.001:
        return max(error,-.002)
    increment=-float(np.clip((target-force)*.0005,-.0002,.0002))
    # Retain unexecuted normal motion across control steps. Re-anchoring every
    # tiny increment at the measured TCP can stall below the requested load.
    return float(np.clip(residual+increment,-.002,.002))


def wiping_orientation(tcp, tool, surface):
    """Correct the measured wiping face through a bounded wrist command."""
    from scipy.spatial.transform import Rotation
    error=(surface*tool.inv()).as_rotvec()
    error*=min(1.,.01/max(float(np.linalg.norm(error)),1e-12))
    return Rotation.from_rotvec(error)*tcp


def wiping_path_complete(progress):
    """Whiteboard Experts finish every visible stroke before withdrawing."""
    cleaned = progress.distance >= progress.goal['cell_sliding_distance_m']
    coverage = (bool(cleaned.all()) if progress.task['episode_config']['task_type'] == 'wipe_whiteboard'
                else cleaned.mean() >= progress.goal['coverage'])
    return coverage and progress.wiping_time >= progress.goal['minimum_duration_s']


def follow_loaded_waypoints(points, servo, position, normal, *, min_force, max_force,
                            tolerance=.001, max_steps=80, until=None):
    """Advance only after measured tangential arrival with surface pressure."""
    for point in points:
        point=np.asarray(point,dtype=float)
        n=normal(point)
        for _ in range(max_steps):
            if until is not None and until():
                return
            state=servo(point)
            error=point-position()
            tangent=error-float(error@n)*n
            if min_force <= state['wipe_force_n'] <= max_force and np.linalg.norm(tangent)<=tolerance:
                break
        else:
            raise RuntimeError('Wiping waypoint could not recover loaded surface contact')


class WipingProgress:
    def __init__(self, task):
        self.task = task
        self.goal, self.config = task['goals'][0], task['success_config']
        self.reset()

    def reset(self):
        self.samples = 0
        self.last_sample = None
        self.elapsed = self.wiping_time = self.stable_time = 0.
        self.distance = np.zeros(len(surface_cells(self.task)))
        self.invalid_reason = None
        self.first_unsafe_contact = None
        self.contact_seen = False
        self.result = {'success': False, 'reason': 'no loaded wiping observed'}

    def update(self, sample, dt, *, sample_id=None, advanced=True):
        if isinstance(dt, bool) or not math.isfinite(dt) or not 0 < dt <= self.task['episode_config']['physics_dt']+1e-12:
            raise ValueError('Wiping requires positive physics-rate samples')
        if not advanced:
            return deepcopy(self.result)
        if type(sample_id) is not int or sample_id < 1:
            raise ValueError('Wiping sample ID must be a positive integer')
        if sample_id == self.last_sample:
            return deepcopy(self.result)
        if sample_id != self.samples+1:
            raise ValueError('Wiping samples must be consecutive')
        fields = ('left_gel_force_n', 'right_gel_force_n', 'wipe_force_n', 'tangent_speed_m_s',
                  'housing_contact_load_n', 'other_robot_force_n', 'rigid_tool_force_n', 'rigid_penetration_m',
                  'felt_compression_m', 'gel_compression_m', 'gel_thickness_m',
                  'mount_error_m', 'mount_angle_rad', 'surface_clearance_m',
                  'object_speed_m_s', 'angular_speed_rad_s')
        for key in fields:
            value = sample.get(key)
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Missing or invalid wiping metric: {key}')
        touched = sample.get('touched_cells')
        if not isinstance(touched, list) or any(type(i) is not int or not 0 <= i < len(self.distance) for i in touched):
            raise ValueError('Invalid native wiping cells')
        self.last_sample = sample_id
        self.samples += 1
        self.elapsed += dt
        g = self.goal
        limits = {'wipe_force_n':g['max_force_n'],
            'housing_contact_load_n':g['max_forbidden_force_n'],
            'other_robot_force_n':g['max_forbidden_force_n'],
            'rigid_tool_force_n':g['max_forbidden_force_n'],
            'rigid_penetration_m':g['max_penetration_m'],
            'felt_compression_m':g['max_compression_m'],
            'gel_compression_m':sample['gel_thickness_m']-.0005,
            'mount_error_m':.001,'mount_angle_rad':.01}
        violations = {key:{'value':sample[key],'limit':limit} for key,limit in limits.items()
                      if sample[key]>limit or (key=='gel_compression_m' and sample[key]==limit)}
        if violations:
            self.invalid_reason = self.invalid_reason or 'unsafe contact or displaced fixture'
            if self.first_unsafe_contact is None:
                self.first_unsafe_contact = {'sample_id':sample_id,'violations':violations}
        if self.elapsed > self.task['episode_config']['timeout_s']:
            self.invalid_reason = self.invalid_reason or 'wiping timed out'
        grasp = min(sample['left_gel_force_n'], sample['right_gel_force_n']) >= self.config['contact_threshold_n']
        speed = sample['tangent_speed_m_s']
        loaded = grasp and g['min_force_n'] <= sample['wipe_force_n'] <= g['max_force_n']
        wiping = loaded and g['min_speed_m_s'] <= speed <= g['max_speed_m_s'] and bool(touched)
        if wiping and self.invalid_reason is None:
            self.distance[list(set(touched))] += speed*dt
            self.wiping_time += dt
            self.contact_seen = True
        cleaned = self.distance >= g['cell_sliding_distance_m']
        coverage = float(cleaned.mean())
        ready = (coverage >= g['coverage'] and self.wiping_time >= g['minimum_duration_s']
                 and sample['surface_clearance_m'] >= g['withdrawal_m']
                 and sample['wipe_force_n'] < g['min_force_n'] and grasp
                 and sample['object_speed_m_s'] <= self.config['max_object_speed_m_s']
                 and sample['angular_speed_rad_s'] <= self.config['max_angular_speed_rad_s']
                 and self.invalid_reason is None)
        self.stable_time = self.stable_time+dt if ready else 0.
        self.result = {'success': bool(ready and self.stable_time+1e-12 >= self.config['stable_duration_s']),
            'contact_seen': self.contact_seen, 'stable_duration_s': self.stable_time,
            'metrics': {**deepcopy(sample), 'coverage': coverage},
            'history': {'physics_sample_count': self.samples, 'wiping_duration_s': self.wiping_time,
                        'cleaned_cells': int(cleaned.sum()), 'invalid_reason': self.invalid_reason,
                        'first_unsafe_contact':deepcopy(self.first_unsafe_contact)}}
        if not self.result['success']:
            self.result['reason'] = self.invalid_reason or 'loaded coverage or withdrawn hold incomplete'
        return deepcopy(self.result)


def configure_contacts(cfg):
    from isaaclab.sensors import ContactSensorCfg
    cls = cfg.scene.object_robot_contact.class_type
    for name, body, filters in (
        ('wipe_contact', 'mechanism/Tool', ['{ENV_REGEX_NS}/surface/Surface']),
        ('surface_robot_contact', 'surface/Surface', cfg.scene.object_robot_contact.filter_prim_paths_expr),
    ):
        setattr(cfg.scene, name, ContactSensorCfg(class_type=cls, prim_path='{ENV_REGEX_NS}/'+body,
            filter_prim_paths_expr=list(filters), update_period=0., history_length=cfg.decimation,
            track_friction_forces=True, max_contact_data_count_per_prim=256))


def spawn_surface(path, cfg, translation=None, orientation=None, **kwargs):
    """Align the world mount before PhysX parses the authored fixed joint."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom, UsdPhysics
    prim = sim_utils.spawn_from_usd(path,cfg,translation=translation,orientation=orientation,**kwargs)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetChild('Surface'))
    joint = UsdPhysics.FixedJoint(prim.GetChild('Mount'))
    joint.CreateLocalPos0Attr(Gf.Vec3f(*matrix.ExtractTranslation()))
    q = matrix.ExtractRotationQuat()
    joint.CreateLocalRot0Attr(Gf.Quatf(q.GetReal(),Gf.Vec3f(*q.GetImaginary())))
    return prim


class WipingRuntime(ContactRuntime):
    def __init__(self, env):
        self.env, self.kind = env, env.task['episode_config']['task_type']
        self.spec = env.task['objects']['mechanism']
        self.goal = env.task['goals'][0]
        self.geometry = {}
        self.initialized = False
        self.last_sample = {}
        self.original_update = env.scene.update

        def update(dt, *args, **kwargs):
            result = self.original_update(dt, *args, **kwargs)
            if dt > 0 and self.initialized and not env._resetting:
                self.physics_steps += 1
                self.last_sample = self.sample(float(dt))
                env.progress.update(self.last_sample, float(dt), sample_id=self.physics_steps)
                self.draw_marks()
            return result
        env.scene.update = update

    def reset(self):
        self.cells = surface_cells(self.env.task)
        if self.kind == 'wipe_whiteboard' and 'ink_text' in self.env.task['episode_config']:
            from benchmark.tasks.extra.whiteboard import configure_whiteboard_visuals
            root = self.env.scene['surface'].cfg.prim_path.replace('/World/envs/env_.*', '/World/envs/env_0')
            configure_whiteboard_visuals(self.env.sim.stage, root)
        self.mark_mask = None
        super().reset()
        self.draw_marks()

    def _extra_snapshot_tensors(self):
        import torch
        data = self.env.scene['surface'].data
        return {'surface/root': torch.cat((data.root_pos_w[0], data.root_quat_w[0],
                    data.root_lin_vel_w[0], data.root_ang_vel_w[0]))}

    def _sample(self, dt):
        from benchmark.tasks.extra.latch import contact_load, contact_penetration
        state = super()._sample(dt)
        p, r = self.pose()
        sp, sr = self.pose('surface')
        sensor = self.env.scene['wipe_contact']
        force, points, normals, distances, counts, starts = sensor.contact_physx_view.get_contact_data(dt=self.env.cfg.sim.dt)
        count, start = int(counts.reshape(-1)[0]), int(starts.reshape(-1)[0])
        loads = force.reshape(-1)[start:start+count].detach().cpu().numpy()
        points = points[start:start+count].detach().cpu().numpy()
        distances = distances.reshape(-1)[start:start+count].detach().cpu().numpy()
        normals = normals[start:start+count].detach().cpu().numpy()
        if not all(np.isfinite(a).all() for a in (loads,points,distances,normals)):
            raise RuntimeError('Nonfinite native wiping contacts')
        local = r.inv().apply(points-p) if count else np.empty((0,3))
        felt = wiping_contact_mask(self.env.task,local)
        compressive = compressive_forces(loads,normals,r.apply([0.,0.,1.]))
        active = felt & (compressive > 1e-5)
        center=(np.average(points[active],axis=0,weights=compressive[active])
                if active.any() else np.zeros(3))
        surface_points = sr.inv().apply(points[active]-sp) if active.any() else np.empty((0,3))
        touched = []
        speed = 0.
        if len(surface_points):
            touched = loaded_patch_cells(self.cells,surface_points,self.goal['patch_radius_m'])
            native, surface = self._cpu_cache['mechanism/root'], self._cpu_cache['surface/root']
            v = native[7:10]+np.cross(native[10:13], points[active]-p)
            v -= surface[7:10]+np.cross(surface[10:13], points[active]-sp)
            n = normals[active]
            tangential = v-(v*n).sum(axis=1)[:,None]*n
            speed = float(np.linalg.norm(tangential,axis=1).mean())
        # Base snapshot already updated these sensors at this physics step.
        # Reuse their friction buffers and query each raw normal buffer once;
        # repeated native queries synchronize the GPU without adding evidence.
        def audit_view(name):
            return snapshot_contact_view(self.env.scene[name],self.env.cfg.sim.dt,
                                         reuse_friction=name!='surface_robot_contact',cpu=True)
        housing = 0.
        for side in ('left','right'):
            view = audit_view(f'gsmini_{side}_case_contact')
            housing += contact_load(view, range(view.filter_count), self.env.cfg.sim.dt)
        robot_sensor = self.env.scene['object_robot_contact']
        robot_view = audit_view('object_robot_contact')
        forbidden = [i for i,path in enumerate(robot_sensor.cfg.filter_prim_paths_expr)
                     if path.rsplit('/',1)[-1] not in {'gelpad_left','gelpad_right'}]
        other = contact_load(robot_view, forbidden, self.env.cfg.sim.dt)
        surface_robot = audit_view('surface_robot_contact')
        other += contact_load(surface_robot, range(surface_robot.filter_count), self.env.cfg.sim.dt)
        rigid = contact_penetration(robot_view, forbidden, self.env.cfg.sim.dt)
        gels = [i for i in range(robot_sensor.contact_physx_view.filter_count) if i not in forbidden]
        gel_compression = contact_penetration(robot_view, gels, self.env.cfg.sim.dt)
        if not hasattr(self, '_gel_thickness'):
            robot = self.env.scene['robot']
            pads = {robot.body_names.index('gelpad_'+side) for side in ('left','right')}
            self._gel_thickness = min(float(np.ptp(corners,axis=0).min())
                for index,corners in self._release_geometry() if index in pads)
        rigid = max(rigid, contact_penetration(surface_robot, range(surface_robot.filter_count), self.env.cfg.sim.dt),
                    max(0., float(-distances[~felt].min())) if (~felt).any() else 0.)
        compression = max(0., float(-distances[felt].min())) if felt.any() else 0.
        requested = self.env.reset_poses['surface']
        mount_error = float(np.linalg.norm(sp-requested['position']))
        mount_angle = float((rotation(requested['quaternion_wxyz']).inv()*sr).magnitude())
        clearance = max(0., float(sr.inv().apply(p-sp)[2]-self.cells[:,2].max()))
        # Keep the signed raw scalar for audits; pressure uses the world force
        # projected from the surface toward the tool, independent of pair order.
        state.update(wipe_force_n=float(compressive[felt].sum()),
            wipe_signed_force_n=float(loads[felt].sum()), tangent_speed_m_s=speed,
            wipe_contact_count=int(active.sum()),
            wipe_contact_x_m=float(center[0]),wipe_contact_y_m=float(center[1]),wipe_contact_z_m=float(center[2]),
            touched_cells=touched, housing_contact_load_n=housing, other_robot_force_n=other,
            rigid_tool_force_n=float(np.abs(loads[~felt]).sum()),
            rigid_penetration_m=rigid, felt_compression_m=compression,
            gel_compression_m=gel_compression, gel_thickness_m=self._gel_thickness,
            mount_error_m=mount_error, mount_angle_rad=mount_angle, surface_clearance_m=clearance)
        return state

    def draw_marks(self):
        from pxr import Gf, UsdGeom
        mask = self.env.progress.distance < self.goal['cell_sliding_distance_m']
        if self.mark_mask is not None and np.array_equal(mask,self.mark_mask):
            return
        self.mark_mask = mask.copy()
        p,r = self.pose('surface')
        marks = UsdGeom.Mesh.Define(self.env.sim.stage, '/World/envs/env_0/WipeMarks')
        if self.kind == 'wipe_whiteboard' and 'ink_text' in self.env.task['episode_config']:
            from benchmark.tasks.extra.whiteboard import draw_ink
            draw_ink(marks, self.env.task['episode_config'], mask, p, r)
            return
        frame=surface_frame_field(self.env.task)
        vertices,faces = [],[]
        for index in np.flatnonzero(mask):
            basis=frame(self.cells[index]).as_matrix()
            u,v,normal=basis.T
            center=self.cells[index]+normal*.0003
            start=len(vertices)
            if self.kind=='wipe_whiteboard':
                half=self.env.task['episode_config']['ink_pitch_m']/2
                vertices.extend(center+half*(a*u+b*v) for a,b in ((-1,-1),(1,-1),(1,1),(-1,1)))
                faces.extend([[start,start+1,start+2],[start,start+2,start+3]])
            else:
                vertices.append(center)
                for angle in np.arange(8)*math.pi/4:
                    vertices.append(center+.0025*(math.cos(angle)*u+math.sin(angle)*v))
                for j in range(8): faces.append([start,start+1+j,start+1+(j+1)%8])
        positions=p+r.apply(np.asarray(vertices).reshape(-1,3)) if vertices else np.empty((0,3))
        marks.CreatePointsAttr([Gf.Vec3f(*point) for point in positions])
        marks.CreateFaceVertexCountsAttr([3]*len(faces))
        marks.CreateFaceVertexIndicesAttr(np.asarray(faces,dtype=int).reshape(-1).tolist())
        marks.CreateSubdivisionSchemeAttr('none')
        color = (.28,.12,.035) if self.kind == 'wipe_bottle' else (.07,.12,.22)
        marks.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    def penetration_report(self):
        result = super().penetration_report()
        for name in ('wipe_contact','surface_robot_contact'):
            _, distances = self.contact_points(name)
            result['pairs'].append({'sensor':name,'contact_count':len(distances),
                'minimum_separation_m':float(distances.min()) if len(distances) else None})
            result['passed'] &= not len(distances) or distances.min() >= -.001
        return result


def wiping_grasp_yaw(task, layout):
    # Pinch the eraser's ends so board strokes stay transverse to the closing
    # axis. Reset IK and the expert must use the same grasp frame.
    return float(layout.as_euler('xyz')[2])+(math.pi/2 if task['episode_config']['task_type']=='wipe_whiteboard' else 0.)


def interaction_path(env):
    from benchmark.tasks.extra.interaction import grasp_point
    p = grasp_point(env)
    r = rotation(env.reset_poses['mechanism']['quaternion_wxyz'])
    yaw = wiping_grasp_yaw(env.task,r)
    sp,sr = env.contact_runtime.pose('surface')
    cells = surface_cells(env.task)
    route = [p+[0.,0.,.07], p, p+[0.,0.,.12]]
    for i in (0,len(cells)//2,len(cells)-1):
        route.append(sp+sr.apply(cells[i])+[0.,0.,env.task['objects']['mechanism']['grasp_point_local_m'][2]])
    route.append(route[-1]+[0.,0.,.04])
    return [(point,yaw) for point in route]


def operation_view_report(env, surfaces, origin):
    from benchmark.tasks.extra.visuals import visible_surface_hits
    p,r = env.contact_runtime.pose('surface')
    points = p+r.apply(surface_cells(env.task))
    rows = []
    for point in points:
        direction = point-origin
        length = float(np.linalg.norm(direction))
        first = min(visible_surface_hits(surfaces,origin,direction/length,length+.003),default=(None,''))
        rows.append({'point_world_m':point.tolist(),'first_surface':first[1],
                     'visible':first[1].startswith('/World/envs/env_0/surface/Surface/')})
    return {'passed':all(row['visible'] for row in rows),'surface_cells':rows,
            'scope':'Every dirty surface cell is visible from the standard front Agentview at settled reset.'}


def relax_wiping_grasp(env, motion):
    """Remove excess airborne sponge preload using measured jaw reactions."""
    from benchmark.tasks.extra.gsmini_contact import _state, _grasp_load, _balance_grasp
    goal=env.task['goals'][0]
    # Cloth/gel combine to mu=0.3. Keep gravity support with 20% margin and
    # both pads above the task's presence threshold while opening gradually.
    target=max(2.*env.task['success_config']['contact_threshold_n'],
               1.2*env.task['objects']['mechanism']['mass_kg']*9.81/(2*.3))
    for _ in range(200):
        state=_state(env)
        if state['surface_clearance_m']<goal['withdrawal_m'] or state['wipe_force_n']>=goal['min_force_n']:
            raise RuntimeError('The wiping tool must be withdrawn before relaxing its grasp')
        _grasp_load(env,state)
        left,right=state['left_gel_force_n'],state['right_gel_force_n']
        if abs(left-right)>.2*max(left,right):
            _balance_grasp(env,motion,limit=.002,step=.0001,relative=.1)
            continue
        if min(left,right)<=target:
            return
        opening=float(np.clip(.04*(min(left,right)-target),.0001,.003))
        env.action[:,-1]=min(1.,float(env.action[0,-1])+opening)
        motion.step()
    raise RuntimeError('Airborne wiping preload did not settle inside its bounded opening stroke')


def execute_wiping(env, motion):
    """Pick, load, raster with force feedback, withdraw; robot actions only."""
    from benchmark.tasks.extra.gsmini_contact import _state, _hold, _grasp_load, _pinch, _balance_grasp, _freeze_arm
    from benchmark.tasks.extra.interaction import grasp_point
    runtime = env.contact_runtime
    yaw = wiping_grasp_yaw(env.task,runtime.layout)
    if runtime.kind=='wipe_bottle':yaw+=math.pi/2
    quaternion = [0.,math.cos(yaw/2),math.sin(yaw/2),0.]
    def move(point, duration=1., converge=False, orientation=None):
        _state(env)
        motion.move(np.asarray(point),yaw,duration=duration,quaternion=quaternion if orientation is None else orientation,
                    converge=converge,rotation_tolerance=.06,interpolate=duration>.1,center_joints=False)
        return _state(env)
    def carry(point, orientation=None):
        # Keep the gel pinch centered during transport. A small roll of a flat
        # eraser otherwise moves one edge onto a case before the lift ends.
        from scipy.spatial.transform import Rotation, Slerp
        target=np.asarray(point,dtype=float).copy()
        tcp=env.scene['ee_frame'].data.target_pos_w[0,0].cpu().numpy().copy()
        start=rotation(env.scene['ee_frame'].data.target_quat_w[0,0].cpu().numpy())
        end=rotation(quaternion if orientation is None else orientation)
        angles=Slerp([0.,1.],Rotation.concatenate([start,end]))
        steps=max(2,math.ceil(float(np.linalg.norm(target-tcp))/.003))
        for i in range(3*steps):
            state=_state(env)
            left,right=state['left_gel_force_n'],state['right_gel_force_n']
            if abs(left-right)>.2*max(left,right):
                before=env.scene['ee_frame'].data.target_pos_w[0,0].cpu().numpy().copy()
                _balance_grasp(env,motion,limit=.002,step=.0001,relative=.1)
                target+=env.scene['ee_frame'].data.target_pos_w[0,0].cpu().numpy()-before
            _grasp_load(env,_state(env))
            tcp=env.scene['ee_frame'].data.target_pos_w[0,0].cpu().numpy()
            error=target-tcp
            distance=float(np.linalg.norm(error))
            if distance<.002 and i>=steps:
                break
            q=angles(min(1.,(i+1)/steps)).as_quat()[[3,0,1,2]]
            move(tcp+error*min(1.,.003/max(distance,1e-9)),env.step_dt,orientation=q)
        else:
            raise RuntimeError('Grasped tool transport did not converge')
        _grasp_load(env,_state(env))
    env.expert_phase = 0
    grip = grasp_point(env)
    grip=grip+runtime.pose()[1].apply([0.,0.,.004])
    motion.move_planned(grip+[0.,0.,.07],yaw,quaternion=quaternion,rotation_tolerance=.06)
    move(grip,1. if runtime.kind=='wipe_whiteboard' else .5,True)
    env.action[:,-1] = 1.
    _pinch(env,motion,minimum_preload=.11 if runtime.kind=='wipe_whiteboard' else .30,fine_step=.00125)
    _grasp_load(env,_state(env))
    env.expert_phase = 1
    if runtime.kind=='wipe_whiteboard':
        carry(grip+[0.,0.,.04])
    else:
        move(grip+[0.,0.,.08],.6,True)
    _grasp_load(env,_state(env))
    sp,sr = runtime.pose('surface')
    cells = surface_cells(env.task)
    frame = surface_frame_field(env.task)
    tcp_q=env.scene['ee_frame'].data.target_quat_w[0,0].cpu().numpy()
    tool_p,tool_r=runtime.pose()
    grasp_rotation=rotation(tcp_q).inv()*tool_r
    grasp_offset=tool_r.inv().apply(env.scene['ee_frame'].data.target_pos_w[0,0].cpu().numpy()-tool_p)
    # Rows run along the bottle's cylindrical axis; each row adapts its height
    # to the measured mesh, using the same controller as the flat board.
    rows = []
    if runtime.kind=='wipe_whiteboard':
        low,high=cells.min(axis=0),cells.max(axis=0)
        # Two overlapping felt-width passes cover the actual TacMind strokes.
        for i,x in enumerate(np.linspace(low[0]+.006,high[0]-.006,2)):
            row=np.array([[x,low[1],low[2]],[x,high[1],high[2]]])
            rows.append(row if i%2==0 else row[::-1])
    else:
        for i,x in enumerate(np.unique(cells[:,0])):
            row=cells[np.isclose(cells[:,0],x)]
            row=row[np.argsort(row[:,1])]
            row=row if i%2==0 else row[::-1]
            # Continue past each ink endpoint so its full native slide is earned.
            margin=.003/abs(float(row[1,1]-row[0,1]))
            rows.append(np.vstack((row[0]+margin*(row[0]-row[1]),row,row[-1]+margin*(row[-1]-row[-2]))))
    first = sp+sr.apply(rows[0][0])
    world=sr*frame(rows[0][0])
    orientation=(world*grasp_rotation.inv()).as_quat()[[3,0,1,2]]
    approach=first+world.apply(grasp_offset)+[0.,0.,.035]
    if runtime.kind=='wipe_whiteboard':
        carry(approach,orientation)
    else:
        move(approach,1.2,True,orientation)
    env.expert_phase = 2
    last_point=None
    contact_established=False
    pressure_target=.10
    loaded_motion=False
    active_normal=[0.,0.,1.]
    def servo(cell):
        nonlocal last_point,contact_established,active_normal
        state = _state(env)
        contact_established |= state['wipe_force_n']>1e-5
        left,right=state['left_gel_force_n'],state['right_gel_force_n']
        imbalance=.2 if runtime.kind=='wipe_whiteboard' else .4
        # Loaded board reactions include the wiping moment. Re-centering from
        # that force split walks the cap onto a case; retain the free-air pinch
        # and keep the jaws aligned with the measured cap instead.
        airborne=state['wipe_force_n']<=1e-5 and (runtime.kind=='wipe_whiteboard' or state['surface_clearance_m']>.01)
        if airborne and abs(left-right)>imbalance*max(left,right):
            _balance_grasp(env,motion,limit=.002,step=.0001,relative=.15)
            state=_state(env)
        # Flattening the pad and shear loading change the pinch. Regulate the
        # measured grasp through jaw commands, without altering tool materials.
        # The supported eraser unloads its carrying pinch on the board. Closing
        # again to recover the airborne preload drives the rigid cap into gel.
        _grasp_load(env,state)
        p,r = runtime.pose()
        tcp = env.scene['ee_frame'].data.target_pos_w[0,0].cpu().numpy()
        world=sr*frame(cell)
        normal=world.apply([0.,0.,1.])
        active_normal=rotation(env.scene['robot'].data.root_quat_w[0].cpu().numpy()).inv().apply(normal)
        if runtime.kind=='wipe_whiteboard':
            # The board supports the flat sole. Preserve the grasp frame as
            # shear turns the eraser, rather than twisting a cap corner into gel.
            orientation=(r*grasp_rotation.inv()).as_quat()[[3,0,1,2]]
        else:
            tcp_r=rotation(env.scene['ee_frame'].data.target_quat_w[0,0].cpu().numpy())
            orientation=wiping_orientation(tcp_r,r,world).as_quat()[[3,0,1,2]]
        error = sp+sr.apply(cell)-p
        normal_error = float(error@normal)
        tangent = error-normal_error*normal
        # Approach through free space, then regulate normal velocity directly.
        # Do not accumulate free-space travel into a penetration target after
        # the wiping pad reaches the surface.
        residual=0. if last_point is None else float((last_point-tcp)@normal)
        advance = wiping_normal_step(normal_error,state['wipe_force_n'],target=pressure_target,residual=residual,
                                     loaded=contact_established)
        delta = tangent+normal*advance
        lead=.0028 if runtime.kind=='wipe_bottle' else .0035 if loaded_motion else .002
        point = tcp+delta*min(1.,lead/max(float(np.linalg.norm(delta)),1e-9))
        last_point=point.copy()
        return move(point,env.step_dt,orientation=orientation)
    quiet=0
    previous_p,previous_r=runtime.pose()
    for _ in range(200):
        state=servo(rows[0][0])
        p,r=runtime.pose()
        # Judge settling over a control interval. A single physics-substep
        # velocity in a compliant contact can oscillate while its pose settles.
        settled=(np.linalg.norm(p-previous_p)<.0002
                 and (r*previous_r.inv()).magnitude()<.005)
        quiet=quiet+1 if (.08 <= state['wipe_force_n'] <= .12 and settled) else 0
        previous_p,previous_r=p.copy(),r
        if quiet>=6:
            break
    else:
        raise RuntimeError('Wiping pad could not establish physical surface contact')
    loaded_motion=True
    if runtime.kind=='wipe_bottle':
        # Leave a pressure margin above the loaded-sliding floor on the curved
        # surface; a target at 0.10 N repeatedly dips below the 0.08 N gate.
        pressure_target=2*runtime.goal['min_force_n']
    else:
        pressure_target=.14
    def covered():
        return wiping_path_complete(env.progress)
    from types import SimpleNamespace
    original_controller=motion.controller
    def slide_compute(position,quaternion,jacobian,joints):
        target=original_controller.ee_pos_des.clone()
        normal=target.new_tensor(active_normal)
        error=target-position
        if runtime.kind=='wipe_bottle':
            from isaaclab.utils.math import quat_apply
            offset=position.new_tensor(env.cfg.scene.ee_frame.target_frames[0].offset.pos).reshape(1,3)
            desired_tcp=target+quat_apply(original_controller.ee_quat_des,offset)
            actual_tcp=position+quat_apply(quaternion,offset)
            error=desired_tcp-actual_tcp
        tangent=error-(error*normal).sum(dim=-1,keepdim=True)*normal
        original_controller.ee_pos_des=target+(2. if runtime.kind=='wipe_bottle' else 1.5)*tangent
        try:
            return original_controller.compute(position,quaternion,jacobian,joints)
        finally:
            original_controller.ee_pos_des=target
    motion.controller=SimpleNamespace(set_command=original_controller.set_command,compute=slide_compute)
    for _ in range(2):
        for row in rows:
            # Transverse moves also run at a bounded Cartesian speed.
            p,_ = runtime.pose()
            previous = sr.inv().apply(p-sp)
            for cell in row:
                count = max(2,math.ceil(float(np.linalg.norm(cell-previous))/.003))
                follow_loaded_waypoints(np.linspace(previous,cell,count+1)[1:],servo,
                    lambda:sr.inv().apply(runtime.pose()[0]-sp),
                    lambda point:frame(point).apply([0.,0.,1.]),
                    min_force=runtime.goal['min_force_n'],max_force=runtime.goal['max_force_n'],
                    until=covered)
                previous = cell
                if covered():
                    break
            if covered():
                break
        if covered():
            break
        # Reverse the actual path for one physical rewipe when native coverage
        # is incomplete. The task timeout still bounds the complete attempt.
        rows=[row[::-1] for row in reversed(rows)]
    motion.controller=original_controller
    if runtime.kind == 'wipe_whiteboard' and not covered():
        raise RuntimeError('Whiteboard still has ink after the loaded wiping passes')
    env.expert_phase = 3
    tcp = env.scene['ee_frame'].data.target_pos_w[0,0].cpu().numpy()
    if runtime.kind=='wipe_whiteboard':
        # The supported wiping pinch is too lightly loaded to carry the cap.
        # Regrasp on the board at the measured affordance before lifting.
        state=_state(env)
        weight=env.task['objects']['mechanism']['mass_kg']*9.81
        if state['surface_clearance_m']>.002 or state['wipe_force_n']<.8*weight:
            raise RuntimeError('Eraser must be supported by the board before regrasping')
        _freeze_arm(env)
        env.action[:,-1]=1.
        _hold(env,motion,.4)
        p,r=runtime.pose()
        q=(r*grasp_rotation.inv()).as_quat()[[3,0,1,2]]
        grip=grasp_point(env)+r.apply([0.,0.,.004])
        move(grip,1.,True,q)
        _pinch(env,motion,minimum_preload=.11,fine_step=.00125)
        carry(grip+[0.,0.,.025],q)
    else:
        move(tcp+[0.,0.,.025],.4,True,env.scene['ee_frame'].data.target_quat_w[0,0].cpu().numpy())
        relax_wiping_grasp(env,motion)
    env.expert_phase = 4
    if runtime.kind=='wipe_bottle':
        for _ in range(math.ceil(.6/env.step_dt)):
            if env.progress.result['success']:break
            _hold(env,motion,env.step_dt,grasp=True)
    else:
        _hold(env,motion,.6,grasp=True)
    if not env.progress.result['success']:
        raise RuntimeError(f'Wiping did not satisfy native coverage: {env.progress.result}')
