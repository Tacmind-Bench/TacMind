"""All-seam food variants in Extra8's accepted GSmini cutting scene.

Only food bodies and their cohesion graph vary. The existing knife, sleeve,
board, robot, sensors, cameras and in-hand reset are reused without retuning.
"""

from copy import deepcopy
import json
import math
from pathlib import Path

from benchmark.tasks.extra.scene_assets import SOURCE
from benchmark.tasks.extra.cutting import CuttingRuntime, cut_gate, point_velocity

PREFIX = "shared_cutting_"
FAST_CONTACT_FOODS = {'cucumber_6', 'carrot'}
FAST_STROKE_FOODS = FAST_CONTACT_FOODS | {'baguette_2', 'cucumber_7'}
GATE_FIELDS = ("plane_distance_m", "flesh_distance_m", "normal_alignment",
               "edge_height_m", "release_height_m", "minimum_height_m",
               "food_speed_m_s", "blade_vz_m_s", "contact_force_n", "grasped")


def motion_recipe(food):
    """Keep the validated baguette stroke; use the lowest edge and force feedback for other foods."""
    if food == "baguette_1":
        return "straight-press"
    return "deep-seam-contact"


def contact_command(recipe, offset, contact_stage, gate):
    """Bound lateral progress and avoid driving a rigid cap from above."""
    if recipe == "deep-seam-contact":
        contact_stage |= (gate.get("edge_height_m", math.inf)
                          < gate.get("release_height_m", -math.inf)-.0005)
        if contact_stage:
            contact = gate.get("contact_force_n", 0.)
            if contact < .08:
                offset = min(.0048, offset+.0001)
            elif contact > .3:
                offset = max(0., offset-.0001)
    return offset, contact_stage


def food_manifest(food):
    path = SOURCE / "robobench/suites/cutting/assets" / food / "manifest.json"
    return json.loads(path.read_text())


def food_pairs(manifest):
    """Original grid-neighbour welds, including both axes of a dice task."""
    pieces = [(i, meta) for i, meta in enumerate(manifest["pieces"]) if meta is not None]
    by_cell = {tuple(meta["cell"]): k for k, (_, meta) in enumerate(pieces)}
    pairs = []
    for a, (_, meta) in enumerate(pieces):
        ci, cj = meta["cell"]
        for axis, neighbour, index in ((0, (ci+1, cj), ci), (1, (ci, cj+1), cj)):
            if neighbour in by_cell:
                b = by_cell[neighbour]
                ca, cb = meta["centroid"], pieces[b][1]["centroid"]
                point = [(ca[d]+cb[d])/2 for d in range(3)]
                point[axis] = manifest[("planes_x", "planes_y")[axis]][index]
                pairs.append(dict(a=a, b=b, axis=axis, point=point))
    return pairs


def resolve_task(original, template):
    """Resolve one source food in the unchanged Extra8 non-food scene."""
    source = original["source"]
    food = source["variant"] or ("tomato" if source["scene"] == "dice" else "carrot")
    manifest = food_manifest(food)
    pairs = food_pairs(manifest)
    task = deepcopy(template)
    for key in ("task_id", "task_name", "language_instruction", "source"):
        task[key] = deepcopy(original[key])
    task["language_instruction"] = original["language_instruction"].replace(
        "Grasp the knife and ", "Use the knife held in the gripper to ")
    task["scene_template"] = {"suite": "tac_extra", "task_id": 8}
    task["reset_contract"]["anchor_m"] = deepcopy(template["objects"]["mechanism"]["initial_position"])
    pieces = [(i, m) for i, m in enumerate(manifest["pieces"]) if m is not None]
    names = ["mechanism" if k == len(pieces)//2 else f"piece_{i}" for k, (i, _) in enumerate(pieces)]
    # Preserve the source food preset: bananas rest on their side; other scans
    # retain their authored flat-bottom orientation and original dimensions.
    banana = food == "banana" or food.startswith("banana_")
    quat = [2**-.5, 2**-.5, 0., 0.] if banana else [1., 0., 0., 0.]
    rotate = (lambda v: [v[0], -v[2], v[1]]) if banana else (lambda v: list(v))
    bounds = manifest["bounds"]
    min_z = bounds[0][1] if banana else bounds[0][2]
    z = task["episode_config"]["board_top_m"] + .001 - min_z
    specimen = deepcopy(template["objects"]["piece_0"])
    objects = {}
    for name, (index, meta) in zip(names, pieces):
        spec = deepcopy(specimen)
        center = meta["centroid"]
        position = rotate(center)
        position[0] += template["objects"]["board"]["initial_position"][0]
        position[1] += template["objects"]["board"]["initial_position"][1]
        position[2] += z
        spec.update(type=f"{PREFIX}{food}_{index}", initial_position=position,
                    initial_quat=quat, initial_region=f"{name}_region",
                    view_bounds_local_m=[[b[d]-center[d] for d in range(3)] for b in bounds])
        objects[name] = spec
    objects.update({name: deepcopy(template["objects"][name]) for name in ("knife", "board")})
    task["objects"] = objects
    task["visual_profile"]["materials"] = {k:v for k,v in task["visual_profile"]["materials"].items()
                                             if k in objects or k in task["fixtures"]}
    cfg = task["episode_config"]
    # Original all-plane tasks retain their 1800 s budget; Extra8/9 are unchanged.
    cfg.update(food=food, shared_scene=True, food_names=names, food_piece_indices=[i for i, _ in pieces],
               food_pairs=pairs,
               original_task_type=source["scene"], timeout_s=original["episode_config"]["timeout_s"],
               cut_plane_index=0, cutting_model="shared-scene-contact-gated-all-seams-v1",
               shared_controller=motion_recipe(food),
               depth_past_center_m=0. if source["scene"] == "dice" or banana or food.startswith("carrot") else .01)
    goal = task["goals"][0]
    goal.update(required_cuts=len(pairs), target_pieces=len(pieces), plane_index=None,
                board_membership_measure="source-body-root")
    goal.pop("minimum_separation_m")
    task["physical_design"] = (
        "User-requested Extra8 scene reuse. Original food meshes and collision geometry are referenced unchanged. "
        "All original x/y grid-neighbour cohesion joints must be physically cut. Board membership retains the "
        "pinned source grader's body-root footprint/above-board test; mesh penetration remains diagnostic. "
        "The accepted knife/sleeve, board, sensors and physical in-hand reset are reused. No episode pose writes or attachments.")
    return task


def food_asset(asset_type):
    """A hierarchy-only entry layer; keep the source collision vertices intact."""
    import os
    import tempfile

    from benchmark.catalog import ASSETS
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    food, index = asset_type.removeprefix(PREFIX).rsplit("_", 1)
    from benchmark.tasks.extra.scene_assets import asset_path
    source = asset_path(SOURCE / "robobench/suites/cutting/assets" / food / f"piece_{int(index)}.usd")
    if not source.is_file():
        raise FileNotFoundError(source)
    output = ASSETS / "objects" / asset_type / f"{asset_type}.usda"
    reference = os.path.relpath(source.resolve(), output.parent)
    if output.is_file():
        layer = Sdf.Layer.OpenAsAnonymous(str(output))
        body = layer.GetPrimAtPath("/Asset/Piece")
        if body and list(body.referenceList.GetAddedOrExplicitItems()) == [Sdf.Reference(reference)]:
            return output
    # A unique stage beside the output anchors relative references correctly;
    # atomic publication keeps parallel writers from sharing the same Sdf layer.
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=asset_type+"_", suffix=".usda", dir=output.parent)
    os.close(fd)
    try:
        stage = Usd.Stage.CreateNew(temporary)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, "Z")
        root = UsdGeom.Xform.Define(stage, "/Asset")
        stage.SetDefaultPrim(root.GetPrim())
        body = stage.DefinePrim("/Asset/Piece", "Xform")
        body.GetReferences().AddReference(reference)
        UsdPhysics.MassAPI.Apply(body).CreateDensityAttr(900.)
        stage.GetRootLayer().Save()
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


class SharedCuttingRuntime(CuttingRuntime):
    """Extra8 physical grasp/cut feedback, generalized to the original food graph."""

    def __init__(self, env):
        import numpy as np
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from benchmark.catalog import object_asset

        self.env, self.config = env, env.task["episode_config"]
        self.manifest = food_manifest(self.config["food"])
        self.names = self.config["food_names"]
        self.pieces = [env.scene[name] for name in self.names]
        self.centroids = np.array([self.manifest["pieces"][i]["centroid"]
                                   for i in self.config["food_piece_indices"]])
        self.pairs = self.config["food_pairs"]
        self.knife = env.scene["knife"]
        self.joints = []
        for k, pair in enumerate(self.pairs):
            a, b, point = pair["a"], pair["b"], np.array(pair["point"])
            joint = UsdPhysics.FixedJoint.Define(env.sim.stage, f"/World/envs/env_0/CutWelds/pair_{k}")
            joint.CreateBody0Rel().SetTargets([f"/World/envs/env_0/{self.names[a]}/Piece"])
            joint.CreateBody1Rel().SetTargets([f"/World/envs/env_0/{self.names[b]}/Piece"])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*(point-self.centroids[a])))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*(point-self.centroids[b])))
            joint.CreateJointEnabledAttr(True)
            self.joints.append(joint)
        knife = Usd.Stage.Open(str(object_asset("extra_slice_knife22r_0")))
        points = np.array(UsdGeom.Mesh(knife.GetPrimAtPath("/Asset/Piece/Collision")).GetPointsAttr().Get())
        self.edge_local = np.array([(x, float(points[abs(points[:, 0]-x)<.008, 1].min()), 0.)
                                   for x in np.linspace(0., .20, 21) if (abs(points[:, 0]-x)<.008).any()])
        self.bounds = []
        for name in self.names:
            source = Usd.Stage.Open(str(object_asset(env.task["objects"][name]["type"])))
            self.bounds.append(np.array(UsdGeom.Mesh(source.GetPrimAtPath("/Asset/Piece/Collision")).GetPointsAttr().Get()))
        self.active_pair = 0
        self.reset()

    def reset(self):
        self.cut = [False]*len(self.joints)
        for joint in self.joints:
            joint.GetJointEnabledAttr().Set(True)
        self.active_pair = 0
        self.tool_lifted = False
        self.last_gate = {}
        self.gates = [[0.]*len(GATE_FIELDS) for _ in self.joints]
        self.gate_valid = [False]*len(self.joints)
        self.knife_reset_z = self.env.task["objects"]["knife"]["initial_position"][2]

    def aim(self, index=None):
        import numpy as np
        index = self.active_pair if index is None else index
        pair = self.pairs[index]
        p, frame = self.pose(self.pieces[pair["a"]])
        local = np.array(pair["point"])
        return p+frame.apply(local-self.centroids[pair["a"]]), frame, local

    def update(self):
        import numpy as np
        env = self.env
        grasped = self.grasp_load_path()["valid"]
        kp, kr = self.pose(self.knife)
        self.tool_lifted |= bool(grasped and kp[2]>self.knife_reset_z+.03)
        if env._resetting:
            return
        edges = self.edge_world()
        velocity = self.knife.data.root_lin_vel_w[0].cpu().numpy()
        omega = self.knife.data.root_ang_vel_w[0].cpu().numpy()
        com = self.knife.data.root_com_pos_w[0].cpu().numpy()
        force = env.scene["knife_food_contact"].data.force_matrix_w_history[0, :, 0].norm(dim=-1)
        board_p, board_r = self.pose(env.scene["board"])
        board_top = float((board_p+board_r.apply([0.,0.,.012]))[2])
        self.gate_valid = [False]*len(self.joints)
        for index, pair in enumerate(self.pairs):
            if self.cut[index]:
                continue
            aim, frame, _ = self.aim(index)
            nearest = edges[np.linalg.norm(edges[:,:2]-aim[:2],axis=1).argmin()]
            delta = frame.inv().apply(nearest-aim)
            normal = frame.inv().apply(kr.apply([0.,0.,1.]))
            axis = pair["axis"]
            # Each seam needs actual force against one of its own two pieces.
            contact = float((force[:,pair["a"]]+force[:,pair["b"]]).max())
            gate = dict(plane_distance_m=float(delta[axis]), flesh_distance_m=float(delta[1-axis]),
                normal_alignment=float(normal[axis]), edge_height_m=float(nearest[2]),
                release_height_m=max(float(aim[2])-self.config["depth_past_center_m"],board_top+.0015),
                minimum_height_m=board_top-.0002,
                food_speed_m_s=max(float(self.pieces[i].data.root_lin_vel_w[0].norm()) for i in (pair["a"],pair["b"])),
                blade_vz_m_s=float(point_velocity(velocity,omega,nearest,com)[2]),
                contact_force_n=contact,grasped=bool(grasped and self.tool_lifted))
            self.gates[index] = [float(gate[name]) for name in GATE_FIELDS]
            self.gate_valid[index] = True
            if index==self.active_pair:
                self.last_gate=gate
            if cut_gate(**gate,config=self.config):
                self.joints[index].GetJointEnabledAttr().Set(False)
                self.cut[index]=True

    def metrics(self):
        import numpy as np
        load=self.grasp_load_path()
        board_p,board_r=self.pose(self.env.scene["board"])
        on_board=True;geometry_on_board=True;clearance=math.inf;world=[]
        for asset,points in zip(self.pieces,self.bounds):
            p,r=self.pose(asset);vertices=p+r.apply(points);world.append(vertices)
            local=board_r.inv().apply(vertices-board_p)
            clearance=min(clearance,float(local[:,2].min()-.012))
            center=board_r.inv().apply(p-board_p)
            # The source slice/dice graders use body roots. Their food material
            # is compliant; raw mesh penetration is not an original goal.
            on_board &= (abs(center[0])<.23 and abs(center[1])<.17 and center[2]>.012)
            geometry_on_board &= (abs(local[:,0]).max()<.23 and abs(local[:,1]).max()<.17
                                  and local[:,2].min()>=.012-self.config["max_food_board_penetration_m"])
        parent=list(range(len(self.pieces)))
        def find(i):
            while parent[i]!=i:
                i=parent[i]
            return i
        for cut,pair in zip(self.cut,self.pairs):
            if not cut:
                parent[find(pair["a"])]=find(pair["b"])
        components=len({find(i) for i in range(len(parent))})
        return dict(target_plane_released=all(self.cut),cut_count=sum(self.cut),pieces_count=components,
            minimum_board_clearance_m=clearance,all_pieces_on_board=bool(on_board),
            all_collision_geometry_on_board=bool(geometry_on_board),
            tool_lifted=bool(self.tool_lifted),tool_grasped=load["valid"],
            tool_housing_force_fraction=load["housing_force_fraction"],
            knife_food_contact_force_n=self.food_contact_force(),
            knife_clearance_m=float(self.edge_world()[:,2].min()-max(v[:,2].max() for v in world)),
            max_piece_speed_m_s=max(float(p.data.root_lin_vel_w[0].norm()) for p in self.pieces),
            cut_mask=list(self.cut),cut_gate=dict(self.last_gate),knife_position_w_m=self.pose(self.knife)[0].tolist())


def execute(env,motion):
    """Lift, physically cut every seam, finish each stroke, withdraw and settle."""
    import numpy as np
    from benchmark.tasks.extra.gsmini import _lift_tool, _wait_for

    from scipy.spatial.transform import Rotation

    runtime=env.cutting
    quaternion=None
    recipe=runtime.config["shared_controller"]
    _,frame,_=runtime.aim(0)
    normal=frame.apply([1.,0.,0.])
    yaw=math.atan2(normal[1],normal[0])+math.pi/2
    env.expert_phase=0
    _lift_tool(env,motion,yaw)
    edge_local=(runtime.edge_local[abs(runtime.edge_local[:,0]-.1).argmin()]
                if recipe=="straight-press" else runtime.edge_local[runtime.edge_world()[:,2].argmin()])
    def move_edge(target,duration):
        kp,kr=runtime.pose(runtime.knife)
        edge=kp+kr.apply(edge_local)
        tcp=env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy()
        # A yaw change rotates the tool lever too, including the nominal
        # top-down grasp. Solve for the edge point in the desired hand frame.
        current=runtime.rotation(env.scene["ee_frame"].data.target_quat_w[0,0].cpu().tolist())
        desired=runtime.rotation(quaternion if quaternion is not None else
                                 [0.,math.cos(yaw/2),math.sin(yaw/2),0.])
        goal=np.asarray(target)-desired.apply(current.inv().apply(edge-tcp))
        motion.move(goal,yaw,duration=duration,quaternion=quaternion)
    # Score light ends/corners while the original food graph supports them.
    # Every original seam and the native physical cut gates remain required.
    food=runtime.config["food"]
    if food=="carrot":
        descent_lead=.00225
    elif food in {"banana","parsnip"}:
        descent_lead=.002
    elif food=="baguette_1":
        descent_lead=.004
    elif food=="zucchini_1":
        descent_lead=.0025
    else:
        descent_lead=.005
    fast_stroke=food in FAST_STROKE_FOODS
    completed_score=food in {"banana","baguette_1","parsnip","potato","zucchini_1"}
    short_transfers=food in {"cucumber_7","banana","baguette_1","parsnip","potato","zucchini_1"}
    ordered_plane_stroke=food in {"banana","banana_2","banana_4","bread_loaf","carrot_2",
                                  "carrot_purple","cucumber_7","parsnip","sweet_potato"}
    small_first=food=="tomato" or ordered_plane_stroke
    follow_plane=food in {"carrot","banana_1"} or ordered_plane_stroke
    plane_feedback=food=="lotus_root"
    order=list(range(len(runtime.pairs)))
    if small_first:
        masses=[float(piece.data.default_mass.sum()) for piece in runtime.pieces]
        order.sort(key=lambda i:masses[runtime.pairs[i]["a"]]+masses[runtime.pairs[i]["b"]])
        print(f"[SHARED CUT ORDER] {order}",flush=True)
    if food=="potato":
        groups={}
        for index in order:
            pair=runtime.pairs[index]
            groups.setdefault((pair["axis"],round(pair["point"][pair["axis"]],6)),[]).append(index)
        leaders=[]
        for (axis,_),indices in sorted(groups.items(),key=lambda item:(-item[0][0],item[0][1])):
            leaders.append((min if axis==0 else max)(indices,key=lambda i:runtime.pairs[i]["point"][1-axis]))
        order=leaders+[i for i in order if i not in leaders]
    for index in order:
        pair=runtime.pairs[index]
        if runtime.cut[index]:
            continue
        runtime.active_pair=index
        aim,frame,_=runtime.aim()
        normal=frame.apply([1.,0.,0.] if pair["axis"]==0 else [0.,1.,0.])
        desired=math.atan2(normal[1],normal[0])+math.pi/2
        desired=yaw+(desired-yaw+math.pi/2)%math.pi-math.pi/2
        tcp=env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy()
        current=runtime.rotation(env.scene["ee_frame"].data.target_quat_w[0,0].cpu().tolist())
        kp,kr=runtime.pose(runtime.knife)
        lever=current.inv().apply(kp+kr.apply(edge_local)-tcp)
        reference=np.asarray(runtime.reset_grip_expected)
        candidates=(desired,desired+math.pi if desired<yaw else desired-math.pi)
        def reach_cost(angle):
            rotation=runtime.rotation([0.,math.cos(angle/2),math.sin(angle/2),0.])
            wrist=aim-rotation.apply(lever)
            return np.linalg.norm(wrist[:2]-reference[:2])
        desired=(max if food=="parsnip" else min)(candidates,key=reach_cost)
        env.expert_phase=1
        clearance=runtime.metrics()["knife_clearance_m"]
        if clearance < .05 and not short_transfers:
            motion.move(tcp+[0.,0.,.05-clearance],yaw,duration=2.,quaternion=quaternion)
        yaw=desired
        current=runtime.rotation(env.scene["ee_frame"].data.target_quat_w[0,0].cpu().tolist())
        _,knife=runtime.pose(runtime.knife)
        nominal=runtime.rotation([0.,math.cos(yaw/2),math.sin(yaw/2),0.])
        predicted=(nominal*current.inv()*knife).apply([0.,0.,1.])
        target_normal=normal if np.dot(normal,predicted)>=0 else -normal
        quaternion=None
        alignment_degrees=12. if food in {"tomato","lotus_root"} else 2.
        if np.dot(predicted,target_normal)<math.cos(math.radians(alignment_degrees)):
            correction,_=Rotation.align_vectors([target_normal],[predicted])
            x,y,z,w=(correction*nominal).as_quat()
            quaternion=[w,x,y,z]
            print(f"[SHARED TILT] pair={index} correction_deg={math.degrees(correction.magnitude()):.3f}",flush=True)
        # Place every blade-edge sample above the actual food before descent.
        actual_hand=runtime.rotation(env.scene["ee_frame"].data.target_quat_w[0,0].cpu().tolist())
        desired_hand=runtime.rotation(quaternion) if quaternion is not None else nominal
        _,actual_knife=runtime.pose(runtime.knife)
        desired_knife=desired_hand*actual_hand.inv()*actual_knife
        edge_offsets=desired_knife.apply(runtime.edge_local-edge_local)[:,2]
        food_top=float(runtime.edge_world()[:,2].min()-runtime.metrics()["knife_clearance_m"])
        approach=aim.copy()
        approach[2]=max(aim[2]+.005,food_top+.010-float(edge_offsets.min()))
        move_edge(approach,3.)
        env.expert_phase=2
        floor=runtime.config["board_top_m"]+.001
        command_offset=(.002 if food=="baguette_1" else .0015) if recipe=="straight-press" else 0.
        contact_stage=False
        plane_correction=0.
        board_recoveries=0
        recovery_z=None
        # The final banana seam remains in contact while progressing slowly.
        # Allow that physical stroke to finish without changing the cut gate.
        stroke_z=float((runtime.pose(runtime.knife)[0]+runtime.pose(runtime.knife)[1].apply(edge_local))[2])
        for _ in range(400 if food == "banana" else 240):
            if runtime.cut[index]:
                break
            kp,kr=runtime.pose(runtime.knife);edge=kp+kr.apply(edge_local)
            aim,live_frame,_=runtime.aim()
            gate=runtime.last_gate
            if recovery_z is not None and gate.get("edge_height_m",-math.inf)>=gate.get("minimum_height_m",math.inf)+.002:
                recovery_z=None
            if recovery_z is None and gate.get("edge_height_m",math.inf)<gate.get("minimum_height_m",-math.inf)+.0003:
                board_recoveries+=1
                if board_recoveries>3:
                    raise RuntimeError(f"Knife repeatedly reached the board before pair {index} released: {gate}")
                recovery_z=float(edge[2])+.005
                print(f"[SHARED BOARD RECOVERY] pair={index} count={board_recoveries}",flush=True)
            target=aim.copy()
            clearance=runtime.metrics()["knife_clearance_m"]
            increment=min(.004,clearance-.006) if clearance>.008 else .002
            lead=.005 if (fast_stroke or completed_score) and clearance>.008 else descent_lead
            stroke_z=max(floor,edge[2]-lead,stroke_z-increment)
            target[2]=recovery_z if recovery_z is not None else stroke_z
            # Source colliders have a real pre-score gap. The offset remains
            # inside the unchanged plane tolerance and earns actual food force.
            command_offset,contact_stage=contact_command(
                recipe,command_offset,contact_stage,runtime.last_gate)
            if follow_plane or plane_feedback:
                live_normal=live_frame.apply([1.,0.,0.] if pair["axis"]==0 else [0.,1.,0.])
                if plane_feedback:
                    # Measure the actual controlled edge, not a gate sample
                    # that may switch to another point along the blade.
                    plane=float(np.dot(edge-aim,live_normal))
                    increment=max(-.0003,min(.0003,.4*(command_offset-plane)))
                    plane_correction=max(-.003,min(.003,plane_correction+increment))
                offset=command_offset+plane_correction
                if follow_plane:
                    horizontal=live_normal[:2]
                    denominator=float(horizontal@horizontal)
                    if denominator<.25:
                        raise RuntimeError("Food cut plane is too horizontal for a vertical board stroke")
                    # Descending at constant XY leaves a tilted food plane.
                    target[:2]+=horizontal*(offset-live_normal[2]*(target[2]-aim[2]))/denominator
                else:
                    target+=live_normal*offset
            else:
                target+=normal*command_offset
            move_edge(target,env.step_dt)
        if not runtime.cut[index]:
            raise RuntimeError(f"Physical cutting did not release pair {index}: {runtime.last_gate}")
        cut_line=aim.copy()
        finish_z=float(edge[2])
        for _ in range(90):
            kp,kr=runtime.pose(runtime.knife);edge=kp+kr.apply(edge_local)
            if runtime.edge_world()[:,2].min()<=floor+.001:
                break
            finish_lead=.008 if completed_score else .005
            finish_increment=.006 if completed_score else .004 if fast_stroke else .002
            finish_z=max(floor,edge[2]-finish_lead,finish_z-finish_increment)
            target=cut_line.copy();target[2]=finish_z
            move_edge(target,env.step_dt)
        else:
            raise RuntimeError(f"Knife could not finish cutting pair {index}")
        env.expert_phase=3
        tcp=env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy()
        clearance_target=(.015 if short_transfers and not all(runtime.cut)
                          else .047 if food in {"banana","baguette_1","potato","zucchini_1"} else .055)
        retreat=max(.02,clearance_target-runtime.metrics()["knife_clearance_m"])
        motion.move(tcp+[0.,0.,retreat],yaw,duration=3.,quaternion=quaternion)
        print(f"[SHARED CUT] pair={index} released={sum(runtime.cut)}/{len(runtime.cut)}",flush=True)
    env.expert_phase=4
    # Respect the original food-top clearance with measured, bounded lifts.
    for _ in range(3):
        clearance=runtime.metrics()["knife_clearance_m"]
        if clearance>=env.task["goals"][0]["knife_clearance_m"]+(.003 if food in {"banana","baguette_1"} else .005):
            break
        tcp=env.scene["ee_frame"].data.target_pos_w[0,0].cpu().numpy()
        motion.move(tcp+[0.,0.,.02],yaw,duration=2.,quaternion=quaternion)
    if not _wait_for(motion,lambda result:result["success"],5.):
        raise RuntimeError(f"Cut food did not settle on the board: {env.read_task_state()}")
