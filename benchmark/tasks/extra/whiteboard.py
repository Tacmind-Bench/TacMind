"""Smooth marker outlines and render-only finishes for the wiping board."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def ink_geometry():
    path = Path(__file__).resolve().parents[2]/'config/whiteboard_ink.json'
    data = json.loads(path.read_text())
    arrays = {name: np.asarray(data[key], dtype=dtype) for name, key, dtype in (
        ('cells', 'cells_local_m', float), ('vertices', 'vertices_xy_m', float),
        ('triangles', 'triangles', np.int32), ('owners', 'triangle_cells', np.int32))}
    cells, vertices, triangles, owners = (arrays[k] for k in ('cells', 'vertices', 'triangles', 'owners'))
    if (cells.ndim != 2 or cells.shape[1] != 3 or vertices.ndim != 2 or vertices.shape[1] != 2
            or triangles.ndim != 2 or triangles.shape[1] != 3 or len(owners) != len(triangles)
            or not np.isfinite(cells).all() or not np.isfinite(vertices).all()
            or triangles.min() < 0 or triangles.max() >= len(vertices)
            or set(owners) != set(range(len(cells)))):
        raise ValueError('Invalid whiteboard ink cells or outline triangles')
    for array in arrays.values():
        array.flags.writeable = False
    return dict(text=data['text'], **arrays)


def _material(stage, path, color, roughness, metallic=0., clearcoat=0.):
    from pxr import Sdf, UsdShade
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path+'/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(tuple(color))
    for name, value in [('roughness', roughness), ('metallic', metallic), ('clearcoat', clearcoat),
                        ('clearcoatRoughness', .16)]:
        shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(value)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def _beveled_rail(stage, path, lo, hi, axis):
    from pxr import UsdGeom, Vt
    cross_axis = 1-axis
    a, b, z0, z1 = lo[cross_axis], hi[cross_axis], lo[2], hi[2]
    bevel = min(.0008, (b-a)/4, (z1-z0)/4)
    outline = [(a+bevel,z0),(b-bevel,z0),(b,z0+bevel),(b,z1-bevel),
               (b-bevel,z1),(a+bevel,z1),(a,z1-bevel),(a,z0+bevel)]
    points = []
    for end in (lo[axis], hi[axis]):
        for cross, height in outline:
            p = [0.,0.,height];p[axis],p[cross_axis] = end,cross;points.append(p)
    faces = [list(reversed(range(8))), list(range(8,16))]
    faces.extend([[i,(i+1)%8,(i+1)%8+8,i+8] for i in range(8)])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(points,dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    return mesh


def configure_whiteboard_visuals(stage, root):
    """Preserve every collider, body and physical material; add only visual detail."""
    from pxr import Usd, UsdGeom, UsdShade
    surface = root+'/Surface'
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        base = surface+'/WhiteboardFinish'
        panel = _material(stage, base+'/Coating', (.90,.92,.91), .22, clearcoat=.4)
        metal = _material(stage, base+'/Aluminum', (.47,.51,.55), .29, metallic=.88)
        gasket = _material(stage, base+'/Gasket', (.045,.05,.055), .55)
        UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(surface+'/Panel')).Bind(panel)
        bounds = [((-.088,-.148,0.),(-.08,.148,.014),1),
                  ((.08,-.148,0.),(.088,.148,.014),1),
                  ((-.08,-.148,0.),(.08,-.14,.014),0),
                  ((-.08,.14,0.),(.08,.148,.014),0)]
        for name,(lo,hi,axis) in zip(('FrameX0','FrameX1','FrameY0','FrameY1'),bounds):
            UsdGeom.Imageable(stage.GetPrimAtPath(surface+'/'+name)).MakeInvisible()
            rail = _beveled_rail(stage, base+'/'+name, lo, hi, axis)
            UsdShade.MaterialBindingAPI.Apply(rail.GetPrim()).Bind(metal)
        # The thin dark seal separates enamel from aluminum inside the existing frame.
        for i,(lo,hi,axis) in enumerate([
            ((-.080,-.140,.0138),(-.0794,.140,.01404),1),
            ((.0794,-.140,.0138),(.080,.140,.01404),1),
            ((-.0794,-.140,.0138),(.0794,-.1394,.01404),0),
            ((-.0794,.1394,.0138),(.0794,.140,.01404),0)]):
            rail = _beveled_rail(stage, base+f'/Seal{i}', lo, hi, axis)
            UsdShade.MaterialBindingAPI.Apply(rail.GetPrim()).Bind(gasket)


def draw_ink(mesh, config, dirty, position, rotation):
    """Clip continuous glyph outlines by cells cleared through native loaded sliding."""
    from pxr import UsdGeom, UsdShade, Vt
    data = ink_geometry()
    triangles = data['triangles'][dirty[data['owners']]]
    local = np.column_stack((data['vertices'],
        np.full(len(data['vertices']), config['ink_surface_z_m']+.00004)))
    points = position+rotation.apply(local)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(triangles))
    mesh.CreateFaceVertexIndicesAttr(triangles.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    material = _material(mesh.GetPrim().GetStage(), '/World/TaskVisuals/WhiteboardInk',
                         config['ink_color_rgb'], .48)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
