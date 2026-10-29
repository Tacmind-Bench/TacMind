"""Coffee scene collision geometry and mass properties."""

from __future__ import annotations

from benchmark.catalog import ASSETS

COLLISIONS = ASSETS / "scenes/coffee/collisions.usda"


def _copy_properties(source, target, *, omit=()):
    from pxr import Sdf

    target.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(source.GetAppliedSchemas()))
    for attribute in source.GetAuthoredAttributes():
        if attribute.GetName() in omit:
            continue
        result = target.CreateAttribute(attribute.GetName(), attribute.GetTypeName(),
                                        custom=attribute.IsCustom(), variability=attribute.GetVariability())
        value = attribute.Get()
        if value is not None:
            result.Set(value)
        if attribute.GetConnections():
            result.SetConnections(attribute.GetConnections())
    for relation in source.GetAuthoredRelationships():
        target.CreateRelationship(relation.GetName(), custom=relation.IsCustom()).SetTargets(relation.GetTargets())


def _complete_floor(root, first_name):
    """Complete the octagonal floor with perpendicular copies of its bars."""
    from pxr import UsdGeom

    stage = root.GetStage()
    source = root.GetChild(first_name)
    if not source or not source.IsA(UsdGeom.Cube):
        raise ValueError("Expected procedural box: " + first_name)
    for angle in (90., 135.):
        path = root.GetPath().AppendChild("collision_floor_complement_" + str(int(angle)))
        if stage.GetPrimAtPath(path):
            raise ValueError("Collider already present: " + str(path))
        shape = UsdGeom.Cube.Define(stage, path)
        _copy_properties(source, shape.GetPrim())
        shape.GetPrim().GetAttribute("xformOp:rotateZ").Set(angle)


def _configure_collisions(root, source, first_name):
    """Use the asset's handle hulls and mass tensor with the spawner's contact settings."""
    from pxr import UsdGeom, UsdPhysics

    _complete_floor(root, first_name)
    stage = root.GetStage()
    for part in source.GetChildren():
        wall = root.GetChild("wall_0")
        if not wall or not part.IsA(UsdGeom.Mesh):
            raise ValueError("Expected cup wall and handle collision mesh")
        path = root.GetPath().AppendChild(part.GetName())
        if stage.GetPrimAtPath(path):
            raise ValueError("Collider already present: " + str(path))
        mesh = UsdGeom.Mesh.Define(stage, path)
        _copy_properties(wall, mesh.GetPrim(),
                         omit=("size", "xformOpOrder", "xformOp:translate", "xformOp:rotateZ", "xformOp:scale"))
        geometry = UsdGeom.Mesh(part)
        mesh.CreatePointsAttr(geometry.GetPointsAttr().Get())
        mesh.CreateFaceVertexCountsAttr(geometry.GetFaceVertexCountsAttr().Get())
        mesh.CreateFaceVertexIndicesAttr(geometry.GetFaceVertexIndicesAttr().Get())
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        UsdGeom.Imageable(mesh).MakeInvisible()
    original = UsdPhysics.MassAPI(source)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(original.GetMassAttr().Get())
    mass.CreateCenterOfMassAttr(original.GetCenterOfMassAttr().Get())
    mass.CreateDiagonalInertiaAttr(original.GetDiagonalInertiaAttr().Get())
    mass.CreatePrincipalAxesAttr(original.GetPrincipalAxesAttr().Get())


def configure_assets(assets, task):
    """Complete coffee object collisions without changing upstream spawners."""
    import functools

    if task["source"]["scene"] != "coffee":
        return assets
    from pxr import Usd

    data = Usd.Stage.Open(str(COLLISIONS))
    for name, first_name in (("cup", "bottom_0"), ("tray", "col_dish_0"), ("pod", "core_0")):
        if name not in assets or not data.GetDefaultPrim().GetChild(name):
            raise ValueError("Missing coffee collision asset: " + name)
        cfg = assets[name].spawn
        original = cfg.func

        @functools.wraps(original)
        def spawn(prim_path, spawn_cfg, translation=None, orientation=None, *,
                  _original=original, _data=data, _name=name, _first_name=first_name, **kwargs):
            root = _original(prim_path, spawn_cfg, translation=translation, orientation=orientation, **kwargs)
            _configure_collisions(root, _data.GetDefaultPrim().GetChild(_name), _first_name)
            return root

        cfg.func = spawn
    return assets
