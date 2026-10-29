"""Render-only detail and shader graph corrections for Extra scenes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from benchmark.catalog import ASSETS, ROOT

PROFILES = ASSETS / "scenes/extra/scene_materials.json"




def _detail_uv(mesh, frequency):
    """Face-varying box projection in asset metres; preserve the source UV set."""
    import numpy as np
    from pxr import Sdf, UsdGeom, Vt

    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    points = points @ transform[:3, :3] + transform[3, :3]
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    if not len(counts) or np.min(counts) < 3:
        raise ValueError(f'Cannot project detail UVs: {mesh.GetPath()}')
    normal = np.cross(points[indices[starts+1]]-points[indices[starts]],
                      points[indices[starts+2]]-points[indices[starts]])
    dominant = np.argmax(np.abs(normal), axis=1)
    axes = np.repeat(dominant, counts)
    vertices = points[indices]
    uv = np.empty((len(indices), 2), dtype=np.float32)
    # Orient each chart consistently with its geometric normal, including
    # negative-facing surfaces, so tangent-space normals do not get mirrored.
    signs = np.repeat(np.where(normal[np.arange(len(counts)), dominant] < 0, -1., 1.), counts)
    for axis, pair in ((0, (1, 2)), (1, (2, 0)), (2, (0, 1))):
        selected = axes == axis
        uv[selected] = vertices[selected][:, pair]
    uv[:, 0] *= signs
    # Isaac's native USD spawner consumes these coordinates directly; stage
    # metersPerUnit metadata is not an additional geometry conversion here.
    uv *= frequency
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        'tacmind_detail_st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    primvar.Set(Vt.Vec2fArray.FromNumpy(uv))


def _add_detail(stage, shader, spec, resources):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    normal = shader.GetInput('normal')
    if normal and (normal.HasConnectedSource() or normal.Get() not in (None, Gf.Vec3f(0, 0, 1))):
        raise ValueError(f'Detail would replace an authored normal: {shader.GetPath()}')
    material = UsdShade.Material(shader.GetPrim().GetParent())
    meshes = []
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        image = UsdGeom.Imageable(prim)
        if image.ComputeVisibility() == 'invisible' or image.ComputePurpose() in ('guide', 'proxy'):
            continue
        binding = UsdShade.MaterialBindingAPI(prim)
        bound, _ = binding.ComputeBoundMaterial()
        subsets = binding.GetMaterialBindSubsets()
        used = {str(bound.GetPath())} if bound else set()
        for subset in subsets:
            bound_subset, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
            if bound_subset:
                used.add(str(bound_subset.GetPath()))
        if str(material.GetPath()) in used:
            meshes.append(UsdGeom.Mesh(prim))
    if not meshes:
        return
    for mesh in meshes:
        _detail_uv(mesh, spec['frequency_per_m'])
    base = str(material.GetPath())
    reader = UsdShade.Shader.Define(stage, base+'/TacMindDetailUV')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.String).Set('tacmind_detail_st')
    texture = UsdShade.Shader.Define(stage, base+'/TacMindDetailNormal')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(ROOT/resources[spec['resource']]['path']))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
    texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
    texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
    strength = spec['strength']
    texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2*strength, 2*strength, 2, 1))
    texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-strength, -strength, -1, 0))
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    shader.CreateInput('normal', Sdf.ValueTypeNames.Normal3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')


def _add_gltf_normal(stage, shader, spec, resources):
    """Keep the glTF material and feed its native normal-lookup structure."""
    from pxr import Sdf, UsdShade

    normal = shader.GetInput('normal_texture')
    if normal and normal.HasConnectedSource():
        raise ValueError(f'Would replace an existing glTF normal: {shader.GetPath()}')
    path = shader.GetPrim().GetParent().GetPath().AppendChild('TacMindGltfNormal')
    lookup = UsdShade.Shader.Define(stage, path)
    lookup.SetSourceAsset(Sdf.AssetPath('gltf/pbr.mdl'), 'mdl')
    lookup.SetSourceAssetSubIdentifier('gltf_normal_texture_lookup', 'mdl')
    attr = lookup.CreateInput('texture', Sdf.ValueTypeNames.Asset).GetAttr()
    attr.Set(Sdf.AssetPath(str((ROOT/resources[spec['resource']]['path']).resolve())))
    attr.SetMetadata('colorSpace', 'raw')
    lookup.CreateInput('normal_scale_factor', Sdf.ValueTypeNames.Float).Set(spec['strength'])
    lookup.CreateInput('scale', Sdf.ValueTypeNames.Float2).Set(tuple(spec['uv_scale']))
    lookup.CreateOutput('out', Sdf.ValueTypeNames.Token)
    shader.CreateInput('normal_texture', Sdf.ValueTypeNames.Token).ConnectToSource(lookup.ConnectableAPI(), 'out')


def build_overlay(composed, output, profile, resources, *, material_references=None):
    """Only shader inputs/new detail UVs may be authored, never physics data."""
    from pxr import Gf, Sdf, Usd, UsdShade

    original = Usd.Stage.Open(str(composed))
    root = original.GetDefaultPrim()
    stage = Usd.Stage.CreateInMemory()
    target = stage.DefinePrim(root.GetPath(), root.GetTypeName())
    target.GetReferences().AddReference(str(Path(composed).resolve()))
    stage.SetDefaultPrim(target)
    for key in ('upAxis', 'metersPerUnit'):
        if original.HasAuthoredMetadata(key):
            stage.SetMetadata(key, original.GetMetadata(key))
    # Edit the referenced visual layer before instancing, so its geometry and
    # material subsets remain instanced and no opinions target proxy prims.
    for path, spec in profile.get('material_references', {}).items():
        prim = stage.GetPrimAtPath(path)
        references = prim.GetMetadata('references').GetAppliedItems() if prim else []
        if not prim or not references:
            raise ValueError(f'Missing visual reference: {path}')
        reference = references[0]
        prim.GetReferences().SetReferences([Sdf.Reference(
            material_references[path], reference.primPath, reference.layerOffset, reference.customData),
            *references[1:]])
    for path, changes in profile.get('inputs', {}).items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(path))
        if not shader:
            raise ValueError(f'Missing shader: {path}')
        for name, spec in changes.items():
            value = (Sdf.AssetPath(str((ROOT/resources[spec['resource']]['path']).resolve()))
                     if spec.get('resource') else spec['value'])
            type_name = getattr(Sdf.ValueTypeNames, spec['type'])
            if spec['type'] in ('Color3f', 'Normal3f', 'Float3'):
                value = Gf.Vec3f(*value)
            elif spec['type'] == 'Float2':
                value = Gf.Vec2f(*value)
            elif spec['type'] == 'Float4':
                value = Gf.Vec4f(*value)
            attribute = shader.CreateInput(name, type_name).GetAttr()
            attribute.Set(value)
            if spec.get('color_space'):
                attribute.SetMetadata('colorSpace', spec['color_space'])
    for path, spec in profile.get('detail', {}).items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(path))
        if not shader:
            raise ValueError(f'Missing detail shader: {path}')
        if spec.get('mdl'):
            shader.CreateInput('normalmap_texture', Sdf.ValueTypeNames.Asset).Set(str(ROOT/resources[spec['resource']]['path']))
            shader.CreateInput('bump_factor', Sdf.ValueTypeNames.Float).Set(spec['strength'])
            shader.CreateInput('project_uvw', Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput('texture_scale', Sdf.ValueTypeNames.Float2).Set((spec['frequency_per_m'],)*2)
        else:
            _add_detail(stage, shader, spec, resources)
    for path, spec in profile.get('gltf_normal', {}).items():
        shader = UsdShade.Shader(stage.GetPrimAtPath(path))
        if not shader:
            raise ValueError(f'Missing glTF shader: {path}')
        _add_gltf_normal(stage, shader, spec, resources)
    stage.GetRootLayer().Export(str(output))


def enhanced_usd(source, composed):
    """Layer visual changes above localization and collision repairs."""
    from benchmark.tasks.extra.scene_assets import MIRROR
    from pxr import Sdf

    if '://' in str(source):
        return str(composed)
    source = Path(source).resolve()
    try:
        relative = source.relative_to(MIRROR.resolve()).as_posix()
    except ValueError:
        return str(composed)
    config = json.loads(PROFILES.read_text())
    profile = config['usd'].get(relative)
    if not profile:
        return str(composed)
    used = {spec['resource'] for spec in profile.get('detail', {}).values()}
    used.update(spec['resource'] for changes in profile.get('inputs', {}).values()
                for spec in changes.values() if spec.get('resource'))
    used.update(spec['resource'] for spec in profile.get('gltf_normal', {}).values())
    resources = {key: config['resources'][key] for key in used}
    for resource in resources.values():
        if not (ROOT/resource['path']).is_file():
            raise FileNotFoundError(ROOT/resource['path'])
    material_references = {}
    for path, spec in profile.get('material_references', {}).items():
        from benchmark.tasks.extra.scene_assets import localized_usd
        visual = MIRROR/spec['source']
        material_references[path] = enhanced_usd(visual, localized_usd(visual))
    output = source.with_suffix('.tacmind-materials.usda')
    with tempfile.NamedTemporaryFile(suffix='.usda', dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        build_overlay(composed, temporary, profile, resources, material_references=material_references)
        os.replace(temporary, output)
        layer = Sdf.Layer.Find(str(output))
        if layer:
            layer.Reload()
    finally:
        temporary.unlink(missing_ok=True)
    return str(output)


def configure_assets(assets):
    """Enhance procedural visual references after their physics cook."""
    import functools
    from pxr import Sdf

    for asset in assets.values():
        cfg = asset.spawn
        source = getattr(cfg, 'visual_usd', None)
        if not source:
            continue
        composed = enhanced_usd(source, source)
        if composed == str(source):
            continue
        original = cfg.func

        @functools.wraps(original)
        def spawn(prim_path, spawn_cfg, translation=None, orientation=None, *,
                  _original=original, _composed=composed, **kwargs):
            root = _original(prim_path, spawn_cfg, translation=translation,
                             orientation=orientation, **kwargs)
            visual = root.GetStage().GetPrimAtPath(root.GetPath().AppendChild('visual'))
            if not visual:
                raise ValueError('Procedural visual prim is missing')
            visual.GetReferences().SetReferences([Sdf.Reference(_composed)])
            return root

        cfg.func = spawn
