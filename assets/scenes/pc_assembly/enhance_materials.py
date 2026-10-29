from __future__ import annotations

import json
import math
import os
import re
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parents[1]
STAGE_PATH = ROOT / 'pc_assembly_gallery.usdc'
MATERIALS = ASSETS / 'materials'
TEXTURE_SIZE = 1024


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]', '_', value)


def read_glb_materials(path: Path):
    data = path.read_bytes()
    offset = 12
    while offset < len(data):
        size, kind = struct.unpack_from('<I4s', data, offset)
        offset += 8
        chunk = data[offset:offset + size]
        offset += size
        if kind == b'JSON':
            gltf = json.loads(chunk.decode('utf-8'))
            return {m.get('name', f'Material_{i}'): m for i, m in enumerate(gltf.get('materials', []))}
    raise ValueError(f'No GLB JSON chunk in {path}')


def material_finish(name: str):
    upper = name.upper()
    if 'GLASS' in upper or 'LABEL' in upper or 'CABLE_SLEEVE' in upper:
        return None
    if 'PCB' in upper or 'SUBSTRATE' in upper:
        return 'soldermask', 0.006, 0.025
    if 'FAN' in upper or 'SHROUD' in upper or 'CONNECTOR' in upper:
        return 'polymer', 0.010, 0.035
    if 'COATED' in upper or 'PSU' in upper:
        return 'powder_coat', 0.014, 0.045
    if 'STEEL' in upper:
        return 'zinc_steel', 0.022, 0.055
    if 'GOLD' in upper:
        return 'polished_gold', 0.012, 0.035
    if any(tag in upper for tag in ('ALU', 'NICKEL', 'HEATSINK', 'SPREADER', 'STORAGE_CASE')):
        return 'brushed_metal', 0.032, 0.045
    return None


def make_normal_roughness_map(material_name: str, roughness: float, finish):
    family, normal_strength, roughness_strength = finish
    size = TEXTURE_SIZE
    rng = np.random.default_rng(zlib.crc32(f'{material_name}:{family}'.encode('utf-8')))
    u = np.arange(size, dtype=np.float32)[None, :] / size
    v = np.arange(size, dtype=np.float32)[:, None] / size
    height = np.zeros((size, size), dtype=np.float32)

    for _ in range(22):
        if family == 'brushed_metal':
            ku = int(rng.integers(0, 5))
            kv = int(rng.integers(28, 220))
        else:
            ku, kv = (int(k) for k in rng.integers(4, 190, size=2))
        frequency = math.hypot(ku, kv)
        phase = float(rng.uniform(0, 2 * math.pi))
        amplitude = 1.0 / max(frequency, 1.0) ** 1.35
        height += amplitude * np.sin(2 * math.pi * (ku * u + kv * v) + phase)

    du = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * (size * 0.5)
    dv = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * (size * 0.5)
    slope_rms = math.sqrt(float(np.mean(du * du + dv * dv))) or 1.0
    nx = -du * (normal_strength / slope_rms)
    ny = -dv * (normal_strength / slope_rms)
    nz = np.ones_like(nx)
    normal_length = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack((nx / normal_length, ny / normal_length, nz / normal_length), axis=-1)

    rgb = np.clip(np.rint((normal * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    roughness_field = height - float(height.mean())
    roughness_field /= float(roughness_field.std()) or 1.0
    if roughness >= 0.97:
        roughness_values = np.full_like(height, roughness)
    else:
        roughness_values = np.clip(roughness + roughness_field * roughness_strength, 0.02, 0.96)
    alpha = np.clip(np.rint(roughness_values * 255), 0, 255).astype(np.uint8)
    rgba = np.dstack((rgb, alpha))

    name = safe_name(material_name.lower())
    path = MATERIALS / name / 'textures' / f'{name}_normal_roughness.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(path, optimize=True)
    return path


def read_obj_uv_faces(path: Path):
    texcoords = []
    faces_by_material = {}
    active_material = 'Default'
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'vt' and len(fields) >= 3:
            texcoords.append((float(fields[1]), float(fields[2])))
        elif fields[0] == 'usemtl' and len(fields) >= 2:
            active_material = fields[1]
        elif fields[0] == 'f' and len(fields) >= 4:
            face_uvs = []
            for spec in fields[1:]:
                indices = spec.split('/')
                if len(indices) < 2 or not indices[1]:
                    raise ValueError(f'Missing OBJ UV index in {path.name}: {spec}')
                uv_index = int(indices[1])
                uv_index = uv_index - 1 if uv_index > 0 else len(texcoords) + uv_index
                face_uvs.append(texcoords[uv_index])
            faces_by_material.setdefault(active_material, []).append(face_uvs)
    return faces_by_material


def author_texture_network(stage, material, shader, texture_path: Path):
    material_path = material.GetPath()
    reader = UsdShade.Shader.Define(stage, material_path.AppendChild('DetailUVReader'))
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)

    texture = UsdShade.Shader.Define(stage, material_path.AppendChild('NormalRoughnessTexture'))
    texture.CreateIdAttr('UsdUVTexture')
    relative = Path(os.path.relpath(texture_path, STAGE_PATH.parent)).as_posix()
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(relative))
    texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
    texture.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set((2.0, 2.0, 2.0, 1.0))
    texture.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set((-1.0, -1.0, -1.0, 0.0))
    texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
    texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
    texture.CreateInput('minFilter', Sdf.ValueTypeNames.Token).Set('linear')
    texture.CreateInput('magFilter', Sdf.ValueTypeNames.Token).Set('linear')
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
    texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    texture.CreateOutput('a', Sdf.ValueTypeNames.Float)
    shader.CreateInput('normal', Sdf.ValueTypeNames.Normal3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), 'a')


def main():
    materials = read_glb_materials(ROOT / 'pc_anatomy_master.glb')
    stage = Usd.Stage.Open(str(STAGE_PATH))
    if stage is None:
        raise RuntimeError(f'Could not open {STAGE_PATH}')

    uv_mesh_count = 0
    textured_materials = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        component = stage.GetPrimAtPath(prim.GetPath().GetParentPath())
        obj_name = component.GetCustomDataByKey('sourceOBJ')
        if not obj_name:
            continue

        mesh_name = prim.GetName()
        material_suffix = mesh_name.split('_', 2)[-1]
        obj = Path(obj_name)
        obj_path = ASSETS / 'objects' / f'pc_{obj.stem}' / obj.name
        uv_groups = read_obj_uv_faces(obj_path)
        source_material_name = next(
            (name for name in uv_groups if safe_name(name) == material_suffix), None,
        )
        if source_material_name is None:
            raise ValueError(f'Could not match OBJ material for {prim.GetPath()}')

        uv_values = [uv for face in uv_groups[source_material_name] for uv in face]
        mesh = UsdGeom.Mesh(prim)
        expected_count = sum(mesh.GetFaceVertexCountsAttr().Get())
        if len(uv_values) != expected_count:
            raise ValueError(f'OBJ/USD face corner mismatch for {prim.GetPath()}')
        st = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            'st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying,
        )
        st.Set(Vt.Vec2fArray([Gf.Vec2f(*uv) for uv in uv_values]))
        uv_mesh_count += 1

        finish = material_finish(source_material_name)
        if finish is None:
            continue
        material, _binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        material_path = str(material.GetPath())
        if material_path in textured_materials:
            continue
        shader = UsdShade.Shader.Get(stage, material.GetPath().AppendChild('PreviewSurface'))
        if not shader:
            raise ValueError(f'Missing PreviewSurface shader on {material.GetPath()}')
        source = materials.get(source_material_name, {})
        roughness = float(source.get('pbrMetallicRoughness', {}).get('roughnessFactor', 1.0))
        texture_path = make_normal_roughness_map(source_material_name, roughness, finish)
        author_texture_network(stage, material, shader, texture_path)
        textured_materials.add(material_path)

    stage.GetRootLayer().Save()
    print(f'Authored UVs on {uv_mesh_count} meshes and detail maps on {len(textured_materials)} materials')


if __name__ == '__main__':
    main()
