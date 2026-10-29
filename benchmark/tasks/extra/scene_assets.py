"""EmbodiedSWE source and local asset paths (no simulator imports)."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import sys
import tempfile

from benchmark.catalog import ASSETS, ROOT
from benchmark.tools.asset_layout import relocate_path

SOURCE = ROOT / "third_party/robobench"
MIRROR = ASSETS
GROUPS = tuple(f"robobench/suites/{suite}/assets" for suite in
               ("assembly", "packing", "puzzle", "cutting")) + tuple(
    f"robobench/assets/rooms/{room}" for room in
    ("factory", "factory001", "kitchen"))


def upstream():
    """Import upstream modules with the benchmark's asset paths."""
    os.environ["COSIGEN_ASSET_DIR"] = str(MIRROR)
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    import robobench
    # Upstream room/environment preparation otherwise downloads missing files.
    # Keep that source unchanged and route its asset boundary to our local bundle.
    from robobench.core import assets as upstream_assets
    original_path = getattr(upstream_assets, "asset_path", None)
    original_ensure = getattr(upstream_assets, "ensure_assets", None)
    upstream_assets.ensure_assets = skip_asset_downloads
    upstream_assets.asset_path = asset_path
    for name, module in tuple(sys.modules.items()):
        if name.startswith("robobench.") and module is not None:
            if original_path is not None and getattr(module, "asset_path", None) is original_path:
                module.asset_path = asset_path
            if original_ensure is not None and getattr(module, "ensure_assets", None) is original_ensure:
                module.ensure_assets = skip_asset_downloads
    return robobench


def asset_path(path):
    """Translate source asset files while retaining upstream directory arithmetic."""
    original = Path(path)
    path = Path(os.path.normpath(original))
    try:
        relative = path.relative_to(SOURCE)
    except ValueError:
        # A same-checkout suite may have been imported before the adapter.
        try:
            relative = path.relative_to(MIRROR)
        except ValueError:
            return original
        if not relative.parts or relative.parts[0] != "robobench":
            return original
        path = SOURCE / relative
    # Upstream builds paths through suite roots and their parents, and reads
    # source JSON metadata there. Resolve only the final file at our boundary.
    if not path.suffix:
        return path
    legacy = "assets/robobench/" + relative.as_posix()
    relocated = relocate_path(legacy)
    return ROOT / relocated if relocated != legacy else path


def relocate_config_assets(value):
    """Resolve scene inputs and spawn paths without changing upstream classes."""
    if isinstance(value, (str, Path)):
        relocated = asset_path(value)
        if relocated == Path(value):
            return value
        return str(relocated) if isinstance(value, str) else relocated
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            setattr(value, field.name, relocate_config_assets(getattr(value, field.name)))
    elif isinstance(value, dict):
        for key, item in value.items():
            value[key] = relocate_config_assets(item)
    elif isinstance(value, list):
        value[:] = [relocate_config_assets(item) for item in value]
    elif isinstance(value, tuple):
        return tuple(relocate_config_assets(item) for item in value)
    return value


def manifest_files(groups=GROUPS):
    """Map source asset groups to their installed paths for texture relocation."""
    groups = tuple(str(group).rstrip("/") for group in groups)
    manifest = json.loads((SOURCE / "robobench/assets_manifest.json").read_text())
    release = json.loads((ROOT / "benchmark/config/asset_manifest.json").read_text())
    released = {name: meta for group in release["groups"].values()
                for name, meta in group["files"].items()}
    authored = json.loads((SOURCE / "source_manifest.json").read_text())["files"]
    records = {}
    for path in manifest["files"].keys() | set(authored):
        relocated = relocate_path("assets/robobench/" + path)
        if not relocated.startswith("assets/"):
            continue  # Optional upstream source fragments are not release assets.
        relative = Path(relocated).relative_to("assets").as_posix()
        selected = {group for group in groups
                    if path == group or path.startswith(group + "/") or relative == group}
        if relocated in released and selected:
            records[relative] = released[relocated]
    return records


def skip_asset_downloads(groups):
    """TacMind assets import/fetch manages installation outside scene startup."""


def localized_usd(path):
    """Resolve local shader assets without changing source or physics bytes."""
    from pxr import Sdf, Usd, UsdGeom
    path = Path(path).resolve()
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    manifest = None
    replacements = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            if (value and attr.GetName() == "info:mdl:sourceAsset"
                    and not Path(value.path).is_absolute() and value.resolvedPath
                    and Path(value.resolvedPath).suffix.lower() == ".mdl"):
                # Fabric pools materials outside their source layer. MDL must
                # retain the resolved module path after that move.
                replacements.append((str(attr.GetPath()), Path(value.resolvedPath), value.path))
                continue
            if not value or not value.path or not Path(value.path).is_absolute() or Path(value.path).is_file():
                continue
            if manifest is None:
                manifest = manifest_files()
            # Match the longest published path suffix, preferring the shortest local path.
            parts = Path(value.path).parts
            matches = []
            for length in range(len(parts)-1, 1, -1):
                suffix = "/".join(parts[-length:])
                matches = [name for name in manifest if name.endswith("/" + suffix)]
                if matches:
                    break
            if not matches:
                continue
            relative = min(matches, key=lambda name: (len(name), name))
            target = MIRROR / relative
            replacements.append((str(attr.GetPath()), target, value.path))
    if not replacements:
        return str(path)
    output = path.with_suffix(".tacmind.usda")
    overlay = Usd.Stage.CreateInMemory()
    referenced = overlay.DefinePrim(root.GetPath(), root.GetTypeName())
    referenced.GetReferences().AddReference(str(path))
    overlay.SetDefaultPrim(referenced)
    UsdGeom.SetStageUpAxis(overlay, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(overlay, UsdGeom.GetStageMetersPerUnit(stage))
    for attr_path, target, _ in replacements:
        attr = overlay.GetAttributeAtPath(attr_path)
        attr.Set(Sdf.AssetPath(str(target) if attr.GetName() == "info:mdl:sourceAsset"
                               else os.path.relpath(target, output.parent)))
    overlay.GetRootLayer().customLayerData = {
        "TacMindTextureRelocations": json.dumps({name: {"original": original, "local": str(target)}
            for name, target, original in replacements}, sort_keys=True)}
    with tempfile.NamedTemporaryFile(suffix=".usda", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        overlay.GetRootLayer().Export(str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return str(output)
