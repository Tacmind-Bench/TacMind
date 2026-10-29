"""Translate the published asset bundle into the current repository layout."""

from functools import lru_cache
import json
import os
from pathlib import Path
import posixpath
import re
import subprocess
import sys
import sysconfig

from benchmark.catalog import ROOT

LAYOUT = Path(__file__).resolve().parents[1] / "config/asset_layout.json"


@lru_cache(maxsize=1)
def _layout():
    return sorted(json.loads(LAYOUT.read_text()).items(), key=lambda item: -len(item[0]))


def relocate_path(relative):
    """Apply the most specific old repository path prefix, if one is declared."""
    relative = str(relative)
    for old, new in _layout():
        if relative == old or relative.startswith(old + "/"):
            return new + relative[len(old):]
    return relative


@lru_cache(maxsize=1)
def _published_materials():
    manifest = json.loads(LAYOUT.with_name("asset_manifest.json").read_text())
    return {name for group in manifest["groups"].values() for name in group["files"]
            if name.endswith(".mdl")}


def _reference(reference, old_path):
    if not reference or "://" in reference:
        return reference
    if reference.startswith("/"):
        if reference.startswith(str(ROOT) + "/"):
            relative = reference[len(str(ROOT)) + 1:]
        elif "/assets/" in reference:
            relative = "assets/" + reference.split("/assets/", 1)[1]
        elif "/materials/" in reference:
            relative = "assets/materials/" + reference.split("/materials/", 1)[1]
            if not any(old != new and (relative == old or relative.startswith(old + "/"))
                       for old, new in _layout()):
                return reference
        else:
            return reference
    else:
        relative = posixpath.normpath(posixpath.join(posixpath.dirname(old_path), reference))
    matching = next(((old, new) for old, new in _layout()
                     if relative == old or relative.startswith(old + "/")), None)
    if matching is None:
        return reference
    if "/" not in reference and reference.endswith(".mdl"):
        # Bare names can resolve through the installed MDL SDK search path.
        # A parent-directory move does not establish a bundled local module.
        if matching[0] != relative and relocate_path(relative) not in _published_materials():
            return reference
    new = posixpath.relpath(relocate_path(relative), posixpath.dirname(relocate_path(old_path)))
    # Keep the original spelling when its location within the bundle is stable.
    if not reference.startswith("/") and posixpath.normpath(reference) == new:
        return reference
    return "./" + new if reference.startswith("./") and not new.startswith(".") else new


def _rewrite_with_isaac(path, old_path):
    """Use the configured SDK's CPU USD libraries without starting Kit."""
    configured = os.environ.get("ISAAC_PATH")
    sdk = Path(configured) if configured else ROOT / "third_party/IsaacLab/_isaac_sim"
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    candidates = [sdk / "exts/omni.usd.libs", *sorted((sdk / "extscache").glob(f"omni.usd.libs-*{abi}*"))]
    library = next((candidate for candidate in candidates
                    if (candidate / "pxr").is_dir() and (candidate / "bin").is_dir()), None)
    if library is None:
        raise ImportError("Relocating an older USD bundle requires the configured Isaac Sim USD libraries. "
                          "Activate the tacmind environment and follow README.md#installation.")
    environment = dict(os.environ)
    for key, paths in (("PYTHONPATH", [str(ROOT), str(library)]),
                       ("LD_LIBRARY_PATH", [str(library / "bin"), sysconfig.get_config_var("LIBDIR")])):
        environment[key] = os.pathsep.join([*paths, environment.get(key, "")])
    # The native loader reads LD_LIBRARY_PATH at process startup, so setting it
    # in this Python process cannot make the SDK extension's libraries visible.
    command = [sys.executable, "-c", "import sys; from pxr import Sdf, UsdUtils; "
               "from benchmark.tools.asset_layout import rewrite_asset; "
               "print(int(rewrite_asset(sys.argv[1], sys.argv[2])))", str(path), old_path]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"Cannot relocate USD asset {old_path}: {result.stderr.strip()}")
    return result.stdout.strip() == "1"


def _repair_material_bindings(layer, canonical_path):
    """Remove known dangling opinions without creating replacement materials."""
    repairs = {
        "assets/scenes/bin_stacking/ur10_bin_stacking_short_suction_no_robot.usda": (
            ("/Stage/staticPlaneActor.material:binding", "/Stage/pallet_holder/Looks/BlackMetallic"),
        ),
        "assets/objects/ur10_mount/ur10_mount.usd": (
            ("/Root/Base/None.material:binding", "/Root/Looks/Extruded_aluminum"),
        ),
        "assets/objects/dex_cube/dex_cube_instanceable.usd": (
            ("/DexCube/collisions.material:binding", "/DexCube/CubeMaterial"),
            ("/DexCube/collisions.material:binding:physics", "/DexCube/CubeMaterial"),
        ),
        "assets/objects/ClothDemo_002/cloth_demo_001.usd": (
            ("/World/sim_scene.material:binding", "/World/table/OmniPBR"),
        ),
        "assets/objects/waterbottle_7/waterbottle_7.usd": (
            ("/root.material:binding", "/root/Looks/bottle_lid"),
            ("/root/emptyenv/Ground/Plane001.material:binding:physics", "/World/PhysicsMaterial"),
        ),
        "assets/objects/SM_P_Flavour_04/SM_P_Flavour_04.usd": (
            ("/World/SM_P_Flavour_04/d.material:binding", "/World/Looks/_"),
            ("/World/SM_P_Flavour_04/_3.material:binding", "/World/Looks/_4"),
        ),
        "assets/objects/SM_P_Tray_01/SM_P_Tray_01.usd": (
            ("/World/SM_P_Tray_01/_3.material:binding", "/World/Looks/_"),
        ),
        "assets/objects/Kitchen_Flowers001/Kitchen_Flowers001.usd": (
            ("/root/Kitchen_Flowers001/Collider/Kitchen_Flowers001_Collider2/boom100214.material:binding",
             "/root/flower01_Collider/Looks/boom100214"),
            ("/root/Kitchen_Flowers001/Collider/Kitchen_Flowers001_Collider2/boom100215.material:binding",
             "/root/flower01_Collider/Looks/boom100215"),
        ),
        "assets/objects/SM_P_Flavour_03_Cover/SM_P_Flavour_03_Cover.usd": (
            ("/World/SM_P_Flavour_03_Cover/AM134_14_Paint.material:binding",
             "/World/Looks/AM134_14_Paint"),
        ),
    }
    obsolete = []
    for property_path, target in repairs.get(canonical_path, ()):
        relationship = layer.GetRelationshipAtPath(property_path)
        if relationship is None:
            continue  # Current bundles have already removed this opinion.
        if [str(path) for path in relationship.targetPathList.GetAppliedItems()] != [target]:
            continue
        obsolete.append(relationship)
    for relationship in obsolete:
        relationship.owner.RemoveProperty(relationship)
    return bool(obsolete)


def rewrite_asset(path, old_path):
    """Migrate published USD/MDL assets, preserving untouched file bytes."""
    path, old_path = Path(path), str(old_path)
    if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
        try:
            from pxr import Sdf, UsdUtils
        except ImportError:
            return _rewrite_with_isaac(path, old_path)

        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None:
            raise ValueError(f"Cannot read relocated USD asset: {old_path}")
        changed = False

        def replace(reference):
            nonlocal changed
            updated = _reference(reference, old_path)
            changed |= updated != reference
            return updated

        UsdUtils.ModifyAssetPaths(layer, replace)
        changed |= _repair_material_bindings(layer, relocate_path(old_path))
        if changed:
            layer.Save()
        return changed
    if path.suffix.lower() != ".mdl":
        return False

    source = path.read_bytes().decode("utf-8")
    # Keep comments opaque so examples and provenance are not treated as assets.
    tokens = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\r\n]*', re.S)
    comments = {}

    def hide_comment(match):
        if match[0].startswith('"'):
            return match[0]
        marker = f"\x00{len(comments)}\x00"
        comments[marker] = match[0]
        return marker

    updated = tokens.sub(hide_comment, source)
    updated = re.sub(
        r'(\b(?:texture_2d|texture_3d|texture_cube|light_profile|bsdf_measurement)'
        r'\s*\(\s*(?:name\s*:\s*)?")([^"\r\n]+)(")',
        lambda match: match[1] + _reference(match[2], old_path) + match[3], updated)
    qualified_modules = {}

    def replace_module(match):
        module = match[2]
        if module.startswith("::"):
            return match[0]
        relative = "/".join(module.split("::")) + ".mdl"
        relocated = _reference(relative, old_path)
        if relocated == relative:
            return match[0]
        relocated = relocated.removesuffix(".mdl")
        if "/" not in relocated:
            relocated = "./" + relocated
        relocated_module = "::".join(relocated.split("/"))
        # MDL qualified identifiers use the imported package/module path,
        # omitting the current/parent package markers from relative imports.
        before = "::".join(part for part in module.split("::") if part not in {".", ".."})
        after = "::".join(part for part in relocated_module.split("::") if part not in {".", ".."})
        if before != after:
            qualified_modules[before] = after
        return match[1] + relocated_module + match[3]

    updated = re.sub(r'(\busing\s+)([\w.:]+)(\s+import\b)', replace_module, updated)
    updated = re.sub(r'(\bimport\s+)([\w.:]+)(::(?:\*|\w+)\s*;)', replace_module, updated)
    if qualified_modules:
        # Keep rewritten declarations and quoted strings opaque. Replacing one
        # lexical identifier at a time prevents repeated namespace expansion.
        code_tokens = re.compile(r'"(?:\\.|[^"\\])*"|\b(?:using|import)\b[^;]*;|'
                                 r'(?<![\w.:])(?:::)?(?:[A-Za-z_]\w*::)+[A-Za-z_]\w*')

        def replace_qualified(match):
            if match[0].startswith('"') or re.match(r'(?:using|import)\b', match[0]):
                return match[0]
            qualified = match[0].removeprefix("::")
            for before, after in sorted(qualified_modules.items(), key=lambda item: -len(item[0])):
                if qualified.startswith(before + "::"):
                    return after + qualified[len(before):]
            return match[0]

        updated = code_tokens.sub(replace_qualified, updated)
    for marker, comment in comments.items():
        updated = updated.replace(marker, comment)
    if updated != source:
        path.write_bytes(updated.encode("utf-8"))
    return updated != source
