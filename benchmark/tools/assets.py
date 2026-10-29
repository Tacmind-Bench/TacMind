"""Install declared assets without overwriting local files."""

import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile

from benchmark.catalog import ROOT
from benchmark.tools.asset_layout import rewrite_asset

MANIFEST = Path(__file__).resolve().parents[1] / "config/asset_manifest.json"


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError(f"Asset path escapes its root: {relative}")
    return path


def inventory(groups=None, manifest=None):
    manifest = manifest or json.loads(MANIFEST.read_text())
    selected = list(manifest["groups"]) if groups is None else groups
    records = {}
    for group in selected:
        spec = manifest["groups"][group]
        files = spec.get("files", {})
        for path, meta in files.items():
            safe_path(ROOT, path)
            records[path] = meta
    return records


def import_bundle(source, records, root=ROOT):
    """Copy missing assets and preserve every existing local file."""
    source, root = Path(source).resolve(), Path(root).resolve()
    missing = {}
    for relative, meta in records.items():
        target = safe_path(root, relative)
        if not target.exists():
            missing[relative] = meta
    inputs = {}
    for relative, meta in missing.items():
        source_name = relative if safe_path(source, relative).exists() else meta.get("source_path", relative)
        inputs[relative] = (safe_path(source, source_name), source_name)
    with tempfile.TemporaryDirectory() as staging:
        for relative, meta in missing.items():
            path, old_name = inputs[relative]
            if "source_path" in meta and path.suffix.lower() in {".usd", ".usda", ".usdc", ".mdl"}:
                converted = safe_path(staging, relative)
                converted.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, converted)
                rewrite_asset(converted, old_name)
                inputs[relative] = (converted, relative)
        copied = 0
        for relative in missing:
            target = safe_path(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                shutil.copyfile(inputs[relative][0], temporary)
                # Publish a complete file without replacing concurrent writes.
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    raise FileExistsError(f"Refusing to replace existing asset: {target}") from None
                copied += 1
            finally:
                temporary.unlink(missing_ok=True)
    return copied


def fetch_bundle(provider, records, manifest):
    """Use one TacMind asset dataset mirror; upstream model origins are not downloads."""
    spec = manifest["providers"][provider]
    records = {name: meta for name, meta in records.items() if not safe_path(ROOT, name).exists()}
    if not records:
        return 0
    local_dir = safe_path(ROOT / "tmp/asset-downloads" / provider, spec["repo_id"])
    # Materialize files locally: hub cache snapshots may use symlinks to blobs
    # outside their root, which the checked importer deliberately rejects.
    local_dir.mkdir(parents=True, exist_ok=True)
    archive = spec.get("archive")
    paths = [archive["path"]] if archive else list(dict.fromkeys(
        name for relative, meta in records.items()
        for name in (relative, meta.get("source_path", relative))))
    if provider == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(repo_id=spec["repo_id"], repo_type="dataset",
                                     revision=spec.get("revision"), allow_patterns=paths,
                                     local_dir=str(local_dir))
    else:
        from modelscope.hub.snapshot_download import dataset_snapshot_download
        snapshot = dataset_snapshot_download(dataset_id=spec["repo_id"], revision=spec.get("revision"),
                                             allow_file_pattern=paths, local_dir=str(local_dir))
    if archive:
        return import_archive(safe_path(snapshot, archive["path"]), records, local_dir)
    return import_bundle(snapshot, records)


def import_archive(path, records, temporary_root):
    """Extract declared regular files without following archive links."""
    path = Path(path)
    records = {name: meta for name, meta in records.items()
               if not safe_path(ROOT, name).exists()}
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary, tarfile.open(path, "r:gz") as bundle:
        expected = {}
        for name, meta in records.items():
            for source_name in (name, meta.get("source_path", name)):
                safe_path(temporary, source_name)
                if source_name in expected and expected[source_name] != name:
                    raise ValueError(f"Conflicting asset archive path: {source_name}")
                expected[source_name] = name
        members, found = {}, set()
        for member in bundle:
            if member.name not in expected:
                continue
            safe_path(temporary, member.name)
            destination = expected[member.name]
            if destination in found or not member.isfile():
                raise ValueError(f"Invalid or duplicate asset archive member: {member.name}")
            members[member.name] = member
            found.add(destination)
        for name, member in members.items():
            target = safe_path(temporary, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
        return import_bundle(temporary, records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "fetch", "import"))
    parser.add_argument("--group", action="append", choices=("tracked", "prepared", "extra"))
    parser.add_argument("--source", type=Path, help="Bundle directory containing assets/, for import")
    parser.add_argument("--provider", choices=("huggingface", "modelscope"), default="huggingface")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    groups = args.group or list(manifest["groups"])
    if args.action == "list":
        print(json.dumps({name: {k: v for k, v in manifest['groups'][name].items() if k != 'files'}
                          for name in groups}, indent=2))
        return
    if args.action == "import" and args.source is None:
        parser.error("import requires --source pointing to a bundle containing assets/")
    try:
        records = inventory(groups, manifest)
        if args.action == "fetch":
            copied = fetch_bundle(args.provider, records, manifest)
        else:
            copied = import_bundle(args.source, records)
    except (OSError, ValueError, KeyError, ImportError, tarfile.TarError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Installed {copied} assets.")


if __name__ == "__main__":
    main()
