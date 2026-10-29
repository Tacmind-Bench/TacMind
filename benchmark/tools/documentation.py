"""Check maintained documentation's local Markdown links without network access."""
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
PATTERNS = (
    "README*.md", "CONTRIBUTING.md", "docs/**/*.md",
    "policy/act_tactile/README*.md",
    "policy/TacMind-0/tacmind0/tactile/lewm/README*.md",
    "policy/optional-assets*.md",
    "examples/learning/README*.md",
    "assets/SOURCE*.md", "assets/materials/SOURCE*.md",
    "assets/materials/*/SOURCE*.md", "assets/scenes/pc_assembly/README*.md",
    "assets/robots/Franka_gsmini/README*.md",
    "third_party/robobench/README*.md",
)


def prose(path):
    """Exclude fenced examples, preserving line numbers for diagnostics."""
    fence = None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            yield number, line


def anchors(path):
    result, counts = set(), {}
    for _, line in prose(path):
        if heading := re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line):
            title = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", heading.group(1))
            slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            count = counts.get(slug, 0)
            counts[slug] = count + 1
            result.add(f"{slug}-{count}" if count else slug)
        result.update(re.findall(r'<a\s+(?:name|id)=["\']([^"\']+)["\']', line))
    return result


def check_links(root=ROOT):
    """Return errors, link count and document count for TacMind-maintained guides."""
    root = Path(root).resolve()
    documents = sorted({path for pattern in PATTERNS for path in root.glob(pattern)})
    errors, checked = [], 0
    anchor_cache = {}
    for document in documents:
        for number, line in prose(document):
            targets = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", line)
            targets += re.findall(
                r'<(?:a|img)\b[^>]*?\s(?:href|src)\s*=\s*["\']([^"\']+)["\']',
                line, flags=re.IGNORECASE)
            for target in targets:
                parsed = urlsplit(target.strip().strip("<>"))
                if parsed.scheme or parsed.netloc:
                    continue
                checked += 1
                path = (document.parent / unquote(parsed.path)).resolve() if parsed.path else document
                error = None
                if not path.exists():
                    error = "missing target"
                elif parsed.fragment and path.suffix == ".md":
                    if path not in anchor_cache:
                        anchor_cache[path] = anchors(path)
                    if unquote(parsed.fragment) not in anchor_cache[path]:
                        error = "missing heading anchor"
                if error:
                    errors.append(f"{document.relative_to(root)}:{number}: {error}: {target}")
    return errors, checked, len(documents)


def main():
    errors, checked, documents = check_links(ROOT)
    for error in errors:
        print(error)
    print(f"Checked {checked} local links in {documents} documents; {len(errors)} errors.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
