#!/usr/bin/env python3
"""Fail closed on operational identifiers, unexpected paths and unsafe archives."""
from pathlib import Path, PurePosixPath
import io
import re
import sys
import tarfile

ALLOWED = {".gitignore", ".gitleaks.toml", "LICENSE", "NOTICE", "CREDITS.md",
           "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "CITATION.cff", "SECURITY.md",
           "README.md", "llms.txt", "cluster.env.example",
           "benchmarks", "docs", "engine", "release", "scripts", "serving", "tools"}
# Split literal detector terms so this file is itself scanned without exemptions.
PATTERNS = {
    "private home": rb"/home/" + b"chimera",
    "private network": rb"192" + rb"\.168\.",
    "private hostname": rb"(?<![A-Za-z0-9/-])spark" + rb"-[A-Za-z0-9]+",
    "assistant internals": rb"(?i)(?:shiki" + rb"gami|clau" + rb"de|asa-" + rb"dot|dot" + rb"pc|ultra" + rb"code|\.co" + rb"dex/sessions)",
    "private project": rb"(?i)(?:spark" + rb"Kitchen|mia-tp3-" + rb"deploy|/GLM" + rb"53/)",
}
FORBIDDEN_PARTS = {".git", ".state", ".venv", "__pycache__", ".pytest_cache",
                   "node_modules", "zig-cache", ".zig-cache", "toolchains"}
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024

CHECKPOINT_SUFFIXES = {".safetensors", ".gguf", ".pt", ".pth", ".ckpt"}
UNSUPPORTED_ARCHIVES = {".zip", ".7z", ".rar", ".bz2", ".xz", ".zst"}

def inspect_name(name, *, archive=False):
    errors = []
    path = PurePosixPath(name)
    for label, pattern in PATTERNS.items():
        if re.search(pattern, name.encode()) or any(re.search(pattern, part.encode()) for part in path.parts):
            errors.append(f"{name}: {label} in path")
    if path.is_absolute() or ".." in path.parts:
        errors.append(f"{name}: unsafe path")
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        errors.append(f"{name}: runtime/cache content")
    if path.suffix.lower() in CHECKPOINT_SUFFIXES:
        errors.append(f"{name}: checkpoint payload is not exported")
    if path.suffix.lower() in UNSUPPORTED_ARCHIVES or (path.suffix.lower() == ".gz" and not name.lower().endswith(".tar.gz")):
        errors.append(f"{name}: unsupported archive format")
    return errors

def inspect_bytes(name, data, depth=0):
    errors = inspect_name(name.split("::")[-1])
    for label, pattern in PATTERNS.items():
        if re.search(pattern, data):
            errors.append(f"{name}: {label}")
    if name.lower().endswith((".tar.gz", ".tgz", ".tar")):
        if depth >= 3:
            return errors + [f"{name}: archive nesting exceeds scan limit"]
        try:
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                total = 0
                for member in archive:
                    path = PurePosixPath(member.name)
                    errors += inspect_name(member.name, archive=True)
                    if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                        errors.append(f"{name}: unsafe archive member")
                        continue
                    if not (member.isfile() or member.isdir()):
                        errors.append(f"{name}: unsupported archive member")
                        continue
                    total += member.size
                    if total > MAX_ARCHIVE_BYTES:
                        errors.append(f"{name}: archive exceeds scan limit")
                        break
                    if member.isfile():
                        errors += inspect_bytes(name + "::" + member.name, archive.extractfile(member).read(), depth+1)
        except (tarfile.TarError, OSError) as error:
            errors.append(f"{name}: archive could not be scanned: {type(error).__name__}")
    return errors

def audit(root):
    errors, files = [], 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts[0] == ".git":
            continue
        if rel.parts[0] not in ALLOWED:
            errors.append(f"{rel}: outside release allowlist")
        errors += inspect_name(str(rel))
        if path.is_symlink():
            errors.append(f"{rel}: symlinks are not exported")
            continue
        if any(part in FORBIDDEN_PARTS for part in rel.parts):
            errors.append(f"{rel}: runtime/cache content")
        if len(rel.parts) >= 3 and rel.parts[:3] == ("engine", "build", "out"):
            errors.append(f"{rel}: build output is not source")
        if path.is_file():
            if path.suffix.lower() in CHECKPOINT_SUFFIXES:
                errors.append(f"{rel}: checkpoint payload is not exported")
            files += 1
            errors += inspect_bytes(str(rel), path.read_bytes())
    return files, errors

def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    files, errors = audit(root)
    for error in errors:
        print("FAIL: " + error, file=sys.stderr)
    if errors:
        return 1
    print(f"Public operational-content scan passed: {files} files (including archive members)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
