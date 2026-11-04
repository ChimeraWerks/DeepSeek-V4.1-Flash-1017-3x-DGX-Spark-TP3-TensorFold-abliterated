#!/usr/bin/env python3
"""Read-only proof that every embedded CUDA fatbin equals a supplied production image."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-empty", type=Path, action="append", default=[],
                        help="Explicit CPU-only binary that may embed no CUDA images")
    parser.add_argument("production", type=Path)
    parser.add_argument("binaries", type=Path, nargs="+")
    args = parser.parse_args()
    sources = {}
    for path in sorted(args.production.glob("*.fatbin")):
        data = path.read_bytes()
        magic, version, header, size = struct.unpack_from("<IHHQ", data)
        if (magic, version, header, header + size) != (0xBA55ED50, 1, 16, len(data)):
            raise ValueError(f"Unsupported fatbin header: {path}")
        sources[sha(data)] = {"name": path.name, "bytes": len(data), "sha256": sha(data)}
    if not sources:
        raise ValueError("No production fatbins found")
    results = []
    for path in args.binaries:
        with path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
            found = []
            at = 0
            while True:
                at = data.find(b"\x50\xed\x55\xba", at)
                if at < 0:
                    break
                if at + 16 > len(data):
                    raise ValueError(f"Truncated fatbin header in {path}")
                _, version, header, size = struct.unpack_from("<IHHQ", data, at)
                # Machine instructions may contain the magic. A CUDA image also has
                # a version-1, 16-byte header. Every such image must be byte-identical.
                if (version, header) != (1, 16):
                    at += 4
                    continue
                end = at + header + size
                if end > len(data):
                    raise ValueError(f"Truncated fatbin payload in {path} at {at}")
                digest = sha(data[at:end])
                source = sources.get(digest)
                if source is None or source["bytes"] != end - at:
                    raise ValueError(f"Non-production fatbin in {path} at {at}: {digest}")
                found.append(dict(source, offset=at))
                at = end
            if not found and path not in args.allow_empty:
                raise ValueError(f"No embedded CUDA images in {path}; explicitly identify CPU-only tools")
            results.append({"binary": str(path), "binary_sha256": sha(data), "embedded": found})
    print(json.dumps({"production": str(args.production), "production_images": list(sources.values()),
                      "verified": results, "all_embedded_byte_identical": True}, indent=2))

if __name__ == "__main__":
    main()
