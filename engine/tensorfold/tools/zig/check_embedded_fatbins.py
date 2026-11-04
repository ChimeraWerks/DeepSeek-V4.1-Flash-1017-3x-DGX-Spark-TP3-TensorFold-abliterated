#!/usr/bin/env python3
"""Read-only proof that every embedded CUDA image is an exact prebuilt input."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

def sha(data):
    return hashlib.sha256(data).hexdigest()

def verify(directory, binaries):
    refs = {sha(p.read_bytes()): (p.name, p.read_bytes())
            for p in sorted(directory.glob("*.fatbin"))}
    if not refs:
        raise ValueError("No reference fatbins")
    reports = []
    for binary in binaries:
        data = binary.read_bytes()
        images = []
        offset = 0
        while True:
            offset = data.find(b"\x50\xed\x55\xba", offset)
            if offset < 0:
                break
            if offset + 16 > len(data):
                raise ValueError("Truncated fatbin header")
            magic, version, header, size = struct.unpack_from("<IHHQ", data, offset)
            if version != 1 or header != 16:
                offset += 4
                continue  # An unrelated instruction/debug byte sequence.
            end = offset + header + size
            if end > len(data):
                raise ValueError("Truncated embedded fatbin")
            image = data[offset:end]
            digest = sha(image)
            if digest not in refs or image != refs[digest][1]:
                raise ValueError(f"Unrecognized or changed embedded fatbin: {binary}:{offset} {digest}")
            images.append({"name": refs[digest][0], "offset": offset, "bytes": len(image), "sha256": digest})
            offset = end
        # Actual inference binaries must retain the pinned head/cast and sampling modules.
        if binary.name in ("tensorfold-native", "tf-dsv41-lanes"):
            names = {x["name"] for x in images}
            if not {"dsv41_ops.fatbin", "dsv41_sample_cand.fatbin", "torch_pointwise.fatbin"} <= names:
                raise ValueError("Missing required embedded inference fatbin")
        reports.append({"binary": str(binary), "sha256": sha(data), "images": images})
    return {"reference_sha256": {name: digest for digest, (name, _) in refs.items()},
            "binaries": reports, "pass": True}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("binaries", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.binaries), indent=2))
