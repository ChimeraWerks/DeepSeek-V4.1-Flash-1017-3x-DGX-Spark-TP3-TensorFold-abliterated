"""Writable overlay metadata without changing read-only Hugging Face blobs.

Adapted from Robert Wojciechowski's Mia ABLIT index fix, commit 534c96b.
"""
import json
import os
from pathlib import Path
import uuid


def writable_index(snapshot, workdir):
    snapshot, workdir = Path(snapshot).resolve(), Path(workdir).resolve()
    if snapshot == workdir:
        raise ValueError("overlay work directory must differ from the snapshot")
    source = snapshot / "model.safetensors.index.json"
    if not source.is_file():
        raise ValueError("snapshot checkpoint index is missing")
    workdir.mkdir(parents=True, exist_ok=True)
    destination = workdir / source.name
    if destination.exists() and not destination.is_symlink():
        if not os.path.samefile(source, destination) and destination.stat().st_mode & 0o200:
            # An earlier overlay may already have edited its independent index.
            return destination
    selected = destination if destination.exists() else source
    with selected.open("rb") as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024 or not isinstance(json.loads(raw), dict):
        raise ValueError("invalid or oversized checkpoint index")
    temporary = workdir / (source.name + ".copy-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(workdir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    if os.path.samefile(source, destination) or not destination.stat().st_mode & 0o200:
        raise ValueError("overlay index is not an independent writable copy")
    return destination
