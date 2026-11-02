"""Read-only Engram input identity matching native rank_identity.gather/listing.

This is deliberately not the capacity filesystem probe or a replacement for the
native persistent fingerprint. Only immediate regular entries are listed; shard
contents and SOURCE.json claims are not used as substitutes for effective files.
"""
import hashlib
import json
import os
from pathlib import Path
import stat


def binding(item):
    payload = {key: item[key] for key in ("argv", "environment", "node")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def option(argv, name, fallback):
    # Native follower CLI options override environment defaults.
    values = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == name]
    return values[-1] if values else fallback


def collect(item):
    env, argv = item["environment"], item["argv"]
    rank = item["node"]["rank"]
    if rank not in (0, 1, 2):
        raise ValueError("invalid Engram rank")
    directory = env.get("TF_DS_ENGRAM")
    kit = env.get("TF_DS_KIT")
    if rank:
        directory = option(argv, "--engram", directory)
        # Follower positional contract: binary, model, cache, rank, world,
        # master, port, kit; this must match the command being launched.
        if len(argv) < 8 or argv[3:5] != [str(rank), "3"]:
            raise ValueError("invalid follower identity command")
        kit = argv[7]
    if not directory:
        identity = b"off"
        revision, kit_hash, listing = "", "", b""
    else:
        revision = env.get("TF_DS_ENGRAM_REVISION", "")
        # A profile's recorded source revision is not an effective native env
        # value. An empty revision is valid for nonpersistent native launches.
        if not revision and item.get("capacity", {}).get("spill"):
            raise ValueError("persistent Engram requires an effective revision")
        if not kit or not Path(directory).is_absolute() or not Path(kit).is_absolute():
            raise ValueError("Engram and kit paths must be explicit and absolute")
        lines = []
        with os.scandir(os.fsencode(directory)) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    lines.append(entry.name + b":" + str(info.st_size).encode("ascii") + b";")
        listing = b"".join(sorted(lines))
        path = Path(kit) / "engram.json"
        # Native exists/fileDigest: absent metadata uses the all-zero digest.
        kit_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "0" * 64
        identity = revision.encode() + b"|" + kit_hash.encode("ascii") + b"|" + listing
    return {"rank": rank, "candidate": binding(item), "engram_revision": revision,
            "kit_engram_sha256": kit_hash, "listing_hex": listing.hex(),
            "identity_sha256": hashlib.sha256(identity).hexdigest()}


def compatible(items, rows):
    if len(items) != 3 or len(rows) != 3 or [x["node"]["rank"] for x in items] != [0, 1, 2]:
        raise ValueError("Engram compatibility requires three ordered candidate ranks")
    for rank, (item, row) in enumerate(zip(items, rows)):
        if row.get("rank") != rank or row.get("candidate") != binding(item):
            raise ValueError("Engram evidence is not bound to this candidate rank")
        revision, kit_hash, listing = row["engram_revision"], row["kit_engram_sha256"], bytes.fromhex(row["listing_hex"])
        identity = b"off" if kit_hash == "" else revision.encode() + b"|" + kit_hash.encode("ascii") + b"|" + listing
        if hashlib.sha256(identity).hexdigest() != row["identity_sha256"]:
            raise ValueError("invalid Engram identity evidence")
    if len({row["identity_sha256"] for row in rows}) != 1:
        raise ValueError("TP3 effective Engram identities differ; candidate replacement refused")
    return rows
