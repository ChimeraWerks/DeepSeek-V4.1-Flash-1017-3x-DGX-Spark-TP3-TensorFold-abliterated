"""Opt-in three-rank native capacity settings and fail-closed launch preflight."""
from __future__ import annotations

import math
from pathlib import PurePosixPath


_KEYS = {"kv_shard", "kv_decode", "kv_ring", "spill_gib", "spill_paths", "spill_floor_gib",
         "spill_min_tokens", "spill_staging_mib", "spill_turns", "spill_flush_ms", "identity", "pool_tokens"}
_IDENTITY = {"TF_DS_ENGINE_ID", "TF_DS_RECIPE_PATCHES", "TF_DS_BUILD_FLAGS",
             "TF_DS_MODEL_REVISION", "TF_DS_ENGRAM_REVISION", "TF_DS_ABLATION_ID"}


def managed_capacity(key):
    return key.startswith(("TF_DS_SPILL_", "TF_DS_KV_SHARD")) or key in {"TF_SPILL_TURNS", "TF_SPILL_FLUSH_MS"}


def plan(cluster, ranks, *, profile_defaults=None):
    raw = cluster.get("capacity", {})
    if not isinstance(raw, dict) or set(raw) - _KEYS:
        raise ValueError("unknown native capacity settings")
    shard = raw.get("kv_shard", False)
    if not isinstance(shard, bool):
        raise ValueError("kv_shard must be a boolean")
    gib = raw.get("spill_gib", 0)
    if isinstance(gib, bool) or not isinstance(gib, (int, float)) or not math.isfinite(gib) or gib < 0:
        raise ValueError("spill_gib must be finite and nonnegative")
    ring = raw.get("kv_ring", True)
    if not isinstance(ring, bool):
        raise ValueError("kv_ring must be a boolean")
    enabled = shard or gib > 0
    # Only the bundled profile may supply default spill. Compare the unmodified
    # launcher output, not cluster.environment, so free-form overrides still fail.
    for index, item in enumerate(ranks):
        baseline = profile_defaults[index] if profile_defaults is not None else {}
        expected = {k: v for k, v in baseline.items() if managed_capacity(k)}
        actual = {k: v for k, v in item["environment"].items() if managed_capacity(k)}
        if actual != expected:
            raise ValueError("configure spill and KV sharding through cluster.capacity")
        if not expected:
            continue
        if "capacity" in cluster:
            # Explicit capacity controls replace, never duplicate, profile flags.
            for key in expected:
                item["environment"].pop(key)
            item["argv"][:] = [arg for arg in item["argv"]
                               if not arg.startswith(("--spill-dir=", "--spill-gib="))]
        else:
            # Planning stays read-only. The existing execution preflight below
            # still requires the real build receipt, rank hash and private disk.
            item["capacity"] = {"spill": True, "kv_shard": False, "floor_gib": 50,
                                "flush_ms": 60000, "spill_path": baseline["TF_DS_SPILL_DIR"]}
    if not enabled:
        if "pool_tokens" in raw:
            raise ValueError("pool_tokens requires an enabled capacity feature")
        return ranks
    if len(ranks) != 3:
        raise ValueError("context capacity requires exactly three ranks")
    if raw.get("kv_decode", "fetch") != "fetch":
        raise ValueError("TP3 KV sharding supports fetch decode only")
    settings = {}
    for name, default, minimum in [("spill_floor_gib", 50, 1), ("spill_min_tokens", 8192, 1),
        ("spill_staging_mib", 128, 16), ("spill_turns", 4, 1), ("spill_flush_ms", 60000, 1)]:
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        settings[name] = value
    if settings["spill_staging_mib"] > 1024 or settings["spill_flush_ms"] > 3600000:
        raise ValueError("spill staging or flush deadline exceeds recipe bounds")
    identity = raw.get("identity", {})
    if gib and (not isinstance(identity, dict) or set(identity) - _IDENTITY or not {
            "TF_DS_ENGINE_ID", "TF_DS_RECIPE_PATCHES", "TF_DS_MODEL_REVISION", "TF_DS_ENGRAM_REVISION"} <= identity.keys()):
        raise ValueError("spill requires explicit engine, patch and model identities")
    if any(not isinstance(v, str) or not v or "\0" in v for v in identity.values()):
        raise ValueError("native capacity identities must be nonempty strings")
    paths = raw.get("spill_paths", [])
    if gib and (not isinstance(paths, list) or len(paths) != 3 or any(
            not isinstance(p, str) or not PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts for p in paths)):
        raise ValueError("spill_paths must name one absolute local directory per rank")
    head = ranks[0]["argv"]
    context = int(head[head.index("--context") + 1])
    if gib and "--prompt-cache-gib" in head and float(head[head.index("--prompt-cache-gib") + 1]) <= 0:
        raise ValueError("spill requires a positive head prompt-cache budget")
    for index, item in enumerate(ranks):
        env = item["environment"]
        argv = item["argv"]
        if "pool_tokens" in raw:
            pool_override = raw["pool_tokens"]
            if isinstance(pool_override, bool) or not isinstance(pool_override, int) or pool_override <= 0:
                raise ValueError("pool_tokens must be a positive integer")
            env["TF_DS_POOL_TOKENS"] = str(pool_override)
            if "--pool" in argv:
                argv[argv.index("--pool") + 1] = str(pool_override)
        if shard:
            env["TF_DS_WINDOW"] = str(context)
            pool = int(env.get("TF_DS_POOL_TOKENS", argv[argv.index("--pool") + 1] if "--pool" in argv else 0))
            window = int(env.get("TF_DS_WINDOW", argv[argv.index("--context") + 1] if "--context" in argv else 0))
            rows = int(env.get("TF_DS_ROUND_ROWS", "48"))
            if not pool or pool % (3 * 2048) or not window or pool // 3 < ((window + rows + 2 + 2047) // 2048) * 2048:
                raise ValueError("KV pool must divide into three whole-block parts, each holding context + round + 2")
            env.update(TF_DS_KV_SHARD="1", TF_DS_KV_SHARD_DECODE="fetch", TF_DS_KV_SHARD_RING="1" if ring else "0")
        item["capacity"] = {"spill": gib > 0, "kv_shard": shard, "floor_gib": settings["spill_floor_gib"], "flush_ms": settings["spill_flush_ms"]}
        receipt = item["node"].get("capacity_build_receipt")
        if not isinstance(receipt, str) or not PurePosixPath(receipt).is_absolute():
            raise ValueError("enabled native capacity requires an absolute capacity_build_receipt on every Spark")
        if gib:
            env.update(identity)
            env.update(TF_DS_SPILL_STAGING_MIB=str(settings["spill_staging_mib"]),
                       TF_SPILL_TURNS=str(settings["spill_turns"]), TF_SPILL_FLUSH_MS=str(settings["spill_flush_ms"]))
            # Each rank weight cache SHA is part of persistent compatibility.
            rank_sha = item["node"].get("rank_cache_sha256", "")
            if len(rank_sha) != 64 or any(c not in "0123456789abcdef" for c in rank_sha):
                raise ValueError("spill requires the verified rank_cache_sha256 of every Spark")
            env["TF_DS_RANK_CACHE_SHA"] = rank_sha
            item["capacity"]["spill_path"] = paths[index]
            if index == 0:
                argv += ["--spill-gib", str(gib), "--spill-dir", paths[index],
                    "--spill-min-free-gib", str(settings["spill_floor_gib"]),
                    "--spill-min-tokens", str(settings["spill_min_tokens"])]
            else:
                env.update(TF_DS_SPILL_GIB=str(gib), TF_DS_SPILL_DIR=paths[index],
                    TF_DS_SPILL_MIN_FREE_GIB=str(settings["spill_floor_gib"]),
                    TF_DS_SPILL_MIN_TOKENS=str(settings["spill_min_tokens"]))
    return ranks


# Runs inside the existing read-before-start remote preflight, before service changes.
REMOTE_CHECK = r'''
cap=x.get("capacity")
if cap:
 receipt_path=n.get("capacity_build_receipt")
 if not isinstance(receipt_path,str) or not pathlib.Path(receipt_path).is_absolute():raise ValueError("enabled native capacity requires an absolute capacity_build_receipt on every Spark")
 receipt=json.loads(pathlib.Path(receipt_path).read_text())
 required=({"spill-v2-tp3"} if cap["spill"] else set())|({"kv-shard-tp3-fetch"} if cap["kv_shard"] else set())
 if not required<=set(receipt.get("capabilities",[])):raise ValueError("native build lacks requested capacity capability")
 binary=receipt.get("binaries",{}).get(pathlib.Path(x["argv"][0]).name,{})
 if binary.get("sha256")!=sha(pathlib.Path(x["argv"][0])):raise ValueError("capacity receipt does not identify the executable")
 if cap["spill"]:
  os.nice(10)
  subprocess.run(["ionice","-c3","-p",str(os.getpid())],check=True,capture_output=True)
  cache=pathlib.Path(x["environment"]["TF_DS_RANK_CACHE"])
  matches=[p for p in cache.glob("rank"+str(n["rank"])+"of3-*.bin") if p.is_file() and not p.is_symlink()]
  if len(matches)!=1 or sha(matches[0])!=n.get("rank_cache_sha256"):raise ValueError("spill rank-cache identity does not match the actual weight file")
  directory=pathlib.Path(cap["spill_path"])
  if any(p.is_symlink() for p in [directory,*directory.parents]):raise ValueError("spill directory cannot traverse symlinks")
  resolved=directory.resolve()
  for protected in [root/"weights",root/"stock",root/"ablit",*map(pathlib.Path,n.get("spill_forbidden_paths",[]))]:
   if resolved==protected.resolve() or protected.resolve() in resolved.parents:raise ValueError("spill directory overlaps protected assets")
  directory.mkdir(parents=True,exist_ok=True,mode=0o700)
  if directory.stat().st_uid!=os.getuid() or directory.stat().st_mode&0o077:raise ValueError("spill directory must be private and owned by the runtime user")
  mounts=[]
  for line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines():
   before,after=line.split(" - ",1);parts=before.split();mount=pathlib.Path(parts[4].replace("\\040"," "))
   if mount==resolved or mount in resolved.parents:mounts.append((len(mount.parts),after.split()[0]))
  if not mounts or max(mounts)[1] not in {"ext4","xfs","btrfs","f2fs"}:raise ValueError("spill requires a local disk filesystem")
  fs=os.statvfs(directory)
  if fs.f_bavail*fs.f_frsize<cap["floor_gib"]*1024**3:raise ValueError("spill free-space floor is unavailable")
  probe=directory/(".preflight-"+uuid.uuid4().hex)
  try:
   with probe.open("xb") as stream:stream.write(b"check");stream.flush();os.fsync(stream.fileno())
  finally:probe.unlink(missing_ok=True)
'''
