#!/usr/bin/env python3
"""Read-only release/default/launch consistency check. No SSH, services or writes."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
from string import Template
import subprocess
import sys
import tarfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path("serving/src/chimera_sparks")
ACTIVE_RELEASE = "v1.2"
ACTIVE_ENGINE = "678ea1b"
ACTIVE_RECIPE_PATCHES = "chimera-tensorfold-patches-20261103"
# Frozen releases: each keeps a manifest in release/history and a bundled native profile.
HISTORY = {"v1": "7d5b4d8", "v1.1": "14803a3"}
RUNTIME_FIELDS = {"streams": "STREAMS", "pool": "POOL_TOKENS", "context": "CONTEXT_TOKENS",
                  "graph_cap": "GRAPH_CAP", "spill_gib": "SPILL_GIB",
                  "memory_reserve_gib": "MEMORY_RESERVE_GIB", "top_k": "TOP_K", "commit": "ENGINE_COMMIT"}
# Split obsolete literals so the scanner also checks its own source, without a
# blanket exemption for tooling or regression fixtures.
STALE_MARKERS = [
    re.compile("TF" + "351", re.I), re.compile("full" + "40", re.I),
    re.compile(r'parallel"\s*:\s*12\b'),
    re.compile(r"\b" + "700" + "0000" + r"\b"),
    re.compile(r"\b" + "401" + "408" + r"\b"),
    re.compile(r"\b12\s+streams\b", re.I),
    re.compile(r"\bC12\s+production\b", re.I),
    re.compile(r"\b12-stream", re.I),
    re.compile(r"\b7," + "000,000" + r"\b"),
    re.compile(r"\b401," + "408" + r"\b"),
    re.compile(r"\bbin-(?:" + "6f7a4b8|56efca0|cebcf1e|93d0b4c|1\\.2\\.5" + r")"),
    re.compile(r"\b12\.5\s*GiB", re.I),
]


def module_at(path, label):
    name = f"_release_check_{label}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves its defining module here.
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def default_native_profile(root):
    tree = ast.parse((root / PACKAGE / "lifecycle.py").read_text())
    rank_plan = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rank_plan")
    for node in ast.walk(rank_plan):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "cluster"
                and node.func.attr == "get" and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "profile"):
            return ast.literal_eval(node.args[1])
    raise ValueError("cannot determine lifecycle's default native profile")


def example_settings(path):
    """Parse literal shell assignments without sourcing or executing the file."""
    result = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line.strip())
        if match:
            key, value = match.groups()
            if key in result:
                raise ValueError("duplicate cluster.env.example setting: " + key)
            words = shlex.split(value, comments=True)
            result[key] = words[0] if len(words) == 1 else "" if not words else None
    return result


def release_defaults(root):
    errors = []
    manifest_path = "release/manifest.json"
    manifest = json.loads((root / manifest_path).read_text())
    engine, serving, weights = (manifest[key] for key in ("engine", "serving", "weights"))

    def same(label, actual, expected):
        if actual != expected:
            errors.append(label + " disagrees with " + manifest_path)

    profile_name = default_native_profile(root)
    same("default native profile", profile_name, serving["native_profile"])
    same("release name", manifest["release"], serving["native_profile"])
    same("active release", manifest["release"], ACTIVE_RELEASE)
    profiles = root / PACKAGE / "assets/profiles"
    same("bundled native profiles", {p.stem for p in profiles.glob("*.json")}, {ACTIVE_RELEASE, *HISTORY})
    profile = json.loads((profiles / (profile_name + ".json")).read_text())
    for field, key in {"streams": "parallel", "pool": "pool_tokens", "context": "context_tokens",
                       "graph_cap": "graph_cap", "spill_gib": "spill_gib",
                       "memory_reserve_gib": "memory_reserve_gib", "top_k": "top_k", "commit": "engine_commit"}.items():
        same("native profile " + key, profile.get(key), engine[field])
    for field, key in {"repo": "hub_repo", "revision": "hub_revision", "model_revision": "source_revision",
                       "arm": "arm", "bits_per_weight": "bits_per_weight"}.items():
        same("native profile " + key, profile.get(key), weights[field])
    same("native profile version", profile["version"], serving["native_profile"])
    same("manifest engine identity", manifest["labels"]["TF_DS_ENGINE_ID"],
         "chimera-tensorfold-tp3-" + engine["commit"])
    same("recipe patches identity", manifest["labels"]["TF_DS_RECIPE_PATCHES"], ACTIVE_RECIPE_PATCHES)
    same("model label", manifest["labels"]["TF_DS_MODEL_REVISION"], weights["model_revision"] + "-chimera-dsv41-flash-1017-abliterated-exl3")
    same("artifact arm", profile.get("artifact_arm"), weights.get("artifact_arm"))
    same("artifact arm production identity", weights.get("artifact_arm"), "V4-MRF-1017")
    same("rank cache layout", profile.get("rank_cache_subdir"), "ranks")
    same("source engine", engine["commit"], ACTIVE_ENGINE)
    same("production pool", engine["pool"], 8000000)
    config = module_at(root / PACKAGE / "config.py", "config").Config()
    for field, expected in {"context_tokens": engine["context"], "engine_streams": engine["streams"],
                            "max_concurrency": serving["max_concurrency"]}.items():
        same("serving Config." + field, getattr(config, field), expected)
        same("manifest serving." + field, serving[field], expected)
    same("manifest concurrency/streams", serving["max_concurrency"], engine["streams"])
    same("serving Config.sampling", config.sampling, manifest["sampler"]["defaults"])
    same("manifest sampler top_k", manifest["sampler"]["defaults"]["top_k"], engine["top_k"])
    same("serving effort map", config.effort_map, manifest["sampler"]["effort_map"])
    same("serving reasoning default", config.reasoning_default, manifest["sampler"]["reasoning_default"])
    same("serving package version", tomllib.loads((root / "serving/pyproject.toml").read_text())["project"]["version"], serving["version"])

    example = example_settings(root / "cluster.env.example")
    for field, key in RUNTIME_FIELDS.items():
        same("cluster.env.example " + key, example.get(key), str(engine[field]))
    same("cluster.env.example HF_REPO", example.get("HF_REPO"), weights["repo"])
    same("cluster.env.example HF_REVISION", example.get("HF_REVISION"), weights["revision"] or "REPLACE_WITH_WEIGHTS_EXPORT_COMMIT")
    scope = json.loads((root / "serving/engine-scope.json").read_text())
    same("engine-scope profile", scope["required_native_profile"], ACTIVE_RELEASE)
    same("engine-scope engine", scope["engine_commit"], engine["commit"])
    same("engine-scope serving", scope["serving_version"], serving["version"])
    for field in RUNTIME_FIELDS.keys() - {"commit"}:
        same("engine-scope " + field, scope["production_defaults"][field], engine[field])

    delta = engine["source_delta"]
    same("source delta base", delta["base_commit"], engine["commit"])
    same("source delta file", sha256(root / delta["path"]), delta["sha256"])
    build = json.loads((root / "release/build-verification.json").read_text())
    same("build verification release", build["release"], manifest["release"])
    same("build verification engine", build["engine_commit"], engine["commit"])
    same("build verification source delta", build["source_delta"], delta)
    errors.extend(launcher_errors(root, manifest, profile_name, manifest_path))
    return errors


def launcher_errors(root, manifest, profile_name, manifest_path):
    """Exercise the actual launcher, not just the JSON. All topology is synthetic;
    the complete manifest env and argv must agree on all three ranks."""
    errors = []
    engine, serving = manifest["engine"], manifest["serving"]

    def same(label, actual, expected):
        if actual != expected:
            errors.append(label + " disagrees with " + manifest_path)

    launcher = module_at(root / PACKAGE / "assets/native_launch.py", "launcher")
    devices = ["rdma0", "rdma1", "rdma2", "rdma3"]
    masks = [[0, 12, 3], [3, 0, 12], [12, 3, 0]]
    base = "/srv/release-check"
    config = {"ranks": [{"rank": rank, "root": base, "rdma_devices": devices,
                          "peer_masks": masks[rank], "gid_index": 3} for rank in range(3)],
              "master_address": "192.0.2.10", "master_port": 29660, "nccl_interface": "fabric0",
              "nccl_hcas": devices, "nccl_address_range": "192.0.2.0/24",
              "api_host": "127.0.0.1", "api_port": 8003, "library_path": "/opt/nccl/lib"}
    substitutions = {"SPARK_LD_LIBRARY_PATH": config["library_path"], "NCCL_IB_ADDR_RANGE": config["nccl_address_range"],
                     "NCCL_IB_HCA": ",".join(devices), "NCCL_SOCKET_IFNAME": "=fabric0", "TF_RDMA_DEVICES": ",".join(devices),
                     "TF_RDMA_GID_INDEX": "3", "MASTER_ADDR": "192.0.2.10", "TP_PORT": "29660",
                     "KIT_DIR": base + "/kit", "ENGRAM_DIR": base + "/weights/engram-tp3",
                     "RANK_CACHE": base + "/ranks", "SPILL_DIR": base + "/spill",
                     **{key: str(engine[field]) for field, key in RUNTIME_FIELDS.items()}}
    nap = os.environ.pop("CHIMERA_RDMA_NAP", None)
    try:
        for rank in range(3):
            argv, env = launcher.command(config, profile_name, rank)
            expected = {key: Template(value).substitute(substitutions) for key, value in engine["environment"].items()}
            if rank == 0:
                expected.update({key: Template(value).substitute(substitutions) for key, value in engine["head_environment"].items()})
            expected.update(manifest["labels"])
            expected["TF_RDMA_PEER_MASKS"] = ",".join(map(str, masks[rank]))
            same(f"native rank {rank} environment", env, expected)
            if rank == 0:
                expected_argv = [base + "/bin/tensorfold-native", "serve", base + "/weights", "--backend", "cuda",
                                 "--host", "127.0.0.1", "--port", "8003", "--name", serving["model"],
                                 "--context", str(engine["context"]), "--parallel", str(engine["streams"]),
                                 *[Template(arg).substitute(substitutions) for arg in engine["head_args"]]]
            else:
                expected_argv = [base + "/bin/tf-dsv41-lanes", base + "/weights", base + "/ranks", str(rank), "3",
                                 "192.0.2.10", "29660", base + "/kit", "--pool", str(engine["pool"]),
                                 "--round-rows", "48", "--streams", str(engine["streams"]), "--drafts", "1",
                                 "--graphs", "1", "--side", "1", "--prefetch", "1", "--aio", "1",
                                 "--graph-cap", str(engine["graph_cap"]), "--engram", base + "/weights/engram-tp3",
                                 "--token-map", base + "/kit/engram-token-map.json", "--rdma", ",".join(devices)]
            same(f"native rank {rank} argv", argv, expected_argv)
    finally:
        if nap is not None:
            os.environ["CHIMERA_RDMA_NAP"] = nap
    return errors


def release_history(root):
    """Frozen releases keep their manifest, their commit and a launcher that still reproduces them."""
    errors = []
    for name, commit in HISTORY.items():
        path = "release/history/" + name + ".json"
        manifest = json.loads((root / path).read_text())
        if (manifest["release"], manifest["serving"]["native_profile"], manifest["engine"]["commit"]) != (name, name, commit):
            errors.append(path + " does not describe " + name + " at " + commit)
        errors.extend(launcher_errors(root, manifest, name, path))
    return errors


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefill_and_assets(root):
    """Declared kernel-source, payload and runtime-asset hashes against the committed files."""
    errors = []
    prefill = root / "engine/tensorfold/zig/kernels/prefill_128"
    source = json.loads((prefill / "SOURCE.json").read_text())
    manifest = json.loads((prefill / "MANIFEST.json").read_text())
    delta = json.loads((root / "release/prefill-source-delta.json").read_text())
    if sha256(prefill / source["attention"]["published_source"]) != source["attention"]["published_file_sha256"]:
        errors.append("prefill_128 kernels.py differs from SOURCE.json")
    moe = source["prompt_moe"]
    if moe["published_file_sha256"] != delta["declared_sha256"]:
        errors.append("release/prefill-source-delta.json declared hash differs from SOURCE.json")
    if sha256(root / delta["path"]) != delta["committed_sha256"]:
        errors.append(delta["path"] + " differs from release/prefill-source-delta.json")
    if sha256(prefill / manifest["archive"]) != manifest["archive_sha256"]:
        errors.append("prefill payload archive differs from MANIFEST.json")
    with tarfile.open(prefill / manifest["archive"]) as tar:
        members = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers() if m.isfile()}
    if members != manifest["payload_files"]:
        errors.append("prefill payload members differ from MANIFEST.json payload_files")
    if moe["measured_cubin_sha256"] not in {members.get("auxiliary/experts_ragged.cubin")} & {delta["served_cubin"]["sha256"]}:
        errors.append("served experts_ragged.cubin differs from SOURCE.json measured_cubin_sha256")
    pins = json.loads((root / "release/runtime-assets.json").read_text())
    manifest_weights = json.loads((root / "release/manifest.json").read_text())["weights"]
    if pins["kit"]["repo"] != manifest_weights["repo"] or not re.fullmatch(r"[0-9a-f]{40}", pins["kit"]["revision"]):
        errors.append("release/runtime-assets.json kit must pin a commit of the weights repository")
    if pins["engram"]["revision"] != manifest_weights["model_revision"]:
        errors.append("release/runtime-assets.json Engram revision differs from the model revision")
    staged = {item["path"]: item["sha256"] for item in pins["kit"]["staged_files"]}
    if staged.get("rope.json") != pins["kit"]["rope"]["rope_json_sha256"]:
        errors.append("release/runtime-assets.json RoPE pins disagree")
    lengths = set()
    for rank, pin in enumerate(pins["engram"]["ranks"]):
        raw = (json.dumps(pin["source_json"], indent=2, sort_keys=True) + "\n").encode()
        serving = (json.dumps(pin["serving_source"], sort_keys=True, separators=(",", ":")) + "\n").encode()
        lengths.add(len(serving))
        if (hashlib.sha256(raw).hexdigest() != pin["source_json_sha256"] or pin["source_json"]["rank"] != rank
                or hashlib.sha256(serving).hexdigest() != pin["serving_source_sha256"] or pin["serving_source"]["rank"] != rank):
            errors.append(f"release/runtime-assets.json Engram rank {rank} pin is inconsistent")
    if len(lengths) != 1:
        errors.append("release/runtime-assets.json serving SOURCE.json lengths differ across ranks")
    return errors


def tracked_text(root):
    # Tracked plus untracked, not ignored: a new document must not escape the check before it is staged.
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                            check=True, capture_output=True)
    paths = {os.fsdecode(name) for name in result.stdout.split(b"\0") if name}
    for name in sorted(paths):
        if name.startswith(("engine/tensorfold/", "serving/src/chimera_sparks/laky/")):
            continue
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            yield name, data.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            continue


def stale_lines(root):
    errors = []
    for name, lines in tracked_text(root):
        history = False
        for number, line in enumerate(lines, 1):
            if name == "CHANGELOG.md" and line.startswith("## "):
                history = bool(re.search(r"\b(?:history|historical)\b", line, re.I))
            historical_line = name == "CHANGELOG.md" and (history or re.search(r"\b(?:historically|previously|formerly)\b", line, re.I))
            for marker in STALE_MARKERS:
                match = marker.search(line)
                if not match or historical_line:
                    continue
                concurrency = (re.fullmatch(r"12\s+streams", match.group(), re.I)
                               and (("benchmark" in line.lower() and "concurrency" in line.lower())
                                    or (name.startswith("benchmarks/") and line.startswith("|")
                                        and re.search(r"\b1 stream\b", line))))
                if not concurrency:
                    errors.append(f"{name}:{number}: stale release marker {match.group()!r}")
    return errors


def check(root=ROOT):
    root = Path(root).resolve()
    errors = []
    try:
        errors.extend(release_defaults(root))
    except (OSError, ValueError, KeyError, StopIteration, SyntaxError, TypeError, AttributeError) as error:
        errors.append("release defaults could not be checked: " + str(error))
    try:
        errors.extend(release_history(root))
    except (OSError, ValueError, KeyError, SyntaxError, TypeError, AttributeError) as error:
        errors.append("release history could not be checked: " + str(error))
    try:
        errors.extend(prefill_and_assets(root))
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append("prefill and runtime-asset hashes could not be checked: " + str(error))
    try:
        errors.extend(stale_lines(root))
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append("tracked text could not be checked: " + str(error))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    errors = check(args.root)
    if errors:
        for error in errors:
            print("FAIL: " + error, file=sys.stderr)
        return 1
    print("Release consistency passed: active v1.2 and historical v1 and v1.1, serving defaults, cluster example, manifest, "
          "prefill and runtime-asset hashes, and tracked text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
