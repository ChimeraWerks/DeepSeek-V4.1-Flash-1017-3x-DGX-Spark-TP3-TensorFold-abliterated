#!/usr/bin/env python3
"""Launch three TensorFold ranks, then arm their invocation-bound memory guards.

Environment (source cluster.env with `set -a` before invoking this file directly):
  Required: SPARK_HOSTS (three SSH hosts, head first), SPARK_USER, SSH_KEY,
            MASTER_ADDR (RoCE rendezvous address), NCCL_IB_ADDR_RANGE, SPARK_ROOT.
  HF_REPO/HF_REVISION identify the export; ENGINE_PORT=8003, API_PORT=18002.
  ENGRAM_SOURCE=ranged|full picks how prepare fetches the Engram tables (docs/runtime-assets.md).
  SPARK_HOME defaults to /home/$SPARK_USER; remote ~/ paths expand against it,
  never against the launcher's home. SSH_KEY is local; HEAD_SSH_KEY defaults to
  ~/.ssh/id_ed25519 on the head for the head-to-worker check.
  Remote path overrides (otherwise under SPARK_ROOT): BIN_DIR=bin,
  MODEL_DIR=weights, RANK_CACHE=ranks, KIT_DIR=kit, ENGRAM_DIR=weights/engram-tp3,
  SPILL_DIR=spill, GUARD_SCRIPT=guard/memguard.py, LOGGER_SCRIPT=service/rotating_run.py.
  SPARK_PYTHON=python3; ENGINE_HOST/API_HOST=127.0.0.1;
  TP_PORT=29660; NCCL_IB_HCA/TF_RDMA_DEVICES and NCCL_SOCKET_IFNAME select links;
  TF_RDMA_GID_INDEX=3; RDMA_PEER_MASKS='0,12,3 3,0,12 12,3,0'.
  SPARK_LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu (add a
  custom NCCL library directory here if needed). ENGINE_LABEL=ds1017.
  Local receipt: DEPLOY_RECEIPT=./.state/deploy-execution.json.
  ENGINE_HEALTH_URL optionally names a local SSH tunnel; otherwise health is
  checked on the head over SSH. OPTIONAL_CONTROLLER_URL is optional: when unset there is
  no dependency on optional affinity controller and no affinity change. SPARK_API_KEY is never
  consumed, forwarded, printed or saved by this engine launcher.

Rank placement belongs to scripts/prepare.sh, not this launcher. Exactly one
rank file must already be present on each Spark. Guards are armed only after warm
health with numeric_effort and ready-<InvocationID> markers. --dry-run performs
no network or writes and prints the exact rank commands and guard commands.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
from string import Template
import subprocess
import sys
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "release/manifest.json"
GUARD_UNIT = re.compile(r"[a-z][a-z0-9-]{0,63}-g[0-9a-f]{32}(?:\.service)?")
CREDENTIAL_NAME = re.compile(r"(?:^|_)(?:TOKEN|PASSWORD|SECRET|API_KEY|ADMIN_KEY|ACCESS_TOKEN|AUTH_TOKEN)$")
SPARKS = {}
SETTINGS = {}
V = {}


def cluster_settings(env=None):
    """Resolve public topology without contacting a host or reading credentials."""
    env = os.environ if env is None else env
    required = ("SPARK_HOSTS", "SPARK_USER", "SSH_KEY", "MASTER_ADDR", "NCCL_IB_ADDR_RANGE", "SPARK_ROOT")
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError("set cluster.env keys: " + ", ".join(missing))
    result = {key: env[key] for key in required}
    hosts = result["SPARK_HOSTS"].split()
    if len(hosts) != 3 or len(set(hosts)) != 3 or any(not re.fullmatch(r"[a-zA-Z0-9_.:-]+", h) or h.startswith("-") for h in hosts):
        raise ValueError("SPARK_HOSTS must contain three distinct SSH hosts, head first")
    if not re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_-]*", result["SPARK_USER"]):
        raise ValueError("invalid SPARK_USER")
    home = env.get("SPARK_HOME", f"/home/{result['SPARK_USER']}")
    if not home.startswith("/"):
        raise ValueError("SPARK_HOME must be absolute on the Sparks")
    result["SPARK_HOME"] = home

    def remote_path(value):
        if value == "~":
            value = home
        elif value.startswith("~/"):
            value = home + value[1:]
        if not value.startswith("/") or "\0" in value or "\n" in value:
            raise ValueError("remote paths must be absolute or start with ~/")
        return value.rstrip("/") or "/"

    root = result["SPARK_ROOT"] = remote_path(result["SPARK_ROOT"])
    for key, suffix in {"BIN_DIR": "bin", "MODEL_DIR": "weights", "RANK_CACHE": "ranks", "KIT_DIR": "kit",
                        "ENGRAM_DIR": "weights/engram-tp3", "SPILL_DIR": "spill", "GUARD_SCRIPT": "guard/memguard.py",
                        "LOGGER_SCRIPT": "service/rotating_run.py",
                        "API_CONFIG": "api/config.json", "API_ENV_FILE": "api/credentials.env"}.items():
        result[key] = remote_path(env.get(key, root + "/" + suffix))
    defaults = {
        "ENGINE_PORT": "8003", "API_PORT": "18002", "TP_PORT": "29660", "ENGINE_HOST": "127.0.0.1", "API_HOST": "127.0.0.1",
        "SPARK_PYTHON": "python3", "HEAD_SSH_KEY": "~/.ssh/id_ed25519", "ENGINE_LABEL": "ds1017", "API_INSTALLATION": "ds1017",
        "NCCL_IB_HCA": "rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1", "NCCL_SOCKET_IFNAME": "=enP7s7",
        "TF_RDMA_GID_INDEX": "3", "RDMA_PEER_MASKS": "0,12,3 3,0,12 12,3,0",
        "SPARK_LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu",
        "OPTIONAL_CONTROLLER_URL": "", "ENGINE_HEALTH_URL": "", "HF_REPO": "", "HF_REVISION": "",
        "ENGRAM_SOURCE": "ranged",
    }
    result.update({key: env.get(key, default) for key, default in defaults.items()})
    result["TF_RDMA_DEVICES"] = env.get("TF_RDMA_DEVICES", result["NCCL_IB_HCA"])
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", result["API_INSTALLATION"]):
        raise ValueError("API_INSTALLATION must follow the serving lifecycle's unit-name contract")
    result["HEAD_SSH_KEY"] = remote_path(result["HEAD_SSH_KEY"])
    for key in ("ENGINE_PORT", "API_PORT", "TP_PORT"):
        if not result[key].isdigit() or not 1 <= int(result[key]) <= 65535:
            raise ValueError(f"invalid {key}")
    masks = result["RDMA_PEER_MASKS"].split()
    if len(masks) != 3 or any(not re.fullmatch(r"\d+,\d+,\d+", mask) for mask in masks):
        raise ValueError("RDMA_PEER_MASKS must contain three comma-separated peer masks")
    result["DEPLOY_RECEIPT"] = str(Path(env.get("DEPLOY_RECEIPT", ROOT / ".state/deploy-execution.json")).expanduser().resolve())
    # The transport's identity file belongs to the caller, not the remote user.
    result["SSH_KEY"] = str(Path(result["SSH_KEY"]).expanduser())
    return result


def configure(env=None):
    global SETTINGS, SPARKS, SSH_KEY, DEPLOY_ROOT, RECEIPT, API, OPTIONAL_CONTROLLER
    SETTINGS = cluster_settings(env)
    SPARKS = dict(enumerate(SETTINGS["SPARK_HOSTS"].split()))
    SSH_KEY, DEPLOY_ROOT = SETTINGS["SSH_KEY"], SETTINGS["SPARK_ROOT"]
    RECEIPT = Path(SETTINGS["DEPLOY_RECEIPT"])
    API, OPTIONAL_CONTROLLER = SETTINGS["ENGINE_HEALTH_URL"], SETTINGS["OPTIONAL_CONTROLLER_URL"]


def ssh_argv(host, cmd):
    return ["ssh", "-i", SSH_KEY, "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oConnectTimeout=10",
            f"{SETTINGS['SPARK_USER']}@{host}", cmd]


def run_ssh(host, cmd, check=True):
    print(f"[{host}] $ {cmd}")
    res = subprocess.run(ssh_argv(host, cmd), capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"SSH command failed on {host} with code {res.returncode}")
    return res.stdout.strip()


def remote_python(code, *args):
    return shlex.join([SETTINGS["SPARK_PYTHON"], "-I", "-c", code, *map(str, args)])


def unit_name(rank, execution):
    unit = f"{V['label']}-rank{rank}-g{execution}"
    if not GUARD_UNIT.fullmatch(unit):
        raise ValueError(f"unit name memguard would refuse: {unit}")
    return unit


def guard_state_dir(unit):
    return f"{DEPLOY_ROOT}/guard/{unit}/state"


def old_units(host):
    pattern = f"{V['label']}-rank[012]-g*.service"
    out = run_ssh(host, shlex.join(["systemctl", "--user", "list-units", "--type=service", "--all", "--plain",
                                  "--no-legend", pattern]))
    # Refuse surprising names returned by systemctl before using them as commands.
    units = [line.split()[0] for line in out.splitlines() if line.strip()]
    valid = re.compile(re.escape(V["label"]) + r"-rank[012]-g[0-9a-f]{32}(?:-guard)?\.service")
    if any(not valid.fullmatch(unit) for unit in units):
        raise RuntimeError(f"unexpected rank unit name on {host}")
    return sorted(units, key=lambda unit: "-guard" not in unit)


def stop_old_services():
    print("Stopping only this recipe's rank units and guards on all three Sparks")
    for host in SPARKS.values():
        units = old_units(host)
        if units:
            run_ssh(host, shlex.join(["systemctl", "--user", "stop", *units]))
    deadline = time.monotonic() + 90
    for host in SPARKS.values():
        while True:
            mem = run_ssh(host, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
            if not mem.isdigit():
                raise RuntimeError(f"could not verify free GPU memory on {host}")
            if int(mem) < 2048:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(f"GPU memory on {host} still {mem} MiB after stopping the engine")
            time.sleep(3)


KIT_FILES = ("aot/aot.json", "auxiliary/experts_ragged.cubin", "engram-token-map.json", "engram.json",
             "cubins/experts.cubin", "vision/gate_bias_vl.safetensors", "rope-plain-cos.f32", "rope-plain-sin.f32",
             "rope-compressed-cos.f32", "rope-compressed-sin.f32")


def preflight():
    """Validate every required artifact on every host before stopping anything."""
    code = '''import json,pathlib,sys
x=json.loads(sys.argv[1])
problems=['missing: '+p for p in x['paths'] if not pathlib.Path(p).exists()]
root=pathlib.Path(x['cache']).resolve()
if root==pathlib.Path('/'):raise ValueError('rank cache must not be the filesystem root')
files=list(root.glob('rank*of3*.bin')) if root.is_dir() else []
expected='rank'+str(x['rank'])+'of3-'
if any(p.is_symlink() for p in files):
 problems.append('symlink rank entries are not supported; use a regular rank file in '+str(root))
elif len(files)!=1 or not files[0].is_file() or not files[0].name.startswith(expected):
 problems.append('expected exactly one rank file in '+str(root)+' (rank'+str(x['rank'])+'of3-*.bin only; found '+str(len(files))+'); rank placement belongs to scripts/prepare.sh')
print('\\n'.join(problems))'''
    for rank, host in SPARKS.items():
        kit = SETTINGS["KIT_DIR"]
        paths = [f"{V['bin_dir']}/" + ("tensorfold-native" if rank == 0 else "tf-dsv41-lanes"),
                 SETTINGS["GUARD_SCRIPT"], SETTINGS["LOGGER_SCRIPT"], SETTINGS["MODEL_DIR"], SETTINGS["ENGRAM_DIR"] + "/SOURCE.json",
                 *(f"{kit}/{name}" for name in KIT_FILES)]
        missing = run_ssh(host, remote_python(code, json.dumps({"paths": paths, "cache": V["rank_cache"], "rank": rank})))
        if missing:
            hint = " (kit and Engram files come from ./scripts/prepare.sh; see docs/runtime-assets.md)" if any(
                line.startswith(("missing: " + kit, "missing: " + SETTINGS["ENGRAM_DIR"])) for line in missing.splitlines()) else ""
            raise FileNotFoundError(f"{host} preflight: {'; '.join(missing.splitlines())}{hint}")


def get_env_json(rank):
    substitutions = {**SETTINGS, "POOL_TOKENS": str(V["pool"]), "STREAMS": str(V["streams"]),
                     "GRAPH_CAP": str(V["graph_cap"]), "MEMORY_RESERVE_GIB": str(V["memory_reserve_gib"]),
                     "SPILL_GIB": str(V["spill_gib"]), "BIN_DIR": V["bin_dir"], "RANK_CACHE": V["rank_cache"]}
    env = {key: Template(str(value)).substitute(substitutions) for key, value in V["base_env"].items()}
    if rank == 0:
        env.update({key: Template(str(value)).substitute(substitutions) for key, value in V["head_env"].items()})
    env["TF_RDMA_PEER_MASKS"] = SETTINGS["RDMA_PEER_MASKS"].split()[rank]
    env.update(V["labels"])
    env.update(V["env"])
    return json.dumps(env)


def unit_props():
    return ["--collect", "--property=Restart=no", "--property=KillMode=control-group", "--property=TimeoutStopSec=15"]


def guard_command(unit):
    return shlex.join(["systemd-run", "--user", f"--unit={unit}-guard", "--collect", "--property=Restart=no",
                       f"--property=Requisite={unit}.service", f"--property=After={unit}.service",
                       SETTINGS["SPARK_PYTHON"], "-I", SETTINGS["GUARD_SCRIPT"], "--unit", unit, "--state-dir", guard_state_dir(unit)])


def start_guard(host, unit):
    if run_ssh(host, f"systemctl --user is-active {unit}", check=False) != "active":
        raise RuntimeError(f"{unit} on {host} is not active; refusing to start its guard")
    run_ssh(host, shlex.join(["mkdir", "-p", guard_state_dir(unit)]))
    run_ssh(host, guard_command(unit))


def mark_ready(units):
    for rank, host in SPARKS.items():
        unit = units[rank]
        invocation = run_ssh(host, f"systemctl --user show {unit} -p InvocationID --value")
        if not re.fullmatch(r"[0-9a-f]{32}", invocation):
            raise RuntimeError(f"{unit} on {host} has no InvocationID ({invocation!r})")
        run_ssh(host, shlex.join(["touch", f"{guard_state_dir(unit)}/ready-{invocation}"]))
    time.sleep(3)
    for rank, host in SPARKS.items():
        state = run_ssh(host, f"systemctl --user is-active {units[rank]}-guard", check=False)
        if state != "active":
            raise RuntimeError(f"{units[rank]}-guard on {host} is {state!r} after arming; check its journal")


def rank_command(rank, unit):
    prefix = ["systemd-run", "--user", f"--unit={unit}", *unit_props(), SETTINGS["SPARK_PYTHON"], "-I", SETTINGS["LOGGER_SCRIPT"],
              "--log", f"{DEPLOY_ROOT}/logs/{V['label']}-rank{rank}.log", "--max-size", "100m", "--max-files", "3",
              "--environment-json", get_env_json(rank), "--"]
    if rank:
        argv = [f"{V['bin_dir']}/tf-dsv41-lanes", SETTINGS["MODEL_DIR"], V["rank_cache"], str(rank), "3", SETTINGS["MASTER_ADDR"],
                SETTINGS["TP_PORT"], SETTINGS["KIT_DIR"], "--pool", str(V["pool"]), "--round-rows", "48", "--streams", str(V["streams"]),
                "--drafts", "1", "--graphs", "1", "--side", "1", "--prefetch", "1", "--aio", "1"]
        if V["graph_cap"]:
            argv += ["--graph-cap", str(V["graph_cap"])]
        argv += ["--engram", SETTINGS["ENGRAM_DIR"], "--token-map", SETTINGS["KIT_DIR"] + "/engram-token-map.json", "--rdma", SETTINGS["TF_RDMA_DEVICES"]]
    else:
        argv = [f"{V['bin_dir']}/tensorfold-native", "serve", SETTINGS["MODEL_DIR"], "--backend", "cuda", "--host", SETTINGS["ENGINE_HOST"],
                "--port", SETTINGS["ENGINE_PORT"], "--name", V["model"], "--context", str(V["context"]), "--parallel", str(V["streams"]),
                *V["head_args"]]
    return shlex.join(prefix + argv)


def launch_cluster(units):
    for rank in (1, 2):
        run_ssh(SPARKS[rank], rank_command(rank, units[rank]))
        start_guard(SPARKS[rank], units[rank])
    time.sleep(2)
    run_ssh(SPARKS[0], rank_command(0, units[0]))
    start_guard(SPARKS[0], units[0])


def health():
    if API:
        with urllib.request.urlopen(API.rstrip("/") + "/health", timeout=3) as response:
            return json.load(response) if response.status == 200 else {}
    code = "import json,sys,urllib.request; r=urllib.request.urlopen(sys.argv[1],timeout=3); print(json.dumps(json.load(r)) if r.status==200 else '{}')"
    url = f"http://{SETTINGS['ENGINE_HOST']}:{SETTINGS['ENGINE_PORT']}/health"
    return json.loads(run_ssh(SPARKS[0], remote_python(code, url)))


def poll_health(timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data = health()
            if data.get("status") == "ok" and data.get("warming") is False and data.get("numeric_effort") is True:
                return True
        except (OSError, ValueError, RuntimeError):
            pass
        time.sleep(3)
    return False


def _optional_controller(path, body=None, timeout=60):
    req = urllib.request.Request(OPTIONAL_CONTROLLER.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def reapply_affinity():
    if not OPTIONAL_CONTROLLER:
        return "unchanged"
    try:
        desired = _optional_controller("/api/affinity").get("desired")
        if not desired:
            return None
        nodes = _optional_controller("/api/affinity/apply", {"mode": desired, "confirmed": True}).get("nodes", {})
        if len(nodes) == 3 and all(node.get("mode") == desired for node in nodes.values()):
            return desired
    except (OSError, ValueError):
        pass
    print("PROBLEM: could not confirm the configured optional affinity controller core mode", file=sys.stderr)
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--label")
    parser.add_argument("--bin-dir")
    parser.add_argument("--rank-cache")
    for key in ("pool", "context", "streams", "graph-cap", "health-timeout"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--dry-run", action="store_true", help="print rank/guard commands and environments; no SSH, HTTP or writes")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--head-arg", action="append", default=[], metavar="FLAG")
    args = parser.parse_args(argv)
    configure()
    manifest = json.loads(args.manifest.read_text())
    engine = manifest["engine"]
    # cluster.env repeats release assertions, not a second mutable engine recipe.
    # Explicit CLI variants remain available; changing the release uses --manifest.
    for field, key in {"pool": "POOL_TOKENS", "context": "CONTEXT_TOKENS", "streams": "STREAMS",
                       "graph_cap": "GRAPH_CAP", "spill_gib": "SPILL_GIB", "commit": "ENGINE_COMMIT",
                       "memory_reserve_gib": "MEMORY_RESERVE_GIB", "top_k": "TOP_K"}.items():
        if os.environ.get(key, str(engine[field])) != str(engine[field]):
            parser.error(key + " disagrees with the release manifest")
    values = {"label": args.label or SETTINGS["ENGINE_LABEL"], "bin_dir": args.bin_dir or SETTINGS["BIN_DIR"],
              "rank_cache": args.rank_cache or SETTINGS["RANK_CACHE"], "base_env": engine["environment"],
              "head_env": engine["head_environment"], "labels": manifest.get("labels", {}),
              "model": manifest["serving"]["model"], "memory_reserve_gib": engine["memory_reserve_gib"], "spill_gib": engine["spill_gib"]}
    for key in ("pool", "context", "streams", "graph_cap", "health_timeout"):
        values[key] = getattr(args, key) if getattr(args, key) is not None else engine[key]
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,40}", values["label"]):
        parser.error("bad --label")
    if any(values[key] <= 0 for key in ("pool", "context", "health_timeout")) or not 1 <= values["streams"] <= 16 or values["graph_cap"] < 0:
        parser.error("positive pool/context/timeout, 1-16 streams and nonnegative graph cap required")
    if any(not re.fullmatch(r"--[a-z][a-z0-9-]*(=[A-Za-z0-9_./:~-]+)?", arg) for arg in args.head_arg):
        parser.error("bad --head-arg")
    extra = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) or CREDENTIAL_NAME.search(key) or "'" in value or "\0" in value or "\n" in value:
            parser.error("bad --env (engine settings only; no credentials)")
        extra[key] = value
    if values["graph_cap"]:
        extra["TF_DS_GRAPH_CAP"] = str(values["graph_cap"])  # one cap for head and followers, as in the production launcher
    values["env"] = extra
    head_vars = {**SETTINGS, "SPILL_GIB": str(engine["spill_gib"]), "TOP_K": str(engine["top_k"])}
    values["head_args"] = [Template(arg).substitute(head_vars) for arg in engine["head_args"]] + args.head_arg
    if any(CREDENTIAL_NAME.search(key) for section in (values["base_env"], values["head_env"], values["labels"]) for key in section):
        parser.error("manifest engine environment must not contain credentials")
    V.clear()
    V.update(values)
    return args


def stop_execution(units):
    """Clean up all minted addresses, including an ambiguously successful SSH launch."""
    errors = {}
    for rank, host in SPARKS.items():
        unit = units[rank]
        try:
            run_ssh(host, f"systemctl --user stop {unit}.service {unit}-guard.service")
        except Exception as exc:
            errors[host] = str(exc)
            print(f"PROBLEM: cleanup failed on {host} for {unit}: {exc}", file=sys.stderr)
    return errors


def write_execution_receipt(execution, units, **result):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps({"execution": execution, "units": units, "launch": V,
                                  "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **result}, indent=2) + "\n")


def main(argv=None, *, ready_status="ready"):
    args = parse_args(argv)
    execution = uuid.uuid4().hex
    units = {rank: unit_name(rank, execution) for rank in SPARKS}
    print(f"Plan: {'print' if args.dry_run else 'launch'} {V['label']} on three Sparks; workers, head, warm health, then guard markers")
    if args.dry_run:
        for rank, host in SPARKS.items():
            print(json.dumps({"rank": rank, "host": host, "unit": units[rank], "environment": json.loads(get_env_json(rank)),
                              "command": rank_command(rank, units[rank]), "guard_command": guard_command(units[rank])}, sort_keys=True))
        return 0
    preflight()
    try:
        stop_old_services()
        launch_cluster(units)
        if not poll_health(timeout=V["health_timeout"]):
            raise RuntimeError("Health check timed out before guards were armed (requires status ok, warming false and numeric_effort true)")
        mark_ready(units)
        affinity = reapply_affinity()
        write_execution_receipt(execution, units, status=ready_status, affinity=affinity)
    except BaseException as exc:
        cleanup_errors = stop_execution(units)
        try:
            write_execution_receipt(execution, units, status="failed", error=str(exc) or type(exc).__name__, cleanup_errors=cleanup_errors)
        except OSError:
            print("PROBLEM: could not record the failed execution receipt", file=sys.stderr)
        raise
    print(f"Cluster {V['label']} is warm with all three memory guards armed")
    return 2 if affinity is None else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAIL: deploy: {error}", file=sys.stderr)
        sys.exit(1)
