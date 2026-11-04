"""Portable owned-unit lifecycle. Topology and artifact identities are operator data."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request

from .assets import native_launch
from .config import Config
from .assets.rotating_run import size_bytes
from . import __version__


def document(path):
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text())
    cfg = Config.load(path)
    return path, raw, cfg


def load_env(path, raw):
    ref = raw.get("env_file")
    if not ref:
        return
    env_file = Path(ref).expanduser()
    if not env_file.is_absolute():
        env_file = path.parent / env_file
    if env_file.is_symlink() or env_file.stat().st_mode & 0o077:
        raise ValueError("credential environment file must be private (0600) and not a symlink")
    for line in env_file.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or not name.isidentifier() or "\0" in value:
            raise ValueError("invalid environment file")
        os.environ[name] = value


def unit_name(raw):
    name = raw.get("installation", "chimera-sparks")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name):
        raise ValueError("installation must be a short lowercase service name")
    return name + "-api"


def ensure_state(cfg):
    state = Path(cfg.state_dir).expanduser()
    if state.is_symlink():
        raise ValueError("state directory must not be a symlink")
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    if state.stat().st_mode & 0o077 or state.stat().st_uid != os.geteuid():
        raise ValueError("state directory must be private and owned by the service user")
    return state


@contextmanager
def locked(path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def remote(node, script, data=None):
    # Script and input are generated locally from a trusted config; no shell
    # interpolation of topology, paths, credentials or arbitrary command text.
    host = node["host"]
    if not re.fullmatch(r"[a-zA-Z0-9_.:@-]+", host):
        raise ValueError("invalid SSH host")
    if host in {"localhost", "127.0.0.1"}:
        argv = ["nice", "-n", "10", "ionice", "-c3", sys.executable, "-c", script, json.dumps(data)]
    else:
        argv = ["ssh", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes", "-oConnectTimeout=8"]
        if node.get("identity_file"):
            argv += ["-i", str(Path(node["identity_file"]).expanduser())]
        argv += [host, shlex.join(["nice", "-n", "10", "ionice", "-c3", "python3", "-c", script, json.dumps(data)])]
    return json.loads(run(argv, timeout=300))


def rank_plan(cluster):
    nodes = cluster["nodes"]
    if len(nodes) != 3 or [n["rank"] for n in nodes] != [0, 1, 2]:
        raise ValueError("exactly three ordered TP3 nodes are required")
    topology = {**cluster["topology"], "ranks": [n["native"] for n in nodes]}
    profile = cluster.get("profile", "v1.2")
    profiles = Path(native_launch.__file__).parent / "profiles"
    if profile not in {p.stem for p in profiles.glob("*.json")}:
        raise ValueError("unknown native profile")
    profile_data = json.loads((profiles / (profile + ".json")).read_text())
    qualification_required = not (profile_data.get("qualified_pool") or profile_data.get("serving_smoke_qualified"))
    logging = cluster.get("logging", {})
    log_size = logging.get("max_size", "100m")
    log_files = logging.get("max_files", 3)
    size_bytes(log_size)
    if isinstance(log_files, bool) or not isinstance(log_files, int) or not 1 <= log_files <= 20:
        raise ValueError("log max_files must be an integer from 1 to 20")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", cluster.get("unit_prefix", "chimera-sparks")):
        raise ValueError("invalid native unit prefix")
    result = []
    profile_defaults = []
    for i, node in enumerate(nodes):
        if not Path(node["native"]["root"]).is_absolute():
            raise ValueError("native roots must be absolute on each Spark")
        argv, env = native_launch.command(topology, profile, i)
        profile_defaults.append(dict(env))
        env.update(cluster.get("environment", {}))
        if any(re.search(r"(?:^|_)(?:TOKEN|PASSWORD|SECRET|API_KEY|ACCESS_TOKEN|AUTH_TOKEN)$", name) for name in env):
            raise ValueError("native environment must contain settings, not credentials")
        log = str(Path(node["native"]["root"]) / "logs" / ("rank" + str(i) + ".log"))
        logger = str(Path(node["native"]["root"]) / "service/rotating_run.py")
        launch = ["python3", "-I", logger, "--log", log, "--max-size", log_size, "--max-files", str(log_files), "--"] + argv
        result.append({"node": node, "argv": argv, "launch": launch, "environment": env, "qualification_required": qualification_required,
                       "engram_compatibility_preflight": profile_data.get("engram_compatibility_preflight", False),
                       "unit": cluster.get("unit_prefix", "chimera-sparks") + "-rank" + str(i)})
    from .capacity import plan
    result = plan(cluster, result, profile_defaults=profile_defaults)
    # Capacity can append flags or replace pool values. Freeze the supervised
    # command only after those mutations so execution and identity checks agree.
    for item in result:
        prefix = item["launch"][:item["launch"].index("--") + 1]
        item["launch"] = prefix[:-1] + ["--environment-json",
            json.dumps(item["environment"], sort_keys=True, separators=(",", ":")), "--"] + item["argv"]
    return result


def engram_preflight(plan):
    """Collect every effective rank identity without provisioning or writes."""
    from .assets.engram_identity import compatible
    if not any(item.get("engram_compatibility_preflight") or item.get("capacity") for item in plan):
        return []
    source = (Path(__file__).parent / "assets/engram_identity.py").read_text()
    script = source + "\nimport sys\nprint(json.dumps(collect(json.loads(sys.argv[1]))))\n"
    rows = [remote(item["node"], script, item) for item in plan]
    return compatible(plan, rows)


def provision_and_check(item):
    item = {**item, "prepare_source": (Path(__file__).parent / "prepare.py").read_text()}
    script = '''import hashlib,json,os,pathlib,subprocess,sys,urllib.request,uuid
x=json.loads(sys.argv[1]);n=x['node'];root=pathlib.Path(n['native']['root']).resolve()
if n.get('writable_index'):
 namespace={};exec(compile(x['prepare_source'],'recipe_prepare.py','exec'),namespace)
 spec=n['writable_index'];work=pathlib.Path(spec['workdir']).resolve()
 if root not in work.parents:raise ValueError('overlay workdir must be within its owned runtime')
 namespace['writable_index'](spec['snapshot'],work)
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
for a in n.get('artifacts',[]):
 p=(root/a['path']).resolve()
 if root not in p.parents or len(a['sha256'])!=64:raise ValueError('invalid artifact path/hash')
 if p.is_file() and sha(p)==a['sha256']:continue
 if not a['url'].startswith('https://'):raise ValueError('artifact downloads require HTTPS')
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.download-'+uuid.uuid4().hex)
 try:
  digest=hashlib.sha256()
  with urllib.request.urlopen(a['url'],timeout=60) as source,tmp.open('xb') as target:
   while True:
    chunk=source.read(4*1024*1024)
    if not chunk:break
    target.write(chunk);digest.update(chunk)
   target.flush();os.fsync(target.fileno())
  if digest.hexdigest()!=a['sha256']:raise ValueError('artifact digest mismatch')
  tmp.chmod(0o755 if a.get('executable') else 0o644);os.replace(tmp,p)
 finally:tmp.unlink(missing_ok=True)
checks=n.get('checksums',{})
if x['argv'][0] not in checks:raise ValueError('binary checksum required before native start')
for name,digest in checks.items():
 p=pathlib.Path(name)
 if not p.is_file() or sha(p)!=digest:raise ValueError('native artifact identity mismatch')
for name in n.get('required_paths',[]):
 if not pathlib.Path(name).exists():raise ValueError('required native asset is missing')
mem={s.split(':')[0]:int(s.split()[1]) for s in pathlib.Path('/proc/meminfo').read_text().splitlines() if ':' in s and s.split()[1].isdigit()}
active=subprocess.run(['systemctl','--user','is-active',x['unit']],capture_output=True,text=True).stdout.strip()=='active'
if active:
 pid=int(subprocess.check_output(['systemctl','--user','show',x['unit'],'-p','MainPID','--value']))
 actual=[s.decode() for s in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\\0') if s]
 if actual[1:]!=x['launch'][1:]:raise ValueError('owned unit already runs a different command; stop it explicitly')
 children=pathlib.Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
 engines=[int(child) for child in children if [v.decode() for v in pathlib.Path(f'/proc/{child}/cmdline').read_bytes().split(b'\\0') if v]==x['argv']]
 if len(engines)!=1:raise ValueError('owned logger does not supervise the expected rank; stop it explicitly')
 engine_pid=engines[0]
 expected=n['checksums'][x['argv'][0]]
 if sha(pathlib.Path(f'/proc/{engine_pid}/exe'))!=expected:raise ValueError('owned rank still runs another executable; stop it explicitly')
 values={s.split(b'=',1)[0].decode():s.split(b'=',1)[1].decode() for s in pathlib.Path(f'/proc/{engine_pid}/environ').read_bytes().split(b'\\0') if b'=' in s}
 def managed(key):return key in x['environment'] or key.startswith(('TF_DS_','TF_SPILL_','TF_TP_','TF_RDMA_','NCCL_','TENSORFOLD_')) or key in ('LD_LIBRARY_PATH','HF_HUB_OFFLINE')
 wanted={key:value for key,value in x['environment'].items() if managed(key)}
 loaded={key:value for key,value in values.items() if managed(key)}
 if loaded!=wanted:raise ValueError('owned rank runs different managed settings; stop it explicitly')
elif mem.get('MemAvailable',0)<int(n.get('minimum_start_gib',100))*1024*1024:raise ValueError('native startup memory gate blocked')
print(json.dumps({'active':active,'verified_files':len(checks)}))'''
    from .capacity import REMOTE_CHECK
    script = script.replace("checks=n.get('checksums',{})", REMOTE_CHECK + "\nchecks=n.get('checksums',{})")
    return remote(item["node"], script, item)


def start_rank(item):
    logger = (Path(__file__).parent / "assets/rotating_run.py").read_text()
    item = {**item, "logger_source": logger}
    script = '''import json,pathlib,subprocess,sys
x=json.loads(sys.argv[1]);root=pathlib.Path(x['node']['native']['root']);root.mkdir(parents=True,exist_ok=True)
(root/'service').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True,mode=0o700)
(root/'service/rotating_run.py').write_text(x['logger_source'])
argv=['systemd-run','--user','--unit='+x['unit'],'--collect','--property=WorkingDirectory='+str(root),'--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=15']
if x.get('capacity',{}).get('spill'):argv+=['--property=TimeoutStopSec='+str((x['capacity']['flush_ms']+999)//1000+15)]

argv+=['--setenv='+k+'='+v for k,v in x['environment'].items()]
subprocess.run(argv+['--']+x['launch'],check=True,capture_output=True)
print(json.dumps({'started':x['unit']}))'''
    return remote(item["node"], script, item)


def stop_rank(item):
    return remote(item["node"], '''import json,subprocess,sys
x=json.loads(sys.argv[1])
for unit in [x['unit']+'-guard',x['unit']]:
 if subprocess.run(['systemctl','--user','is-active',unit],capture_output=True).returncode==0:subprocess.run(['systemctl','--user','stop',unit],check=True,capture_output=True)
print('{}')''', item)


def start_guard(item, ready=False):
    source = Path(__file__).parent / "assets/memguard.py"
    payload = {**item, "guard_source": source.read_text(), "ready": ready}
    script = '''import json,pathlib,subprocess,sys
x=json.loads(sys.argv[1]);root=pathlib.Path(x['node']['native']['root'])/'guard';root.mkdir(parents=True,exist_ok=True)
state=root/'state';state.mkdir(exist_ok=True)
unit=x['unit']+'-guard'
if not x['ready']:
 (root/'memguard.py').write_text(x['guard_source'])
 active=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True).returncode==0
 if not active:subprocess.run(['systemd-run','--user','--unit='+unit,'--collect','--','python3','-I',str(root/'memguard.py'),'--unit',x['unit'],'--state-dir',str(state)],check=True,capture_output=True)
else:
 invocation=subprocess.check_output(['systemctl','--user','show',x['unit'],'-p','InvocationID','--value'],text=True).strip()
 if not invocation:raise ValueError('rank invocation missing')
 (state/('ready-'+invocation)).touch()
print(json.dumps({'guard':unit,'ready':x['ready']}))'''
    return remote(item["node"], script, payload)


def wait_native(cfg, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(cfg.upstream + "/health", timeout=5) as response:
                value = json.load(response)
            if not value.get("warming", False) and value.get("status", "ok") == "ok":
                return value
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("native readiness deadline exceeded")


def start_api(path, raw, cfg, *, executable=None):
    unit = unit_name(raw)
    state = ensure_state(cfg)
    active = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True).returncode == 0
    record_path = state / "installation.json"
    resolved = dict(raw)
    resolved["api"] = cfg.public()
    if raw.get("env_file"):
        ref = Path(raw["env_file"]).expanduser()
        resolved["env_file"] = str((ref if ref.is_absolute() else path.parent / ref).resolve())
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    old = json.loads(record_path.read_text()) if record_path.exists() else None
    if active:
        if old and old["config_sha256"] == digest and old["executable"] == (executable or sys.executable):
            return unit
        raise ValueError("API unit is active with another config/release; stop it explicitly before upgrade")
    # Copy the nonsecret configuration into the installation's durable state.
    snapshot = state / ("config-" + digest[:16] + ".json")
    snapshot.write_text(json.dumps(resolved, indent=2) + "\n")
    argv = [executable or sys.executable, "-m", "chimera_sparks.cli", "serve", "--config", str(snapshot)]
    run(["systemd-run", "--user", "--unit=" + unit, "--collect", "--property=Restart=on-failure",
         "--property=RestartSec=3", "--property=UMask=0077", "--property=NoNewPrivileges=yes",
         "--property=PrivateTmp=yes", "--property=ProtectSystem=strict",
         "--property=ReadWritePaths=" + str(state), "--"] + argv)
    record = {"config": str(path), "snapshot": str(snapshot), "config_sha256": digest,
              "executable": executable or sys.executable, "unit": unit, "previous": old}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return unit


def up(path):
    path, raw, cfg = document(path)
    load_env(path, raw)
    cfg.secrets()
    state = Path(cfg.state_dir).expanduser()
    try:
        if raw.get("cluster"):
            from .supervisor import native_up
            native_up(raw, cfg)
        else:
            wait_native(cfg, raw.get("ready_timeout", 60))
        unit = start_api(path, raw, cfg)
        if raw.get("cluster"):
            start_supervisor(raw, cfg)
        deadline = time.monotonic() + (raw.get("cluster", {}).get("ready_timeout", 1800) if raw.get("cluster") else 30)
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{cfg.port}/ready", timeout=3) as response:
                    if json.load(response).get("ok") and group_ready(raw, cfg):
                        return {"unit": unit, "endpoint": f"http://{cfg.host}:{cfg.port}/v1", "ready": True}
            except (OSError, ValueError):
                pass
            time.sleep(.25)
        raise RuntimeError("recipe API did not become ready; inspect its user journal")
    except BaseException:
        raise


def rollback(path):
    path, raw, cfg = document(path)
    state = Path(cfg.state_dir).expanduser()
    current = json.loads((state / "installation.json").read_text())
    previous = current.get("previous")
    if not previous:
        raise ValueError("no previous API installation is recorded")
    # Ledger stays in place. API-only rollback never stops the native model.
    run(["systemctl", "--user", "stop", current["unit"]])
    saved, old, old_cfg = document(previous["snapshot"])
    start_api(saved, old, old_cfg, executable=previous["executable"])
    return {"rolled_back": previous["unit"], "native_unchanged": True}


def start_supervisor(raw, cfg):
    unit = unit_name(raw).removesuffix("-api") + "-group"
    if subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True).returncode == 0:
        return unit
    state = Path(cfg.state_dir).expanduser()
    current = json.loads((state / "installation.json").read_text())
    command = ["systemd-run", "--user", "--unit=" + unit, "--collect",
         "--property=Restart=on-failure", "--property=RestartSec=5",
         "--property=StartLimitIntervalSec=300", "--property=StartLimitBurst=3",
         "--property=TimeoutStopSec=15", "--property=UMask=0077",
         "--property=Nice=10", "--property=IOSchedulingClass=idle", "--property=CPUWeight=20", "--",
         sys.executable, "-m", "chimera_sparks.cli", "supervise", "--config", current["snapshot"]]
    if os.environ.get("CHIMERA_QUALIFICATION_WINDOW") == "1":
        command.insert(command.index("--"), "--setenv=CHIMERA_QUALIFICATION_WINDOW=1")
    run(command)
    return unit


def group_ready(raw, cfg):
    if not raw.get("cluster"):
        return True
    record = Path(cfg.state_dir).expanduser() / "native-supervisor.json"
    if not record.is_file():
        return False
    value = json.loads(record.read_text())
    group = unit_name(raw).removesuffix("-api") + "-group"
    if subprocess.run(["systemctl", "--user", "is-active", group], capture_output=True).returncode != 0:
        return False
    invocation = run(["systemctl", "--user", "show", group, "-p", "InvocationID", "--value"])
    return bool(invocation) and value.get("ready") is True and value.get("controller_invocation") == invocation and value.get("package_version") == __version__


def stop_installation(path, with_engine=False):
    path, raw, cfg = document(path)
    if with_engine:
        if not raw.get("cluster"):
            raise ValueError("attach mode does not own the native engine")
        group = unit_name(raw).removesuffix("-api") + "-group"
        run(["systemctl", "--user", "stop", group])
        from .supervisor import cleanup, inspect
        with locked(raw["cluster"].get("engine_lock", str(Path(cfg.state_dir) / "native.lock"))):
            plan = rank_plan(raw["cluster"])
            current = [inspect(item) for item in plan]
            cleanup(plan, [row["invocation"] for row in current])
    run(["systemctl", "--user", "stop", unit_name(raw)])
    return {"stopped": unit_name(raw), "native_stopped": with_engine}
