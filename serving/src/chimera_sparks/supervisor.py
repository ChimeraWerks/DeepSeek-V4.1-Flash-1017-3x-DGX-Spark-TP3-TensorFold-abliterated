"""Restart the owned TP3 group after a failed head, with invocation fencing.

Inspired by Mia 0020. Native rank 0 exits 75; systemd restarts this controller,
which reconstructs all three ranks rather than joining stale follower state.
"""
import json
import os
from pathlib import Path
import time

from . import lifecycle as L
from . import __version__


class OwnershipChanged(RuntimeError):
    pass


def inspect(item):
    return L.remote(item["node"], '''import json,subprocess,sys
x=json.loads(sys.argv[1]);p=subprocess.run(['systemctl','--user','show',x['unit'],'-p','ActiveState','-p','InvocationID','-p','ExecMainStatus'],capture_output=True,text=True)
v=dict(s.split('=',1) for s in p.stdout.splitlines() if '=' in s)
print(json.dumps({'active':v.get('ActiveState')=='active','invocation':v.get('InvocationID',''),'status':int(v.get('ExecMainStatus','0'))}))''', item)


def check_ownership(states, expected):
    if len(states) != 3 or len(expected) != 3:
        raise ValueError("three rank identities required")
    for state, wanted in zip(states, expected):
        if state["active"] and state["invocation"] != wanted:
            raise OwnershipChanged("rank invocation changed outside this supervisor; leaving it alone")


def cleanup(plan, expected):
    states = [inspect(item) for item in plan]
    # Validate every survivor before stopping any, so an unrelated replacement
    # cannot be partially dismantled by a stale controller.
    check_ownership(states, expected)
    for item in reversed(plan):
        L.stop_rank(item)
    if any(state["active"] for state in [inspect(item) for item in plan]):
        raise RuntimeError("owned ranks have not all stopped; refuse a new GPU world")


def native_up(raw, cfg):
    cluster = raw["cluster"]
    L.ensure_state(cfg)
    with L.locked(cluster.get("engine_lock", str(Path(cfg.state_dir) / "native.lock"))):
        plan = L.rank_plan(cluster)
        if any(item["qualification_required"] for item in plan) and os.environ.get("CHIMERA_QUALIFICATION_WINDOW") != "1":
            raise ValueError("native candidate needs an explicit qualification window; no GPU process started")
        evidence = L.engram_preflight(plan)
        checks = [L.provision_and_check(item) for item in plan]
        if L.engram_preflight(plan) != evidence:
            raise ValueError("Engram candidate changed during provisioning; replacement refused")
        states = [inspect(item) for item in plan]
        expected = [state["invocation"] for state in states]
        if any(row["active"] for row in checks) and not all(row["active"] for row in checks):
            # An incomplete old world cannot be repaired by attaching one new head.
            cleanup(plan, expected)
            checks = [L.provision_and_check(item) for item in plan]
            if L.engram_preflight(plan) != evidence:
                raise ValueError("Engram candidate changed during provisioning; replacement refused")
            if any(row["active"] for row in checks):
                raise RuntimeError("partial native world did not stop")
        try:
            for index in [1, 2, 0]:
                item = plan[index]
                if not checks[index]["active"]:
                    L.start_rank(item)
                    expected[index] = inspect(item)["invocation"]
                L.start_guard(item)
            L.wait_native(cfg, cluster.get("ready_timeout", 1800))
            for item in plan:
                L.start_guard(item, ready=True)
            states = [inspect(item) for item in plan]
            if any(not state["active"] or not state["invocation"] for state in states):
                raise RuntimeError("native readiness lacks all three rank identities")
            check_ownership(states, expected)
            return plan, [state["invocation"] for state in states]
        except BaseException:
            cleanup(plan, expected)
            raise


def supervise(path):
    path, raw, cfg = L.document(path)
    record = Path(cfg.state_dir) / "native-supervisor.json"
    if record.exists():
        previous = json.loads(record.read_text())
        plan = L.rank_plan(raw["cluster"])
        states = [inspect(item) for item in plan]
        try:
            check_ownership(states, previous["invocations"])
        except OwnershipChanged:
            return 0
        if any(not row["active"] for row in states):
            with L.locked(raw["cluster"].get("engine_lock", str(Path(cfg.state_dir) / "native.lock"))):
                cleanup(plan, previous["invocations"])
    plan, expected = native_up(raw, cfg)
    if not raw["cluster"].get("api_external", False):
        L.start_api(path, raw, cfg)
    record.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = record.with_suffix(".new")
    with temporary.open("w") as stream:
        json.dump({"invocations": expected, "controller_invocation": os.environ.get("INVOCATION_ID", "manual"),
                   "package_version": __version__, "ready": True}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(record)
    directory = os.open(record.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    while True:
        time.sleep(1)
        head = inspect(plan[0])
        if head["active"]:
            if head["invocation"] != expected[0]:
                return 0  # an operator replaced the world; this controller stands down
            continue
        try:
            with L.locked(raw["cluster"].get("engine_lock", str(Path(cfg.state_dir) / "native.lock"))):
                cleanup(plan, expected)
        except OwnershipChanged:
            return 0
        print(json.dumps({"event": "rank_group_failed", "head_status": head["status"], "exit": 75}), flush=True)
        return 75  # Restart=on-failure reconstructs the entire group on the next invocation
