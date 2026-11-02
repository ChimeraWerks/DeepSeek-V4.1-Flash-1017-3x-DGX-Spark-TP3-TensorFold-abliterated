#!/usr/bin/env python3
"""Head-only API staging/start, with JSON on stdin and no credential in argv.

Staging validates private files but keeps them pending until API replacement.
Starting attaches only to a warm numeric-effort engine; rank guards stay separate.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.request


def configuration(settings, manifest, previous=None):
    """Build attach-mode config, preserving existing client identities and roles."""
    sampler, serving = manifest["sampler"], manifest["serving"]
    old = (previous or {}).get("api", {})
    clients = dict(old.get("clients", {}))
    clients.setdefault("default", "SPARK_API_KEY")
    admins = list(old.get("admin_clients", []))
    if "default" in admins or any(clients.get(name) == "SPARK_API_KEY" for name in admins):
        raise ValueError("the default serving client must not have admin access")
    if not any(clients.get(name) == "SPARK_ADMIN_KEY" for name in admins):
        if "SPARK_ADMIN_KEY" in clients.values():
            raise ValueError("SPARK_ADMIN_KEY must not belong to a serving client")
        name, suffix = "admin", 1
        while name in clients:
            suffix += 1
            name = "admin-" + str(suffix)
        clients[name] = "SPARK_ADMIN_KEY"
        admins.append(name)
    return {
        "installation": settings["API_INSTALLATION"], "env_file": settings["API_ENV_FILE"],
        "api": {"upstream": f"http://{settings['ENGINE_HOST']}:{settings['ENGINE_PORT']}",
                "model": serving["model"], "host": settings["API_HOST"], "port": int(settings["API_PORT"]),
                "state_dir": str(Path(settings["SPARK_ROOT"]) / "api/state"), "clients": clients, "admin_clients": admins,
                "sampling": sampler["defaults"], "profiles": sampler["profiles"], "reasoning_default": sampler["reasoning_default"],
                "effort_map": sampler["effort_map"], "context_tokens": serving["context_tokens"],
                "engine_streams": manifest["engine"]["streams"], "max_concurrency": serving["max_concurrency"]}}


def private_temp(path, content):
    """Write a complete same-filesystem 0600 file before making it visible."""
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    temp = Path(name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def replace_files(documents):
    """Stage both files first; restore old contents if publishing either fails."""
    staged, previous, replaced = {}, {}, []
    try:
        for path, content in documents.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            previous[path] = path.read_bytes() if path.exists() else None
            staged[path] = private_temp(path, content)
        for path, temp in staged.items():
            os.replace(temp, path)
            replaced.append(path)
    except BaseException:
        for path in reversed(replaced):
            if previous[path] is None:
                path.unlink(missing_ok=True)
            else:
                temp = private_temp(path, previous[path])
                try:
                    os.replace(temp, path)
                finally:
                    temp.unlink(missing_ok=True)
        raise
    finally:
        for temp in staged.values():
            temp.unlink(missing_ok=True)


@contextmanager
def publishing_files(documents):
    """Keep the exact prior files until the lifecycle records a successful start."""
    backups, reserved, committed = {}, [], False
    try:
        for path in documents:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.is_symlink():
                backup = private_temp(path, b"")
                reserved.append(backup)
                os.replace(path, backup)
                backups[path] = backup
            else:
                backups[path] = None
        replace_files(documents)
        yield
        committed = True
    except BaseException:
        for path, backup in reversed(list(backups.items())):
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(backup, path)
        raise
    finally:
        # Never discard an original whose restoration itself failed.
        for backup in reserved:
            if committed or backup not in backups.values():
                backup.unlink(missing_ok=True)


def pending_path(path):
    return path.with_name("." + path.name + ".pending")


def validate_secrets(cfg, values):
    """Use the package's exact credential validation without retaining its env."""
    saved = {name: os.environ.get(name) for name in cfg.clients.values()}
    try:
        for name in saved:
            os.environ[name] = values.get(name, "")
        cfg.secrets()
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def stage(payload):
    from chimera_sparks import __version__, lifecycle
    from chimera_sparks.config import Config

    settings, manifest = payload["settings"], payload["manifest"]
    if __version__ != manifest["serving"]["version"]:
        raise RuntimeError("installed serving version differs from the release manifest")
    # Reject caller input before any credential/config file is opened for writing.
    provided = {"SPARK_API_KEY": payload.get("key", ""), "SPARK_ADMIN_KEY": payload.get("admin_key", "")}
    for value in provided.values():
        if not isinstance(value, str) or (value and len(value) < 16) or any(c in value for c in ("\n", "\r", "\0")):
            raise ValueError("API keys must be at least 16 characters and fit on one environment line")
    config, credentials = Path(settings["API_CONFIG"]).expanduser().resolve(), Path(settings["API_ENV_FILE"]).expanduser()
    if credentials.is_symlink():
        raise ValueError("credential file must not be a symlink")
    values = {}
    if credentials.exists():
        if credentials.stat().st_mode & 0o077:
            raise ValueError("existing API credential file must be private (0600)")
        for line in credentials.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if not separator or not name.isidentifier() or any(c in value for c in ("\r", "\0")):
                raise ValueError("invalid credential environment file")
            values[name] = value
    previous = json.loads(config.read_text()) if config.exists() else None
    raw = configuration(settings, manifest, previous)
    cfg = Config(**raw["api"]).validate()
    lifecycle.unit_name(raw)
    for name, supplied in provided.items():
        values[name] = supplied or values.get(name) or secrets.token_urlsafe(32)
    validate_secrets(cfg, values)
    lifecycle.ensure_state(cfg)
    replace_files({pending_path(credentials): "".join(name + "=" + value + "\n" for name, value in values.items()).encode(),
                   pending_path(config): (json.dumps(raw, indent=2) + "\n").encode()})
    return {"ok": True, "staged": True, "credential_file": str(credentials), "config": str(config)}


def warm_engine(health):
    return (isinstance(health, dict) and health.get("status") == "ok" and health.get("warming") is False
            and health.get("numeric_effort") is True)


def wait_native(cfg, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(cfg.upstream + "/health", timeout=5) as response:
                if response.status == 200 and warm_engine(json.load(response)):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("engine readiness requires status ok, warming false and numeric_effort true")


class ApiStartFailure(RuntimeError):
    def __init__(self, result):
        super().__init__("API start/readiness failed")
        self.result = result


class ApiOwnershipRefusal(RuntimeError):
    pass


def require_owned_api(lifecycle, path, raw, cfg):
    """Keep the lifecycle's active config/release check, plus recipe identity."""
    try:
        record = json.loads((lifecycle.ensure_state(cfg) / "installation.json").read_text())
        if (record.get("config") != str(path) or record.get("executable") != sys.executable
                or record.get("unit") != lifecycle.unit_name(raw)):
            raise ValueError("installation ledger mismatch")
        # For an active unit this is a read-only ownership check, not a restart.
        lifecycle.start_api(path, raw, cfg, executable=sys.executable)
    except Exception:
        raise ApiOwnershipRefusal("active API unit does not match this installation/config/release; refusing to stop it") from None


def start(payload):
    from chimera_sparks import __version__, lifecycle

    settings, manifest = payload["settings"], payload["manifest"]
    if __version__ != manifest["serving"]["version"]:
        raise RuntimeError("installed serving version differs from the release manifest")
    path = Path(settings["API_CONFIG"]).expanduser().resolve()
    credentials = Path(settings["API_ENV_FILE"]).expanduser()
    pending_config, pending_credentials = pending_path(path), pending_path(credentials)
    pending = pending_config.exists() or pending_credentials.exists()
    _, raw, cfg = lifecycle.document(pending_config if pending else path)
    lifecycle.load_env(path, {**raw, "env_file": str(pending_credentials)} if pending else raw)
    cfg.secrets()
    # Read the complete validated pair before readiness/ownership checks or stops.
    documents = {credentials: pending_credentials.read_bytes(), path: pending_config.read_bytes()} if pending else {}
    wait_native(cfg, manifest["engine"]["health_timeout"])
    unit = lifecycle.unit_name(raw)
    if subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit], capture_output=True).returncode == 0:
        require_owned_api(lifecycle, path, raw, cfg)
        subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=60, capture_output=True)
    # Do not turn a lifecycle/busy refusal into permission to stop a unit.
    # Cleanup owns only a start that the lifecycle actually confirmed.
    started, launched = False, False
    lifecycle_run = lifecycle.run
    def run_and_track(command, **kwargs):
        nonlocal launched
        output = lifecycle_run(command, **kwargs)
        if command[0] == "systemd-run" and "--unit=" + unit in command:
            launched = True
        return output
    try:
        # Observe the real launch separately from the subsequent ledger write.
        lifecycle.run = run_and_track
        try:
            with publishing_files(documents):
                lifecycle.start_api(path, raw, cfg, executable=sys.executable)
                started = True
        finally:
            lifecycle.run = lifecycle_run
        if pending:
            pending_config.unlink()
            pending_credentials.unlink()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://{settings['API_HOST']}:{settings['API_PORT']}/ready", timeout=5) as response:
                    health = json.load(response)
                if response.status == 200 and health.get("ok") is True and warm_engine(health.get("native")):
                    return {"ok": True, "unit": unit, "ready": True, "started": True, "version": __version__,
                            "max_concurrency": cfg.max_concurrency, "credential_file": settings["API_ENV_FILE"]}
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError("API did not become ready within 120 seconds")
    except BaseException as error:
        result = {"ok": False, "unit": unit, "ready": False, "start_attempted": True, "started": started,
                  "launched": launched or started,
                  "error": "API start/readiness failed (" + type(error).__name__ + ")", "stopped": False}
        if launched or started:
            try:
                subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=60, capture_output=True)
                result["stopped"] = True
            except Exception as cleanup:
                result["cleanup_error"] = type(cleanup).__name__
        raise ApiStartFailure(result) from None


def main():
    try:
        payload = json.load(sys.stdin)
        action = payload["action"]
        if action not in ("stage", "start"):
            raise ValueError("expected stage or start")
        print(json.dumps(stage(payload) if action == "stage" else start(payload)))
        return 0
    except ApiStartFailure as error:
        print(json.dumps(error.result))
    except ApiOwnershipRefusal as error:
        print(json.dumps({"ok": False, "ready": False, "stopped": False, "error": str(error)}))
    except Exception as error:
        # Tracebacks and arbitrary exception text could contain credential data.
        print(json.dumps({"ok": False, "ready": False, "error": "API setup failed (" + type(error).__name__ + ")"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
