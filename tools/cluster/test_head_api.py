"""Head API staging/readiness regressions using only disposable placeholder keys."""
import io
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from test_deploy import ROOT, load

SERVING_KEY = "test-serving-placeholder-00000001"
ADMIN_KEY = "test-admin-placeholder-000000002"
OPS_KEY = "test-existing-ops-placeholder-003"
HEALTH = {"status": "ok", "warming": False, "numeric_effort": True}


def response(data, status=200):
    stream = io.StringIO(json.dumps(data))
    stream.status = status
    return stream


@pytest.fixture
def head(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "serving/src"))
    mod = load("head_api", "head_api.py")
    for name in ("SPARK_API_KEY", "SPARK_ADMIN_KEY"):
        monkeypatch.setenv(name, "")
    clock = [0]
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda n: clock.__setitem__(0, clock[0] + n)))
    generated = []
    def placeholder(_):
        generated.append(True)
        return "test-generated-placeholder-" + str(len(generated)).zfill(8)
    monkeypatch.setattr(mod.secrets, "token_urlsafe", placeholder)
    return mod


@pytest.fixture
def payload(tmp_path):
    return {"settings": {"SPARK_ROOT": str(tmp_path), "API_CONFIG": str(tmp_path / "api/config.json"),
                         "API_ENV_FILE": str(tmp_path / "api/credentials.env"), "API_INSTALLATION": "ds1017",
                         "ENGINE_HOST": "127.0.0.1", "ENGINE_PORT": "9003", "API_HOST": "127.0.0.1", "API_PORT": "19002"},
            "manifest": json.loads((ROOT / "release/manifest.json").read_text()), "key": SERVING_KEY, "admin_key": ADMIN_KEY}


def publish_stage(head, payload):
    """Set up a previously installed pair without running a service."""
    paths = tuple(Path(payload["settings"][name]) for name in ("API_CONFIG", "API_ENV_FILE"))
    for path in paths:
        head.pending_path(path).replace(path)
    return paths


@pytest.fixture
def staged(head, payload):
    assert head.stage(payload)["staged"] is True
    return publish_stage(head, payload)


def contents(paths):
    return [p.read_bytes() for p in paths]


@pytest.mark.parametrize("name,value", [("key", "short"), ("admin_key", "short"), ("key", "x" * 16 + "\n"),
                                       ("admin_key", "x" * 16 + "\r"), ("key", "x" * 16 + "\0")])
def test_bad_supplied_key_preserves_both_valid_files(head, payload, staged, name, value):
    before = contents(staged)
    payload[name] = value
    with pytest.raises(ValueError, match="at least 16 characters"):
        head.stage(payload)
    assert contents(staged) == before


def test_invalid_first_key_creates_no_files(head, payload):
    payload["key"] = "short"
    with pytest.raises(ValueError):
        head.stage(payload)
    assert not Path(payload["settings"]["API_CONFIG"]).exists()
    assert not Path(payload["settings"]["API_ENV_FILE"]).exists()


@pytest.mark.parametrize("failure", ["config", "duplicate_keys"])
def test_full_validation_precedes_any_file_replacement(head, payload, staged, failure):
    before = contents(staged)
    if failure == "config":
        payload["manifest"]["serving"]["max_concurrency"] = 17
    else:
        payload["admin_key"] = payload["key"]
    with pytest.raises(ValueError):
        head.stage(payload)
    assert contents(staged) == before


def test_config_and_credentials_stage_in_private_same_directory_temps(head, payload, staged, monkeypatch):
    before = contents(staged)
    pending = [head.pending_path(path) for path in staged]
    replace, seen = head.os.replace, []
    def checked(source, destination):
        source, destination = Path(source), Path(destination)
        assert source != destination and source.parent == destination.parent
        assert stat.S_IMODE(source.stat().st_mode) == 0o600
        seen.append(destination)
        return replace(source, destination)
    monkeypatch.setattr(head.os, "replace", checked)
    payload.update(key=SERVING_KEY + "-rotated", admin_key=ADMIN_KEY + "-rotated")
    assert head.stage(payload)["ok"] is True
    assert seen == [pending[1], pending[0]]
    assert contents(staged) == before and contents(pending) != before
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in pending)
    assert set(staged[0].parent.glob(".*")) == set(pending)


@pytest.mark.parametrize("failure", ["temp_write", "second_publish"])
def test_io_failure_preserves_previous_pair_and_cleans_temps(head, payload, staged, monkeypatch, failure):
    before = contents(staged)
    if failure == "temp_write":
        original = head.private_temp
        def write(path, data):
            if path == head.pending_path(staged[0]):
                raise OSError("test disk full")
            return original(path, data)
        monkeypatch.setattr(head, "private_temp", write)
    else:
        replace = head.os.replace
        failed = []
        def publish(source, destination):
            if Path(destination) == head.pending_path(staged[0]) and not failed:
                failed.append(True)
                raise OSError("test rename failure")
            return replace(source, destination)
        monkeypatch.setattr(head.os, "replace", publish)
    payload.update(key=SERVING_KEY + "-new", admin_key=ADMIN_KEY + "-new")
    with pytest.raises(OSError):
        head.stage(payload)
    assert contents(staged) == before
    assert not list(staged[0].parent.glob(".*"))


def test_stage_keeps_credential_privacy_guards(head, payload, staged):
    config, credentials = staged
    before = config.read_bytes()
    credentials.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        head.stage(payload)
    assert config.read_bytes() == before
    credentials.chmod(0o600)
    alias = credentials.with_name("alias.env")
    alias.symlink_to(credentials)
    payload["settings"]["API_ENV_FILE"] = str(alias)
    with pytest.raises(ValueError, match="symlink"):
        head.stage(payload)
    assert config.read_bytes() == before


def test_generated_admin_is_separate_and_credentials_are_stable_across_starts(head, payload, monkeypatch):
    payload.update(key="", admin_key="")
    head.stage(payload)
    paths = publish_stage(head, payload)
    raw = json.loads(paths[0].read_text())
    values = dict(line.split("=", 1) for line in paths[1].read_text().splitlines())
    assert raw["api"]["admin_clients"] == ["admin"]
    assert raw["api"]["clients"] == {"default": "SPARK_API_KEY", "admin": "SPARK_ADMIN_KEY"}
    assert len(values["SPARK_ADMIN_KEY"]) >= 16 and values["SPARK_ADMIN_KEY"] != values["SPARK_API_KEY"]
    before = contents(paths)
    monkeypatch.setattr(head.secrets, "token_urlsafe", lambda _: pytest.fail("restart rotates a stable key"))
    head.stage(payload)
    assert contents(paths) == contents([head.pending_path(path) for path in paths]) == before


def test_existing_admin_and_serving_clients_keep_keys_and_roles(head, payload, staged, monkeypatch):
    config, credentials = staged
    raw = json.loads(config.read_text())
    raw["api"].update(clients={"default": "SPARK_API_KEY", "admin": "SERVING_ONLY_KEY", "ops": "OPS_KEY"}, admin_clients=["ops"])
    config.write_text(json.dumps(raw))
    credentials.write_text("SPARK_API_KEY=" + SERVING_KEY + "\nSERVING_ONLY_KEY=" + ADMIN_KEY + "\nOPS_KEY=" + OPS_KEY + "\n")
    payload.update(key="", admin_key="")
    head.stage(payload)
    publish_stage(head, payload)
    raw = json.loads(config.read_text())
    assert raw["api"]["clients"] == {"default": "SPARK_API_KEY", "admin": "SERVING_ONLY_KEY", "ops": "OPS_KEY", "admin-2": "SPARK_ADMIN_KEY"}
    assert raw["api"]["admin_clients"] == ["ops", "admin-2"]
    values = dict(line.split("=", 1) for line in credentials.read_text().splitlines())
    assert values["SPARK_API_KEY"] == SERVING_KEY and values["SERVING_ONLY_KEY"] == ADMIN_KEY and values["OPS_KEY"] == OPS_KEY
    before = contents(staged)
    monkeypatch.setattr(head.secrets, "token_urlsafe", lambda _: pytest.fail("restart changes an existing admin"))
    head.stage(payload)
    assert contents(staged) == contents([head.pending_path(path) for path in staged]) == before


@pytest.mark.parametrize("conflict", ["default_role", "shared_env"])
def test_serving_clients_are_never_promoted_to_admin(head, payload, staged, conflict):
    raw = json.loads(staged[0].read_text())
    if conflict == "default_role":
        raw["api"]["admin_clients"].append("default")
    else:
        raw["api"]["admin_clients"] = []
    staged[0].write_text(json.dumps(raw))
    before = contents(staged)
    with pytest.raises(ValueError, match="serving client"):
        head.stage(payload)
    assert contents(staged) == before


@pytest.mark.parametrize("health", [{"status": "ok", "warming": False}, {**HEALTH, "numeric_effort": False},
                                   {"warming": False, "numeric_effort": True}, {"status": "ok", "numeric_effort": True},
                                   {**HEALTH, "warming": True}, {**HEALTH, "status": "error"}])
def test_native_wait_rejects_missing_numeric_effort_and_implicit_readiness(head, health, monkeypatch):
    monkeypatch.setattr(head.urllib.request, "urlopen", lambda *a, **kw: response(health))
    with pytest.raises(RuntimeError, match="numeric_effort true"):
        head.wait_native(SimpleNamespace(upstream="http://fake-engine:9003"), 2)


def fake_lifecycle(head, monkeypatch, api_health, *, stop_error=False, native_health=None):
    from chimera_sparks import lifecycle
    calls = []
    def systemctl(argv, **kwargs):
        calls.append(argv)
        if "stop" in argv and stop_error:
            raise OSError("test cleanup failure")
        return SimpleNamespace(returncode=3 if "is-active" in argv else 0)
    def start(*args, **kwargs):
        calls.append(["start-api"])
    def urlopen(url, **kwargs):
        if url.endswith("/health"):
            return response(HEALTH if native_health is None else native_health)
        assert url.endswith("/ready")
        return response(api_health)
    monkeypatch.setattr(head.subprocess, "run", systemctl)
    monkeypatch.setattr(lifecycle, "start_api", start)
    monkeypatch.setattr(head.urllib.request, "urlopen", urlopen)
    return calls


def test_start_requires_both_api_and_strict_native_readiness(head, payload, staged, monkeypatch):
    calls = fake_lifecycle(head, monkeypatch, {"ok": True, "native": HEALTH})
    row = head.start(payload)
    assert row["ready"] is True and row["unit"] == "ds1017-api"
    assert ["start-api"] in calls and not any("stop" in c for c in calls)


@pytest.mark.parametrize("failure", ["unready", "numeric_missing", "cleanup_exception"])
def test_api_failure_stops_only_the_unit_it_started(head, payload, staged, monkeypatch, failure):
    health = {"ok": False, "native": HEALTH} if failure != "numeric_missing" else {"ok": True, "native": {"status": "ok", "warming": False}}
    calls = fake_lifecycle(head, monkeypatch, health, stop_error=failure == "cleanup_exception")
    with pytest.raises(head.ApiStartFailure) as error:
        head.start(payload)
    row = error.value.result
    assert row["ready"] is False and row["stopped"] is (failure != "cleanup_exception")
    assert row["start_attempted"] is True
    assert [c for c in calls if "stop" in c] == [["systemctl", "--user", "stop", "ds1017-api"]]
    assert SERVING_KEY not in json.dumps(row) and ADMIN_KEY not in json.dumps(row)
    if failure == "cleanup_exception":
        assert row["cleanup_error"] == "OSError"


@pytest.mark.parametrize("refusal", [ValueError, RuntimeError])
def test_unconfirmed_or_refused_api_start_never_stops_an_unowned_unit(head, payload, staged, monkeypatch, refusal):
    from chimera_sparks import lifecycle
    calls = fake_lifecycle(head, monkeypatch, {"ok": True, "native": HEALTH})
    def fail(*args, **kwargs):
        raise refusal("test refusal containing " + SERVING_KEY)
    monkeypatch.setattr(lifecycle, "start_api", fail)
    with pytest.raises(head.ApiStartFailure) as error:
        head.start(payload)
    assert error.value.result["started"] is False and error.value.result["stopped"] is False
    assert not any("stop" in c for c in calls)
    assert SERVING_KEY not in json.dumps(error.value.result)


def test_bad_engine_never_stops_or_starts_any_api_unit(head, payload, staged, monkeypatch):
    before = contents(staged)
    payload.update(key=SERVING_KEY + "-pending", admin_key=ADMIN_KEY + "-pending")
    head.stage(payload)
    payload["manifest"]["engine"]["health_timeout"] = 2
    calls = fake_lifecycle(head, monkeypatch, {"ok": True, "native": HEALTH}, native_health={"status": "ok", "warming": False})
    with pytest.raises(RuntimeError, match="numeric_effort true"):
        head.start(payload)
    assert calls == [] and contents(staged) == before


@pytest.mark.parametrize("mismatch", [None, "missing_ledger", "config", "executable", "unit", "config_sha256", "changed_config"])
def test_active_api_restart_requires_matching_installation_and_lifecycle_ownership(head, payload, staged, monkeypatch, capsys, mismatch):
    from chimera_sparks import lifecycle
    calls, active = [], [False]
    before = contents(staged)
    replacing = [False]
    def systemctl(argv, **kwargs):
        calls.append(argv)
        if "is-active" in argv:
            return SimpleNamespace(returncode=0 if active[0] else 3)
        if "stop" in argv:
            assert contents(staged) == before  # Publish only after the owned API stops.
            replacing[0] = True
            active[0] = False
        elif argv[0] == "systemd-run":
            if replacing[0]:
                assert contents(staged) != before
            active[0] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(head.subprocess, "run", systemctl)
    path, raw, cfg = lifecycle.document(staged[0])
    lifecycle.start_api(path, raw, cfg, executable=head.sys.executable)
    record_path = Path(cfg.state_dir) / "installation.json"
    if mismatch == "missing_ledger":
        record_path.unlink()
    elif mismatch in ("config", "executable", "unit", "config_sha256"):
        record = json.loads(record_path.read_text())
        record[mismatch] = "test-another-installation-or-release"
        record_path.write_text(json.dumps(record))
    elif mismatch == "changed_config":
        payload["manifest"]["serving"]["max_concurrency"] = 8
    payload.update(key=SERVING_KEY + "-replacement", admin_key=ADMIN_KEY + "-replacement")
    head.stage(payload)
    calls.clear()
    monkeypatch.setattr(head.urllib.request, "urlopen", lambda url, **kw: response(HEALTH if url.endswith("/health") else {"ok": True, "native": HEALTH}))
    if mismatch is None:
        assert head.start(payload)["ready"] is True
        assert replacing[0] is True and active[0] is True
        assert [c for c in calls if "stop" in c] == [["systemctl", "--user", "stop", "ds1017-api"]]
        assert not any(head.pending_path(path).exists() for path in staged)
    else:
        payload["action"] = "start"
        monkeypatch.setattr(head.sys, "stdin", io.StringIO(json.dumps(payload)))
        assert head.main() == 1
        out = capsys.readouterr()
        assert len(out.out.splitlines()) == 1 and out.err == ""
        row = json.loads(out.out)
        assert row["stopped"] is False and "refusing to stop" in row["error"]
        assert not any("stop" in c or c[0] == "systemd-run" for c in calls)
        assert active[0] is True and contents(staged) == before
        assert SERVING_KEY not in out.out and ADMIN_KEY not in out.out


def test_pending_pair_publish_failure_restores_live_files_before_api_start(head, payload, staged, monkeypatch):
    before = contents(staged)
    payload.update(key=SERVING_KEY + "-replacement", admin_key=ADMIN_KEY + "-replacement")
    head.stage(payload)
    calls = fake_lifecycle(head, monkeypatch, {"ok": True, "native": HEALTH})
    replace, failed = head.os.replace, []
    def publish(source, destination):
        if Path(destination) == staged[0] and not failed:
            failed.append(True)
            raise OSError("test config publish failure")
        return replace(source, destination)
    monkeypatch.setattr(head.os, "replace", publish)
    with pytest.raises(head.ApiStartFailure) as error:
        head.start(payload)
    assert error.value.result["started"] is False
    assert contents(staged) == before
    assert not any("stop" in c or c == ["start-api"] for c in calls)


@pytest.mark.parametrize("failure", ["systemd_run", "installation_record", "cleanup"])
def test_lifecycle_start_failure_restores_exact_files_and_tracks_confirmed_launch(head, payload, staged, monkeypatch, failure):
    from chimera_sparks import lifecycle
    staged[0].chmod(0o640)
    staged[1].chmod(0o400)
    before = contents(staged)
    identities = [(path.stat().st_ino, path.stat().st_mode, path.stat().st_mtime_ns) for path in staged]
    payload.update(key=SERVING_KEY + "-replacement", admin_key=ADMIN_KEY + "-replacement")
    head.stage(payload)
    calls, active = [], [False]
    original_run, write_text = lifecycle.run, Path.write_text
    def systemctl(argv, **kwargs):
        calls.append(argv)
        if "is-active" in argv:
            return SimpleNamespace(returncode=0 if active[0] else 3, stdout="", stderr="")
        if argv[0] == "systemd-run":
            assert contents(staged) != before
            if failure == "systemd_run":
                raise head.subprocess.CalledProcessError(1, argv)
            active[0] = True
        if "stop" in argv:
            if failure == "cleanup":
                raise OSError("test cleanup failure")
            active[0] = False
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    def fail_record(path, *args, **kwargs):
        if path.name == "installation.json":
            raise OSError("test installation ledger write failure")
        return write_text(path, *args, **kwargs)
    monkeypatch.setattr(head.subprocess, "run", systemctl)
    monkeypatch.setattr(Path, "write_text", fail_record)
    monkeypatch.setattr(head.urllib.request, "urlopen", lambda *a, **kw: response(HEALTH))
    with pytest.raises(head.ApiStartFailure) as error:
        head.start(payload)
    row = error.value.result
    assert row["started"] is False and row["launched"] is (failure != "systemd_run")
    assert row["stopped"] is (failure == "installation_record")
    assert active[0] is (failure == "cleanup")
    assert contents(staged) == before
    assert [(path.stat().st_ino, path.stat().st_mode, path.stat().st_mtime_ns) for path in staged] == identities
    assert lifecycle.run is original_run
    stops = [command for command in calls if "stop" in command]
    assert stops == ([] if failure == "systemd_run" else [["systemctl", "--user", "stop", "ds1017-api"]])
    if failure == "cleanup":
        assert row["cleanup_error"] == "OSError"
    assert SERVING_KEY not in json.dumps(row) and ADMIN_KEY not in json.dumps(row)


def test_head_cli_returns_failure_metadata_without_arbitrary_error_text(head, payload, monkeypatch, capsys):
    payload["action"] = "stage"
    monkeypatch.setattr(head.sys, "stdin", io.StringIO(json.dumps(payload)))
    def fail(_):
        raise ValueError("test exception with " + SERVING_KEY)
    monkeypatch.setattr(head, "stage", fail)
    assert head.main() == 1
    out = capsys.readouterr()
    assert len(out.out.splitlines()) == 1 and out.err == ""
    row = json.loads(out.out)
    assert row["ok"] is False and "ValueError" in row["error"]
    assert SERVING_KEY not in out.out
