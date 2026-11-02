"""Offline start ordering, receipts and direct-engine warmup tests."""
import hashlib
import io
import itertools
import json
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_deploy import ROOT, d, recipe
from test_head_api import contents, head, payload, staged


@pytest.fixture
def starting(recipe, monkeypatch):
    events, statuses = [], []
    manifest = json.loads((ROOT / "release/manifest.json").read_text())
    units = {rank: recipe.deploy.unit_name(rank, "a" * 32) for rank in recipe.deploy.SPARKS}
    original_write = recipe.deploy.write_execution_receipt
    def write(execution, addresses, **result):
        statuses.append(result["status"])
        original_write(execution, addresses, **result)
    def engine(argv, *, ready_status):
        events.append("engine")
        assert argv == ["--manifest", str(ROOT / "release/manifest.json")]
        write("a" * 32, units, status=ready_status, affinity="unchanged")
        return 0
    def api(action, manifest, venv):
        events.append("api-" + action)
        return {"ok": True, "staged": action == "stage", "ready": action == "start", "unit": "ds1017-api", "version": "0.1.5"}
    def remote(host, code, data):
        assert host == "spark1"
        if code == recipe.INSTALL_API:
            events.append("install")
            return {}
        assert code == recipe.WARMUP
        events.append("warmup")
        return {"warmups": [{"completion_tokens": 16, "tokens_per_second": 20 + i, "rate_basis": "engine decode"} for i in range(3)]}
    monkeypatch.setattr(recipe, "mkdirs", lambda *a: events.append("directories"))
    monkeypatch.setattr(recipe, "rsync", lambda *a: events.append("source"))
    monkeypatch.setattr(recipe, "remote", remote)
    monkeypatch.setattr(recipe, "head_api", api)
    monkeypatch.setattr(recipe.deploy, "main", engine)
    monkeypatch.setattr(recipe.deploy, "write_execution_receipt", write)
    monkeypatch.setattr(recipe.deploy, "stop_execution", lambda *a: pytest.fail("API failure stops warm ranks"))
    return recipe, manifest, events, statuses


def test_start_stages_api_before_engine_and_only_finishes_after_warmups(starting, capsys):
    recipe, manifest, events, statuses = starting
    recipe.start(ROOT / "release/manifest.json", manifest)
    assert events == ["directories", "source", "install", "api-stage", "engine", "api-start", "warmup"]
    assert statuses == ["engine-ready", "ready"]
    receipt = json.loads(recipe.deploy.RECEIPT.read_text())
    assert receipt["status"] == "ready" and receipt["engine_status"] == "ready" and receipt["api"]["ready"] is True
    assert receipt["ranks_status"] == "running" and len(receipt["warmups"]) == 3
    assert capsys.readouterr().out.count("tokens/s (engine decode)") == 3


@pytest.mark.parametrize("failure", ["install", "config"])
def test_api_staging_failure_never_replaces_ranks_or_overwrites_existing_receipt(starting, monkeypatch, failure):
    recipe, manifest, events, statuses = starting
    recipe.deploy.RECEIPT.write_text('{"status":"ready","execution":"previous"}')
    before = recipe.deploy.RECEIPT.read_bytes()
    if failure == "install":
        def fail(*a, **kw):
            raise recipe.RecipeFailure("test install failure")
        monkeypatch.setattr(recipe, "remote", fail)
    else:
        monkeypatch.setattr(recipe, "head_api", lambda *a: {"ok": False, "staged": False})
    with pytest.raises(recipe.RecipeFailure):
        recipe.start(ROOT / "release/manifest.json", manifest)
    assert "engine" not in events and "api-start" not in events and "warmup" not in events
    assert statuses == [] and recipe.deploy.RECEIPT.read_bytes() == before


@pytest.mark.parametrize("failure", ["preflight", "deployment", "exit_code"])
def test_engine_failure_keeps_pending_api_changes_out_of_live_files(starting, head, payload, staged, monkeypatch, failure):
    recipe, manifest, events, statuses = starting
    recipe.deploy.RECEIPT.write_text('{"status":"ready","execution":"previous"}')
    receipt_before = recipe.deploy.RECEIPT.read_bytes()
    before = contents(staged)
    identities = [(path.stat().st_ino, path.stat().st_mode, path.stat().st_mtime_ns) for path in staged]
    payload.update(key=payload["key"] + "-replacement", admin_key=payload["admin_key"] + "-replacement")
    payload["settings"]["API_PORT"] = "19003"
    def api(action, *args):
        assert action == "stage"
        return head.stage(payload)
    def fail(*args, **kwargs):
        if failure == "exit_code":
            return 1
        raise FileNotFoundError("test engine " + failure + " failed")
    monkeypatch.setattr(recipe, "head_api", api)
    monkeypatch.setattr(recipe.deploy, "main", fail)
    with pytest.raises((FileNotFoundError, recipe.RecipeFailure)):
        recipe.start(ROOT / "release/manifest.json", manifest)
    assert contents(staged) == before
    assert [(path.stat().st_ino, path.stat().st_mode, path.stat().st_mtime_ns) for path in staged] == identities
    assert contents([head.pending_path(path) for path in staged]) != before
    assert recipe.deploy.RECEIPT.read_bytes() == receipt_before and statuses == []
    assert "api-start" not in events and "warmup" not in events


@pytest.mark.parametrize("stopped", [True, False])
def test_api_failure_records_failed_start_with_cleanup_outcome_and_running_ranks(starting, monkeypatch, stopped):
    recipe, manifest, events, statuses = starting
    original = recipe.head_api
    def api(action, *args):
        if action == "stage":
            return original(action, *args)
        return {"ok": False, "ready": False, "unit": "ds1017-api", "stopped": stopped,
                **({"cleanup_error": "TimeoutExpired"} if not stopped else {})}
    monkeypatch.setattr(recipe, "head_api", api)
    with pytest.raises(recipe.RecipeFailure, match="engine ranks left running") as error:
        recipe.start(ROOT / "release/manifest.json", manifest)
    assert "\n" not in str(error.value)
    assert ("API unit stopped" if stopped else "API stop not confirmed") in str(error.value)
    receipt = json.loads(recipe.deploy.RECEIPT.read_text())
    assert statuses == ["engine-ready", "failed"] and "warmup" not in events
    assert receipt["status"] == "failed" and receipt["phase"] == "api" and receipt["engine_status"] == "ready"
    assert receipt["ranks_status"] == "left-running" and receipt["api"]["stopped"] is stopped


def test_warmup_failure_never_leaves_a_successful_start_receipt(starting, monkeypatch):
    recipe, manifest, events, statuses = starting
    original = recipe.remote
    def remote(host, code, data):
        if code == recipe.WARMUP:
            raise recipe.RecipeFailure("test warmup failure")
        return original(host, code, data)
    monkeypatch.setattr(recipe, "remote", remote)
    with pytest.raises(recipe.RecipeFailure, match="warmup failed; API and engine ranks left running"):
        recipe.start(ROOT / "release/manifest.json", manifest)
    receipt = json.loads(recipe.deploy.RECEIPT.read_text())
    assert statuses == ["engine-ready", "failed"]
    assert receipt["phase"] == "warmup" and receipt["api"]["ready"] is True and receipt["ranks_status"] == "left-running"


def test_keys_only_travel_on_stage_stdin_not_argv_or_start(recipe, monkeypatch, capsys):
    serving_key, admin_key = "test-serving-placeholder-key", "test-admin-placeholder-key-2"
    monkeypatch.setenv("SPARK_API_KEY", serving_key)
    monkeypatch.setenv("SPARK_ADMIN_KEY", admin_key)
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, json.loads(kwargs["input"])))
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=0, stdout='{"ok":true,"staged":true}', stderr="")
    monkeypatch.setattr(recipe.subprocess, "run", run)
    for action in ("stage", "start"):
        assert recipe.head_api(action, {}, "/head/venv")["ok"] is True
    assert calls[0][1]["key"] == serving_key and calls[0][1]["admin_key"] == admin_key
    assert "key" not in calls[1][1] and "admin_key" not in calls[1][1]
    for argv, _ in calls:
        assert serving_key not in shlex.join(argv) and admin_key not in shlex.join(argv)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("stdout,code", [("", 255), ('{"ok":true,"ready":true}', 1), ("not json", 0), ("[]", 0)])
def test_unconfirmed_head_operation_is_never_success(recipe, monkeypatch, stdout, code):
    monkeypatch.setattr(recipe.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=code, stdout=stdout, stderr="private test diagnostics"))
    assert recipe.head_api("start", {}, "/head/venv")["ok"] is False


@pytest.mark.parametrize("native_metrics", [True, False])
def test_warmup_script_sends_three_greedy_non_thinking_requests_directly_to_engine(recipe, monkeypatch, capsys, native_metrics):
    requests = []
    data = {**recipe.deploy.SETTINGS, "model": "deepseek-v4.1-flash"}
    monkeypatch.setattr(recipe.sys, "argv", ["warmup", json.dumps(data)])
    import time
    import urllib.request
    ticks = itertools.count()
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    def urlopen(request, **kwargs):
        requests.append(request)
        assert request.full_url == "http://127.0.0.1:8003/v1/chat/completions"
        assert "Authorization" not in request.headers
        body = json.loads(request.data)
        assert body["max_tokens"] == 16 and body["temperature"] == 0 and body["reasoning_effort"] == "none"
        assert body["chat_template_kwargs"]["enable_thinking"] is False and body["stream"] is False
        result = {"usage": {"completion_tokens": 8}}
        if native_metrics:
            result["tensorfold"] = {"tokens_per_second": 24.5}
        return io.StringIO(json.dumps(result))
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    exec(compile(recipe.WARMUP, "warmup-probe", "exec"), {})
    result = json.loads(capsys.readouterr().out)
    assert len(requests) == len(result["warmups"]) == 3
    assert all(row["tokens_per_second"] == (24.5 if native_metrics else 8) for row in result["warmups"])
    assert all(row["rate_basis"] == ("engine decode" if native_metrics else "end-to-end wall time") for row in result["warmups"])


def test_transport_exception_is_reported_as_unconfirmed_not_a_success(recipe, monkeypatch):
    def fail(*a, **kw):
        raise OSError("test SSH process failure")
    monkeypatch.setattr(recipe.subprocess, "run", fail)
    row = recipe.head_api("start", {}, "/head/venv")
    assert row["ok"] is False and row["ready"] is False and "not confirmed" in row["error"]


def test_smoke_uses_a_serving_client_even_when_an_admin_was_configured_first(recipe, monkeypatch, tmp_path, capsys):
    import time
    import urllib.request
    config, credentials = tmp_path / "config.json", tmp_path / "credentials.env"
    config.write_text(json.dumps({"env_file": str(credentials), "api": {"clients": {"ops": "SPARK_ADMIN_KEY", "default": "SPARK_API_KEY"}, "admin_clients": ["ops"]}}))
    credentials.write_text("SPARK_ADMIN_KEY=test-admin-placeholder-key\nSPARK_API_KEY=test-serving-placeholder-key\n")
    credentials.chmod(0o600)
    data = {**recipe.deploy.SETTINGS, "API_CONFIG": str(config), "API_ENV_FILE": str(credentials)}
    monkeypatch.setattr(recipe.sys, "argv", ["smoke", json.dumps(data)])
    ticks = itertools.count()
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    def urlopen(request, **kwargs):
        assert request.get_header("Authorization") == "Bearer test-serving-placeholder-key"
        assert request.full_url == "http://127.0.0.1:18002/v1/chat/completions"
        return io.StringIO(json.dumps({"choices": [{"message": {"content": "deployment ready"}}], "usage": {"completion_tokens": 3}}))
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    exec(compile(recipe.SMOKE, "smoke-probe", "exec"), {})
    row = json.loads(capsys.readouterr().out)
    assert row["ok"] is True and row["completion_tokens"] == 3


@pytest.fixture
def rank_input(recipe, tmp_path, monkeypatch):
    root = tmp_path / "download with spaces"
    (root / "tp3").mkdir(parents=True)
    payload = struct.pack("<3e", 1., 2., -0.)
    shard = "model-00001-of-00001.safetensors"
    spec = {"dtype": "F16", "shape": [3], "data_offsets": [0, len(payload)]}
    header = json.dumps({"weight": spec}, separators=(",", ":")).encode()
    (root / shard).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": shard}}))
    (root / "tp3/rank0.keys").write_text("weight|None\n")
    (root / "plan.json").write_text("private plan is not an input")
    (root / "json.py").write_text("raise RuntimeError('model code imported')")
    index = [["weight|None", "float16", [3], 0, len(payload)]]
    artifact = payload + json.dumps(index, separators=(",", ":")).encode() + struct.pack("<Q", len(payload)) + b"TFDSRK01"
    expected = hashlib.sha256(artifact).hexdigest()
    recipe.deploy.SETTINGS.update(SPARK_ROOT=str(tmp_path / "spark"), MODEL_DIR=str(root),
                                 RANK_CACHE=str(tmp_path / "rank cache"), SPARK_PYTHON=sys.executable,
                                 HF_REPO="example/export", HF_REVISION="a" * 40)
    def mkdirs(host, paths):
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(recipe, "mkdirs", mkdirs)
    monkeypatch.setattr(recipe, "rsync", lambda source, host, dest: shutil.copyfile(source, dest))
    tools_root = recipe.stage_weight_tools("offline")
    for name in recipe.WEIGHT_TOOLS:
        assert (Path(tools_root) / name).read_bytes() == (ROOT / name).read_bytes()
    return {**recipe.deploy.SETTINGS, "rank": 0, "arm": "SYNTHETIC", "expected_sha256": expected, "tools_root": tools_root}


def run_split(recipe, data, monkeypatch, capfd):
    monkeypatch.setattr(sys, "argv", ["offline-split", json.dumps(data)])
    exec(compile(recipe.SPLIT, "offline-split", "exec"), {})
    return json.loads(capfd.readouterr().out)


def test_staged_split_builds_a_byte_identical_rank_then_skips_it(recipe, rank_input, monkeypatch, capfd):
    row = run_split(recipe, rank_input, monkeypatch, capfd)
    assert row["ok"] is True and row["status"] == "WRITTEN" and row["sha256"] == rank_input["expected_sha256"]
    path = Path(row["path"])
    assert list(path.parent.glob("rank*of3*.bin")) == [path]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rank_input["expected_sha256"]
    assert path.with_suffix(".partial").stat().st_ino == path.stat().st_ino
    plan = json.loads((Path(row["metadata"]) / "plan.json").read_bytes())
    receipt = json.loads((Path(row["metadata"]) / "source.receipt.json").read_bytes())
    assert set(plan["sources"]) == {"mia"} and receipt["verification"] == "local-snapshot"
    assert receipt["plan_sha256"] == hashlib.sha256((Path(row["metadata"]) / "plan.json").read_bytes()).hexdigest()
    assert "0/1 6 bytes" in (Path(row["metadata"]) / "prepare.log").read_text()
    assert json.loads(path.with_suffix(".receipt.json").read_bytes())["status"] == "WRITTEN_NOT_QUALIFIED"
    before = path.stat()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("matching rank was rebuilt"))
    skipped = run_split(recipe, rank_input, monkeypatch, capfd)
    assert skipped["ok"] is True and skipped["status"] == "SKIPPED" and skipped["path"] == str(path)
    assert path.stat() == before


def test_existing_rank_mismatch_fails_loudly_without_overwriting(recipe, rank_input, monkeypatch, capfd):
    cache = Path(rank_input["RANK_CACHE"])
    cache.mkdir()
    path = cache / "rank0of3-existing.bin"
    path.write_bytes(b"not this release")
    before = path.stat()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("mismatched rank was overwritten"))
    row = run_split(recipe, rank_input, monkeypatch, capfd)
    assert row["ok"] is False and "rank 0 SHA-256 mismatch" in row["error"]
    assert rank_input["expected_sha256"] in row["error"] and hashlib.sha256(path.read_bytes()).hexdigest() in row["error"]
    assert path.stat() == before


def test_new_rank_must_match_the_manifest_not_just_its_writer_receipt(recipe, rank_input, monkeypatch, capfd):
    row = run_split(recipe, {**rank_input, "expected_sha256": "0" * 64}, monkeypatch, capfd)
    assert row["ok"] is False and "SHA-256 mismatch" in row["error"] and rank_input["expected_sha256"] in row["error"]


@pytest.mark.parametrize("names", [["rank1of3-wrong.bin"], ["rank0of3-a.bin", "rank0of3-b.bin"], ["rank0of3.bin"]])
def test_split_cannot_succeed_with_rank_placement_that_deploy_will_reject(recipe, rank_input, monkeypatch, capfd, names):
    cache = Path(rank_input["RANK_CACHE"])
    cache.mkdir()
    for name in names:
        (cache / name).write_bytes(b"rank")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("invalid rank placement was modified"))
    row = run_split(recipe, rank_input, monkeypatch, capfd)
    assert row["ok"] is False and "fix rank placement" in row["error"]
    assert sorted(path.name for path in cache.iterdir()) == sorted(names)


@pytest.mark.parametrize("where", ["inside", "same", "around"])
def test_split_refuses_a_rank_cache_that_overlaps_the_weights_folder(recipe, rank_input, monkeypatch, capfd, where):
    root = Path(rank_input["MODEL_DIR"])
    cache = {"inside": root / "ranks", "same": root, "around": root.parent}[where]
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("overlapping cache was written"))
    row = run_split(recipe, {**rank_input, "RANK_CACHE": str(cache)}, monkeypatch, capfd)
    assert row["ok"] is False and "must be separate folders" in row["error"]
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*")) == before


def test_interrupted_rank_is_not_overwritten_and_has_a_durable_failure_log(recipe, rank_input, monkeypatch, capfd):
    cache = Path(rank_input["RANK_CACHE"])
    cache.mkdir()
    target = cache / ("rank0of3-" + rank_input["expected_sha256"][:12] + ".bin")
    partial = target.with_suffix(".partial")
    partial.write_bytes(b"keep interrupted work")
    row = run_split(recipe, rank_input, monkeypatch, capfd)
    assert row["ok"] is False and "rank_cache.py failed" in row["error"] and "prepare.log" in row["error"]
    assert partial.read_bytes() == b"keep interrupted work" and not target.exists()
    log = next((cache / "preparation").iterdir()) / "prepare.log"
    assert "FileExistsError" in log.read_text()


@pytest.mark.parametrize("bad", [None, [], ["0" * 64], ["x" * 64] * 3])
def test_prepare_rejects_missing_or_invalid_rank_digests_before_remote_writes(recipe, monkeypatch, bad):
    recipe.deploy.SETTINGS.update(HF_REPO="example/export", HF_REVISION="a" * 40)
    manifest = json.loads((ROOT / "release/manifest.json").read_bytes())
    manifest["weights"]["rank_sha256"] = bad
    monkeypatch.setattr(recipe, "remote", lambda *a, **kw: pytest.fail("invalid manifest reached a Spark"))
    with pytest.raises(recipe.RecipeFailure, match="three rank SHA-256"):
        recipe.prepare(manifest)


@pytest.mark.parametrize("result", [{"ok": False, "error": "rank 0 SHA-256 mismatch: test"}, {"ok": True, "sha256": "0" * 64}])
def test_prepare_stops_after_an_unconfirmed_or_mismatched_rank(recipe, monkeypatch, result):
    recipe.deploy.SETTINGS.update(HF_REPO="example/export", HF_REVISION="a" * 40)
    manifest = json.loads((ROOT / "release/manifest.json").read_bytes())
    calls = []
    def remote(host, code, data):
        calls.append(host)
        if code == recipe.ASSETS:
            return {"ok": True, "kit": "SKIPPED", "engram": "SKIPPED"}
        return {"downloaded": data["HF_REPO"], "revision": data["HF_REVISION"]} if code == recipe.DOWNLOAD else result
    monkeypatch.setattr(recipe, "remote", remote)
    monkeypatch.setattr(recipe, "stage_weight_tools", lambda host: "/tools")
    monkeypatch.setattr(recipe, "stage_engine", lambda: pytest.fail("failed preparation staged an engine"))
    with pytest.raises(recipe.RecipeFailure, match="SHA-256 mismatch"):
        recipe.prepare(manifest)
    assert calls == ["spark1", "spark1", "spark1"]


def test_prepare_fails_on_asset_errors_before_downloading_weights(recipe, monkeypatch):
    recipe.deploy.SETTINGS.update(HF_REPO="example/export", HF_REVISION="a" * 40)
    seen = []
    def remote(host, code, data):
        seen.append(code)
        return {"ok": False, "error": "runtime assets need 70.0 GB free under /srv"}
    monkeypatch.setattr(recipe, "remote", remote)
    monkeypatch.setattr(recipe, "stage_weight_tools", lambda host: "/tools")
    with pytest.raises(recipe.RecipeFailure, match="GB free.*docs/runtime-assets.md"):
        recipe.prepare(json.loads((ROOT / "release/manifest.json").read_text()))
    assert seen == [recipe.ASSETS]


def test_prepare_rejects_unknown_engram_source(recipe, monkeypatch):
    recipe.deploy.SETTINGS.update(HF_REPO="example/export", HF_REVISION="a" * 40, ENGRAM_SOURCE="mirror")
    monkeypatch.setattr(recipe, "remote", lambda *a, **kw: pytest.fail("bad ENGRAM_SOURCE reached a Spark"))
    with pytest.raises(recipe.RecipeFailure, match="ENGRAM_SOURCE"):
        recipe.prepare(json.loads((ROOT / "release/manifest.json").read_text()))


def test_assets_script_checks_disk_before_any_download(recipe, monkeypatch, tmp_path, capfd):
    tools = tmp_path / "tools"
    (tools / "release").mkdir(parents=True)
    shutil.copyfile(ROOT / "release/runtime-assets.json", tools / "release/runtime-assets.json")
    data = {**recipe.deploy.SETTINGS, "KIT_DIR": str(tmp_path / "kit"), "ENGRAM_DIR": str(tmp_path / "engram"),
            "SPARK_ROOT": str(tmp_path), "rank": 0, "tools_root": str(tools), "kit_bytes": 1 << 60, "engram_bytes": 1}
    monkeypatch.setattr(shutil, "which", lambda name: pytest.fail("download attempted without disk space"))
    monkeypatch.setattr(sys, "argv", ["offline-assets", json.dumps(data)])
    exec(compile(recipe.ASSETS, "offline-assets", "exec"), {})
    row = json.loads(capfd.readouterr().out)
    assert row["ok"] is False and "GB free" in row["error"]
