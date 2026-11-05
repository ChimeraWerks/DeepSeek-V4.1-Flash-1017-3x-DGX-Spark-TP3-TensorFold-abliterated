"""Focused, transport-isolated deploy/recipe regression tests. Never contact a Spark."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def d(monkeypatch, tmp_path):
    for key, value in {"SPARK_HOSTS": "spark1 spark2 spark3", "SPARK_USER": "ubuntu", "SSH_KEY": "/keys/test-identity",
                       "MASTER_ADDR": "10.0.0.1", "NCCL_IB_ADDR_RANGE": "10.0.0.0/24", "SPARK_ROOT": "/srv/ds1017",
                       "DEPLOY_RECEIPT": str(tmp_path / "execution.json"), "OPTIONAL_CONTROLLER_URL": ""}.items():
        monkeypatch.setenv(key, value)
    mod = load("deploy", "deploy.py")
    mod.parse_args([])
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    for key in ("SPARK_API_KEY", "SPARK_ADMIN_KEY"):
        monkeypatch.delenv(key, raising=False)
    return mod


class FakeSsh:
    def __init__(self, invocation="0123456789abcdef0123456789abcdef", active=True, guard_active=True):
        self.calls, self.invocation, self.active, self.guard_active = [], invocation, active, guard_active

    def __call__(self, host, cmd, check=True):
        self.calls.append((host, cmd))
        if cmd.startswith("systemctl --user is-active"):
            ok = self.guard_active if cmd.split()[-1].endswith("-guard") else self.active
            return "active" if ok else "failed"
        if "-p InvocationID" in cmd:
            return self.invocation
        return ""


def test_unit_names_are_single_use_execution_addresses(d):
    pattern = r"[a-z][a-z0-9-]{0,63}-g[0-9a-f]{32}(?:\.service)?"
    assert pattern == d.GUARD_UNIT.pattern
    unit = d.unit_name(0, "a" * 32)
    assert re.fullmatch(pattern, unit) and unit == "ds1017-rank0-g" + "a" * 32
    with pytest.raises(ValueError):
        d.unit_name(0, "not-hex")


def test_launch_binds_each_guard_to_its_active_rank(d, monkeypatch):
    ssh = FakeSsh()
    monkeypatch.setattr(d, "run_ssh", ssh)
    units = {r: d.unit_name(r, "b" * 32) for r in d.SPARKS}
    d.launch_cluster(units)
    starts = [c for _, c in ssh.calls if c.startswith("systemd-run")]
    ranks = [c for c in starts if "-guard" not in shlex.split(c)[2]]
    guards = [c for c in starts if "-guard" in shlex.split(c)[2]]
    assert [shlex.split(c)[2] for c in ranks] == [f"--unit={units[r]}" for r in (1, 2, 0)]
    for rank, cmd in zip((1, 2, 0), guards):
        unit = units[rank]
        assert f"--unit={unit}-guard" in cmd and f"--property=Requisite={unit}.service" in cmd
        assert f"--property=After={unit}.service" in cmd and f"--unit {unit} " in cmd
        assert f"--state-dir {d.DEPLOY_ROOT}/guard/{unit}/state" in cmd
    assert all("--property=Restart=no" in c and "--collect" in c for c in ranks)
    seq = [c for _, c in ssh.calls]
    for rank in d.SPARKS:
        assert seq.index(f"systemctl --user is-active {units[rank]}") < next(i for i, c in enumerate(seq) if f"--unit={units[rank]}-guard" in c)


def test_guard_refused_when_rank_not_active(d, monkeypatch):
    monkeypatch.setattr(d, "run_ssh", FakeSsh(active=False))
    with pytest.raises(RuntimeError, match="not active"):
        d.start_guard("h", d.unit_name(1, "c" * 32))


def test_mark_ready_touches_marker_for_current_invocation(d, monkeypatch):
    ssh = FakeSsh()
    monkeypatch.setattr(d, "run_ssh", ssh)
    units = {r: d.unit_name(r, "d" * 32) for r in d.SPARKS}
    d.mark_ready(units)
    assert [c for _, c in ssh.calls if c.startswith("touch")] == [f"touch {d.DEPLOY_ROOT}/guard/{units[r]}/state/ready-{ssh.invocation}" for r in d.SPARKS]


def test_mark_ready_fails_without_invocation_or_live_guard(d, monkeypatch):
    units = {r: d.unit_name(r, "e" * 32) for r in d.SPARKS}
    monkeypatch.setattr(d, "run_ssh", FakeSsh(invocation=""))
    with pytest.raises(RuntimeError, match="InvocationID"):
        d.mark_ready(units)
    monkeypatch.setattr(d, "run_ssh", FakeSsh(guard_active=False))
    with pytest.raises(RuntimeError, match="after arming"):
        d.mark_ready(units)


def test_reapply_affinity(d, monkeypatch):
    assert d.reapply_affinity() == "unchanged"  # no private controller dependency
    monkeypatch.setattr(d, "OPTIONAL_CONTROLLER", "http://controller.invalid")

    def api(desired, modes):
        def call(path, body=None, timeout=60):
            if body is None:
                return {"desired": desired}
            assert body == {"mode": desired, "confirmed": True}
            return {"nodes": {f"n{i}": {"mode": m} for i, m in enumerate(modes)}}
        return call

    monkeypatch.setattr(d, "_optional_controller", api(None, []))
    assert d.reapply_affinity() is None
    monkeypatch.setattr(d, "_optional_controller", api("efficiency", ["efficiency"] * 3))
    assert d.reapply_affinity() == "efficiency"
    monkeypatch.setattr(d, "_optional_controller", api("efficiency", ["efficiency", "all", "efficiency"]))
    assert d.reapply_affinity() is None


def test_variant_flags_reach_the_launch(d, monkeypatch):
    ssh = FakeSsh()
    monkeypatch.setattr(d, "run_ssh", ssh)
    # Explicit nonproduction overrides exercise CLI plumbing, not release defaults.
    d.parse_args(["--label", "ds1017-alt", "--bin-dir", "/x/bin", "--rank-cache", "/x/rc", "--pool", "6000000", "--context", "524288",
                  "--streams", "8", "--graph-cap", "64", "--head-arg=--top-k=30"])
    units = {r: d.unit_name(r, "f" * 32) for r in d.SPARKS}
    d.launch_cluster(units)
    head = next(c for _, c in ssh.calls if f"--unit={units[0]} " in c)
    worker = next(c for _, c in ssh.calls if f"--unit={units[1]} " in c)
    assert "/x/bin/tensorfold-native serve" in head and "--context 524288" in head and "--parallel 8" in head
    assert '"TF_DS_POOL_TOKENS": "6000000"' in head and '"TF_DS_RANK_CACHE": "/x/rc"' in head
    assert "/x/bin/tf-dsv41-lanes" in worker and " /x/rc " in worker and "--pool 6000000" in worker
    assert "--streams 8" in worker and "--graph-cap 64" in worker and '"TF_DS_GRAPH_CAP": "64"' in head
    assert "logs/ds1017-alt-rank1.log" in worker and "--top-k=30" in head
    with pytest.raises(SystemExit):
        d.parse_args(["--label", "Bad Label"])


@pytest.mark.parametrize("key", ["ENGINE_COMMIT", "STREAMS", "POOL_TOKENS", "CONTEXT_TOKENS",
                                 "GRAPH_CAP", "SPILL_GIB", "MEMORY_RESERVE_GIB", "TOP_K"])
def test_cluster_release_assertions_cannot_silently_fork_defaults(d, monkeypatch, key):
    monkeypatch.setenv(key, "999")
    with pytest.raises(SystemExit):
        d.parse_args([])


def test_matching_cluster_release_assertions_preserve_production(d, monkeypatch):
    for key, value in {"ENGINE_COMMIT": "30a9647", "STREAMS": "16", "POOL_TOKENS": "8000000",
                       "CONTEXT_TOKENS": "1048576", "GRAPH_CAP": "128", "SPILL_GIB": "256",
                       "MEMORY_RESERVE_GIB": "4", "TOP_K": "20"}.items():
        monkeypatch.setenv(key, value)
    d.parse_args([])
    assert (d.V["streams"], d.V["pool"], d.V["context"], d.V["spill_gib"]) == (16, 8000000, 1048576, 256)


def test_defaults_match_production(d):
    assert (d.V["pool"], d.V["context"], d.V["streams"], d.V["graph_cap"]) == (8000000, 1048576, 16, 128)
    assert d.V["memory_reserve_gib"] == 4 and d.V["spill_gib"] == 256
    for rank in d.SPARKS:
        env = json.loads(d.get_env_json(rank))
        assert env["TF_DS_PREFILL_OWNED"] == env["TF_DS_DIRECT_GATHER"] == env["TF_DS_SAMPLE_CAND"] == "1"
        assert env["TF_DS_SPILL_DIR"] == "/srv/ds1017/spill" and env["TENSORFOLD_MEMORY_RESERVE_GIB"] == "4"
        assert env["TF_TP_MASTER"] == "10.0.0.1"
    assert "--spill-gib=256" in d.rank_command(0, d.unit_name(0, "a" * 32))
    assert "--top-k=20" in d.rank_command(0, d.unit_name(0, "a" * 32))


def test_extra_env_reaches_every_rank(d, monkeypatch):
    d.parse_args(["--env", "TF_DS_ENGRAM_KEEPALIVE_MS=2000"])
    assert all(json.loads(d.get_env_json(r))["TF_DS_ENGRAM_KEEPALIVE_MS"] == "2000" for r in d.SPARKS)
    for bad in ("lower=1", "NOEQUALS", "K=it's", "API_KEY=not-allowed", "SPARK_ADMIN_KEY=test-placeholder-denied"):
        with pytest.raises(SystemExit):
            d.parse_args(["--env", bad])


@pytest.mark.parametrize("code", [1, 255])
def test_preflight_ssh_failure_never_stops_production(d, monkeypatch, code):
    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=code if "ubuntu@spark2" in cmd else 0, stdout="", stderr="connection failed")
    monkeypatch.setattr(d.subprocess, "run", run)
    stopped = []
    monkeypatch.setattr(d, "stop_old_services", lambda: stopped.append(True))
    monkeypatch.setattr(d, "launch_cluster", lambda _: pytest.fail("launch after failed preflight"))
    with pytest.raises(RuntimeError, match="SSH command failed on spark2"):
        d.main([])
    assert stopped == []


def test_preflight_reports_missing_paths(d, monkeypatch):
    monkeypatch.setattr(d, "run_ssh", lambda *args, **kwargs: "/missing/rank.bin")
    with pytest.raises(FileNotFoundError, match="spark1 preflight"):
        d.preflight()


def test_preflight_points_missing_kit_files_to_prepare(d, monkeypatch):
    seen = []
    def ssh(host, command, check=True):
        seen.append(json.loads(shlex.split(command)[-1])["paths"])
        return "missing: " + d.SETTINGS["KIT_DIR"] + "/aot/aot.json"
    monkeypatch.setattr(d, "run_ssh", ssh)
    with pytest.raises(FileNotFoundError, match="aot.json.*prepare.sh; see docs/runtime-assets.md"):
        d.preflight()
    assert d.SETTINGS["ENGRAM_DIR"] + "/SOURCE.json" in seen[0]
    assert d.SETTINGS["KIT_DIR"] + "/auxiliary/experts_ragged.cubin" in seen[0]


@pytest.mark.parametrize("kind", ["regular", "symlink", "dangling_symlink", "directory"])
def test_remote_rank_listing_rejects_nonregular_entries_before_stopping(d, monkeypatch, tmp_path, kind):
    cache = tmp_path / "rank-cache"
    cache.mkdir()
    entry = cache / "rank0of3-test.bin"
    if kind == "regular":
        entry.write_bytes(b"test rank")
    elif kind == "directory":
        entry.mkdir()
    else:
        target = tmp_path / "rank-target.bin"
        if kind == "symlink":
            target.write_bytes(b"test rank")
        entry.symlink_to(target)
    def ssh(host, command, check=True):
        argv = shlex.split(command)
        assert argv[:3] == ["python3", "-I", "-c"]
        data = json.loads(argv[4])
        data.update(paths=[], cache=str(cache))
        output = io.StringIO()
        with monkeypatch.context() as context, redirect_stdout(output):
            context.setattr(d.sys, "argv", ["remote-probe", json.dumps(data)])
            exec(compile(argv[3], "remote-preflight", "exec"), {})
        return output.getvalue().strip()
    monkeypatch.setattr(d, "run_ssh", ssh)
    monkeypatch.setattr(d, "SPARKS", {0: "spark1"})
    monkeypatch.setattr(d, "stop_old_services", lambda: pytest.fail("invalid rank entry stopped a service"))
    monkeypatch.setattr(d, "launch_cluster", lambda _: pytest.fail("invalid rank entry launched a rank"))
    if kind == "regular":
        d.preflight()
    else:
        with pytest.raises(FileNotFoundError, match="symlink rank entries" if "symlink" in kind else "exactly one rank file"):
            d.main([])
        assert entry.is_symlink() if "symlink" in kind else entry.is_dir()
        assert not d.RECEIPT.exists()


@pytest.mark.parametrize("flag", ["--sync", "--production", "--settings"])
def test_retired_options_are_rejected_without_network(d, monkeypatch, flag):
    monkeypatch.setattr(d.subprocess, "run", lambda *a, **kw: pytest.fail("retired option contacts hosts"))
    with pytest.raises(SystemExit) as error:
        d.main([flag])
    assert error.value.code == 2


@pytest.fixture
def deployment(d, monkeypatch):
    ssh = FakeSsh()
    monkeypatch.setattr(d, "run_ssh", ssh)
    monkeypatch.setattr(d, "preflight", lambda: None)
    monkeypatch.setattr(d, "stop_old_services", lambda: None)
    monkeypatch.setattr(d, "poll_health", lambda **kw: True)
    monkeypatch.setattr(d, "reapply_affinity", lambda: "efficiency")
    return d, ssh


@pytest.mark.parametrize("failure", ["launch", "timeout", "arming", "interrupt"])
def test_failed_deployment_stops_its_execution(deployment, monkeypatch, failure):
    d, ssh = deployment
    if failure in ("launch", "interrupt"):
        original = d.run_ssh
        def fail_second_rank(host, cmd, check=True):
            if cmd.startswith("systemd-run") and "-rank2-" in cmd:
                if failure == "interrupt":
                    raise KeyboardInterrupt()
                raise RuntimeError("rank2 failed")
            return original(host, cmd, check)
        monkeypatch.setattr(d, "run_ssh", fail_second_rank)
    elif failure == "timeout":
        monkeypatch.setattr(d, "poll_health", lambda **kw: False)
    else:
        ssh.guard_active = False
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        d.main([])
    receipt = json.loads(d.RECEIPT.read_text())
    assert receipt["status"] == "failed" and receipt["cleanup_errors"] == {}
    stops = [(h, c) for h, c in ssh.calls if c.startswith("systemctl --user stop")]
    assert stops == [(d.SPARKS[int(rank)], f"systemctl --user stop {unit}.service {unit}-guard.service") for rank, unit in receipt["units"].items()]
    assert all(receipt["execution"] in cmd and "*" not in cmd for _, cmd in stops)


def test_cleanup_failure_preserves_original_error_and_tries_other_hosts(deployment, monkeypatch):
    d, ssh = deployment
    original = d.run_ssh
    def fail_stop(host, cmd, check=True):
        result = original(host, cmd, check)
        if host == "spark1" and cmd.startswith("systemctl --user stop"):
            raise RuntimeError("spark1 unreachable during cleanup")
        return result
    monkeypatch.setattr(d, "run_ssh", fail_stop)
    monkeypatch.setattr(d, "poll_health", lambda **kw: False)
    with pytest.raises(RuntimeError, match="Health check timed out"):
        d.main([])
    receipt = json.loads(d.RECEIPT.read_text())
    assert receipt["cleanup_errors"] == {"spark1": "spark1 unreachable during cleanup"}
    assert len([c for _, c in ssh.calls if c.startswith("systemctl --user stop")]) == 3


@pytest.mark.parametrize("affinity", ["efficiency", None])
def test_success_keeps_armed_ranks_running(deployment, monkeypatch, affinity):
    d, ssh = deployment
    monkeypatch.setattr(d, "reapply_affinity", lambda: affinity)
    assert d.main([]) == (2 if affinity is None else 0)
    assert not any(cmd.startswith("systemctl --user stop") for _, cmd in ssh.calls)
    receipt = json.loads(d.RECEIPT.read_text())
    assert receipt["status"] == "ready" and receipt["affinity"] == affinity


def test_dry_run_does_not_contact_hosts_or_write_receipts(d, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("network or write during dry run")
    monkeypatch.setattr(d.subprocess, "run", forbidden)
    monkeypatch.setattr(d.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(d, "write_execution_receipt", forbidden)
    assert d.main(["--dry-run"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert [row["host"] for row in rows] == ["spark1", "spark2", "spark3"]
    assert all(row["command"] == d.rank_command(row["rank"], row["unit"]) for row in rows)
    assert not d.RECEIPT.exists()


def test_failed_preflight_does_not_replace_existing_receipt(deployment, monkeypatch):
    d, ssh = deployment
    d.RECEIPT.write_text('{"status":"ready","execution":"previous"}')
    before = d.RECEIPT.read_bytes()
    def fail():
        raise RuntimeError("missing artifact")
    monkeypatch.setattr(d, "preflight", fail)
    with pytest.raises(RuntimeError, match="missing artifact"):
        d.main([])
    assert d.RECEIPT.read_bytes() == before and ssh.calls == []


def test_custom_topology_paths_ports_and_shell_quoting(d, monkeypatch):
    for key, value in {"SPARK_USER": "operator", "SPARK_ROOT": "~/models with spaces", "SPARK_HOME": "/users/operator",
                       "MASTER_ADDR": "10.20.1.7", "TP_PORT": "30303", "ENGINE_PORT": "9003", "NCCL_IB_ADDR_RANGE": "10.20.0.0/16"}.items():
        monkeypatch.setenv(key, value)
    d.parse_args([])
    for rank in d.SPARKS:
        argv = shlex.split(d.rank_command(rank, d.unit_name(rank, "a" * 32)))
        env = json.loads(argv[argv.index("--environment-json") + 1])
        assert env["TF_TP_MASTER"] == "10.20.1.7" and env["TF_TP_PORT"] == "30303"
        assert env["TF_DS_RANK_CACHE"] == "/users/operator/models with spaces/ranks"
        assert env["NCCL_IB_ADDR_RANGE"] == "10.20.0.0/16"
        assert "/users/operator/models with spaces/bin/" in " ".join(argv)
        if not rank:
            assert argv[argv.index("--port") + 1] == "9003"
    assert "operator@spark1" in d.ssh_argv("spark1", "true")


def test_old_units_only_matches_the_selected_recipe(d, monkeypatch):
    calls = []
    unit = d.unit_name(0, "a" * 32) + ".service"
    def ssh(host, cmd, check=True):
        calls.append(cmd)
        return unit + " loaded active running description"
    monkeypatch.setattr(d, "run_ssh", ssh)
    assert d.old_units("spark1") == [unit]
    assert shlex.split(calls[0])[-1] == "ds1017-rank[012]-g*.service"
    monkeypatch.setattr(d, "run_ssh", lambda *a, **kw: "some-other-api.service loaded active running")
    with pytest.raises(RuntimeError, match="unexpected rank unit"):
        d.old_units("spark1")


def test_manifest_and_example_contract(d):
    manifest = json.loads((ROOT / "release/manifest.json").read_text())
    assert manifest["engine"]["commit"] == "30a9647" and manifest["engine"]["tensorfold_base"] == "cebcf1e"
    assert manifest["serving"]["version"] == "0.1.5"
    assert manifest["weights"]["model_revision"] == "1bae0e1088993b9b82de08f59de5be5a34e42170"
    # The published Hugging Face weights commit, verified on the hub after upload; never a placeholder.
    assert manifest["weights"]["revision"] == "2fa62d701ce4a98abafd0be51ad2048dc88b8180"
    assert manifest["weights"]["repo"] == "ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated"
    assert manifest["sampler"]["effort_map"] == {"minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 88, "max": 100}
    example = (ROOT / "cluster.env.example").read_text()
    for key in ("SPARK_HOSTS", "SPARK_USER", "SSH_KEY", "MASTER_ADDR", "NCCL_IB_ADDR_RANGE", "SPARK_ROOT", "HF_REPO", "HF_REVISION", "ENGINE_PORT", "API_PORT", "SPARK_API_KEY", "SPARK_ADMIN_KEY"):
        assert re.search(r"^" + key + "=", example, re.M)


@pytest.fixture
def recipe(d, monkeypatch):
    mod = load("recipe", "recipe.py")
    monkeypatch.setattr(mod, "deploy", d)
    return mod


@pytest.mark.parametrize("manifest_name", ["manifest.json", "history/v1.1.json", "history/v1.json"])
def test_prepare_downloads_and_verifies_each_rank_before_staging_engine(recipe, monkeypatch, manifest_name):
    recipe.deploy.SETTINGS.update(HF_REPO="example/export", HF_REVISION="a" * 40)
    manifest = json.loads((ROOT / "release" / manifest_name).read_text())
    calls = []
    def remote(host, code, data):
        calls.append((host, {recipe.DOWNLOAD: "download", recipe.ASSETS: "assets"}.get(code, "split")))
        if code == recipe.ASSETS:
            assert data["tools_root"] == "/srv/ds1017/prepare-tools" and data["ENGRAM_SOURCE"] == "ranged"
            assert 67e9 < data["engram_bytes"] < 68e9 and data["rank"] == int(host[-1]) - 1
            return {"ok": True, "kit": "WRITTEN", "engram": "WRITTEN"}
        if code == recipe.DOWNLOAD:
            return {"downloaded": data["HF_REPO"], "revision": data["HF_REVISION"]}
        assert code == recipe.SPLIT and data["rank"] == int(host[-1]) - 1
        assert data["expected_sha256"] == manifest["weights"]["rank_sha256"][data["rank"]]
        assert data["tools_root"] == "/srv/ds1017/prepare-tools" and data["arm"] == manifest["weights"].get("artifact_arm", manifest["weights"]["arm"])
        return {"ok": True, "status": "WRITTEN", "path": "/rank.bin", "sha256": data["expected_sha256"], "elapsed_seconds": 1}
    monkeypatch.setattr(recipe, "remote", remote)
    monkeypatch.setattr(recipe, "stage_weight_tools", lambda host: calls.append((host, "tools")) or "/srv/ds1017/prepare-tools")
    monkeypatch.setattr(recipe, "stage_engine", lambda: calls.append(("all", "stage")))
    recipe.prepare(manifest)
    assert calls == [(host, action) for host in ("spark1", "spark2", "spark3") for action in ("tools", "assets", "download", "split")] + [("all", "stage")]


def test_missing_export_revision_never_downloads(recipe, monkeypatch):
    recipe.deploy.SETTINGS.update(HF_REPO="example/export", HF_REVISION="REPLACE_WITH_WEIGHTS_EXPORT_COMMIT")
    monkeypatch.setattr(recipe, "remote", lambda *a, **kw: pytest.fail("unpublished export downloaded"))
    with pytest.raises(recipe.RecipeFailure, match="set HF_REVISION to the pinned weights commit"):
        recipe.prepare(json.loads((ROOT / "release/manifest.json").read_text()))


def test_remote_script_syntax(recipe):
    for name in ("PREFLIGHT", "STATUS", "SMOKE", "DOWNLOAD", "SPLIT", "ASSETS", "INSTALL_API", "WARMUP"):
        compile(getattr(recipe, name), name, "exec")


def test_read_only_actions_only_use_remote_probes(recipe, monkeypatch):
    seen = []
    def remote(host, code, data):
        seen.append(code)
        if code == recipe.SMOKE:
            return {"completion_tokens": 3, "elapsed_seconds": 1, "tokens_per_second": 3}
        return {"problems": []}
    monkeypatch.setattr(recipe, "remote", remote)
    monkeypatch.setattr(recipe.deploy, "main", lambda *a: pytest.fail("read-only action launches engine"))
    monkeypatch.setattr(recipe, "stage_engine", lambda: pytest.fail("read-only action stages files"))
    for action in ("preflight", "status", "smoke-test"):
        assert recipe.main([action]) == 0
    assert seen == [recipe.PREFLIGHT] * 3 + [recipe.STATUS] * 3 + [recipe.SMOKE]


def test_scripts_work_from_any_directory_and_fail_once_without_env(tmp_path):
    result = subprocess.run([str(ROOT / "scripts/status.sh")], cwd=tmp_path,
                            env={"PATH": "/usr/bin:/bin", "CLUSTER_ENV": str(tmp_path / "absent")}, capture_output=True, text=True)
    assert result.returncode == 1 and result.stdout.startswith("Plan:")
    assert len(result.stderr.splitlines()) == 1 and result.stderr.startswith("FAIL:")


def test_generated_api_configuration_validates_with_the_actual_package(d, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "serving/src"))
    from chimera_sparks.config import Config
    head = load("head_api", "head_api.py")
    manifest = json.loads((ROOT / "release/manifest.json").read_text())
    raw = head.configuration(d.SETTINGS, manifest)
    cfg = Config(**raw["api"])
    cfg.validate()
    assert "cluster" not in raw  # API attach mode must not also launch/supervise fixed-name engine ranks
    assert cfg.sampling["top_k"] == 20 and cfg.max_concurrency == cfg.engine_streams == 16
    assert cfg.context_tokens == 1048576 and cfg.reasoning_default == "high"
    assert cfg.effort_map == manifest["sampler"]["effort_map"]
    assert raw["env_file"] == "/srv/ds1017/api/credentials.env"
    assert cfg.clients == {"default": "SPARK_API_KEY", "admin": "SPARK_ADMIN_KEY"}
    assert cfg.admin_clients == ["admin"]


def test_graph_cap_override_cannot_split_head_and_worker_caps(d):
    d.parse_args(["--graph-cap", "64", "--env", "TF_DS_GRAPH_CAP=32"])
    assert all(json.loads(d.get_env_json(r))["TF_DS_GRAPH_CAP"] == "64" for r in d.SPARKS)
    assert "--graph-cap 64" in d.rank_command(1, d.unit_name(1, "a" * 32))


def test_receipt_failure_cleans_up_execution(deployment, monkeypatch):
    d, ssh = deployment
    def fail(*args, **kwargs):
        raise OSError("receipt disk full")
    monkeypatch.setattr(d, "write_execution_receipt", fail)
    with pytest.raises(OSError, match="receipt disk full"):
        d.main([])
    assert len([cmd for _, cmd in ssh.calls if cmd.startswith("systemctl --user stop")]) == 3


def test_status_does_not_return_execstart_or_engine_environment(recipe):
    assert "line.split()[:4]" in recipe.STATUS and "'sub'" in recipe.STATUS
    assert "expected one active memory guard" in recipe.STATUS


def test_stop_stops_api_before_only_rank_addresses(recipe, monkeypatch):
    calls = []
    monkeypatch.setattr(recipe, "remote", lambda host, code, data: calls.append(("api", data["unit"])))
    units = {rank: recipe.deploy.unit_name(rank, "a" * 32) + ".service" for rank in range(3)}
    monkeypatch.setattr(recipe.deploy, "old_units", lambda host: [units[recipe.deploy.SETTINGS["SPARK_HOSTS"].split().index(host)]])
    monkeypatch.setattr(recipe.deploy, "run_ssh", lambda host, cmd: calls.append((host, cmd)))
    recipe.stop()
    assert calls[0] == ("api", "ds1017-api")
    assert [cmd for host, cmd in calls[1:]] == ["systemctl --user stop " + unit for unit in units.values()]


def test_v12_default_labels_and_configuration_are_rank_consistent_and_history_is_explicit(d):
    historical = json.loads((ROOT / "release/history/v1.json").read_text())
    current = json.loads(d.DEFAULT_MANIFEST.read_text())
    assert current["weights"]["arm"] == "chimera-dsv41-flash-1017-abliterated-exl3"
    assert current["weights"]["artifact_arm"] == historical["weights"]["arm"]
    current_envs = [json.loads(d.get_env_json(rank)) for rank in range(3)]
    assert all(env["TF_DS_ENGINE_ID"] == "chimera-tensorfold-tp3-30a9647" for env in current_envs)
    assert all(env["TF_DS_RECIPE_PATCHES"] == "chimera-tensorfold-patches-20261103" for env in current_envs)
    assert all(env["TF_DS_POOL_TOKENS"] == "8000000" for env in current_envs)
    assert all(env["TF_DS_PREFILL_COMPUTE_ROWS"] == "8192" for env in current_envs)
    assert all(env["TF_DS_PREFILL_SLICE"] == "5" for env in current_envs)
    assert d.SETTINGS["ENGINE_LABEL"] == d.SETTINGS["API_INSTALLATION"] == "ds1017"
    d.parse_args(["--manifest", str(ROOT / "release/history/v1.json")])
    for rank in range(3):
        old = json.loads(d.get_env_json(rank))
        assert old["TF_DS_ENGINE_ID"] == historical["labels"]["TF_DS_ENGINE_ID"]
        assert old["TF_DS_RECIPE_PATCHES"] == historical["labels"]["TF_DS_RECIPE_PATCHES"]
        assert old["TF_DS_POOL_TOKENS"] == "10000000"
        assert "TF_DS_PREFILL_COMPUTE_ROWS" not in old
    d.parse_args(["--manifest", str(ROOT / "release/history/v1.1.json")])
    for rank in range(3):
        previous = json.loads(d.get_env_json(rank))
        assert previous["TF_DS_ENGINE_ID"] == "chimera-tensorfold-tp3-14803a3"
        assert previous["TF_DS_RECIPE_PATCHES"] == "chimera-tensorfold-patches-20261102"
        assert previous["TF_DS_PREFILL_COMPUTE_ROWS"] == "8192"
        assert "TF_DS_PREFILL_SLICE" not in previous
