"""Read-only rank-layout and numeric health probes, with no Spark connection."""
import json
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_deploy import d


@pytest.mark.parametrize("health,ready", [({"status": "ok", "warming": False, "numeric_effort": True}, True),
                                         ({"status": "ok", "warming": False}, False),
                                         ({"status": "ok", "warming": False, "numeric_effort": False}, False),
                                         ({"status": "ok", "warming": True, "numeric_effort": True}, False),
                                         ({"warming": False, "numeric_effort": True}, False),
                                         ({"status": "ok", "numeric_effort": True}, False)])
def test_engine_readiness_requires_all_three_explicit_fields(d, monkeypatch, health, ready):
    clock = [0]
    monkeypatch.setattr(d, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda n: clock.__setitem__(0, clock[0] + n)))
    monkeypatch.setattr(d, "health", lambda: health)
    assert d.poll_health(timeout=2) is ready


@pytest.mark.parametrize("files,exists,valid", [([], False, False), ([], True, False),
                                               (["rank0of3-model.bin"], True, True),
                                               (["rank0of3.bin"], True, False),
                                               (["rank1of3-model.bin"], True, False),
                                               (["rank0of3-old.bin", "rank0of3-new.bin"], True, False),
                                               (["rank0of3-model.bin", "rank0of3.bin"], True, False),
                                               (["rank0of3-model.bin", "rank1of3-model.bin"], True, False)])
def test_preflight_requires_exactly_one_correct_rank_file(d, monkeypatch, tmp_path, files, exists, valid):
    cache = tmp_path / "ranks"
    if exists:
        cache.mkdir()
    for name in files:
        (cache / name).write_bytes(b"test rank placeholder")
    monkeypatch.setattr(d, "SPARKS", {0: "spark1"})
    d.V["rank_cache"] = str(cache)
    # Run the exact generated preflight on disposable files only. Required engine
    # assets are covered by the separate missing-path regression, not this probe.
    def local_probe(host, command):
        argv = shlex.split(command)
        data = json.loads(argv[-1])
        assert data["cache"] == str(cache)
        data["paths"] = []
        result = subprocess.run([sys.executable, "-I", "-c", argv[argv.index("-c") + 1], json.dumps(data)],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    monkeypatch.setattr(d, "run_ssh", local_probe)
    if valid:
        d.preflight()
    else:
        with pytest.raises(FileNotFoundError, match="expected exactly one rank file.*rank placement belongs to scripts/prepare.sh"):
            d.preflight()
    assert sorted(p.name for p in cache.iterdir()) == sorted(files) if exists else not cache.exists()


def test_preflight_rejects_two_hardlinks_even_when_the_bytes_are_shared(d, monkeypatch, tmp_path):
    cache = tmp_path / "ranks"
    cache.mkdir()
    source = cache / "rank0of3-model.bin"
    source.write_bytes(b"test shared rank")
    (cache / "rank0of3-other.bin").hardlink_to(source)
    monkeypatch.setattr(d, "SPARKS", {0: "spark1"})
    d.V["rank_cache"] = str(cache)
    def local_probe(host, command):
        argv = shlex.split(command)
        data = json.loads(argv[-1])
        data["paths"] = []
        result = subprocess.run([sys.executable, "-I", "-c", argv[argv.index("-c") + 1], json.dumps(data)],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    monkeypatch.setattr(d, "run_ssh", local_probe)
    with pytest.raises(FileNotFoundError, match="found 2"):
        d.preflight()
