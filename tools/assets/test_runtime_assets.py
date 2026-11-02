"""Offline kit staging and Engram fetch tests with synthetic pins; no network."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import urllib.request

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("runtime_assets", HERE / "runtime_assets.py")
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)
ROOT = HERE.parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tar_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network in an offline test"))


@pytest.fixture
def kit_inputs(tmp_path):
    rows, half = 3, 2
    rope = {name: bytes([i]) * (rows * half * 4) for i, name in enumerate(ra.ROPE)}
    base = {"aot/aot.json": b"base", "aot/cubins/a.cubin": b"A", "engram.json": b"E", "engram-token-map.json": b"[0]",
            "rope.json": b"R", "models/m/config.json": b"{}"}
    payload = {"aot/aot.json": b"merged", "aot/cubins/b.cubin": b"B", "auxiliary/experts_ragged.cubin": b"X"}
    runtime, prefill = tmp_path / "dl", tmp_path / "prefill"
    (runtime / "runtime/rope").mkdir(parents=True)
    prefill.mkdir()
    archive = tar_bytes(base)
    (runtime / "runtime/tensorfold-kernels.tar.gz").write_bytes(archive)
    files = {"runtime/tensorfold-kernels.tar.gz": {"bytes": len(archive), "sha256": sha(archive)}}
    for name, data in rope.items():
        (runtime / "runtime/rope" / name).write_bytes(data)
        files["runtime/rope/" + name] = {"bytes": len(data), "sha256": sha(data)}
    packed = tar_bytes(payload)
    (prefill / "payload.tar.gz").write_bytes(packed)
    (prefill / "MANIFEST.json").write_text(json.dumps({
        "archive": "payload.tar.gz", "archive_sha256": sha(packed), "base_aot_sha256": sha(b"base"),
        "base_files": {"aot/cubins/a.cubin": sha(b"A")}, "payload_files": {k: sha(v) for k, v in payload.items()}}))
    pins = {"kit": {"repo": "o/r", "revision": "c" * 40, "files": files, "rope": {"rows": rows, "half": half},
                    "staged_files": [{"sha256": sha(base[k]), "path": k} for k in ("engram.json", "engram-token-map.json", "rope.json")]}}
    return runtime, prefill, pins


def test_kit_stages_verifies_and_skips(kit_inputs, tmp_path):
    runtime, prefill, pins = kit_inputs
    out = tmp_path / "kit"
    assert ra.stage_kit(runtime, prefill, out, pins)["status"] == "WRITTEN"
    assert (out / "aot/aot.json").read_bytes() == b"merged"
    assert (out / "auxiliary/experts_ragged.cubin").read_bytes() == b"X"
    assert ra.check_kit(out, pins, prefill, full=True) == []
    assert ra.stage_kit(runtime, prefill, out, pins)["status"] == "SKIPPED"
    assert not list(tmp_path.glob("kit.partial-*"))


def test_kit_refuses_tampered_download_and_foreign_kit(kit_inputs, tmp_path):
    runtime, prefill, pins = kit_inputs
    (runtime / "runtime/rope" / ra.ROPE[0]).write_bytes(b"\7" * 24)
    with pytest.raises(ra.AssetError, match="download it again"):
        ra.stage_kit(runtime, prefill, tmp_path / "kit", pins)
    assert not (tmp_path / "kit").exists() and not list(tmp_path.glob("kit.partial-*"))
    foreign = tmp_path / "other"
    foreign.mkdir()
    with pytest.raises(ra.AssetError, match="move it aside"):
        ra.stage_kit(runtime, prefill, foreign, pins)


def test_kit_check_names_missing_and_changed_files(kit_inputs, tmp_path):
    runtime, prefill, pins = kit_inputs
    out = tmp_path / "kit"
    ra.stage_kit(runtime, prefill, out, pins)
    (out / "engram-token-map.json").write_bytes(b"[1]")
    (out / "aot/cubins/b.cubin").unlink()
    (out / "aot/cubins/a.cubin").write_bytes(b"Z")
    quick = ra.check_kit(out, pins, prefill)
    assert any("differs" in p and "engram-token-map" in p for p in quick)
    assert any("missing" in p and "b.cubin" in p for p in quick)
    assert not any("a.cubin" in p for p in quick)
    assert any("a.cubin" in p for p in ra.check_kit(out, pins, prefill, full=True))


def test_safe_extract_refuses_links_and_parent_paths(tmp_path):
    for member in (tarfile.TarInfo("../x"), tarfile.TarInfo("link")):
        if member.name == "link":
            member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            tar.addfile(member, io.BytesIO(b""))
        archive = tmp_path / "a.tar"
        archive.write_bytes(buffer.getvalue())
        with pytest.raises(ra.AssetError, match="refusing"):
            ra.safe_extract(archive, tmp_path / "out")


@pytest.fixture
def engram_pins():
    size = 4096
    blob = {name: bytes((i * 7 + j) % 251 for j in range(size)) for i, name in enumerate(("a.safetensors", "b.safetensors"))}
    files = {}
    for name, data in blob.items():
        ranges = [{"offset": 1000, "bytes": 1500, "sha256": sha(data[1000:2500]), "tensor": "w", "row_range": [0, 1]},
                  {"offset": 3000, "bytes": 200, "sha256": sha(data[3000:3200]), "tensor": "s", "row_range": [0, 1]}]
        files[name] = {"size": size, "header_bytes": 16, "header_sha256": sha(data[:16]), "ranges": ranges}
    source = {"files": files, "rank": 1, "world": 3, "original_revision": "r"}
    serving = {"files": {name: {"size": size} for name in files}, "rank": 1, "world": 3, "original_revision": "r"}
    raw = (json.dumps(serving, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ranks = [None, {"source_json": source, "serving_source": serving, "serving_source_sha256": sha(raw)}, None]
    return blob, {"engram": {"repo": "deepseek-ai/m", "revision": "r", "ranks": ranks}}, raw


class Server:
    """Serves byte ranges of blob; optionally cuts one response short to exercise resume."""

    def __init__(self, blob, cut_after=None):
        self.blob, self.cut_after, self.requests = blob, cut_after, []

    def __call__(self, url, start, end):
        name = url.rsplit("/", 1)[1]
        assert url == f"https://huggingface.co/deepseek-ai/m/resolve/r/{name}"
        self.requests.append((name, start, end))
        data = self.blob[name][start:end]
        if self.cut_after is not None and len(data) > self.cut_after:
            data, self.cut_after = data[:self.cut_after], None
        return io.BytesIO(data)


def test_ranged_fetch_writes_sparse_owned_bytes_and_marker(engram_pins, tmp_path, monkeypatch):
    blob, pins, raw = engram_pins
    monkeypatch.setattr(ra.time, "sleep", lambda s: None)
    server = Server(blob, cut_after=700)
    result = ra.fetch_engram(tmp_path / "e", 1, pins, opener=server, log=lambda m: None)
    assert result["status"] == "WRITTEN"
    assert (tmp_path / "e/SOURCE.json").read_bytes() == raw
    for name, data in blob.items():
        got = (tmp_path / "e" / name).read_bytes()
        assert len(got) == len(data)
        assert got[:16] == data[:16] and got[1000:2500] == data[1000:2500] and got[3000:3200] == data[3000:3200]
        assert got[2500:3000] == b"\0" * 500
    # The cut response resumed from where it stopped instead of starting over.
    assert ("a.safetensors", 1700, 2500) in server.requests
    assert ra.check_engram(tmp_path / "e", 1, pins, full=True) == []
    assert not (tmp_path / "e/.fetch-progress.json").exists()
    assert ra.fetch_engram(tmp_path / "e", 1, pins, opener=server, log=lambda m: None)["status"] == "SKIPPED"


def test_ranged_fetch_rejects_wrong_bytes(engram_pins, tmp_path):
    blob, pins, _ = engram_pins
    bad = {name: data[:1000] + b"\xff" + data[1001:] for name, data in blob.items()}
    with pytest.raises(ra.AssetError, match="differ from the pin"):
        ra.fetch_engram(tmp_path / "e", 1, pins, opener=Server(bad), log=lambda m: None)
    assert not (tmp_path / "e/SOURCE.json").exists()


def test_full_shards_are_checked_before_the_marker(engram_pins, tmp_path):
    blob, pins, raw = engram_pins
    directory = tmp_path / "e"
    directory.mkdir()
    for name, data in blob.items():
        (directory / name).write_bytes(data)
    assert ra.adopt_full(directory, 1, pins)["mode"] == "full"
    assert (directory / "SOURCE.json").read_bytes() == raw
    (directory / "b.safetensors").write_bytes(blob["b.safetensors"][:3001] + b"\xff" + blob["b.safetensors"][3002:])
    (directory / "SOURCE.json").unlink()
    with pytest.raises(ra.AssetError, match="differ from DeepSeek"):
        ra.adopt_full(directory, 1, pins)
    assert ra.check_engram(directory, 1, pins) == ["missing " + str(directory / "SOURCE.json")]


def test_stray_file_fails_the_check_because_it_changes_the_shared_listing(engram_pins, tmp_path):
    blob, pins, _ = engram_pins
    ra.fetch_engram(tmp_path / "e", 1, pins, opener=Server(blob), log=lambda m: None)
    (tmp_path / "e/rank1.sparse.tar").write_bytes(b"x")
    assert ra.check_engram(tmp_path / "e", 1, pins) == ["unexpected file " + str(tmp_path / "e/rank1.sparse.tar")]


def test_every_rank_lists_identical_engram_names_and_sizes():
    # rank_identity.listing feeds a common fingerprint field, so a size difference stops the ranks at startup.
    pins = ra.load_pins()
    rows = [(len(ra.source_json_bytes(pin)), sorted((n, m["size"]) for n, m in pin["serving_source"]["files"].items()))
            for pin in pins["engram"]["ranks"]]
    assert rows[0] == rows[1] == rows[2]


def test_published_pins_describe_each_rank():
    pins = ra.load_pins()
    for rank, pin in enumerate(pins["engram"]["ranks"]):
        source = pin["source_json"]
        serving = pin["serving_source"]
        assert (serving["rank"], serving["world"], serving["original_revision"]) == (rank, 3, pins["engram"]["revision"])
        assert serving["original_manifest_sha256"] == pin["source_json_sha256"]
        assert (source["rank"], source["world"], source["original_revision"]) == (rank, 3, pins["engram"]["revision"])
        assert set(source["files"]) == set(pins["engram"]["files"])
        for name, meta in source["files"].items():
            assert meta["size"] == pins["engram"]["files"][name]["bytes"]
            assert meta["original_full_source_sha256"] == pins["engram"]["files"][name]["sha256"]
        ra.source_json_bytes(pin)
    owned = sum(length for pin in pins["engram"]["ranks"] for _, _, length, _ in ra.owned(pin))
    assert 202e9 < owned < 203e9
