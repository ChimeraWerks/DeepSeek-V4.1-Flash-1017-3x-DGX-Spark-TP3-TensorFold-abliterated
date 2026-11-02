"""Offline mutation tests for release defaults, argv/env and the tracked-text gate."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_consistency", ROOT / "tools/check_release_consistency.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def release_copy(tmp_path):
    files = ["serving/src/chimera_sparks/assets/profiles/v1.1.json", "release/manifest.json", "cluster.env.example", "serving/engine-scope.json", "serving/pyproject.toml",
             "serving/src/chimera_sparks/config.py", "serving/src/chimera_sparks/lifecycle.py",
             "serving/src/chimera_sparks/assets/native_launch.py", "serving/src/chimera_sparks/assets/profiles/v1.json",
             "release/runtime-assets.json", "release/prefill-source-delta.json",
             *("engine/tensorfold/zig/kernels/prefill_128/" + name for name in
               ("SOURCE.json", "MANIFEST.json", "kernels.py", "experts_ragged.cu", "payload.tar.gz"))]
    for name in files:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", *files], check=True, capture_output=True)
    return tmp_path


def change_json(root, name, field, value):
    path = root / name
    data = json.loads(path.read_text())
    current = data
    for key in field[:-1]:
        current = current[key]
    current[field[-1]] = value
    path.write_text(json.dumps(data, indent=2) + "\n")


def track(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    subprocess.run(["git", "-C", str(root), "add", "--", name], check=True, capture_output=True)


def test_current_release_is_consistent():
    assert checker.check(ROOT) == []


@pytest.mark.parametrize("field,value", [("parallel", 8), ("pool_tokens", 6000000), ("context_tokens", 524288),
                                         ("graph_cap", 64), ("spill_gib", 64), ("memory_reserve_gib", 8),
                                         ("top_k", 30), ("engine_commit", "badc0de"),
                                         ("hub_repo", "example/not-the-served-pack"), ("arm", "wrong-pack"),
                                         ("bits_per_weight", 4.0)])
def test_default_profile_disagreement_fails(release_copy, field, value):
    change_json(release_copy, "serving/src/chimera_sparks/assets/profiles/v1.1.json", [field], value)
    assert any("native profile " + field in error for error in checker.check(release_copy))


@pytest.mark.parametrize("old,new,label", [
    ("context_tokens: int = 1048576", "context_tokens: int = 524288", "Config.context_tokens"),
    ("engine_streams: int = 16", "engine_streams: int = 8", "Config.engine_streams"),
    ('"top_k": 20', '"top_k": 30', "Config.sampling"),
])
def test_serving_defaults_disagreement_fails(release_copy, old, new, label):
    path = release_copy / "serving/src/chimera_sparks/config.py"
    path.write_text(path.read_text().replace(old, new))
    assert any(label in error for error in checker.check(release_copy))


@pytest.mark.parametrize("field,key", list(checker.RUNTIME_FIELDS.items()))
def test_cluster_example_disagreement_fails(release_copy, field, key):
    manifest = json.loads((release_copy / "release/manifest.json").read_text())
    path = release_copy / "cluster.env.example"
    path.write_text(path.read_text().replace(f"{key}='{manifest['engine'][field]}'", f"{key}='999'"))
    assert any("cluster.env.example " + key in error for error in checker.check(release_copy))


@pytest.mark.parametrize("field,value", [("streams", 8), ("pool", 6000000), ("context", 524288),
                                         ("top_k", 30), ("commit", "badc0de")])
def test_manifest_disagreement_fails(release_copy, field, value):
    change_json(release_copy, "release/manifest.json", ["engine", field], value)
    assert checker.check(release_copy)


def test_implicit_profile_and_scope_cannot_drift(release_copy):
    path = release_copy / "serving/src/chimera_sparks/lifecycle.py"
    path.write_text(path.read_text().replace('cluster.get("profile", "v1.1")', 'cluster.get("profile", "unbundled")'))
    assert any("could not be checked" in error for error in checker.check(release_copy))
    path.write_bytes((ROOT / "serving/src/chimera_sparks/lifecycle.py").read_bytes())
    change_json(release_copy, "serving/engine-scope.json", ["required_native_profile"], "unbundled")
    assert any("engine-scope profile" in error for error in checker.check(release_copy))


def test_correct_json_is_not_enough_when_launcher_drops_a_flag(release_copy):
    path = release_copy / "serving/src/chimera_sparks/assets/native_launch.py"
    old = "'--graph-cap', str(profile['graph_cap'])"
    assert old in path.read_text()
    path.write_text(path.read_text().replace(old, "'--graph-cap', '64'"))
    assert any("rank 1 argv" in error for error in checker.check(release_copy))


@pytest.mark.parametrize("marker", ["TF" + "351", "full" + "40", 'parallel' + '": 12',
                                    'parallel' + '":12', "700" + "0000", "401" + "408",
                                    "12" + " streams", "C12" + " production"])
def test_stale_markers_in_any_tracked_text_fail(release_copy, marker):
    track(release_copy, "docs/regression.txt", "Active release: " + marker + "\n")
    assert any("docs/regression.txt:1: stale release marker" in error for error in checker.check(release_copy))


def test_exemptions_are_narrow_and_do_not_hide_current_release_claims(release_copy):
    marker = "TF" + "351"
    track(release_copy, "engine/tensorfold/upstream.txt", marker)
    track(release_copy, "serving/src/chimera_sparks/laky/upstream.txt", marker)
    track(release_copy, "benchmarks/table.md", "| Mode | 1 stream | 12" + " streams | 16 streams |\n")
    track(release_copy, "docs/bench.md", "Benchmark concurrency: 12" + " streams\n")
    track(release_copy, "CHANGELOG.md", "## History\nPreviously used " + marker + "\n")
    assert checker.check(release_copy) == []
    track(release_copy, "CHANGELOG.md", "## Current release\nServed pack: " + marker + "\n")
    assert any("CHANGELOG.md:2" in error for error in checker.check(release_copy))
    track(release_copy, "CHANGELOG.md", "## History\nPreviously used " + marker + "\n")
    track(release_copy, "benchmarks/table.md", "Production pack: " + marker + "\n")
    assert any("benchmarks/table.md:1" in error for error in checker.check(release_copy))


def test_cli_is_read_only_and_reports_failure(release_copy):
    command = [sys.executable, "-I", str(ROOT / "tools/check_release_consistency.py"), "--root", str(release_copy)]
    good = subprocess.run(command, capture_output=True, text=True, check=False)
    assert good.returncode == 0, good.stderr
    before = (release_copy / "release/manifest.json").read_bytes()
    track(release_copy, "docs/current.txt", "Active pack: TF" + "351")
    bad = subprocess.run(command, capture_output=True, text=True, check=False)
    assert bad.returncode == 1 and "stale release marker" in bad.stderr
    assert (release_copy / "release/manifest.json").read_bytes() == before


@pytest.mark.parametrize("name,field,value", [
    ("release/manifest.json", ["weights", "artifact_arm"], "wrong-pack"),
    ("release/manifest.json", ["labels", "TF_DS_RECIPE_PATCHES"], "old-patches"),
    ("serving/src/chimera_sparks/assets/profiles/v1.1.json", ["artifact_arm"], "wrong-pack"),
    ("serving/src/chimera_sparks/assets/profiles/v1.1.json", ["rank_cache_subdir"], "renamed"),
])
def test_current_artifact_identity_cannot_drift(release_copy, name, field, value):
    change_json(release_copy, name, field, value)
    assert checker.check(release_copy)


@pytest.mark.parametrize("name,field,value,label", [
    ("release/prefill-source-delta.json", ["committed_sha256"], "0" * 64, "differs from release/prefill-source-delta.json"),
    ("release/prefill-source-delta.json", ["declared_sha256"], "0" * 64, "declared hash differs from SOURCE.json"),
    ("engine/tensorfold/zig/kernels/prefill_128/MANIFEST.json", ["payload_files", "auxiliary/experts_ragged.cubin"], "0" * 64,
     "payload members differ"),
    ("release/runtime-assets.json", ["kit", "revision"], "REPLACE_AFTER_UPLOAD", "must pin a commit"),
    ("release/runtime-assets.json", ["engram", "ranks", 1, "source_json", "rank"], 2, "rank 1 pin is inconsistent"),
])
def test_prefill_and_runtime_asset_hashes_must_agree(release_copy, name, field, value, label):
    change_json(release_copy, name, field, value)
    assert any(label in error for error in checker.check(release_copy))
