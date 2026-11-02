"""Offline positive and negative privacy/export gates; no network."""
import importlib.util
import io
from pathlib import Path
import tarfile

SPEC = importlib.util.spec_from_file_location("audit_public", Path(__file__).with_name("audit_public.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

def test_operational_identifier_classes_fail():
    examples = [b"/home/" + b"chimera/private", b"192" + b".168.1.234",
                b"host=" + b"spark" + b"-abcd", b"shiki" + b"gami",
                b"clau" + b"de", b"asa-" + b"dot", b"dot" + b"pc",
                b"ultra" + b"code", b".co" + b"dex/sessions",
                b"spark" + b"Kitchen", b"mia-tp3-" + b"deploy",
                b"/GLM" + b"53/private"]
    for data in examples:
        assert audit.inspect_bytes("README.md", data), data

def test_upstream_architecture_and_repository_credit_are_not_private_hosts():
    assert not audit.inspect_bytes("CREDITS.md", b"GLM53 source architecture; https://example.org/dgx-spark-memory-saver")

def archive(name, data):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return output.getvalue()

def test_archives_are_scanned_and_cannot_escape():
    assert audit.inspect_bytes("kernels.tar.gz", archive("kernel.cu", b"/home/" + b"chimera/private"))
    assert audit.inspect_bytes("kernels.tar.gz", archive("../escape", b"safe"))
    assert not audit.inspect_bytes("kernels.tar.gz", archive("kernel.cu", b"public source"))

def test_allowlist_symlinks_runtime_and_weights_fail(tmp_path):
    (tmp_path / "README.md").write_text("public release")
    assert audit.audit(tmp_path) == (1, [])
    (tmp_path / "private.txt").write_text("outside allowlist")
    assert audit.audit(tmp_path)[1]
    (tmp_path / "private.txt").unlink()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/link").symlink_to("/etc/passwd")
    assert any("symlinks" in error for error in audit.audit(tmp_path)[1])
    (tmp_path / "docs/link").unlink()
    (tmp_path / "docs/model.safetensors").write_bytes(b"model")
    assert any("checkpoint" in error for error in audit.audit(tmp_path)[1])

def test_scanner_and_test_source_have_no_self_exemption():
    for path in (Path(__file__), Path(__file__).with_name("audit_public.py")):
        assert not audit.inspect_bytes("tools/release/" + path.name, path.read_bytes())

def test_private_paths_and_forbidden_archive_members_fail():
    private = "spark" + "-private.txt"
    assert audit.inspect_bytes("docs/" + private, b"safe")
    for name in (private, ".git/config", "model.safetensors", "__pycache__/cache.pyc"):
        assert audit.inspect_bytes("kernel.tar.gz", archive(name, b"safe")), name
    for name in ("weights.zip", "unknown.7z", "unknown.gz"):
        assert audit.inspect_bytes(name, b"unknown")

def test_archive_symlink_is_rejected():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    assert audit.inspect_bytes("kernel.tar.gz", output.getvalue())

def test_uppercase_archive_suffix_is_scanned():
    assert audit.inspect_bytes("payload.TGZ", archive("safe.txt", b"clau" + b"de"))
    assert audit.inspect_bytes("payload.TAR.GZ", archive("model.safetensors", b"model"))
