"""Offline tests: prepare stages the pinned arm64 PDFium beside tensorfold-native, checked by SHA-256."""
import hashlib
import io
import json
import tarfile

import pytest

from test_deploy import ROOT, d, recipe  # noqa: F401  (the fixtures)


def archive_of(library, extra=()):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in [("lib/libpdfium.so", library), *extra]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def pin_for(archive, library):
    return {"release": "chromium/7725", "archive": "pdfium-linux-arm64.tgz", "url": "https://example.invalid/pdfium.tgz",
            "archive_bytes": len(archive), "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "member": "lib/libpdfium.so", "library_bytes": len(library), "library_sha256": hashlib.sha256(library).hexdigest()}


def test_the_release_pin_names_the_arm64_chromium_7725_build():
    pin = json.loads((ROOT / "release/runtime-assets.json").read_text())["pdfium"]
    assert (pin["repo"], pin["release"], pin["archive"], pin["member"]) == (
        "bblanchon/pdfium-binaries", "chromium/7725", "pdfium-linux-arm64.tgz", "lib/libpdfium.so")
    assert pin["archive_sha256"] == "a0dab89cdbee57832292034eaad27e010f84dea1c367e1e0535730933ead2315"
    assert pin["library_sha256"] == "5cfe82b0f16d33c0117d9df10790ba83b47d0b10deb74b1e7f540e607452a752"
    assert pin["url"].endswith("/chromium%2F7725/pdfium-linux-arm64.tgz")


def test_a_fetched_archive_is_checked_then_its_library_staged(recipe, monkeypatch, tmp_path):
    library = b"\x7fELF pdfium"
    archive = archive_of(library, [("include/fpdfview.h", b"/* header */")])
    pin = pin_for(archive, library)
    fetched = []
    monkeypatch.setattr(recipe, "fetch", lambda url, limit: fetched.append((url, limit)) or archive)
    lib = recipe.pdfium_library(tmp_path, pin)
    assert lib == tmp_path / "lib/libpdfium.so" and lib.read_bytes() == library
    assert fetched == [(pin["url"], len(archive))]
    assert (tmp_path / "downloads/pdfium-linux-arm64.tgz").read_bytes() == archive
    assert not (tmp_path / "include").exists()  # only the library leaves the archive
    # staged once: the pinned file is reused without another download
    monkeypatch.setattr(recipe, "fetch", lambda *a: pytest.fail("a staged library was downloaded again"))
    assert recipe.pdfium_library(tmp_path, pin) == lib
    lib.unlink()
    assert recipe.pdfium_library(tmp_path, pin).read_bytes() == library  # from the kept archive


def test_a_file_that_differs_from_its_pin_is_never_staged(recipe, monkeypatch, tmp_path):
    library = b"\x7fELF pdfium"
    archive = archive_of(library)
    pin = pin_for(archive, library)
    monkeypatch.setattr(recipe, "fetch", lambda *a: archive_of(b"\x7fELF other"))
    with pytest.raises(recipe.RecipeFailure, match="pinned SHA-256"):
        recipe.pdfium_library(tmp_path, pin)
    assert not (tmp_path / "lib").exists() and not (tmp_path / "downloads").exists()
    swapped = dict(pin, library_sha256="0" * 64)
    monkeypatch.setattr(recipe, "fetch", lambda *a: archive)
    with pytest.raises(recipe.RecipeFailure, match="pinned SHA-256"):
        recipe.pdfium_library(tmp_path, swapped)
    assert not (tmp_path / "lib").exists()
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/libpdfium.so").write_bytes(b"someone else's build")
    with pytest.raises(recipe.RecipeFailure, match="move it aside"):
        recipe.pdfium_library(tmp_path, pin)


def test_without_network_the_engine_stages_without_pdfium(recipe, monkeypatch, tmp_path, capsys):
    library = b"\x7fELF pdfium"
    pin = pin_for(archive_of(library), library)
    def offline(url, limit):
        raise OSError("no route to host")
    monkeypatch.setattr(recipe, "fetch", offline)
    assert recipe.pdfium_library(tmp_path, pin) is None
    assert "HTTP 501" in capsys.readouterr().err


def test_stage_engine_sends_pdfium_into_bin_dir_beside_the_server(recipe, monkeypatch, tmp_path):
    for name in ("native/bin/tensorfold-native", "bin/tf-dsv41-lanes", "bin/tf-dsv41-check"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"binary")
    monkeypatch.setenv("OUT", str(tmp_path))
    library = tmp_path / "lib/libpdfium.so"
    monkeypatch.setattr(recipe, "pdfium_library", lambda out: library)
    sent = []
    monkeypatch.setattr(recipe, "mkdirs", lambda *a: None)
    monkeypatch.setattr(recipe, "rsync", lambda source, host, destination: sent.append((str(source), host, destination)))
    recipe.stage_engine()
    bin_dir = recipe.deploy.SETTINGS["BIN_DIR"] + "/"
    for host in recipe.deploy.SPARKS.values():
        assert (str(library), host, bin_dir) in sent
