"""spark_prep's boot, package and swap logic, offline."""
import spark_prep as prep

UUID = "0f6c1b2a-1111-2222-3333-444455556666"
GRUB = f"""submenu 'Advanced options for DGX OS' $menuentry_id_option 'gnulinux-advanced-{UUID}' {{
	menuentry 'DGX OS, with Linux 7.0.0-1019-nvidia-64k' --class ubuntu $menuentry_id_option 'gnulinux-7.0.0-1019-nvidia-64k-advanced-{UUID}' {{
	}}
	menuentry 'DGX OS, with Linux 7.0.0-1019-nvidia' --class ubuntu $menuentry_id_option 'gnulinux-7.0.0-1019-nvidia-advanced-{UUID}' {{
	}}
}}"""


def test_grub_entry_matches_exact_kernel():
    assert prep.grub_entry("7.0.0-1019-nvidia", GRUB) == f"gnulinux-advanced-{UUID}>gnulinux-7.0.0-1019-nvidia-advanced-{UUID}"
    assert prep.grub_entry("7.0.0-1019-nvidia-64k", GRUB) == f"gnulinux-advanced-{UUID}>gnulinux-7.0.0-1019-nvidia-64k-advanced-{UUID}"
    assert prep.grub_entry("7.0.0-1020-nvidia", GRUB) is None


def test_simulation_allows_only_the_pinned_kernel_packages():
    ok = "\n".join(f"Inst {p.split('=')[0]} (x Ubuntu:24.04 [arm64])" for p in prep.KERNEL_PKGS) + "\nInst dkms (3.0 Ubuntu [all])"
    assert prep.unexpected_driver_changes(ok) == []
    bad = ok + "\nInst nvidia-driver-580-open [580.178.04-0ubuntu1] (580.200.01-0ubuntu1 Ubuntu [arm64])\nRemv mokutil [1]"
    assert [line.split()[1] for line in prep.unexpected_driver_changes(bad)] == ["nvidia-driver-580-open", "mokutil"]


def test_swap_fstab_rewrites_only_swap_lines():
    lines = ["UUID=abc / ext4 defaults 0 1", "/swap.img\tnone\tswap\tsw\t0\t0", "# keep me"]
    got = prep.swap_fstab(lines, {"/swap.img", "/swap-64k.img"})
    assert got[0] == lines[0] and got[2] == "# keep me"
    assert got[1] == "/swap.img\tnone\tswap\tsw,nofail\t0\t0"
    assert got[3] == "/swap-64k.img\tnone\tswap\tsw,nofail\t0\t0"
    assert prep.swap_fstab(got, {"/swap.img", "/swap-64k.img"}) == got


def test_firmware_floor(monkeypatch):
    floor = {k: (v, None) for k, v in prep.FIRMWARE.items()}
    monkeypatch.setattr(prep, "fw_versions", lambda: floor)
    assert prep.fw_check()[0] == prep.APPLIED
    monkeypatch.setattr(prep, "fw_versions", lambda: {**floor, "SoC": (0x02009E00, 0x02009E0F)})
    assert prep.fw_check()[0] == prep.STAGED
    monkeypatch.setattr(prep, "fw_versions", lambda: {**floor, "SoC": (0x02009E00, None)})
    assert prep.fw_check()[0] == prep.MISSING


def test_kernel_install_waits_for_the_base_system():
    rows = [{"id": "base", "state": prep.MANUAL}, {"id": "kernel_pkgs", "state": prep.MISSING}]
    assert prep.blocked_by("kernel_pkgs", rows, []) == ["base"]
    rows[0]["state"] = prep.APPLIED
    assert prep.blocked_by("kernel_pkgs", rows, []) == []
    assert prep.blocked_by("display_auto", [{"id": "firmware", "state": prep.MISSING}], []) == []


def test_json_apply_needs_yes(monkeypatch, capsys):
    monkeypatch.setattr(prep.os, "geteuid", lambda: 0)
    monkeypatch.setattr(prep, "status_rows", lambda: [{"id": "no_sleep", "state": prep.MISSING, "scriptable": True}])
    applied = []
    monkeypatch.setitem(prep.BY_ID, "no_sleep", prep.Item("no_sleep", "t", "w", "y", "g", "l", lambda: (prep.MISSING, ""),
                                                          lambda: applied.append(1) or "done"))
    assert prep.main(["apply", "--json"]) == 2 and not applied
    assert prep.main(["apply", "--json", "--yes"]) == 0 and applied


def test_pinned_package_version_is_checked(monkeypatch):
    replies = {"${Status}": "install ok installed", "${Version}": "7.0.0-1019.18~24.04.1"}
    monkeypatch.setattr(prep, "out", lambda *cmd: replies[cmd[2].removeprefix("-f=")])
    assert prep.pkg_installed("linux-image-7.0.0-1019-nvidia-64k")
    assert not prep.pkg_installed(f"linux-image-7.0.0-1019-nvidia-64k={prep.KERNEL_PKG_VERSION}")
    replies["${Version}"] = prep.KERNEL_PKG_VERSION
    assert prep.pkg_installed(f"linux-image-7.0.0-1019-nvidia-64k={prep.KERNEL_PKG_VERSION}")


def test_memory_policy_checks_every_setting(tmp_path):
    thp = tmp_path / "thp"
    (thp / "hugepages-64kB").mkdir(parents=True)
    for f in ("enabled", "defrag", "hugepages-64kB/enabled"):
        (thp / f).write_text("always madvise [never]\n")
    reserve = tmp_path / "min_free_kbytes"
    reserve.write_text("1048576\n")
    assert prep.policy_live(thp, reserve)[0] == prep.APPLIED
    (thp / "defrag").write_text("[madvise] never\n")
    reserve.write_text("6291456\n")
    state, detail = prep.policy_live(thp, reserve)
    assert state == prep.MISSING and "defrag" in detail and "6291456" in detail
