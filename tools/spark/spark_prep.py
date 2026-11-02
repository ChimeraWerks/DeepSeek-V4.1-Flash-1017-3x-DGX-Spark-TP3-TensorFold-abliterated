#!/usr/bin/env python3
"""spark_prep: check and stage the DGX Spark host setup this release runs on.

Runs on one Spark, standard library only. It never reboots. `apply` stages changes that take effect on the next boot
(firmware, kernel, display reservation) or immediately where that is safe (packages, services, swap entries).
Every file it edits keeps its first original as `<name>.pre-spark-prep`; firmware and EFI settings are not files and are not backed up.

  python3 spark_prep.py status                what is applied, staged for the next boot, or missing
  python3 spark_prep.py plan                  what `apply` would change
  sudo python3 spark_prep.py apply [--only ID ...] [--yes]
  python3 spark_prep.py explain               every item: what, why, measured gain, how long it lasts

Add --json for machine-readable output (apply --json also needs --yes). Items that need a person at the console print instructions instead.
Run it again after every reboot until every item reads applied.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

VERSION = "1"
APPLIED, STAGED, MISSING, MANUAL, NA = "applied", "staged for next boot", "missing", "needs a person", "not applicable"

# October 2026 (OTA2610) Founders Edition firmware: the set this cluster was qualified on.
FIRMWARE = {"Embedded Controller": 0x03000D07, "SoC": 0x02009E0F, "USB-PD": 0x00000519}
DISP_GUID = "284e19b9-235f-4aab-ae4c-190456d1bd76"
EFIVARS = Path("/sys/firmware/efi/efivars")
DISP_REQUEST = EFIVARS / f"NvidiaDispFrmSizeRequest-{DISP_GUID}"
DISP_STATUS = EFIVARS / f"NvidiaDispFrmStatus-{DISP_GUID}"

# The tested package set, from the pinned memory-saver kernel guide (v0.4.1, ec1b17a8).
DRIVER = "580.178.04"
KERNEL_STOCK = "7.0.0-1019-nvidia"
KERNEL_64K = "7.0.0-1019-nvidia-64k"
KERNEL_PKG_VERSION = "7.0.0-1019.19~24.04.2"
NVIDIA_SOURCE_VERSION = "580.178.04-0ubuntu0.24.04.1"
KERNEL_PKGS = [f"linux-image-{KERNEL_64K}={KERNEL_PKG_VERSION}", f"linux-modules-{KERNEL_64K}={KERNEL_PKG_VERSION}",
               f"linux-headers-{KERNEL_64K}={KERNEL_PKG_VERSION}", f"linux-tools-{KERNEL_64K}={KERNEL_PKG_VERSION}",
               f"linux-modules-nvidia-580-open-{KERNEL_64K}", f"nvidia-kernel-source-580-open={NVIDIA_SOURCE_VERSION}"]
BUILD_PKGS = ["git", "build-essential", "gcc-13", "patch", "coreutils", "kmod", "python3", "openssl", "mokutil",
              "dkms", "initramfs-tools", "util-linux", "psmisc"]
# Packages the simulated install may add or upgrade; anything else NVIDIA-related means a driver change.
DRIVER_PKG = re.compile(r"^(Inst|Remv) (\S*nvidia\S*)")

GRUB_CFG = Path("/boot/grub/grub.cfg")
GRUB_SAVER = Path("/etc/default/grub.d/zz-dgx-spark-memory-saver.cfg")
GRUB_BACKUP = Path("/var/lib/dgx-spark-memory-saver/preparation")
FSTAB = Path("/etc/fstab")
SWAP_4K, SWAP_64K = "/swap.img", "/swap-64k.img"
POLICY_BIN = Path("/usr/local/sbin/spark-memory-policy")
POLICY_UNIT = Path("/etc/systemd/system/spark-memory-policy.service")
CX7_HOTPLUG = Path("/etc/nvidia/cx7-hotplug-enabled")
SLEEP_TARGETS = ["sleep.target", "suspend.target", "hibernate.target", "hybrid-sleep.target"]
UNUSED_SERVICES = ["cups.service", "cups-browsed.service", "multipathd.service"]
SUFFIX = ".pre-spark-prep"

POLICY_SCRIPT = """#!/bin/sh
# 64 KiB Spark memory policy (spark_prep): THP off, then the explicit reserve.
# Reserve must be written after THP changes; khugepaged otherwise sets ~6.2 GiB.
set -eu
[ "$(getconf PAGESIZE)" = 65536 ] || exit 0
thp=/sys/kernel/mm/transparent_hugepage
printf 'never\\n' > "$thp/enabled"
for f in "$thp"/hugepages-*kB/enabled; do [ -f "$f" ] && printf 'never\\n' > "$f"; done
printf 'never\\n' > "$thp/defrag"
printf '1048576\\n' > /proc/sys/vm/min_free_kbytes
"""
POLICY_SERVICE = f"""[Unit]
Description=64 KiB Spark THP-off and 1 GiB memory reserve policy
After=systemd-sysctl.service systemd-tmpfiles-setup.service
Before=docker.service

[Service]
Type=oneshot
ExecStart={POLICY_BIN}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""


def run(*cmd: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def out(*cmd: str) -> str:
    try:
        return run(*cmd).stdout.strip()
    except FileNotFoundError:
        return ""


def must(*cmd: str, env: dict | None = None) -> str:
    r = run(*cmd, env=env)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd[:3])} failed: {(r.stderr or r.stdout)[-400:]}")
    return r.stdout


def read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def backup(path: Path) -> None:
    """Keep the first original of every file this script edits."""
    dst = path.with_name(path.name + SUFFIX)
    if path.exists() and not dst.exists():
        shutil.copy2(path, dst)


@dataclass
class Item:
    id: str
    title: str
    what: str
    why: str
    gain: str
    lasts: str
    check: Callable[[], tuple[str, str]]
    apply: Callable[[], str] | None = None
    instructions: str = ""
    depends: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- base system
def driver_version() -> str:
    return read(Path("/sys/module/nvidia/version")).strip() or out("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")


def base_check():
    arch, kernel, driver = platform.machine(), os.uname().release, driver_version()
    detail = f"{arch}, kernel {kernel}, driver {driver or 'not loaded'}"
    if arch == "aarch64" and kernel in (KERNEL_STOCK, KERNEL_64K) and driver == DRIVER:
        return APPLIED, detail
    return MANUAL, detail


# ---------------------------------------------------------------- firmware
def fw_versions(ids: dict[str, str] | None = None) -> dict[str, tuple[int, int | None]]:
    """name -> (installed, pending update target or None), from fwupd; fills `ids` with name -> fwupd device id."""
    found: dict[str, tuple[int, int | None]] = {}
    try:
        devices = json.loads(out("fwupdmgr", "get-devices", "--json", "--no-unreported-check") or "{}").get("Devices", [])
    except json.JSONDecodeError:
        return found
    for d in devices:
        name, ver = d.get("Name", ""), d.get("Version", "")
        if d.get("Plugin") != "uefi_capsule" or not ver.startswith("0x"):
            continue
        cur = int(ver, 16)
        key = ("Embedded Controller" if "Embedded" in name else
               "USB-PD" if cur < 0x00010000 else "SoC")
        pending = cur if d.get("UpdateState") == 4 else None  # 4: needs reboot (a capsule is staged)
        found[key] = (cur, pending)
        if ids is not None and d.get("DeviceId"):
            ids[key] = d["DeviceId"]
    return found


def fw_check():
    v = fw_versions()
    if not v:
        return MISSING, "fwupd reports no NVIDIA capsule devices"
    low = [k for k, want in FIRMWARE.items() if v.get(k, (0, None))[0] < want]
    staged = [k for k in low if v.get(k, (0, None))[1] is not None]
    if not low:
        return APPLIED, ", ".join(f"{k} {v[k][0]:#010x}" for k in FIRMWARE)
    if staged and len(staged) == len(low):
        return STAGED, "capsules staged: " + ", ".join(staged)
    return MISSING, "older than October 2026: " + ", ".join(f"{k} {v.get(k, (0, None))[0]:#010x}" for k in low)


def fw_apply():
    """Stage fwupd's current NVIDIA capsule for each device below the floor; never touches other devices."""
    run("fwupdmgr", "refresh", "--force")
    ids: dict[str, str] = {}
    v = fw_versions(ids)
    low = [k for k, want in FIRMWARE.items() if v.get(k, (0, None))[0] < want]
    if not low:
        return "firmware already at or above the October 2026 floor"
    for k in low:
        if k not in ids:
            raise RuntimeError(f"fwupd lists no device id for {k}; update it with NVIDIA's DGX Spark firmware tool")
        r = run("fwupdmgr", "update", ids[k], "-y", "--no-reboot-check", "--no-unreported-check")
        if r.returncode not in (0, 2):  # 2: nothing to do
            raise RuntimeError(f"fwupdmgr update {k} failed: {(r.stderr or r.stdout)[-300:]}")
    return ("staged the capsule fwupd offers for " + ", ".join(low) +
            " (the October 2026 release or newer); it flashes on the next boot, do not cut power while it runs")


# ---------------------------------------------------------------- display reservation
def disp_status() -> tuple[int, int] | None:
    try:
        b = DISP_STATUS.read_bytes()[4:]
        return b[0], int.from_bytes(b[1:5], "little")
    except (OSError, IndexError):
        return None


def disp_check():
    st = disp_status()
    pending = DISP_REQUEST.exists()
    if st and st[0] == 0:
        return APPLIED, f"Auto, {st[1]} MiB reserved"
    if pending:
        return STAGED, "Auto requested; firmware applies it on the next boot"
    if st is None:
        return MISSING, "firmware has no display reservation option yet (needs October firmware)"
    return MISSING, f"mode {st[0]}, {st[1]} MiB reserved"


def disp_apply():
    if DISP_REQUEST.exists():
        run("chattr", "-i", str(DISP_REQUEST))
    DISP_REQUEST.write_bytes(b"\x07\x00\x00\x00\x00")  # attributes NV|BS|RT, then 0 = Auto
    return "requested Auto display reservation for the next boot"


# ---------------------------------------------------------------- 64K kernel packages
def grub_entry(kernel: str, cfg: str | None = None) -> str | None:
    """GRUB's `submenu>entry` id for a kernel in grub.cfg, or None."""
    cfg = read(GRUB_CFG) if cfg is None else cfg
    sub = re.search(r"submenu '[^']*' \$menuentry_id_option '([^']+)'", cfg)
    ent = re.search(rf"menuentry '[^']*{re.escape(kernel)}'[^$]*\$menuentry_id_option '([^']+)'", cfg)
    return f"{sub.group(1)}>{ent.group(1)}" if sub and ent else None


def pkg_installed(spec: str) -> bool:
    """`name` or `name=version`: installed, and at that exact version when one is pinned."""
    name, _, version = spec.partition("=")
    if "install ok installed" not in out("dpkg-query", "-W", "-f=${Status}", name):
        return False
    return not version or out("dpkg-query", "-W", "-f=${Version}", name) == version


def kpkg_check():
    have = [p for p in KERNEL_PKGS if pkg_installed(p)]
    if len(have) == len(KERNEL_PKGS) and Path(f"/boot/vmlinuz-{KERNEL_64K}").exists():
        return APPLIED, f"{KERNEL_64K} installed beside the 4K kernel"
    return MISSING, f"{len(have)} of {len(KERNEL_PKGS)} {KERNEL_64K} packages installed"


def unexpected_driver_changes(simulation: str) -> list[str]:
    """NVIDIA package installs or removals in an apt simulation that are not the expected new 64K modules."""
    allowed = {p.split("=")[0] for p in KERNEL_PKGS}  # the 64K kernel names contain "nvidia" too
    bad = []
    for line in simulation.splitlines():
        m = DRIVER_PKG.match(line)
        if m and (m.group(1) == "Remv" or m.group(2) not in allowed):
            bad.append(line.strip())
        elif line.startswith("Remv "):
            bad.append(line.strip())
    return bad


def pin_stock_boot() -> str:
    """Keep the working 4K kernel as GRUB's default before a newer kernel is installed (upstream step 2)."""
    if GRUB_SAVER.exists():
        return "boot default already pinned"
    stock = grub_entry(KERNEL_STOCK)
    if not stock:
        raise RuntimeError(f"no GRUB entry for {KERNEL_STOCK}; inspect /boot/grub/grub.cfg before installing")
    GRUB_BACKUP.mkdir(parents=True, exist_ok=True)
    for src in (Path("/etc/default/grub"), Path("/etc/default/grub.d"), FSTAB, GRUB_CFG):
        if src.is_dir():
            shutil.copytree(src, GRUB_BACKUP / src.name, dirs_exist_ok=True)
        elif src.exists():
            shutil.copy2(src, GRUB_BACKUP / src.name)
    GRUB_SAVER.write_text(f'GRUB_DEFAULT="{stock}"\n')
    must("update-grub")
    return f"pinned {KERNEL_STOCK} as the boot default"


def kpkg_apply():
    if platform.machine() != "aarch64" or os.uname().release != KERNEL_STOCK or os.sysconf("SC_PAGE_SIZE") != 4096:
        raise RuntimeError(f"run this from the stock {KERNEL_STOCK} 4K kernel")
    if driver_version() != DRIVER:
        raise RuntimeError(f"driver {driver_version() or 'not loaded'}; this package set needs {DRIVER}")
    notes = [pin_stock_boot()]
    env = dict(os.environ, NEEDRESTART_MODE="l", DEBIAN_FRONTEND="noninteractive")
    must("apt-get", "update", env=env)
    args = ["--no-remove", "--no-install-recommends", "install", *KERNEL_PKGS, *BUILD_PKGS]
    bad = unexpected_driver_changes(must("apt-get", "--simulate", *args, env=env))
    if bad:
        raise RuntimeError("the install would change the NVIDIA driver or remove packages; nothing installed: " + "; ".join(bad[:5]))
    must("apt-get", "-y", *args, env=env)
    must("dkms", "autoinstall", "-k", KERNEL_64K)
    must("depmod", KERNEL_64K)
    must("update-initramfs", "-u", "-k", KERNEL_64K)
    must("update-grub")
    if "CONFIG_ARM64_64K_PAGES=y" not in read(Path(f"/boot/config-{KERNEL_64K}")):
        raise RuntimeError(f"/boot/config-{KERNEL_64K} is not a 64 KiB page kernel")
    got = out("modinfo", "-k", KERNEL_64K, "-F", "version", "nvidia")
    if got != DRIVER:
        raise RuntimeError(f"NVIDIA module for {KERNEL_64K} is {got or 'missing'}, expected {DRIVER}")
    return "; ".join(notes + [f"installed {KERNEL_64K} and its NVIDIA {DRIVER} modules; the 4K kernel stays the default"])


# ---------------------------------------------------------------- memory saver + 64K default
def saver_ready() -> bool:
    info = out("modinfo", "-k", KERNEL_64K, "nvidia_uvm")
    return "uvm_pack_sysmem_leaf_tables" in info and "signer" in info


def saver_check():
    if not saver_ready():
        return MANUAL, f"signed memory-saver UVM module for {KERNEL_64K} not installed"
    running = os.uname().release == KERNEL_64K
    packing = Path("/sys/module/nvidia_uvm/parameters/uvm_pack_sysmem_leaf_tables")
    if running and read(packing).strip() == "Y":
        return APPLIED, "packing on"
    return (STAGED if not running else MISSING), "installed; active when the 64K kernel boots"


def grub_default() -> str | None:
    m = re.search(r'^GRUB_DEFAULT="?([^"\n]*)', read(GRUB_SAVER), re.M)
    return m.group(1) if m else None


def next_entry() -> str:
    m = re.search(r"^next_entry=(.*)$", out("grub-editenv", "list"), re.M)
    return m.group(1).strip() if m else ""


def k64_check():
    if not saver_ready():
        return MANUAL, "install the memory saver first"
    want = grub_entry(KERNEL_64K)
    running = os.uname().release == KERNEL_64K
    is_default = bool(want and grub_default() == want)
    if running and is_default:
        return APPLIED, f"running {KERNEL_64K}, default on boot"
    if is_default:
        return STAGED, f"{KERNEL_64K} boots next time"
    if running:
        return MISSING, f"running {KERNEL_64K} once; apply makes it the default"
    if want and next_entry() == want:
        return STAGED, f"one trial boot into {KERNEL_64K} queued"
    return MISSING, f"running {os.uname().release}; apply queues one trial boot into {KERNEL_64K}"


def k64_apply():
    want = grub_entry(KERNEL_64K)
    if not want or not saver_ready():
        raise RuntimeError("64K kernel entry or signed memory-saver module not found; see the memory_saver instructions")
    if os.uname().release != KERNEL_64K:
        must("grub-reboot", want)
        return f"next boot only: {KERNEL_64K}; check it, then run apply again to make it the default"
    backup(GRUB_SAVER)
    text = read(GRUB_SAVER)
    line = f'GRUB_DEFAULT="{want}"'
    text = re.sub(r"^GRUB_DEFAULT=.*$", line, text, flags=re.M) if "GRUB_DEFAULT=" in text else text + line + "\n"
    GRUB_SAVER.write_text(text)
    must("update-grub")
    return f"default boot set to {KERNEL_64K} (the 4K kernel stays installed as fallback)"


# ---------------------------------------------------------------- swap
def fstab_lines() -> list[str]:
    return read(FSTAB).splitlines()


def swap_entry(path: str) -> str | None:
    for line in fstab_lines():
        f = line.split()
        if f and f[0] == path and len(f) >= 4 and f[2] == "swap":
            return f[3]
    return None


def swap_check():
    opts = {p: swap_entry(p) for p in (SWAP_4K, SWAP_64K)}
    bad = [p for p, o in opts.items() if o is None or "noauto" in o.split(",") or "nofail" not in o.split(",")]
    have64 = Path(SWAP_64K).exists()
    if not bad and have64:
        active = out("swapon", "--show=NAME", "--noheadings").split()
        return APPLIED, "both swap files mount at boot; active: " + (", ".join(active) or "none")
    return MISSING, ("no 64K swap file; " if not have64 else "") + ("boot entries need sw,nofail: " + ", ".join(bad) if bad else "")


def swap_fstab(lines: list[str], present: set[str]) -> list[str]:
    """Both swap files as `sw,nofail`; every other line untouched."""
    result, seen = [], set()
    for line in lines:
        f = line.split()
        if f and f[0] in (SWAP_4K, SWAP_64K) and len(f) >= 4 and f[2] == "swap":
            seen.add(f[0])
            line = f"{f[0]}\tnone\tswap\tsw,nofail\t0\t0"
        result.append(line)
    return result + [f"{p}\tnone\tswap\tsw,nofail\t0\t0" for p in (SWAP_4K, SWAP_64K) if p not in seen and p in present]


def swap_apply():
    done = []
    if not Path(SWAP_64K).exists():
        must("fallocate", "-l", "16G", SWAP_64K)
        os.chmod(SWAP_64K, 0o600)
        must("mkswap", "--pagesize", "65536", SWAP_64K)
        done.append("created 16 GiB 64K-page swap file")
    backup(FSTAB)
    present = {p for p in (SWAP_4K, SWAP_64K) if Path(p).exists()}
    FSTAB.write_text("\n".join(swap_fstab(fstab_lines(), present)) + "\n")
    run("systemctl", "daemon-reload")
    return "; ".join(done + ["each kernel now mounts the swap file that matches its page size at boot"])


# ---------------------------------------------------------------- memory policy
def policy_check():
    ok = POLICY_BIN.exists() and POLICY_UNIT.exists() and out("systemctl", "is-enabled", POLICY_UNIT.name) == "enabled"
    if not ok:
        return MISSING, "policy service not installed"
    if os.sysconf("SC_PAGE_SIZE") == 65536:
        return policy_live(Path("/sys/kernel/mm/transparent_hugepage"), Path("/proc/sys/vm/min_free_kbytes"))
    return STAGED, "installed; acts only under the 64K kernel"


def policy_live(thp: Path, reserve: Path) -> tuple[str, str]:
    """Whether THP, per-size THP, defrag and the 1 GiB reserve all match the policy on this boot."""
    wrong = [f.name if f.parent == thp else f"{f.parent.name}/{f.name}"
             for f in [thp / "enabled", thp / "defrag", *sorted(thp.glob("hugepages-*kB/enabled"))]
             if f.exists() and "[never]" not in read(f)]
    if read(reserve).strip() != "1048576":
        wrong.append(f"min_free_kbytes {read(reserve).strip() or 'unknown'}")
    return (MISSING, "not as set: " + ", ".join(wrong)) if wrong else (APPLIED, "THP off, 1 GiB reserve")


def policy_apply():
    backup(POLICY_BIN)
    backup(POLICY_UNIT)
    POLICY_BIN.write_text(POLICY_SCRIPT)
    POLICY_BIN.chmod(0o755)
    POLICY_UNIT.write_text(POLICY_SERVICE)
    run("systemctl", "daemon-reload")
    must("systemctl", "enable", POLICY_UNIT.name)
    return "installed and enabled the 64K memory policy service"


# ---------------------------------------------------------------- small host items
def kho_check():
    installed = pkg_installed("nvidia-spark-grub-kho")
    if installed and "kho=off" in read(Path("/proc/cmdline")):
        return APPLIED, "kho=off on this boot"
    return (STAGED, "package installed, takes effect next boot") if installed else (MISSING, "nvidia-spark-grub-kho not installed")


def kho_apply():
    env = dict(os.environ, NEEDRESTART_MODE="l", DEBIAN_FRONTEND="noninteractive")
    must("apt-get", "-y", "--no-remove", "install", "nvidia-spark-grub-kho", env=env)
    return "installed NVIDIA's kho=off boot flag package; active after reboot"


def headless_check():
    tgt = out("systemctl", "get-default")
    dm = out("systemctl", "is-enabled", "display-manager.service")
    return (APPLIED if tgt == "multi-user.target" and dm != "enabled" else MISSING), f"default {tgt}, display manager {dm or 'absent'}"


def headless_apply():
    must("systemctl", "set-default", "multi-user.target")
    if out("systemctl", "is-enabled", "display-manager.service") == "enabled":
        must("systemctl", "disable", "display-manager.service")
    return "boots to the console target without a desktop"


def sleep_check():
    states = {t: out("systemctl", "is-enabled", t) for t in SLEEP_TARGETS}
    bad = [t for t, s in states.items() if s != "masked"]
    return (APPLIED, "all sleep targets masked") if not bad else (MISSING, "not masked: " + ", ".join(bad))


def sleep_apply():
    must("systemctl", "mask", *SLEEP_TARGETS)
    return "masked sleep, suspend and hibernate"


def cx7_check():
    return (MISSING, f"{CX7_HOTPLUG} present") if CX7_HOTPLUG.exists() else (APPLIED, "a peer link loss cannot remove the adapter")


def cx7_apply():
    if CX7_HOTPLUG.exists():
        backup(CX7_HOTPLUG)
        CX7_HOTPLUG.unlink()
    return "disabled ConnectX-7 hot-unplug on link loss"


def services_check():
    on = [s for s in UNUSED_SERVICES if out("systemctl", "is-enabled", s) == "enabled"]
    return (APPLIED, "printing and multipath disabled") if not on else (MISSING, "enabled: " + ", ".join(on))


def services_apply():
    for svc in UNUSED_SERVICES:
        if out("systemctl", "is-enabled", svc) == "enabled":
            must("systemctl", "disable", "--now", svc)
    return "disabled printing and multipath services"


def governor_check():
    res = out("systemctl", "show", "-p", "Result", "--value", "nv-cpu-governor.service")
    gov = read(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")).strip()
    return (APPLIED if gov == "performance" else MISSING), f"cpu0 governor {gov or 'unknown'}, NVIDIA service {res or 'absent'}"


BASE_HOWTO = f"""Base system: update DGX OS with NVIDIA's DGX Spark system update until the node runs kernel {KERNEL_STOCK}
  with NVIDIA driver {DRIVER} on aarch64, then run this script again. The rest of the setup is pinned to that set."""

SAVER_HOWTO = f"""Memory saver (one time, needs a person at the console for Secure Boot key enrollment):
  1. Run `sudo python3 spark_prep.py apply --only kernel_pkgs` first: it installs the {KERNEL_64K} packages.
  2. Follow https://github.com/christopherowen/dgx-spark-memory-saver at v0.4.1 (commit ec1b17a8),
     docs/installation.md: generate a signing key, `sudo mokutil --import <key>.der`, reboot with a keyboard and
     monitor attached, choose Enroll MOK -> Continue -> Yes, enter the password, then install its signed DKMS module.
  3. Run apply again: kernel_64k queues one trial boot into the 64K kernel; after that boot, apply makes it the default.
Before any NVIDIA driver upgrade, remove the memory saver first (its docs/maintenance.md), then reinstall it."""

ITEMS = [
    Item("base", "DGX OS base", f"aarch64, kernel {KERNEL_STOCK} (or its 64K twin), NVIDIA driver {DRIVER}.",
         "Every pinned package and module below was built and tested against this set.",
         "Prerequisite; no direct gain.", "Recheck after any OS update.", base_check, None, BASE_HOWTO),
    Item("firmware", "October 2026 firmware",
         "Stages fwupd's NVIDIA capsule for each device below SoC/UEFI 2.158.15, EC 3.13.7, USB-PD 0.5.25; other devices are left alone.",
         "Adds the Display Reserved Memory option, fixes fans stopping on idle headless nodes and the UEFI password bypass (CVE-2026-47624). Note: community EC fan-floor tools stop working on this firmware.",
         "Enables the +1.5 GiB display item; no direct speed change.",
         "Once. After the floor is met, newer firmware is never flashed by this script.", fw_check, fw_apply),
    Item("display_auto", "Display Reserved Memory: Auto",
         "Writes NVIDIA's NvidiaDispFrmSizeRequest (porting guide 5.5.1) so a headless boot reserves 512 MiB instead of 2 GiB.",
         "Nothing uses the display on a headless node.",
         "+1.5 GiB RAM per node (measured 2,046 -> 512 MiB).",
         "Kept in firmware settings across reboots; recheck after firmware updates.", disp_check, disp_apply, depends=["firmware"]),
    Item("kernel_pkgs", "64K page kernel packages",
         f"Pins the working 4K boot entry, then installs {KERNEL_64K} {KERNEL_PKG_VERSION}, its NVIDIA {DRIVER} modules "
         "and the memory-saver build tools, exactly as the memory-saver kernel guide lists them. Refuses if apt would "
         "change the driver or remove anything.",
         "The memory saver only works on the 64K kernel.", "Prerequisite for the memory saver.",
         "Once; the 4K kernel stays installed and stays the default until kernel_64k.", kpkg_check, kpkg_apply, depends=["base"]),
    Item("memory_saver", "Memory saver (UVM page-table packing)",
         "Signed patched nvidia-uvm for the 64K kernel packs GPU page tables 16 to a page.",
         "The stock 64K driver wastes a 64 KiB page per 256-byte GPU page table.",
         "+1.8 GiB usable per node vs the 4K kernel (upstream 3-node measurement); no speed change.",
         "Rebuild on kernel updates; remove before NVIDIA driver upgrades.", saver_check, None, SAVER_HOWTO, depends=["kernel_pkgs"]),
    Item("kernel_64k", "64K page kernel by default",
         f"First apply queues one trial boot into {KERNEL_64K}; the next apply, from that kernel, makes it the default. "
         "The 4K kernel stays installed as fallback.",
         "Fewer page-table and memmap pages for 128 GB of unified memory; the memory saver only works here.",
         "Together with the memory saver and display Auto: MemTotal 121.7 -> 125.3 GiB per node (measured).",
         "Pinned to this kernel version: after a kernel update run spark_prep again.", k64_check, k64_apply, depends=["memory_saver"]),
    Item("swap", "Swap for both page sizes",
         f"{SWAP_4K} (4K header) and {SWAP_64K} (64K header) both mount at boot with nofail.",
         "A swap file only works on the kernel whose page size made it; without this a reboot leaves no swap.",
         "16 GiB of swap safety on whichever kernel boots.", "Once.", swap_check, swap_apply),
    Item("memory_policy", "64K memory policy",
         "At boot under 64K: transparent huge pages off and a fixed 1 GiB free-memory reserve.",
         "On 64K, khugepaged otherwise raises the reserve to about 6.2 GiB and compaction stalls long prompts.",
         "About 5 GiB less RAM held back by the kernel reserve on 64K.", "Runs every boot automatically.",
         policy_check, policy_apply),
    Item("kho_off", "kho=off boot flag",
         "Installs NVIDIA's nvidia-spark-grub-kho package, which adds kho=off to the kernel command line.",
         f"Kernel {KERNEL_STOCK} otherwise fails RDMA memory registration (NCCL/RoCE ENOMEM).",
         "Required for multi-node RoCE on this kernel.", "Package-managed; survives updates.", kho_check, kho_apply),
    Item("headless", "Headless boot", "Console target, no display manager.",
         "No desktop memory or GPU use; display Auto needs no display at boot.", "A few hundred MiB RAM.", "Once.",
         headless_check, headless_apply),
    Item("no_sleep", "Never sleep", "Masks sleep, suspend, hibernate and hybrid sleep.",
         "A serving node must not suspend.", "Availability.", "Once.", sleep_check, sleep_apply),
    Item("cx7_hotplug", "Keep ConnectX-7 on link loss", f"Removes {CX7_HOTPLUG} (kept as a backup).",
         "Otherwise a peer rebooting can hot-unplug this node's NIC and break RDMA until reboot.", "Stability.", "Once.",
         cx7_check, cx7_apply),
    Item("services", "Unused services off", "Disables printing (cups) and multipath.",
         "Not used on an inference node.", "Small memory and CPU.", "Once.", services_check, services_apply),
    Item("governor", "Performance CPU governor", "NVIDIA's nv-cpu-governor service sets performance at boot.",
         "Stable host-side latency for the engine.", "Lower latency jitter.", "Package-managed.", governor_check),
]
BY_ID = {i.id: i for i in ITEMS}


def status_rows():
    rows = []
    for i in ITEMS:
        try:
            state, detail = i.check()
        except Exception as e:  # noqa: BLE001 - a broken probe is reported, not fatal
            state, detail = MISSING, f"check failed: {e}"
        rows.append({"id": i.id, "title": i.title, "state": state, "detail": detail, "gain": i.gain,
                     "lasts": i.lasts, "scriptable": i.apply is not None, "depends": i.depends})
    return rows


def explain_rows():
    return [{"id": i.id, "title": i.title, "what": i.what, "why": i.why, "gain": i.gain, "lasts": i.lasts,
             "scriptable": i.apply is not None, "instructions": i.instructions} for i in ITEMS]


def blocked_by(item_id: str, rows: list[dict], results: list[dict]) -> list[str]:
    """Dependencies still missing or manual, unless this run just applied them."""
    unmet = {x["id"] for x in rows if x["state"] in (MISSING, MANUAL)}
    done = {x["id"] for x in results if x["ok"]}
    blocked = [d for d in BY_ID[item_id].depends if d in unmet and d not in done]
    return [] if item_id == "display_auto" else blocked  # display Auto may be queued before the firmware that reads it


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check and stage the DGX Spark host setup this release runs on.")
    p.add_argument("command", choices=["status", "plan", "apply", "explain"])
    p.add_argument("--only", nargs="+", choices=list(BY_ID), help="limit apply/plan to these items")
    p.add_argument("--yes", action="store_true", help="apply without the interactive prompt")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    host = os.uname().nodename

    if a.command == "explain":
        rows = explain_rows()
        if a.json:
            print(json.dumps({"version": VERSION, "items": rows}, indent=1))
        else:
            for r in rows:
                print(f"{r['title']} [{r['id']}]\n  What: {r['what']}\n  Why:  {r['why']}\n  Gain: {r['gain']}\n  Lasts: {r['lasts']}")
                if r["instructions"]:
                    print("  " + r["instructions"].replace("\n", "\n  "))
        return 0

    rows = status_rows()
    todo = [r for r in rows if r["state"] == MISSING and r["scriptable"] and (not a.only or r["id"] in a.only)]
    manual = [r for r in rows if r["state"] == MANUAL]
    if a.command in ("status", "plan"):
        if a.json:
            print(json.dumps({"version": VERSION, "host": host, "kernel": os.uname().release, "items": rows,
                              "would_apply": [r["id"] for r in todo]}, indent=1))
        else:
            print(f"{host}: kernel {os.uname().release}")
            for r in rows:
                print(f"  {r['state']:22} {r['title']:36} {r['detail']}")
            if a.command == "plan":
                print("\napply would change: " + (", ".join(r["id"] for r in todo) or "nothing"))
            for r in manual:
                print(f"\n{BY_ID[r['id']].instructions}")
        return 0

    if os.geteuid() != 0:
        print("apply needs root: sudo python3 spark_prep.py apply", file=sys.stderr)
        return 2
    if not todo:
        print(json.dumps({"host": host, "applied": [], "note": "nothing to change"}) if a.json else f"{host}: nothing to change")
        return 0
    if a.json and not a.yes:
        print("apply --json changes the host without asking; add --yes to confirm", file=sys.stderr)
        return 2
    if not a.yes:
        print(f"{host} will change: " + ", ".join(r["id"] for r in todo))
        if input("continue? [y/N] ").strip().lower() != "y":
            return 1
    results = []
    for r in sorted(todo, key=lambda r: [i.id for i in ITEMS].index(r["id"])):
        blocked = blocked_by(r["id"], rows, results)
        if blocked:
            results.append({"id": r["id"], "ok": False, "note": "waiting on " + ", ".join(blocked)})
            continue
        try:
            apply = BY_ID[r["id"]].apply
            assert apply is not None  # todo holds scriptable items only
            results.append({"id": r["id"], "ok": True, "note": apply()})
        except Exception as e:  # noqa: BLE001 - report each item, keep going
            results.append({"id": r["id"], "ok": False, "note": str(e)})
    if a.json:
        print(json.dumps({"host": host, "applied": results, "reboot_needed": True}, indent=1))
    else:
        for x in results:
            print(f"  {'ok' if x['ok'] else 'FAILED':7} {x['id']:14} {x['note']}")
        print(f"\n{host}: reboot to finish (firmware, kernel and display changes apply at boot), then run status again.")
    return 0 if all(x["ok"] for x in results) else 1


if __name__ == "__main__":
    sys.exit(main())
