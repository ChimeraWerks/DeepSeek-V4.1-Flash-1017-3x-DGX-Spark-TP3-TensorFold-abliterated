#!/usr/bin/env python3
"""Stage and check the runtime inputs the engine loads besides the weights: the kernel kit and Engram tables.

Pins live in release/runtime-assets.json. Standard library only; runs on each Spark.

  kit     --runtime DIR --prefill DIR --out KIT_DIR
          DIR holds the mirrored runtime/ files (hf download <repo> --revision <rev> --include 'runtime/*').
          Verifies them, unpacks the base kit, adds the RoPE tables and the prefill 1.2.8 supplement from
          engine/tensorfold/zig/kernels/prefill_128, checks the result and only then renames it to KIT_DIR.
  engram  --rank N --dir ENGRAM_DIR [--mode ranged|full]
          ranged: downloads this rank's header and row ranges of DeepSeek's two Engram shards into sparse files
          (about 67.6 GB per rank) and resumes after an interruption. full: checks two complete shards that
          `hf download` already put in ENGRAM_DIR (about 203 GB). Both check every owned range's SHA-256 and
          then write the rank's SOURCE.json, the marker the launcher reads.
  check   --kit KIT_DIR --engram ENGRAM_DIR --rank N [--full]
          Read-only. Sizes and small-file hashes by default; --full also hashes kernels, RoPE tables and ranges.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "release/runtime-assets.json"
PREFILL = ROOT / "engine/tensorfold/zig/kernels/prefill_128"
DOCS = "docs/runtime-assets.md"
CHUNK = 8 << 20
ROPE = ("rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32")


class AssetError(RuntimeError):
    pass


def sha256_file(path, offset=0, length=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        left = os.fstat(f.fileno()).st_size - offset if length is None else length
        while left > 0:
            data = f.read(min(CHUNK, left))
            if not data:
                raise AssetError(f"{path} is shorter than expected")
            h.update(data)
            left -= len(data)
    return h.hexdigest()


def load_pins(path=PINS):
    return json.loads(Path(path).read_text())


def source_json_bytes(rank_pin):
    """The rank's serving SOURCE.json. Same length on every rank: the engine compares Engram file names and sizes."""
    raw = (json.dumps(rank_pin["serving_source"], sort_keys=True, separators=(",", ":")) + "\n").encode()
    if hashlib.sha256(raw).hexdigest() != rank_pin["serving_source_sha256"]:
        raise AssetError("pinned Engram SOURCE.json does not serialize to its pinned SHA-256")
    return raw


# Kit

def safe_extract(archive, dest):
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise AssetError(f"refusing archive member {member.name!r} in {archive}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)


def stage_kit(runtime, prefill, out, pins):
    kit = pins["kit"]
    out = Path(out)
    if out.exists():
        problems = check_kit(out, pins, prefill, full=True)
        if problems:
            raise AssetError(f"{out} exists but is not the pinned kit ({problems[0]}); move it aside, then rerun")
        return {"status": "SKIPPED", "kit": str(out)}
    for name, pin in kit["files"].items():
        path = Path(runtime) / name
        if not path.is_file() or path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
            raise AssetError(f"{path} is missing or differs from {kit['repo']}@{kit['revision']}; download it again")
    work = out.with_name(out.name + f".partial-{os.getpid()}")
    work.mkdir(parents=True)
    try:
        safe_extract(Path(runtime) / "runtime/tensorfold-kernels.tar.gz", work)
        for name in ROPE:
            shutil.copyfile(Path(runtime) / "runtime/rope" / name, work / name)
        manifest = json.loads((Path(prefill) / "MANIFEST.json").read_text())
        payload = Path(prefill) / manifest["archive"]
        if sha256_file(payload) != manifest["archive_sha256"]:
            raise AssetError(f"{payload} differs from its MANIFEST.json")
        if sha256_file(work / "aot/aot.json") != manifest["base_aot_sha256"]:
            raise AssetError("base kit aot/aot.json is not the one the prefill supplement extends")
        safe_extract(payload, work)
        problems = check_kit(work, pins, prefill, full=True)
        if problems:
            raise AssetError("staged kit failed its check: " + "; ".join(problems[:5]))
        work.rename(out)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return {"status": "WRITTEN", "kit": str(out)}


def check_kit(kit, pins, prefill=PREFILL, full=False):
    kit, problems = Path(kit), []
    manifest = json.loads((Path(prefill) / "MANIFEST.json").read_text())
    staged = {item["path"]: item["sha256"] for item in pins["kit"]["staged_files"]}
    expected = dict(staged)
    expected.update(manifest["payload_files"])
    small = {"aot/aot.json", "auxiliary/experts_ragged.cubin", *staged}
    if full:
        expected.update({k: v for k, v in manifest["base_files"].items() if k not in expected})
    for name, digest in sorted(expected.items()):
        path = kit / name
        if not path.is_file():
            problems.append("missing " + str(path))
        elif (full or name in small) and sha256_file(path) != digest:
            problems.append("differs " + str(path))
    rope = pins["kit"]["rope"]
    for name in ROPE:
        path = kit / name
        pin = pins["kit"]["files"]["runtime/rope/" + name]
        if not path.is_file() or path.stat().st_size != rope["rows"] * rope["half"] * 4:
            problems.append("missing or short " + str(path))
        elif full and sha256_file(path) != pin["sha256"]:
            problems.append("differs " + str(path))
    return problems


# Engram

def hf_url(repo, revision, name):
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo}/resolve/{revision}/{name}"


def open_range(url, start, end):
    """Bytes [start, end) of url as a readable response; end is exclusive."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": "ds1017-runtime-assets"})
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    response = urllib.request.urlopen(request, timeout=60)
    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end - 1}/"):
        response.close()
        raise AssetError(f"server ignored the byte range request for {url}")
    return response


def owned(rank_pin):
    """(file, offset, bytes, sha256) for the header and every owned range, in file order."""
    for name, meta in sorted(rank_pin["source_json"]["files"].items()):
        yield name, 0, meta["header_bytes"], meta["header_sha256"]
        for r in sorted(meta["ranges"], key=lambda r: r["offset"]):
            yield name, r["offset"], r["bytes"], r["sha256"]


def fetch_engram(directory, rank, pins, opener=open_range, retries=8, log=None):
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    eng, pin = pins["engram"], pins["engram"]["ranks"][rank]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "SOURCE.json"
    raw = source_json_bytes(pin)
    if marker.is_file() and marker.read_bytes() == raw:
        return {"status": "SKIPPED", "engram": str(directory)}
    marker.unlink(missing_ok=True)
    progress_path = directory / ".fetch-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    for name, meta in pin["source_json"]["files"].items():
        path = directory / name
        with open(path, "ab"):
            pass
        if path.stat().st_size != meta["size"]:
            os.truncate(path, meta["size"])
    for name, offset, length, digest in owned(pin):
        key = f"{name}:{offset}"
        done = progress.get(key, 0)
        h = hashlib.sha256()
        if done:
            with open(directory / name, "rb") as f:
                f.seek(offset)
                left = done
                while left:
                    data = f.read(min(CHUNK, left))
                    h.update(data)
                    left -= len(data)
        if done == length:
            if h.hexdigest() != digest:
                raise AssetError(f"{name} bytes at {offset} differ from the pin; delete {directory} and rerun")
            continue
        url = hf_url(eng["repo"], eng["revision"], name)
        fd = os.open(directory / name, os.O_WRONLY)
        try:
            attempt, last = 0, time.monotonic()
            while done < length:
                try:
                    with opener(url, offset + done, offset + length) as response:
                        while done < length:
                            data = response.read(min(CHUNK, length - done))
                            if not data:
                                raise AssetError("connection closed early")
                            os.pwrite(fd, data, offset + done)
                            h.update(data)
                            done += len(data)
                            if time.monotonic() - last > 30:
                                os.fsync(fd)
                                progress[key] = done
                                progress_path.write_text(json.dumps(progress))
                                log(f"{name} @{offset}: {done / 1e9:.1f} of {length / 1e9:.1f} GB")
                                last = time.monotonic()
                except (OSError, AssetError, urllib.error.URLError) as error:
                    attempt += 1
                    if attempt > retries:
                        raise AssetError(f"download of {name} failed after {retries} retries: {error}; rerun to resume") from None
                    log(f"retry {attempt}/{retries} for {name} at byte {offset + done}: {error}")
                    time.sleep(min(60, 2 ** attempt))
            os.fsync(fd)
        finally:
            os.close(fd)
        if h.hexdigest() != digest:
            progress.pop(key, None)
            progress_path.write_text(json.dumps(progress))
            raise AssetError(f"{name} bytes at {offset} differ from the pin after download; rerun to fetch them again")
        progress[key] = length
        progress_path.write_text(json.dumps(progress))
    marker.write_bytes(raw)
    progress_path.unlink(missing_ok=True)
    return {"status": "WRITTEN", "engram": str(directory), "mode": "ranged"}


def adopt_full(directory, rank, pins):
    pin = pins["engram"]["ranks"][rank]
    directory = Path(directory)
    raw = source_json_bytes(pin)
    marker = directory / "SOURCE.json"
    if marker.is_file() and marker.read_bytes() == raw:
        return {"status": "SKIPPED", "engram": str(directory), "mode": "full"}
    for name, offset, length, digest in owned(pin):
        path = directory / name
        if not path.is_file() or path.stat().st_size != pin["source_json"]["files"][name]["size"]:
            raise AssetError(f"{path} is missing or not the complete shard; download it with hf download first")
        if sha256_file(path, offset, length) != digest:
            raise AssetError(f"{path} bytes at {offset} differ from DeepSeek's pinned revision")
    marker.write_bytes(raw)
    return {"status": "WRITTEN", "engram": str(directory), "mode": "full"}


def check_engram(directory, rank, pins, full=False):
    pin, directory, problems = pins["engram"]["ranks"][rank], Path(directory), []
    marker = directory / "SOURCE.json"
    if not marker.is_file():
        return ["missing " + str(marker)]
    if marker.read_bytes() != source_json_bytes(pin):
        return [f"{marker} is not rank {rank}'s pinned SOURCE.json"]
    for name, meta in pin["source_json"]["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != meta["size"]:
            problems.append("missing or wrong size " + str(path))
    # Every regular file here is part of the identity all ranks must share.
    allowed = {"SOURCE.json", *pin["source_json"]["files"]}
    problems.extend("unexpected file " + str(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name not in allowed)
    for name, offset, length, digest in owned(pin):
        if not problems and (full or offset == 0) and sha256_file(directory / name, offset, length) != digest:
            problems.append(f"differs {directory / name} at byte {offset}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pins", type=Path, default=PINS)
    sub = parser.add_subparsers(dest="action", required=True)
    kit = sub.add_parser("kit")
    kit.add_argument("--runtime", type=Path, required=True)
    kit.add_argument("--prefill", type=Path, default=PREFILL)
    kit.add_argument("--out", type=Path, required=True)
    engram = sub.add_parser("engram")
    engram.add_argument("--rank", type=int, choices=(0, 1, 2), required=True)
    engram.add_argument("--dir", type=Path, required=True)
    engram.add_argument("--mode", choices=("ranged", "full"), default="ranged")
    check = sub.add_parser("check")
    check.add_argument("--kit", type=Path, required=True)
    check.add_argument("--engram", type=Path, required=True)
    check.add_argument("--rank", type=int, choices=(0, 1, 2), required=True)
    check.add_argument("--prefill", type=Path, default=PREFILL)
    check.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    pins = load_pins(args.pins)
    try:
        if args.action == "kit":
            result = stage_kit(args.runtime, args.prefill, args.out, pins)
        elif args.action == "engram":
            result = (fetch_engram if args.mode == "ranged" else adopt_full)(args.dir, args.rank, pins)
        else:
            problems = check_kit(args.kit, pins, args.prefill, args.full) + check_engram(args.engram, args.rank, pins, args.full)
            result = {"ok": not problems, "problems": problems}
            if problems:
                result["fix"] = f"run ./scripts/prepare.sh, or stage the files by hand as {DOCS} describes"
    except (OSError, ValueError, KeyError, AssetError) as error:
        print(json.dumps({"ok": False, "error": str(error), "fix": f"see {DOCS}"}))
        return 1
    result.setdefault("ok", True)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
