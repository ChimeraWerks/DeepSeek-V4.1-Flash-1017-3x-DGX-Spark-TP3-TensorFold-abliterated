"""Bounded per-engine logs, inspired by MiaAI-Lab's 7ed7988 log rotation.

Run as the rank unit's parent; systemd KillMode=control-group owns both processes.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess


def size_bytes(text):
    if not isinstance(text, str) or not re.fullmatch(r"[1-9][0-9]*[kmg]", text):
        raise ValueError("log size must be a positive size such as 100m")
    value = int(text[:-1]) * {"k": 1024, "m": 1024**2, "g": 1024**3}[text[-1]]
    if value > 10 * 1024**3:
        raise ValueError("log size exceeds 10 GiB")
    return value


class Rotator:
    def __init__(self, path, limit, keep):
        self.path, self.limit, self.keep = Path(path), limit, keep
        if not 1 <= keep <= 20 or limit < 1:
            raise ValueError("log retention must keep 1 to 20 bounded files")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.stream = None

    def rotate(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.keep == 1:
            self.path.unlink(missing_ok=True)
            return
        for index in range(self.keep - 1, 0, -1):
            source = self.path if index == 1 else Path(str(self.path) + "." + str(index - 1))
            target = Path(str(self.path) + "." + str(index))
            if source.exists():
                os.replace(source, target)

    def write(self, data):
        while data:
            if self.stream is None:
                if self.path.is_symlink():
                    raise ValueError("log path must not be a symlink")
                self.stream = self.path.open("ab", buffering=0)
            length = self.stream.tell()
            if length >= self.limit:
                self.rotate()
                continue
            piece, data = data[: self.limit - length], data[self.limit - length :]
            self.stream.write(piece)

    def close(self):
        if self.stream:
            self.stream.close()
            self.stream = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-json", default="{}")
    parser.add_argument("--log", required=True)
    parser.add_argument("--max-size", default="100m")
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("engine command is required")
    env = json.loads(args.environment_json)
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("engine environment must map strings to strings")
    unbound = sorted(key for key in os.environ
                     if key.startswith(("TF_DS_", "TF_SPILL_")) and key not in env)
    if unbound:
        raise ValueError("unbound inherited native settings: " + ", ".join(unbound))
    child_env = {**os.environ, **env}
    os.umask(0o077)
    log = Rotator(args.log, size_bytes(args.max_size), args.max_files)
    print("Owned engine output: " + str(log.path), flush=True)
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda signum, frame: child.send_signal(signum) if child.poll() is None else None)
    try:
        while True:
            block = child.stdout.read1(65536)
            if not block:
                break
            log.write(block)
        status = child.wait()
    except BaseException:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        raise
    finally:
        log.close()
    # In particular, retain the watchdog's exit 75 for the group supervisor.
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    raise SystemExit(main())
