"""Rank-local memory guard for one TP3 DeepSeek rank.

Policy decides; the daemon samples MemAvailable, stays paused until the rank's
current systemd invocation has a ready marker, logs dips and recoveries, and
SIGKILLs only that rank unit. Events go to stdout (the guard unit's journal,
identifier chimera-memguard) and to <state>/events.jsonl.
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

GIB = 1024 ** 3


@dataclass(frozen=True)
class Config:
    warn: int = 4 * GIB
    soft: int = 2 * GIB
    hard: int = 3 * GIB // 4
    soft_seconds: float = 10.0
    hard_samples: int = 2
    sample_seconds: float = 0.25
    retry_seconds: float = 2.0   # a stop that failed is decided again after this long

    def __post_init__(self):
        if not 0 < self.hard < self.soft < self.warn:
            raise ValueError('require 0 < hard < soft < warn')
        if self.soft_seconds <= 0 or self.hard_samples < 1 or self.sample_seconds <= 0 or self.retry_seconds <= 0:
            raise ValueError('durations and sample count must be positive')


class Policy:
    def __init__(self, config=Config()):
        self.config = config
        self.reset()

    def reset(self):
        self.start = None
        self.minimum = None
        self.soft_start = None
        self.hard_count = 0
        self.last = None
        self.killed = False      # set by confirm_stop once a stop actually worked
        self.last_stop = None    # when the last stop decision was made

    def sample(self, now, available, ready):
        if available < 0 or (self.last is not None and now < self.last):
            raise ValueError('invalid memory or nonmonotonic time')
        if not ready:
            self.reset()
            return []
        self.last = now
        events = []
        c = self.config
        if available < c.warn:
            if self.start is None:
                self.start, self.minimum = now, available
                events.append(dict(kind='warning', available=available))
            self.minimum = min(self.minimum, available)
        elif self.start is not None:
            events.append(dict(kind='recovered', minimum=self.minimum, duration=now-self.start))
            self.start = self.minimum = None
        if available < c.soft:
            if self.soft_start is None:
                self.soft_start = now
        else:
            self.soft_start = None
        self.hard_count = self.hard_count + 1 if available < c.hard else 0
        reason = None
        if self.hard_count >= c.hard_samples:
            reason = 'hard'
        elif self.soft_start is not None and now-self.soft_start >= c.soft_seconds:
            reason = 'soft'
        if reason and not self.killed and (self.last_stop is None or now - self.last_stop >= c.retry_seconds):
            self.last_stop = now
            events.append(dict(kind='stop_rank', reason=reason, available=available,
                               minimum=self.minimum, duration=now-self.start))
        return events

    def confirm_stop(self):
        """The stop worked: decide no more stops for this invocation."""
        self.killed = True


def mem_available(path='/proc/meminfo'):
    for line in Path(path).read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise ValueError('no MemAvailable')


def unit_state(unit):
    out = subprocess.run(['systemctl', '--user', 'show', '-p', 'ActiveState', '-p', 'InvocationID', unit],
                         capture_output=True, text=True, timeout=10).stdout
    v = dict(x.split('=', 1) for x in out.splitlines() if '=' in x)
    return v.get('ActiveState', ''), v.get('InvocationID', '')


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--unit', required=True, help='the rank\'s systemd user unit, the only thing this guard may kill')
    a.add_argument('--state-dir', type=Path, required=True)
    a.add_argument('--warn-gib', type=float, default=4.0)
    a.add_argument('--soft-gib', type=float, default=2.0)
    a.add_argument('--hard-gib', type=float, default=0.75)
    a.add_argument('--soft-seconds', type=float, default=10.0)
    a.add_argument('--hard-samples', type=int, default=2)
    a.add_argument('--sample-seconds', type=float, default=0.25)
    a.add_argument('--retry-seconds', type=float, default=2.0, help='retry a failed stop after this long')
    a.add_argument('--unit-check-seconds', type=float, default=2.0)
    a.add_argument('--dry-run', action='store_true', help='log a stop decision without killing')
    args = a.parse_args(argv)
    cfg = Config(warn=int(args.warn_gib * GIB), soft=int(args.soft_gib * GIB), hard=int(args.hard_gib * GIB),
                 soft_seconds=args.soft_seconds, hard_samples=args.hard_samples, sample_seconds=args.sample_seconds,
                 retry_seconds=args.retry_seconds)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    policy, invocation, ready, checked = Policy(cfg), '', False, -1e9

    def emit(event):
        event = dict(event, unit=args.unit, invocation=invocation, time=time.time())
        line = json.dumps(event, sort_keys=True)
        print(line, flush=True)
        with (args.state_dir / 'events.jsonl').open('a') as f:
            f.write(line + '\n')

    emit(dict(kind='guard_start', config=cfg.__dict__, dry_run=args.dry_run))
    while True:
        now = time.monotonic()
        if now - checked >= args.unit_check_seconds:
            checked = now
            state, inv = unit_state(args.unit)
            if inv != invocation:
                invocation = inv
                policy.reset()
            was, ready = ready, state == 'active' and bool(inv) and (args.state_dir / f'ready-{inv}').exists()
            if ready != was:
                emit(dict(kind='armed' if ready else 'paused', state=state))
        for e in policy.sample(now, mem_available(), ready):
            if e['kind'] == 'stop_rank':
                state, inv = unit_state(args.unit)
                if state != 'active' or inv != invocation:
                    emit(dict(e, kind='stop_skipped', why='unit changed before the stop'))
                    continue
                if args.dry_run:
                    policy.confirm_stop()
                else:
                    r = subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', args.unit],
                                       capture_output=True, text=True, timeout=10)
                    e = dict(e, returncode=r.returncode, stderr=r.stderr.strip())
                    if r.returncode == 0:
                        policy.confirm_stop()
                    else:
                        e['retry_after_s'] = cfg.retry_seconds
            emit(e)
        time.sleep(cfg.sample_seconds)


if __name__ == '__main__':
    sys.exit(main())
