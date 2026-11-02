"""Live operator settings: validated as a whole config, applied to the running config and admission gate, and saved
beside the state so a restart keeps them. The JSON config file stays the base; reset returns to it."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path

# Everything a request reads per call, plus the admission limits. Ports, keys, context and timeouts need a restart.
LIVE_FIELDS = ("sampling", "profiles", "client_profiles", "default_profile", "reasoning_default",
               "max_concurrency", "max_queue", "effort_map")
FILE = "live-settings.json"


def snapshot(cfg):
    return {name: copy.deepcopy(getattr(cfg, name)) for name in LIVE_FIELDS}


def merged(cfg, update):
    """The config this update would produce, validated. Maps merge per key (null removes a profile or client
    assignment); sampling merges per setting; everything else replaces."""
    if not isinstance(update, dict) or set(update) - set(LIVE_FIELDS):
        raise ValueError(f"live settings are limited to {', '.join(LIVE_FIELDS)}")
    values = snapshot(cfg)
    for name, value in update.items():
        if name in ("sampling", "profiles", "client_profiles", "effort_map"):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be an object")
            target = values[name]
            for key, item in value.items():
                if item is None and name not in ("sampling", "effort_map"):
                    target.pop(key, None)
                else:
                    target[key] = item
        else:
            values[name] = value
    return replace(cfg, **values).validate()


def from_base(cfg, base):
    """The config with every live setting back at the config file's values (maps replaced, not merged)."""
    return replace(cfg, **copy.deepcopy(base)).validate()


def _copy_into(cfg, new):
    for name in LIVE_FIELDS:
        value = copy.deepcopy(getattr(new, name))
        current = getattr(cfg, name)
        if isinstance(current, dict):
            current.clear()
            current.update(value)
        else:
            object.__setattr__(cfg, name, value)  # the frozen config changes only through this path


async def apply(cfg, new, gate=None):
    """Copy the validated settings into the running config (shared by the API, policy and engine) and the gate."""
    _copy_into(cfg, new)
    if gate is not None:
        await gate.resize(cfg.max_concurrency, cfg.max_queue)


def path(cfg):
    return Path(cfg.state_dir).expanduser() / FILE


def save(cfg, base):
    """Write only what differs from the base config (0600, atomic)."""
    changed = {name: value for name, value in snapshot(cfg).items() if value != base[name]}
    target = path(cfg)
    if not changed:
        target.unlink(missing_ok=True)
        return changed
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = target.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(changed, f, indent=2, sort_keys=True)
    os.replace(tmp, target)
    return changed


def load(cfg):
    """Startup: apply saved live settings over the base config; an invalid file stops the start."""
    target = path(cfg)
    if not target.exists():
        return {}
    saved = json.loads(target.read_text())
    if not isinstance(saved, dict) or set(saved) - set(LIVE_FIELDS):
        raise ValueError(f"live settings are limited to {', '.join(LIVE_FIELDS)}")
    # save() writes whole field values, so they replace the base; merging would revive removed profiles
    _copy_into(cfg, replace(cfg, **copy.deepcopy(saved)).validate())
    return saved
