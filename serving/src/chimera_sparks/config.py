"""Portable, nonsecret JSON configuration. Credentials are environment references."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Config:
    upstream: str = "http://127.0.0.1:8003"
    model: str = "deepseek-v4.1-flash"
    host: str = "127.0.0.1"
    port: int = 18002
    state_dir: str = "~/.local/state/chimera-sparks"
    clients: dict[str, str] = field(default_factory=lambda: {"default": "SPARK_API_KEY"})
    # Client IDs allowed to read and change the live settings (/admin/settings); none by default.
    admin_clients: list[str] = field(default_factory=list)
    upstream_key_env: str | None = None
    # top_k 20 keeps the engine's sampler on its GPU candidates; top_k 0 samples the whole vocabulary on the CPU.
    sampling: dict[str, float | int] = field(default_factory=lambda: {
        "temperature": 1.0, "top_p": 1.0, "top_k": 20, "presence_penalty": 0.0})
    # Labelled presets (README); top_k 20 is this recipe's speed choice. Selected per request (X-Recipe-Profile or
    # model "<model>@<name>"), per client or as default_profile; add, label and edit more live.
    profiles: dict[str, dict] = field(default_factory=lambda: {
        "standard": {"label": "DeepSeek 1017 card: temperature 1, top_p 1", "temperature": 1.0, "top_p": 1.0,
                     "top_k": 20},
        "focused": {"label": "temperature 1, top_p 0.95, min_p 0.05", "temperature": 1.0, "top_p": 0.95,
                    "top_k": 20, "min_p": 0.05}})
    client_profiles: dict[str, str] = field(default_factory=dict)
    # The profile a request gets when neither its X-Recipe-Profile header nor its client names one.
    default_profile: str | None = None
    context_tokens: int = 1048576
    output_tokens: int = 32768
    max_output_tokens: int = 131072
    safety_tokens: int = 1024
    # Decode streams the native engine runs (TF_DS_STREAMS); the admission cap may be lower, never higher.
    engine_streams: int = 16
    max_concurrency: int = 16
    max_queue: int = 64
    admitted_tokens: int = 5336064
    max_body_bytes: int = 64 * 1024 * 1024
    max_response_bytes: int = 64 * 1024 * 1024
    media_concurrency: int = 2
    image_token_reserve: int = 16384
    request_timeout: float = 3590.0
    queue_timeout: float = 1800.0
    heartbeat: float = 15.0
    # Thinking for requests and profiles that name no effort; 1017's encoder defaults to high (75) as well.
    reasoning_default: str = "high"
    # The effort number (1-100) each named level sends; starts at the engine's own table, so levels go by name until
    # one is moved. Moving one needs an engine whose /health reports numeric_effort.
    effort_map: dict[str, int] = field(default_factory=lambda: {
        "minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 88, "max": 100})
    reasoning_output: str = "preserve"
    vision_effort: str = "medium"
    vision_thinking_tokens: int = 3072
    vision_final_tokens: int = 32768

    auto_compact: bool = False
    compact_trigger_tokens: int = 300000
    compact_target_tokens: int = 240000
    compact_recent_tokens: int = 16000
    compact_chunk_chars: int = 24000
    checkpoint_output_tokens: int = 4096
    checkpoint_max_chunks: int = 16
    checkpoint_timeout: float = 180.0

    def validate(self):
        parsed = urlsplit(self.upstream)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("upstream must be an HTTP URL without credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("upstream must be an origin without a path/query")
        if not self.clients or any(not k or not isinstance(v, str) or not v.isidentifier()
                                   for k, v in self.clients.items()):
            raise ValueError("clients must map stable client IDs to environment variable names")
        if not isinstance(self.admin_clients, list) or any(name not in self.clients for name in self.admin_clients):
            raise ValueError("admin_clients must name configured clients")
        for name in ("port", "context_tokens", "output_tokens", "max_output_tokens", "safety_tokens",
                     "engine_streams", "max_concurrency", "max_queue", "admitted_tokens", "max_body_bytes",
                     "max_response_bytes", "image_token_reserve", "vision_thinking_tokens", "vision_final_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.port > 65535 or self.engine_streams > 16 or self.media_concurrency not in {1, 2}:
            raise ValueError("invalid port or concurrency; this TP3 recipe runs at most 16 engine streams/two media requests")
        if self.max_concurrency > self.engine_streams:
            raise ValueError("max_concurrency cannot exceed the engine's decode streams")
        if self.output_tokens > self.max_output_tokens or self.max_output_tokens + self.safety_tokens >= self.context_tokens:
            raise ValueError("output limits must fit the context")
        if self.reasoning_default not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("invalid reasoning_default")
        if self.reasoning_output not in {"preserve", "hide"} or self.vision_effort not in {"medium", "max"}:
            raise ValueError("invalid reasoning output or vision effort")
        for value in (self.request_timeout, self.queue_timeout, self.heartbeat):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 3600:
                raise ValueError("timeouts must be finite and between 0 and 3600 seconds")
        if not isinstance(self.auto_compact, bool):
            raise ValueError("auto_compact must be a boolean")
        if self.auto_compact:
            for name in ("compact_trigger_tokens", "compact_target_tokens", "compact_recent_tokens",
                         "compact_chunk_chars", "checkpoint_output_tokens", "checkpoint_max_chunks"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if not self.compact_recent_tokens < self.compact_target_tokens < self.compact_trigger_tokens < self.context_tokens:
                raise ValueError("shortening requires recent < target < trigger < context")
            if self.checkpoint_output_tokens > self.max_output_tokens or self.checkpoint_max_chunks > 64:
                raise ValueError("checkpoint work exceeds recipe limits")
            if not isinstance(self.checkpoint_timeout, (int, float)) or isinstance(self.checkpoint_timeout, bool) or not math.isfinite(self.checkpoint_timeout) or not 0 < self.checkpoint_timeout <= self.request_timeout:
                raise ValueError("checkpoint timeout must fit the request deadline")
        from .policy import effort_map_values, profile_values, sampling_values
        effort_map_values(self.effort_map)
        if not {"temperature", "top_p", "top_k", "presence_penalty"} <= self.sampling.keys():
            raise ValueError("sampling must specify temperature, top_p, top_k and presence_penalty")
        if any(value is None for value in self.sampling.values()):
            raise ValueError("configured sampling defaults cannot be null")
        sampling_values(self.sampling)
        for name, values in self.profiles.items():
            if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,47}", name):
                raise ValueError("profile names are 1-48 lowercase letters, digits, '.', '_' or '-'")
            profile_values(values)
        if any(name not in self.profiles for name in self.client_profiles.values()):
            raise ValueError("client profile is undefined")
        if self.default_profile is not None and self.default_profile not in self.profiles:
            raise ValueError("default profile is undefined")
        return self

    @classmethod
    def load(cls, path: str | Path):
        raw = json.loads(Path(path).read_text())
        raw = raw.get("api", raw)
        cfg = cls(**raw).validate()
        state = Path(cfg.state_dir).expanduser()
        if not state.is_absolute():
            state = Path(path).resolve().parent / state
        from dataclasses import replace
        return replace(cfg, state_dir=str(state))

    def secrets(self):
        keys = {name: os.environ.get(env, "") for name, env in self.clients.items()}
        if any(len(value) < 16 for value in keys.values()) or len(set(keys.values())) != len(keys):
            raise ValueError("each configured client needs a distinct secret of at least 16 characters in its environment variable")
        return keys

    def public(self):
        return asdict(self)
