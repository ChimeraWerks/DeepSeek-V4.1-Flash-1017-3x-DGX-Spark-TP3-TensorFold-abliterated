"""One sampling/reasoning resolver; native TensorFold owns rendering and generation."""
import math
import secrets


class PolicyError(ValueError):
    pass


def sampling_values(values):
    allowed = {"temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "min_p"}
    if not isinstance(values, dict) or set(values) - allowed:
        raise PolicyError("unsupported sampling settings")
    result = {}
    for name, value in values.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PolicyError(f"{name} must be finite")
        good = ((name == "temperature" and 0 <= value <= 2)
                or (name == "top_p" and 0 < value <= 1)
                or (name == "min_p" and 0 <= value <= 1)
                or (name == "top_k" and isinstance(value, int) and 0 <= value <= 129408)
                or (name in {"presence_penalty", "frequency_penalty"} and -2 <= value <= 2))
        if not good:
            raise PolicyError(f"invalid {name}")
        result[name] = value
    return result


EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
# The number the engine's 1017 template hears for each named level (TensorFold families/dsv41/text.zig).
ENGINE_EFFORTS = {"minimal": 25, "low": 50, "medium": 62, "high": 75, "xhigh": 88, "max": 100}


def effort_number(value):
    """A request's own effort number (a whole number 1 to 100, as a number or a digit string), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    return value if isinstance(value, int) and 1 <= value <= 100 else None


def effort_map_values(mapping):
    """A validated level -> number map: every named level once, each a whole number 1 to 100."""
    if not isinstance(mapping, dict) or set(mapping) != set(ENGINE_EFFORTS):
        raise PolicyError(f"effort_map must give a number for each of {', '.join(ENGINE_EFFORTS)}")
    if any(effort_number(v) is None or not isinstance(v, int) for v in mapping.values()):
        raise PolicyError("effort_map numbers must be whole numbers from 1 to 100")
    return dict(mapping)


LABEL_CHARS = 120


def profile_values(values):
    """A profile: sampling settings, an optional default reasoning_effort for requests that name none, and an
    optional display label."""
    if not isinstance(values, dict):
        raise PolicyError("a profile must be an object")
    label = values.get("label")
    if label is not None and (not isinstance(label, str) or len(label) > LABEL_CHARS):
        raise PolicyError(f"a profile label must be text of at most {LABEL_CHARS} characters")
    sampling = {k: v for k, v in values.items() if k not in ("reasoning_effort", "label")}
    effort = values.get("reasoning_effort")
    if effort is not None and effort not in EFFORTS:
        raise PolicyError("invalid profile reasoning_effort")
    return sampling_values(sampling), effort


def normalize(raw, cfg, client="default", api="chat", profile=None):
    if not isinstance(raw, dict):
        raise PolicyError("request must be an object")
    if api == "responses" and raw.get("previous_response_id") is not None and len(cfg.clients) > 1:
        # Stored responses are not tied to a client, so with several clients one could continue another's history.
        raise PolicyError("previous_response_id needs a single configured client; send the full input instead")
    body = dict(raw)
    model = body.get("model", cfg.model)
    # "<model>@<profile>" selects a profile for clients that cannot send the X-Recipe-Profile header
    name, _, suffix = model.partition("@") if isinstance(model, str) else (None, "", "")
    if name != cfg.model:
        raise PolicyError("unknown model")
    body["model"] = cfg.model
    selected = profile or suffix or cfg.client_profiles.get(client) or cfg.default_profile
    if selected and selected not in cfg.profiles:
        raise PolicyError("unknown sampling profile")
    values, profile_effort = dict(cfg.sampling), None
    if selected:
        sampling, profile_effort = profile_values(cfg.profiles[selected])
        values.update(sampling)
    values.update({k: v for k, v in raw.items() if k in {
        "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "min_p"} and v is not None})
    body.update(sampling_values(values))
    # The engine seeds sampling from the prompt when a request names no seed, so identical sampled requests would
    # return identical replies. Give each sampled request its own seed unless the caller pins one.
    seed = raw.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise PolicyError("seed must be a non-negative integer")
        body["seed"] = seed
    elif body.get("temperature", 0) > 0:
        body["seed"] = secrets.randbelow(2**31 - 1) + 1
    if api == "chat" and (not isinstance(body.get("messages"), list) or not body["messages"]):
        raise PolicyError("messages must be a nonempty array")
    if api == "responses" and "input" not in body:
        raise PolicyError("Responses input is required")
    if body.get("n", 1) != 1:
        raise PolicyError("only n=1 is supported")
    field = "max_output_tokens" if api == "responses" else "max_tokens"
    cap = body.get(field, body.get("max_completion_tokens"))
    cap = cfg.output_tokens if cap is None else cap
    if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= cfg.max_output_tokens:
        raise PolicyError(f"{field} must be between 1 and {cfg.max_output_tokens}")
    body[field] = cap
    body.pop("max_completion_tokens", None)
    kw = raw.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise PolicyError("chat_template_kwargs must be an object")
    kw = dict(kw)
    effort = raw.get("reasoning_effort", (raw.get("reasoning") or {}).get("effort") if isinstance(raw.get("reasoning"), dict) else None)
    explicit_effort = effort if effort is not None else kw.get("reasoning_effort")
    # DeepSeek's request shape: a thinking object decides the mode before any effort does (deepseek-recipe reasoning.rs).
    thinking = raw.get("thinking")
    kind = thinking.get("type") if isinstance(thinking, dict) else None
    if isinstance(thinking, dict) and kind not in {"enabled", "adaptive", "disabled"}:
        raise PolicyError("thinking.type must be enabled, adaptive or disabled")
    enabled = kw.get("thinking") is True or kw.get("enable_thinking") is True or thinking is True or kind in {"enabled", "adaptive"}
    default = profile_effort or cfg.reasoning_default
    effort = explicit_effort if explicit_effort is not None else ("low" if enabled and default == "none" else default)
    if kind in {"enabled", "adaptive"} and (explicit_effort is None or explicit_effort == "none"):
        effort = "high"
    if isinstance(effort, bool) or not isinstance(effort, (str, int)):
        raise PolicyError("unsupported reasoning effort; use a named level or a whole number from 1 to 100")
    aliases = {"off": "none", "disabled": "none", "false": "none", "standard": "medium", "ultra": "max"}
    number = effort_number(effort)
    effort = str(number) if number is not None else aliases.get(str(effort).lower(), effort)
    if number is None and effort not in EFFORTS:
        raise PolicyError("unsupported reasoning effort; use a named level or a whole number from 1 to 100")
    disabled = kw.get("thinking") is False or kw.get("enable_thinking") is False or thinking is False or kind == "disabled"
    if kind is None and explicit_effort and ((disabled and effort != "none") or (enabled and effort == "none")):
        raise PolicyError("conflicting thinking controls")
    if disabled:
        effort = "none"
    # A named level goes as its name unless the live effort_map moved it off the engine's own number; then, like a
    # request's own number, the engine hands the number to the template as is (needs /health numeric_effort).
    sent = effort
    if number is not None:
        sent = number
    elif effort in ENGINE_EFFORTS and cfg.effort_map.get(effort, ENGINE_EFFORTS[effort]) != ENGINE_EFFORTS[effort]:
        sent = cfg.effort_map[effort]
    kw.update(thinking=effort != "none", reasoning_effort=sent)
    kw.pop("enable_thinking", None)
    body.update(reasoning_effort=sent, chat_template_kwargs=kw)
    body.pop("thinking", None)
    if "thinking_budget" in body:
        budget = body["thinking_budget"]
        if isinstance(budget, bool) or not isinstance(budget, int) or not 0 < budget <= cap:
            raise PolicyError("thinking_budget must fit the total output cap")
    # These internals are owned exclusively by the API, never by callers.
    for name in ("dgx_vision_control", "request_id", "rid", "priority"):
        body.pop(name, None)
    return body


def normalize_messages(raw, cfg, client="default", profile=None):
    """A Messages request for the engine's native route: what the client set passes through, the gateway fills the rest."""
    if not isinstance(raw, dict):
        raise PolicyError("request must be an object")
    config = raw.get("output_config")
    if config is not None and not isinstance(config, dict):
        raise PolicyError("output_config must be an object")
    config = config or {}
    sampled = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed")
    view = {k: raw[k] for k in ("model", "messages", "max_tokens", "thinking", *sampled) if k in raw}
    if config.get("effort") is not None:
        view["reasoning_effort"] = config["effort"]
    chat = normalize(view, cfg, client, "chat", profile)
    body = dict(raw)
    body.update({k: chat[k] for k in ("model", "max_tokens", *sampled) if k in chat})
    effort = chat["reasoning_effort"]
    if "thinking" not in raw and config.get("effort") is None:
        body["thinking"] = {"type": "disabled" if effort == "none" else "enabled"}
    if effort != "none" and config.get("effort") is None and isinstance(effort, str) and not effort.isdigit():
        body["output_config"] = {**config, "effort": effort}
    for name in ("dgx_vision_control", "request_id", "rid", "priority"):
        body.pop(name, None)
    return body
