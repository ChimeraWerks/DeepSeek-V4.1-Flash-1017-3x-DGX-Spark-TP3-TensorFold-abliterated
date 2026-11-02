"""The sole recipe API boundary; native model and job work have distinct owners."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import copy
import hmac
import json
import os
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import httpx

from . import __version__, live
from .config import Config
from .engine import Engine, GatewayError
from .laky.body_limit import read_limited_body, BodyTooLarge
from .laky.queue import QueueFullError, AdmissionWeightError
from .laky.vision_control import NativeControl
from .laky.vision_jobs import Jobs, AdmissionError, standalone, contaminated, job_headers, stream_result
from .policy import ENGINE_EFFORTS, normalize, PolicyError


def error(status, message, code="invalid_request", **extra):
    return JSONResponse({"error": {"message": message, "type": "recipe_error", "code": code, **extra}},
                        status_code=status, headers={"Cache-Control": "no-store", "X-Should-Retry": "false"})


async def until_disconnect(request, operation):
    """Close the owned native socket when an ordinary waiting client leaves."""
    task = asyncio.create_task(operation)

    async def watch():
        while not task.done():
            if await request.is_disconnected():
                return
            await asyncio.sleep(.05)

    watcher = asyncio.create_task(watch())
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return error(499, "client disconnected; no generation was replayed", "client_disconnected")
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)


REASONING_ITEMS = {"reasoning", "reasoning_text"}


def is_reasoning(value):
    return isinstance(value, dict) and value.get("type") in REASONING_ITEMS


def reasoning_event(value):
    """A Responses stream event that only carries reasoning: its own type, or the item or part it announces."""
    return (str(value.get("type", "")).startswith("response.reasoning")
            or is_reasoning(value.get("item")) or is_reasoning(value.get("part")))


def project_reasoning(payload, mode):
    if mode == "preserve":
        return payload
    if isinstance(payload, list):
        return [project_reasoning(v, mode) for v in payload if not is_reasoning(v)]
    if isinstance(payload, dict):
        return {k: project_reasoning(v, mode) for k, v in payload.items()
                if k not in {"reasoning_content", "reasoning", "thinking"} and not is_reasoning(v)}
    return payload


def create_app(cfg: Config, *, transport=None):
    cfg.validate()
    secrets = cfg.secrets()
    base = live.snapshot(cfg)  # the config file's values; /admin/settings/reset returns here
    live.load(cfg)

    @asynccontextmanager
    async def lifespan(app):
        os.umask(0o077)
        state = Path(cfg.state_dir).expanduser()
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        if state.is_symlink() or (state.stat().st_mode & 0o077):
            raise ValueError("state directory must be private (0700) and not a symlink")
        app.state.engine = Engine(cfg, transport)
        try:
            await app.state.engine.validate_features()
        except BaseException:
            await app.state.engine.close()
            raise
        # One configured app/worker per process. These retained Laky settings
        # are derived from our typed config, not hidden operator overrides.
        os.environ.update(DGX_MEDIA_MAX_CONCURRENCY=str(cfg.media_concurrency),
            DGX_VISION_REASONING_EFFORT=cfg.vision_effort,
            DGX_VISION_THINKING_BUDGET=str(cfg.vision_thinking_tokens),
            DGX_VISION_FINAL_ANSWER_TOKENS=str(cfg.vision_final_tokens),
            DGX_VISION_JOB_TIMEOUT_SECONDS=str(cfg.request_timeout),
            DGX_VISION_QUEUE_TIMEOUT_SECONDS=str(cfg.queue_timeout),
            DGX_VISION_JOBS_REVISION="chimera-sparks-v1")
        app.state.jobs = Jobs(state / "vision.sqlite3", app.state.engine,
                              control=NativeControl(app.state.engine.control_client))
        await app.state.jobs.start()
        try:
            yield
        finally:
            await app.state.jobs.close()
            await app.state.engine.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def principal(request):
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return None
        supplied = auth[7:].encode()
        for name, key in secrets.items():
            if hmac.compare_digest(supplied, key.encode()):
                return name
        return None

    async def body(request):
        try:
            raw = await read_limited_body(request, cfg.max_body_bytes)
            result = json.loads(raw)
        except BodyTooLarge as exc:
            raise GatewayError(413, "request body exceeds recipe limit") from exc
        except (ValueError, UnicodeError) as exc:
            raise PolicyError("invalid JSON") from exc
        if not isinstance(result, dict):
            raise PolicyError("request body must be an object")
        return result

    @app.exception_handler(PolicyError)
    async def policy_error(request, exc):
        return error(400, str(exc))

    @app.exception_handler(GatewayError)
    async def upstream_error(request, exc):
        return error(exc.status_code, str(exc), exc.code)

    @app.exception_handler(QueueFullError)
    async def queue_error(request, exc):
        return error(429, "recipe queue is full", "queue_full")

    @app.exception_handler(AdmissionWeightError)
    async def admission_error(request, exc):
        return error(400, "request exceeds the admitted-token budget", "admission_oversized")

    @app.exception_handler(httpx.HTTPError)
    async def transport_error(request, exc):
        return error(502, "native transport failed; request was not automatically replayed", "upstream_transport")

    @app.exception_handler(TimeoutError)
    async def timeout_error(request, exc):
        return error(504, "recipe deadline exceeded; check retained job identity before retrying", "deadline")

    @app.get("/health")
    @app.get("/ready")
    async def health(request: Request):
        engine = request.app.state.engine
        try:
            response = await engine.control_client.get("/health")
            native = response.json()
            ready = response.is_success and not native.get("warming", False)
        except (httpx.HTTPError, ValueError):
            native, ready = {"available": False}, False
        payload = {"ok": ready, "version": __version__, "native": native,
                   "active": engine.gate.active, "waiting": engine.gate.waiting,
                   "max_concurrency": cfg.max_concurrency, "default_profile": cfg.default_profile,
                   "sampling": cfg.sampling, "reasoning_output": cfg.reasoning_output}
        jobs = request.app.state.jobs
        if jobs:
            states = jobs.ledger.counts()
            payload["vision_jobs"] = {"states": states, "automatic_replay": False,
                                     "worker_alive": not jobs.runner.done(), "output_budget": jobs.output_budget()}
            ready = ready and not jobs.runner.done()
            payload["ok"] = ready
        return JSONResponse(payload, status_code=200 if ready else 503)

    @app.get("/metrics")
    async def metrics(request: Request):
        engine = request.app.state.engine
        lines = [f"recipe_requests_total {engine.requests}", f"recipe_stream_errors_total {engine.errors}",
                 f"recipe_active {engine.gate.active}", f"recipe_waiting {engine.gate.waiting}",
                 f"recipe_admitted_tokens {engine.gate.active_weight}"]
        if request.app.state.jobs:
            lines += [f'recipe_vision_jobs{{state="{state}"}} {count}' for state, count in request.app.state.jobs.ledger.counts().items()]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    async def numeric_effort(request):
        """Whether the running engine takes effort numbers (its /health numeric_effort); False when it cannot say."""
        try:
            response = await request.app.state.engine.control_client.get("/health")
            return bool(response.is_success and response.json().get("numeric_effort"))
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def settings_view(request):
        gate = request.app.state.engine.gate
        return {"settings": live.snapshot(cfg), "base": base, "engine_streams": cfg.engine_streams,
                "engine_efforts": ENGINE_EFFORTS, "numeric_effort": await numeric_effort(request),
                "active": gate.active, "waiting": gate.waiting, "saved": live.path(cfg).exists()}

    def admin_denied(request):
        name = principal(request)
        if name is None:
            return error(401, "invalid API credential", "invalid_api_key")
        if name not in cfg.admin_clients:
            return error(403, "this client cannot change live settings", "forbidden")
        return None

    @app.get("/admin/settings")
    async def get_settings(request: Request):
        return admin_denied(request) or await settings_view(request)

    @app.put("/admin/settings")
    async def put_settings(request: Request):
        if denied := admin_denied(request):
            return denied
        update = await body(request)
        try:
            new = live.merged(cfg, update)
        except (ValueError, TypeError) as exc:
            return error(400, str(exc), "invalid_settings")
        # A moved level is sent as a number, which only an engine reporting numeric_effort accepts.
        if new.effort_map != ENGINE_EFFORTS and new.effort_map != cfg.effort_map and not await numeric_effort(request):
            return error(409, "the running engine does not take effort numbers; deploy an engine whose /health "
                         "reports numeric_effort before remapping effort levels", "engine_capability")
        await live.apply(cfg, new, request.app.state.engine.gate)
        live.save(cfg, base)
        return await settings_view(request)

    @app.post("/admin/settings/reset")
    async def reset_settings(request: Request):
        if denied := admin_denied(request):
            return denied
        await live.apply(cfg, live.from_base(cfg, base), request.app.state.engine.gate)
        live.save(cfg, base)
        return await settings_view(request)

    @app.get("/v1/models")
    async def models(request: Request):
        if principal(request) is None:
            return error(401, "invalid API credential", "invalid_api_key")
        presets = [{"id": f"{cfg.model}@{name}", "object": "model", "owned_by": "recipe",
                    "description": values.get("label") or name} for name, values in sorted(cfg.profiles.items())]
        return {"object": "list", "data": [{"id": cfg.model, "object": "model", "owned_by": "recipe"}, *presets]}

    @app.get("/v1/profiles")
    async def profiles(request: Request):
        """The live presets any client may read: what each sets and how to select it."""
        if principal(request) is None:
            return error(401, "invalid API credential", "invalid_api_key")
        return {"defaults": {**cfg.sampling, "reasoning_effort": cfg.reasoning_default},
                "default_profile": cfg.default_profile, "profiles": copy.deepcopy(cfg.profiles),
                "select": {"header": "X-Recipe-Profile: <name>", "model": f"{cfg.model}@<name>"}}

    @app.post("/tokenize")
    @app.post("/v1/tokenize")
    async def tokenize(request: Request):
        if principal(request) is None:
            return error(401, "invalid API credential", "invalid_api_key")
        raw = await body(request)
        response = await request.app.state.engine.control_client.post("/tokenize", json=raw, timeout=30)
        return Response(response.content, status_code=response.status_code, media_type="application/json")

    async def accept_job(request, raw, client):
        if not standalone(raw):
            raise PolicyError("durable vision requires one user image question without assistant history or tools")
        if contaminated(raw):
            raise PolicyError("repetitive incoming reasoning; no image generation dispatched")
        normalize(raw, cfg, client, profile=request.headers.get("X-Recipe-Profile"))
        key = request.headers.get("Idempotency-Key") or request.headers.get("X-Client-Request-Id")
        if key and len(key) > 256:
            raise PolicyError("idempotency key exceeds 256 characters")
        jobs = request.app.state.jobs
        generated = jobs.generation_body(raw)
        generated = normalize(generated, cfg, client, profile=request.headers.get("X-Recipe-Profile"))
        # Resolve header/client defaults before generation-body normalization
        # fills fields, while retaining original input as the dedup identity.
        explicit_sampling = {k: v for k, v in raw.items() if k in cfg.sampling and v is not None}
        defaults = normalize({**raw, **explicit_sampling}, cfg, client,
                             profile=request.headers.get("X-Recipe-Profile"))
        generated.update({k: defaults[k] for k in cfg.sampling})
        try:
            return jobs.accept(client, raw, key, generation_body=generated)
        except AdmissionError as exc:
            raise GatewayError(409, str(exc), "vision_not_accepted") from exc

    @app.post("/v1/vision/jobs")
    async def submit_job(request: Request):
        client = principal(request)
        if client is None:
            return error(401, "invalid API credential", "invalid_api_key")
        identity, created = await accept_job(request, await body(request), client)
        return JSONResponse({"job_id": identity, "status_url": "/v1/vision/jobs/" + identity,
                             "created": created}, status_code=202, headers=job_headers(identity))

    @app.get("/v1/vision/jobs/{identity}")
    async def job_status(identity: str, request: Request):
        client = principal(request)
        if client is None:
            return error(401, "invalid API credential", "invalid_api_key")
        if not request.app.state.jobs.ledger.read(identity, client):
            return error(404, "unknown job", "not_found")
        return JSONResponse(request.app.state.jobs.ledger.status(identity, client), headers=job_headers(identity))

    async def relay(engine, normalized, api):
        handle = None
        task = asyncio.create_task(engine.begin_stream(normalized, api=api))
        try:
            async with asyncio.timeout(cfg.request_timeout):
                while not task.done():
                    done, _ = await asyncio.wait({task}, timeout=cfg.heartbeat)
                    if not done:
                        yield b": recipe waiting\n\n"
                handle = task.result()
                if not handle.response.is_success:
                    # Native error text is structured data, never an exception containing a URL/key.
                    data = await limited_read(handle.response, cfg.max_response_bytes)
                    try:
                        payload = json.loads(data)
                    except ValueError:
                        payload = {"error": {"message": "native request rejected", "code": "upstream_rejected"}}
                    yield ("data: " + json.dumps(payload) + "\n\n").encode()
                    return
                pending = b""
                async for chunk in engine.iter_stream(handle):
                    if cfg.reasoning_output == "preserve":
                        yield chunk
                        continue
                    pending = (pending + chunk).replace(b"\r\n", b"\n")
                    while b"\n\n" in pending:
                        frame, pending = pending.split(b"\n\n", 1)
                        data = b"\n".join(line[5:].strip() for line in frame.splitlines() if line.startswith(b"data:"))
                        if not data or data == b"[DONE]":
                            yield frame + b"\n\n"
                        else:
                            value = json.loads(data)
                            if reasoning_event(value):
                                continue
                            prefix = b"\n".join(line for line in frame.splitlines() if not line.startswith(b"data:"))
                            yield prefix + b"\ndata: " + json.dumps(project_reasoning(value, "hide")).encode() + b"\n\n"
        except (Exception, asyncio.CancelledError) as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            engine.errors += 1
            yield b'data: {"error":{"type":"recipe_error","code":"stream_incomplete","message":"Native stream did not complete; it was not replayed"}}\n\n'
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if handle is None and task.done() and not task.cancelled() and task.exception() is None:
                handle = task.result()
            if handle:
                await asyncio.shield(engine.close_stream(handle))

    async def complete(request, api):
        client = principal(request)
        if client is None:
            return error(401, "invalid API credential", "invalid_api_key")
        raw = await body(request)
        if api == "chat" and not isinstance(raw.get("messages"), list):
            raise PolicyError("messages must be an array")
        if api == "chat" and contaminated(raw):
            return error(422, "repetitive incoming assistant reasoning; no generation dispatched", "reasoning_repetitive")
        if api == "chat" and standalone(raw):
            identity, _ = await accept_job(request, raw, client)
            if raw.get("stream") is True:
                return StreamingResponse(stream_result(request.app.state.jobs, identity, client, cfg.heartbeat),
                                         media_type="text/event-stream", headers=job_headers(identity))
            # Accepted work belongs to the durable worker. Disconnecting this
            # waiter cannot cancel or replay its generation.
            result = await request.app.state.jobs.wait(identity, client)
            return JSONResponse(result["data"], status_code=result["status_code"],
                                headers={**result.get("headers", {}), **job_headers(identity)})
        normalized = normalize(raw, cfg, client, api, request.headers.get("X-Recipe-Profile"))
        engine = request.app.state.engine
        if raw.get("stream") is True:
            return StreamingResponse(relay(engine, normalized, api), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no", "X-Recipe-Version": __version__})
        async def execute():
            handle = None
            try:
                async with asyncio.timeout(cfg.request_timeout):
                    handle = await engine.begin_stream(normalized, api=api)
                    data = await limited_read(handle.response, cfg.max_response_bytes)
                    payload = json.loads(data)
                    if handle.response.is_success:
                        if api == "chat":
                            choices = payload.get("choices") or []
                            message = choices[0].get("message", {}) if choices else {}
                            if not message.get("content") and not message.get("tool_calls"):
                                return error(502, "native returned no final text or tool call", "empty_final")
                        payload = project_reasoning(payload, cfg.reasoning_output)
                    return JSONResponse(payload, status_code=handle.response.status_code,
                                        headers=engine._response_headers(handle.prepared, queue_wait_seconds=handle.queue_wait_seconds))
            finally:
                if handle:
                    await asyncio.shield(engine.close_stream(handle))

        return await until_disconnect(request, execute())

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await complete(request, "chat")

    @app.post("/v1/responses")
    async def responses(request: Request):
        return await complete(request, "responses")

    # Explicit hooks used by the durable job extension without exposing secrets.
    app.state.principal = principal
    app.state.read_body = body
    return app


async def limited_read(response, limit):
    data = bytearray()
    async for chunk in response.aiter_bytes():
        if len(data) + len(chunk) > limit:
            raise GatewayError(502, "native response exceeded recipe byte limit")
        data.extend(chunk)
    return bytes(data)
