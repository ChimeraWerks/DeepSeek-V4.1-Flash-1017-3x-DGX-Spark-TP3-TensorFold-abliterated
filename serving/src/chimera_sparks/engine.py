"""Small TensorFold transport; no model execution, abort-all or generation retries."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import time
import uuid
import httpx

from . import __version__
from .laky.media_admission import has_images
from .laky.media_types import DOCUMENT_BLOCK_TYPES
from .laky.queue import FairPriorityGate
from .laky.stream_validation import ChatStreamValidator, MessagesStreamValidator, ResponsesStreamValidator
from .policy import CAP_OMITTED, normalize


def upstream_message(response):
    """The error message in an engine reply (OpenAI's or Anthropic's shape), or ""."""
    try:
        error = response.json().get("error")
    except ValueError:
        return ""
    message = error.get("message") if isinstance(error, dict) else error
    return message if isinstance(message, str) else ""


# The engine's default PDF page limit (TensorFold --vision-pdf, TENSORFOLD_MAX_PDF_PAGES): what one document can take
# when the engine cannot count its pages.
PDF_PAGE_RESERVE = 16


class GatewayError(RuntimeError):
    def __init__(self, status_code, message, code="upstream_error"):
        super().__init__(message)
        self.status_code, self.code = status_code, code


@dataclass
class Handle:
    response: httpx.Response
    prepared: dict
    upstream_request_id: str
    weight: int
    queue_wait_seconds: float
    media: bool
    api: str
    closed: bool = False


class Engine:
    def __init__(self, cfg, transport=None):
        self.cfg = cfg
        headers = {}
        if cfg.upstream_key_env:
            key = os.environ.get(cfg.upstream_key_env)
            if not key:
                raise ValueError("upstream credential environment variable is missing")
            headers["Authorization"] = "Bearer " + key
        self.client = httpx.AsyncClient(base_url=cfg.upstream, transport=transport, headers=headers,
            trust_env=False, timeout=httpx.Timeout(cfg.request_timeout, connect=5, pool=10),
            # Sized for the engine's streams, so a live raise of the admission cap never waits on the pool.
            limits=httpx.Limits(max_connections=cfg.engine_streams + 4, max_keepalive_connections=cfg.engine_streams + 4))
        self.control_client = httpx.AsyncClient(base_url=cfg.upstream, transport=transport, headers=headers,
            trust_env=False, timeout=httpx.Timeout(5), limits=httpx.Limits(max_connections=4))
        self.gate = FairPriorityGate(cfg.max_concurrency, cfg.max_queue, cfg.admitted_tokens)
        self.media_gate = asyncio.Semaphore(cfg.media_concurrency)
        self.requests = 0
        self.errors = 0
        self.native_window = None
        # Whether the engine's count routes span images exactly: None until an image request shows it.
        self.exact_images, self.exact_images_checked = None, None
        self.exact_recheck_seconds = 300.0

    async def close(self):
        await self.client.aclose()
        await self.control_client.aclose()

    def normalise_request(self, body):
        return normalize(body, self.cfg), None, 0, "default", True

    async def count_input(self, body, api="chat"):
        """The prompt's tokens as the native engine will hold them, and how they were counted.

        The engine spans each image exactly in its count routes (from the image's header, without decoding it), and
        each PDF page as generation will hold it (the page's size from PDFium, no pixels drawn), so images and PDF
        documents go to the count as they are. An engine that counts text only refuses them there; for it, each image
        becomes the text "[image]" plus image_token_reserve, the most one image can take on this model, and each
        document "[document]" plus PDF_PAGE_RESERVE such pages.
        """
        images, documents = [], []
        def text_only(value):
            if isinstance(value, dict):
                kind = value.get("type")
                if isinstance(kind, str) and kind in {"image_url", "input_image", "image"}:
                    images.append(1)
                    return {"type": "text", "text": "[image]"}
                if isinstance(kind, str) and kind in DOCUMENT_BLOCK_TYPES:
                    documents.append(1)
                    return {"type": "text", "text": "[document]"}
                return {k: text_only(v) for k, v in value.items()}
            if isinstance(value, list):
                return [text_only(v) for v in value]
            return value
        legacy_body = text_only(body)
        if (images or documents) and self._exact_images_possible():
            data = await self._native_count(body, api, with_images=True)
            if data is not None:
                self.exact_images, self.exact_images_checked = True, None
                return self._count_of(data), "native-exact"
            self.exact_images, self.exact_images_checked = False, time.monotonic()
        count = self._count_of(await self._native_count(legacy_body, api))
        pages = len(images) + len(documents) * PDF_PAGE_RESERVE
        return count + pages * self.cfg.image_token_reserve, "native-text+media-reserve"

    def _exact_images_possible(self):
        # An engine found counting text only is asked again after a while: it may have been upgraded in place.
        if self.exact_images is not False:
            return True
        return time.monotonic() - self.exact_images_checked >= self.exact_recheck_seconds

    async def _native_count(self, body, api, *, with_images=False):
        token_body = body
        if api == "chat":
            token_body = {k: v for k, v in token_body.items() if k in {"model", "messages", "tools", "tool_choice", "chat_template_kwargs", "reasoning_effort", "thinking_budget"}}
            token_body.update(add_generation_prompt=True, return_token_strs=False)
        elif api == "responses":
            token_body = {**token_body, "api": "responses", "stream": False, "store": False}
        path = "/v1/messages/count_tokens" if api == "messages" else "/tokenize"
        response = await self.control_client.post(path, json=token_body, timeout=30)
        if with_images:
            if response.status_code == 400:
                message = upstream_message(response)
                if "accepts text" in message:
                    return None  # an engine that counts text only
                # the image itself is refused (bad data, too many pixels): generation would refuse it the same way
                raise GatewayError(400, message or "the engine refused an image", "invalid_request_error")
            if response.status_code == 501:
                # an enabled media feature without its runtime library (PDF inputs without libpdfium)
                raise GatewayError(501, upstream_message(response) or "the engine cannot read this input", "not_implemented")
            if response.is_success and "image_tokens" not in response.json():
                return None  # counted without spanning its images
        if not response.is_success:
            raise GatewayError(503, "native tokenizer is unavailable; no generation dispatched")
        data = response.json()
        self.native_window = data.get("max_model_len", self.native_window)
        return data

    @staticmethod
    def _count_of(data):
        count = data.get("count", data.get("num_tokens", data.get("input_tokens")))
        if count is None and isinstance(data.get("tokens"), list):
            count = len(data["tokens"])
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise GatewayError(503, "native tokenizer did not return a valid count")
        return count

    async def prepare(self, body, api="chat"):
        body = dict(body)
        omitted = body.pop(CAP_OMITTED, False) is True
        count, method = await self.count_input(body, api)
        return self.check_count(body, count, api, method, cap_omitted=omitted)

    def check_count(self, body, count, api, method, *, cap_omitted=False):
        """Admit a counted prompt with its output cap, or refuse it.

        A cap the client gave must fit beside the prompt and safety_tokens. A cap it left out (the configured default)
        is lowered to the context left, when at least one token is left, like deepseek-recipe's resolve_max_tokens; the
        lowered cap is what the engine gets and what admission weighs.
        """
        field = "max_output_tokens" if api == "responses" else "max_tokens"
        cap = body[field]
        room = self.cfg.context_tokens - count - self.cfg.safety_tokens
        if cap_omitted and 1 <= room < cap:
            body, cap = {**body, field: room}, room
        if count + cap + self.cfg.safety_tokens > self.cfg.context_tokens:
            raise GatewayError(400, "input and output reserve exceed recipe context; compact the conversation explicitly", "context_length_exceeded")
        return {"body": body, "count": count, "weight": count + cap, "token_method": method}

    async def begin_stream(self, request_body, *, vision_dispatch=None, api="chat"):
        prepared = await self.prepare(request_body, api)
        media = has_images(request_body)
        acquired_media = acquired_gate = False
        try:
            if media:
                await self.media_gate.acquire()
                acquired_media = True
            async with asyncio.timeout(self.cfg.queue_timeout):
                wait = await self.gate.acquire(weight=prepared["weight"])
            acquired_gate = True
            request_id = vision_dispatch["request_id"] if vision_dispatch else "recipe-" + uuid.uuid4().hex
            body = {**prepared["body"], "request_id": request_id}
            if vision_dispatch:
                body["dgx_vision_control"] = vision_dispatch
            path = {"responses": "/v1/responses", "messages": "/v1/messages"}.get(api, "/v1/chat/completions")
            request = self.client.build_request("POST", path, json=body)
            response = await self.client.send(request, stream=True)
            self.requests += 1
            return Handle(response, prepared, request_id, prepared["weight"], wait, media, api)
        except BaseException:
            if acquired_gate:
                await self.gate.release(weight=prepared["weight"])
            if acquired_media:
                self.media_gate.release()
            raise

    async def close_stream(self, handle, *, outcome="success"):
        if handle.closed:
            return
        handle.closed = True
        try:
            # TensorFold owns disconnect cancellation; never send vLLM abort requests.
            await handle.response.aclose()
        finally:
            await self.gate.release(weight=handle.weight)
            if handle.media:
                self.media_gate.release()

    async def iter_stream(self, handle):
        kinds = {"chat": ChatStreamValidator, "messages": MessagesStreamValidator}
        validator = kinds.get(handle.api, ResponsesStreamValidator)(self.cfg.max_response_bytes)
        pending = b""
        meaningful = truncated = False
        try:
            async for chunk in handle.response.aiter_bytes():
                validator.feed(chunk)
                if handle.api == "chat":
                    pending = (pending + chunk).replace(b"\r\n", b"\n")
                    while b"\n\n" in pending:
                        frame, pending = pending.split(b"\n\n", 1)
                        data = b"\n".join(line[5:].strip() for line in frame.splitlines() if line.startswith(b"data:"))
                        # A reply cut by max_tokens while thinking is a normal length finish, not a failure.
                        if data == b"[DONE]" and not meaningful and not truncated:
                            raise GatewayError(502, "native returned no final text or tool call", "empty_final")
                        if data and data != b"[DONE]":
                            for choice in json.loads(data).get("choices", []):
                                truncated = truncated or choice.get("finish_reason") == "length"
                                delta = choice.get("delta") or {}
                                content = delta.get("content")
                                meaningful = meaningful or bool(delta.get("tool_calls")) or (isinstance(content, str) and bool(content.strip()))
                yield chunk
            validator.finish()
        finally:
            await asyncio.shield(self.close_stream(handle))

    def _response_headers(self, prepared, *, queue_wait_seconds=0):
        body = prepared["body"]
        return {"X-Recipe-Version": __version__, "X-Recipe-Token-Method": prepared["token_method"],
                "X-Recipe-Queue-Seconds": f"{queue_wait_seconds:.6f}",
                "X-Recipe-Sampling": json.dumps({k: body[k] for k in self.cfg.sampling if k in body}, separators=(",", ":")),
                "Cache-Control": "no-store"}
