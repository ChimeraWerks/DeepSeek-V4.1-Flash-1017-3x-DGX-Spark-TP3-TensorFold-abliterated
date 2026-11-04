# Copyright (c) 2026 LakyFx. MIT; see LICENSES/LakyFx-MIT.txt.
# Adapted for the recipe-owned TensorFold API; see recipe/UPSTREAM.json.
"""Validate Chat SSE termination without retaining a transcript."""

import json


class ChatStreamValidator:
    def __init__(self, max_event_bytes: int = 8 * 1024 * 1024) -> None:
        self.pending = b""
        self.finished = False
        self.done = False
        self.max_event_bytes = max_event_bytes

    def feed(self, chunk: bytes) -> None:
        self.pending = (self.pending + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in self.pending:
            block, self.pending = self.pending.split(b"\n\n", 1)
            if len(block) > self.max_event_bytes:
                raise ValueError("upstream SSE event exceeds the size limit")
            data = b"\n".join(
                line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith(b"data:")
            )
            if not data:
                continue
            if self.done:
                raise ValueError("upstream sent model data after DONE")
            if data == b"[DONE]":
                if not self.finished:
                    raise ValueError("upstream ended without a finish reason")
                self.done = True
                continue
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeError) as exc:
                raise ValueError("upstream returned malformed SSE JSON") from exc
            if not isinstance(payload, dict) or payload.get("error"):
                raise ValueError("upstream returned a streaming error")
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    raise TypeError("upstream returned an invalid streaming choice")
                finish = choice.get("finish_reason")
                if finish:
                    if finish not in {"stop", "length", "tool_calls", "content_filter"}:
                        raise ValueError(
                            "upstream returned an unsuccessful finish reason"
                        )
                    self.finished = True
        if len(self.pending) > self.max_event_bytes:
            raise ValueError("upstream SSE event exceeds the size limit")

    def finish(self) -> None:
        if not self.done or not self.finished or self.pending.strip():
            raise ValueError("upstream stream closed before a complete terminal event")


class ResponsesStreamValidator:
    def __init__(self, max_event_bytes: int = 8 * 1024 * 1024) -> None:
        self.pending = b""
        self.terminal: str | None = None
        self.max_event_bytes = max_event_bytes

    def feed(self, chunk: bytes) -> None:
        self.pending = (self.pending + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in self.pending:
            block, self.pending = self.pending.split(b"\n\n", 1)
            if len(block) > self.max_event_bytes:
                raise ValueError("upstream Responses event exceeds the size limit")
            data = b"\n".join(
                line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith(b"data:")
            )
            if not data:
                continue
            if data == b"[DONE]":
                if self.terminal is None:
                    raise ValueError("upstream Responses ended without terminal state")
                continue
            if self.terminal is not None:
                raise ValueError("upstream Responses sent data after terminal state")
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(
                    "upstream Responses returned malformed SSE JSON"
                ) from exc
            if (
                not isinstance(payload, dict)
                or payload.get("error")
                or payload.get("type") == "error"
            ):
                raise ValueError("upstream Responses returned a streaming error")
            kind = payload.get("type")
            if kind in {"response.completed", "response.incomplete", "response.failed"}:
                response = payload.get("response") or {}
                if (
                    not isinstance(response, dict)
                    or response.get("status") != kind.split(".")[1]
                ):
                    raise ValueError(
                        "upstream Responses terminal status is inconsistent"
                    )
                self.terminal = kind
        if len(self.pending) > self.max_event_bytes:
            raise ValueError("upstream Responses event exceeds the size limit")

    def finish(self) -> None:
        if self.terminal is None or self.pending.strip():
            raise ValueError("upstream Responses stream closed without terminal state")


class MessagesStreamValidator:
    """A Messages stream must end in message_stop, with no data after it and no error event."""

    def __init__(self, max_event_bytes: int = 8 * 1024 * 1024) -> None:
        self.pending = b""
        self.stopped = False
        self.max_event_bytes = max_event_bytes

    def feed(self, chunk: bytes) -> None:
        self.pending = (self.pending + chunk).replace(b"\r\n", b"\n")
        while b"\n\n" in self.pending:
            block, self.pending = self.pending.split(b"\n\n", 1)
            if len(block) > self.max_event_bytes:
                raise ValueError("upstream Messages event exceeds the size limit")
            data = b"\n".join(line[5:].lstrip() for line in block.splitlines() if line.startswith(b"data:"))
            if not data:
                continue
            if self.stopped:
                raise ValueError("upstream Messages sent data after message_stop")
            try:
                payload = json.loads(data)
            except (ValueError, UnicodeError) as exc:
                raise ValueError("upstream Messages returned malformed SSE JSON") from exc
            if not isinstance(payload, dict) or payload.get("type") == "error":
                raise ValueError("upstream Messages returned a streaming error")
            self.stopped = payload.get("type") == "message_stop"
        if len(self.pending) > self.max_event_bytes:
            raise ValueError("upstream Messages event exceeds the size limit")

    def finish(self) -> None:
        if not self.stopped:
            raise ValueError("upstream Messages ended without message_stop")


class MessagesFinalGuard:
    """Replace the closing events of a Messages stream that ended with thinking only, as Chat answers empty_final.

    A max_tokens stop is a normal finish. Nothing is held once final text or a tool call has appeared.
    """

    ERROR = json.dumps({"type": "error", "error": {"type": "api_error", "message": "native returned no final text or tool call"}})

    def __init__(self) -> None:
        self.pending = b""
        self.answered = False

    def feed(self, chunk: bytes) -> bytes:
        if self.answered:
            return chunk
        self.pending = (self.pending + chunk).replace(b"\r\n", b"\n")
        out = []
        while b"\n\n" in self.pending and not self.answered:
            frame, self.pending = self.pending.split(b"\n\n", 1)
            data = b"\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:"))
            try:
                event = json.loads(data) if data else {}
            except ValueError:
                event = {}
            event = event if isinstance(event, dict) else {}
            block, delta = event.get("content_block") or {}, event.get("delta") or {}
            self.answered = (block.get("type") == "tool_use" or delta.get("stop_reason") == "max_tokens"
                             or bool(str(block.get("text") or "").strip() if block.get("type") == "text" else "")
                             or bool(str(delta.get("text") or "").strip() if delta.get("type") == "text_delta" else ""))
            if self.answered or event.get("type") not in ("message_delta", "message_stop"):
                out.append(frame + b"\n\n")
            elif event.get("type") == "message_stop":
                out.append(b"event: error\ndata: " + self.ERROR.encode() + b"\n\n")
        if self.answered:
            out.append(self.pending)
            self.pending = b""
        return b"".join(out)
