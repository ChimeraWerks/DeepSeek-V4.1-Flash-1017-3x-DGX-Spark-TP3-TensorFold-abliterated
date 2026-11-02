import asyncio
from dataclasses import replace
import json
import unittest

from chimera_sparks.config import Config
from chimera_sparks.shortening import Shortener, CompactionError
from chimera_sparks.laky import responses_checkpoint as cp
from chimera_sparks.engine import Engine, GatewayError
from chimera_sparks.policy import normalize
import httpx


class ShorteningTests(unittest.IsolatedAsyncioTestCase):
    def cfg(self):
        return replace(Config(), auto_compact=True, compact_trigger_tokens=16000,
            compact_target_tokens=12000, compact_recent_tokens=200,
            compact_chunk_chars=6000, context_tokens=24000,
            max_output_tokens=8192, output_tokens=512).validate()

    def history(self):
        return [{"role": "system", "content": "Preserve /srv/state and job_id=ABC-123."},
            {"role": "user", "content": "Original task: keep exact id ABC-123."},
            {"role": "assistant", "content": "old result " * 300},
            {"role": "user", "content": "Earlier request"},
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function",
                "function": {"name": "inspect", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "verified receipt"},
            {"role": "user", "content": "Continue the current task, exactly."}]

    async def count(self, body, api):
        return len(json.dumps(body)) // 2, "test-measurement"

    async def completion(self, system, prompt, material):
        value = json.loads(prompt)
        result = dict(value["checkpoint_control"])
        result.pop("allowed_evidence_ids", None)
        result.pop("required_literals", None)
        for name in cp._LIST_FIELDS:
            result[name] = []
        for name in cp._CLAIM_FIELDS:
            result[name] = {"text": "verified task", "evidence_ids": [material.allowed_evidence_ids[0]]}
        result["exact_literals"] = []
        result["critical_evidence_ids"] = []
        return result

    async def test_disabled_requests_never_generate_a_checkpoint(self):
        seen = []
        async def upstream(request):
            seen.append(request.url.path)
            return httpx.Response(200, json={"count": 10})
        engine = Engine(Config(), httpx.MockTransport(upstream))
        try:
            body = normalize({"messages": self.history()}, engine.cfg)
            prepared = await engine.prepare(body)
            self.assertEqual(prepared["body"], body)
            self.assertFalse(prepared["compacted"])
            self.assertEqual(seen, ["/tokenize"])
        finally:
            await engine.close()

    async def test_chat_keeps_current_turn_instructions_and_atomic_tool_group(self):
        cfg = self.cfg()
        body = normalize({"messages": self.history()}, cfg)
        result, compacted = await Shortener(cfg, self.count, self.completion).compact(body, 20000, "chat")
        self.assertTrue(compacted)
        self.assertEqual(result["messages"][0], body["messages"][0])
        self.assertEqual(result["messages"][-1], body["messages"][-1])
        calls = [m for m in result["messages"] if m.get("tool_calls")]
        if calls:
            self.assertIn(body["messages"][-2], result["messages"])
        text = json.dumps(result)
        self.assertIn("ABC-123", text)
        self.assertIn("critical_evidence_jsonl", text)
        self.assertEqual(body["messages"], self.history())

    async def test_unknown_checkpoint_evidence_is_refused(self):
        async def invented(system, prompt, material):
            value = await self.completion(system, prompt, material)
            value["goal"]["evidence_ids"] = ["invented"]
            return value
        body = normalize({"messages": self.history()}, self.cfg())
        with self.assertRaises(CompactionError):
            await Shortener(self.cfg(), self.count, invented).compact(body, 20000, "chat")

    async def test_checkpoint_work_bound_and_live_only_refusal(self):
        cfg = replace(self.cfg(), checkpoint_max_chunks=1)
        history = self.history()
        history[2]["content"] *= 20
        body = normalize({"messages": history}, cfg)
        with self.assertRaises(CompactionError):
            await Shortener(cfg, self.count, self.completion).compact(body, 20000, "chat")
        only = normalize({"messages": [{"role": "user", "content": "current " * 3000}]}, cfg)
        with self.assertRaises(CompactionError):
            await Shortener(cfg, self.count, self.completion).compact(only, 20000, "chat")

    async def test_responses_preserves_full_live_tail_and_refuses_previous_id(self):
        cfg = self.cfg()
        body = normalize({"input": [{"type": "message", **m} for m in self.history()[:4]] + [
            {"type": "message", "role": "user", "content": "Current task"},
            {"type": "function_call", "call_id": "c1", "name": "inspect", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "receipt"}]}, cfg, api="responses")
        result, compacted = await Shortener(cfg, self.count, self.completion).compact(body, 20000, "responses")
        self.assertTrue(compacted)
        self.assertEqual(result["input"][-3:], body["input"][-3:])
        with self.assertRaises(CompactionError):
            await Shortener(cfg, self.count, self.completion).compact({**body, "previous_response_id": "old"}, 20000, "responses")

    async def test_enabled_feature_refuses_start_without_tokenizer(self):
        async def upstream(request):
            return httpx.Response(503)
        engine = Engine(self.cfg(), httpx.MockTransport(upstream))
        try:
            with self.assertRaises(GatewayError):
                await engine.validate_features()
        finally:
            await engine.close()

    def test_enabled_configuration_is_typed_and_bounded(self):
        for changes in [{"auto_compact": 1}, {"checkpoint_timeout": float("nan")},
                        {"compact_target_tokens": 17000}, {"checkpoint_max_chunks": True}]:
            with self.assertRaises(ValueError):
                replace(self.cfg(), **changes).validate()

class ShorteningTransportTests(unittest.IsolatedAsyncioTestCase):
    cfg = ShorteningTests.cfg
    history = ShorteningTests.history
    async def exercise(self, outcome):
        calls = []
        checkpoint_calls = []
        async def upstream(request):
            body = json.loads(request.content)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 100})
            if (body.get("response_format") or {}).get("json_schema", {}).get("name") == "responses_checkpoint":
                checkpoint_calls.append(body)
                control = json.loads(body["messages"][-1]["content"])["checkpoint_control"]
                value = {k: v for k, v in control.items() if k not in {"allowed_evidence_ids", "required_literals"}}
                value.update(exact_literals=[], critical_evidence_ids=[])
                for name in cp._LIST_FIELDS:
                    value[name] = []
                for name in cp._CLAIM_FIELDS:
                    value[name] = {"text": "verified task", "evidence_ids": [control["allowed_evidence_ids"][0]]}
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]})
            calls.append(body)
            if len(calls) == 1:
                if outcome == "timeout":
                    raise httpx.ReadTimeout("accepted outcome is unknown")
                return httpx.Response(400 if outcome == "context" else 503,
                    json={"error": {"code": "context_length_exceeded" if outcome == "context" else "busy"}})
            return httpx.Response(200, json={"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]})
        engine = Engine(self.cfg(), httpx.MockTransport(upstream))
        try:
            body = normalize({"messages": self.history()}, engine.cfg)
            if outcome == "timeout":
                with self.assertRaises(httpx.ReadTimeout):
                    await engine.begin_stream(body)
            else:
                handle = await engine.begin_stream(body)
                await engine.close_stream(handle)
                self.assertEqual(handle.prepared["compacted"], outcome == "context")
            self.assertEqual(engine.gate.active_weight, 0)
            self.assertEqual(len(calls), 2 if outcome == "context" else 1)
            self.assertEqual(bool(checkpoint_calls), outcome == "context")
            if outcome == "context":
                self.assertEqual(calls[1]["messages"][-1], body["messages"][-1])
        finally:
            await engine.close()

    async def test_one_retry_only_after_typed_context_rejection(self):
        await self.exercise("context")

    async def test_no_retry_after_ambiguous_timeout_or_other_rejection(self):
        await self.exercise("timeout")
        await self.exercise("busy")


class ShorteningStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_enabled_startup_refuses_missing_or_smaller_native_context(self):
        for window in (None, 1024):
            async def upstream(request):
                return httpx.Response(200, json={"count": 10, "max_model_len": window})
            engine = Engine(ShorteningTests.cfg(self), httpx.MockTransport(upstream))
            try:
                with self.assertRaises(GatewayError):
                    await engine.validate_features()
            finally:
                await engine.close()
