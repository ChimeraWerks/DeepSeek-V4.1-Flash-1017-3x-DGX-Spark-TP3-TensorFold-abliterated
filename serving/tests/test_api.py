import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import httpx

from chimera_sparks.api import create_app
from chimera_sparks.config import Config
from chimera_sparks.policy import normalize, PolicyError


class PolicyTests(unittest.TestCase):
    def test_default_override_and_zero(self):
        cfg = Config()
        raw = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0, "top_k": 0, "max_tokens": 4096}
        result = normalize(raw, cfg)
        self.assertEqual((result["temperature"], result["top_p"], result["top_k"], result["max_tokens"]), (0, 1, 0, 4096))
        self.assertNotIn("top_p", raw)
        result = normalize({**raw, "temperature": None}, cfg, profile="focused")
        self.assertEqual((result["temperature"], result["top_p"], result["min_p"]), (1, .95, .05))
        self.assertNotIn("min_p", normalize({"messages": raw["messages"]}, cfg))  # no min_p by default
        result = normalize({**raw, "temperature": None, "model": "deepseek-v4.1-flash@focused"}, cfg)
        self.assertEqual((result["model"], result["top_p"], result["min_p"]), ("deepseek-v4.1-flash", .95, .05))
        with self.assertRaises(PolicyError):
            normalize({**raw, "model": "deepseek-v4.1-flash@missing"}, cfg)

    def test_previous_response_id_needs_a_single_client(self):
        body = {"input": "hi", "previous_response_id": "resp_1"}
        single = Config(clients={"a": "A_KEY"})
        self.assertEqual(normalize(body, single, "a", "responses")["previous_response_id"], "resp_1")
        with self.assertRaises(PolicyError):
            normalize(body, Config(clients={"a": "A_KEY", "b": "B_KEY"}), "a", "responses")

    def test_thinking_object_decides_mode_like_deepseek_recipe(self):
        base = {"messages": [{"role": "user", "content": "hello"}]}
        cfg = replace(Config(), reasoning_default="none")
        cases = [({"thinking": {"type": "enabled"}}, "high"), ({"thinking": {"type": "adaptive"}}, "high"),
                 ({"thinking": {"type": "enabled"}, "reasoning_effort": "low"}, "low"),
                 ({"thinking": {"type": "enabled"}, "reasoning_effort": "none"}, "high"),
                 ({"thinking": {"type": "disabled"}, "reasoning_effort": "max"}, "none"),
                 ({"reasoning_effort": "ultra"}, "max")]
        for extra, effort in cases:
            result = normalize({**base, **extra}, cfg)
            self.assertEqual(result["reasoning_effort"], effort, extra)
            self.assertEqual(result["chat_template_kwargs"]["thinking"], effort != "none", extra)
            self.assertNotIn("thinking", result)
        with self.assertRaises(PolicyError):
            normalize({**base, "thinking": {"type": "sometimes"}}, cfg)

    def test_responses_admission_uses_the_native_tokenizer_like_chat(self):
        seen = []
        def native(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"count": 7, "max_model_len": 1048576})
        async def run():
            from chimera_sparks.engine import Engine
            engine = Engine(Config(), httpx.MockTransport(native))
            try:
                return await engine.count_input({"model": "deepseek-v4.1-flash", "input": "x" * 5000}, "responses")
            finally:
                await engine.close()
        self.assertEqual(asyncio.run(run())[0], 7)
        self.assertEqual(seen[0]["api"], "responses")

    def test_bad_settings_and_internal_ownership(self):
        base = {"messages": [{"role": "user", "content": "hello"}]}
        for bad in [{"temperature": float("nan")}, {"top_k": True}, {"top_p": 0}, {"max_tokens": -1}]:
            with self.assertRaises(PolicyError):
                normalize({**base, **bad}, Config())
        with self.assertRaises(PolicyError):
            normalize({**base, "reasoning_effort": "max", "chat_template_kwargs": {"thinking": False}}, Config())
        with self.assertRaises(PolicyError):
            normalize({**base, "reasoning_effort": "none", "chat_template_kwargs": {"thinking": True}}, Config())
        value = normalize({**base, "dgx_vision_control": {"request_id": "foreign"}, "request_id": "foreign"}, Config())
        self.assertNotIn("dgx_vision_control", value)
        self.assertNotIn("request_id", value)


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["RECIPE_TEST_KEY"] = "test-credential-only-123456789"
        self.seen = []
        self.truncate = False
        async def upstream(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True, "warming": False})
            raw = json.loads(request.content)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 10})
            if request.url.path == "/v1/messages/count_tokens":
                return httpx.Response(200, json={"input_tokens": 10})
            self.seen.append(raw)
            if request.url.path == "/v1/messages":
                self.paths.append(request.url.path)
                if raw.get("stream"):
                    events = [{"type": "message_start"}, {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}}, {"type": "message_stop"}]
                    data = b"".join(("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n").encode() for e in events[: 2 if self.truncate else 3])
                    return httpx.Response(200, content=data, headers={"content-type": "text/event-stream"})
                return httpx.Response(200, json={"type": "message", "role": "assistant", "content": [{"type": "text", "text": "hello"}]})
            if raw.get("stream"):
                events = [{"choices": [{"delta": {"reasoning_content": "private thought"}, "finish_reason": None}]},
                          {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
                          {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
                data = b"".join(("data: " + json.dumps(e) + "\n\n").encode() for e in events)
                if not self.truncate:
                    data += b"data: [DONE]\n\n"
                return httpx.Response(200, content=data, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello", "reasoning_content": "thought"}, "finish_reason": "stop"}]})
        self.cfg = Config(state_dir=self.temp.name, clients={"test": "RECIPE_TEST_KEY"})
        self.app = create_app(self.cfg, transport=httpx.MockTransport(upstream))
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
                                       headers={"Authorization": "Bearer " + os.environ["RECIPE_TEST_KEY"]})

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.life.__aexit__(None, None, None)
        self.temp.cleanup()

    async def test_length_cut_while_thinking_is_a_normal_finish(self):
        def native(request):
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 10})
            if json.loads(request.content).get("stream"):
                events = [{"choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}]},
                          {"choices": [{"delta": {}, "finish_reason": "length"}]}]
                data = b"".join(("data: " + json.dumps(e) + "\n\n").encode() for e in events) + b"data: [DONE]\n\n"
                return httpx.Response(200, content=data, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json={"choices": [{"message": {"reasoning_content": "thinking"}, "finish_reason": "length"}]})
        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)
        app = create_app(replace(self.cfg, state_dir=state.name), transport=httpx.MockTransport(native))
        async with app.router.lifespan_context(app):
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                                       headers={"Authorization": "Bearer " + os.environ["RECIPE_TEST_KEY"]})
            base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "reasoning_effort": "high"}
            try:
                plain = await client.post("/v1/chat/completions", json=base)
                stream = await client.post("/v1/chat/completions", json={**base, "stream": True})
            finally:
                await client.aclose()
        self.assertEqual(plain.status_code, 200, plain.text)
        self.assertIn("[DONE]", stream.text)
        self.assertNotIn("stream_incomplete", stream.text)

    async def test_messages_forwards_natively_and_client_settings_win(self):
        self.paths = []
        base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}
        keyed = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
                                  headers={"x-api-key": os.environ["RECIPE_TEST_KEY"]})
        try:
            answer = await keyed.post("/v1/messages", json=base)
        finally:
            await keyed.aclose()
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()["content"][0]["text"], "hello")
        self.assertEqual(self.paths, ["/v1/messages"])
        sent = self.seen[-1]
        wanted = "disabled" if self.cfg.reasoning_default == "none" else "enabled"
        self.assertEqual((sent["top_k"], sent["thinking"]), (20, {"type": wanted}))
        self.assertEqual(sent["messages"], base["messages"])
        self.assertGreater(sent["seed"], 0)
        await self.client.post("/v1/messages", json={**base, "temperature": .3, "top_k": 5,
                               "thinking": {"type": "enabled", "budget_tokens": 32}, "output_config": {"effort": "low"}})
        sent = self.seen[-1]
        self.assertEqual((sent["temperature"], sent["top_k"], sent["output_config"]), (.3, 5, {"effort": "low"}))
        self.assertEqual(sent["thinking"], {"type": "enabled", "budget_tokens": 32})
        self.assertEqual(self.app.state.engine.gate.active_weight, 0)

    async def test_messages_stream_and_count_tokens(self):
        self.paths = []
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "stream": True}
        ok = await self.client.post("/v1/messages", json=body)
        self.assertIn("message_stop", ok.text)
        self.truncate = True
        failed = await self.client.post("/v1/messages", json=body)
        self.assertNotIn("message_stop", failed.text)
        self.assertEqual(self.app.state.engine.gate.active, 0)
        count = await self.client.post("/v1/messages/count_tokens", json={"model": "x", "messages": []})
        self.assertEqual(count.json(), {"input_tokens": 10})
        bad = await self.client.post("/v1/messages/count_tokens", json={}, headers={"Authorization": "Bearer incorrect"})
        self.assertEqual(bad.status_code, 401)

    async def test_auth_sampler_and_admission_release(self):
        bad = await self.client.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer incorrect"})
        self.assertEqual(bad.status_code, 401)
        answer = await self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64})
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(self.seen[-1]["top_p"], 1)
        self.assertEqual(self.seen[-1]["top_k"], 20)
        self.assertEqual(self.app.state.engine.gate.active_weight, 0)

    async def test_stream_reasoning_terminal_and_no_replay(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "stream": True}
        ok = await self.client.post("/v1/chat/completions", json=body)
        self.assertIn("private thought", ok.text)
        self.assertIn("[DONE]", ok.text)
        self.truncate = True
        failed = await self.client.post("/v1/chat/completions", json=body)
        self.assertIn("stream_incomplete", failed.text)
        self.assertEqual(len(self.seen), 2)
        self.assertEqual(self.app.state.engine.gate.active, 0)

    async def test_context_rejected_before_generation(self):
        self.app.state.engine.cfg = replace(self.cfg, context_tokens=100)
        answer = await self.client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64})
        self.assertEqual(answer.status_code, 400)
        self.assertEqual(self.seen, [])

    async def test_malformed_messages_and_effort_are_400_not_500(self):
        for bad in ({"messages": 123}, {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": {}},
                    {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": True}):
            answer = await self.client.post("/v1/chat/completions", json=bad)
            self.assertEqual(answer.status_code, 400, (bad, answer.text))
        self.assertEqual(self.seen, [])


if __name__ == "__main__":
    unittest.main()


def test_hide_drops_reasoning_items_and_parts_from_the_responses_stream():
    import json, re
    from pathlib import Path
    from chimera_sparks.api import project_reasoning, reasoning_event
    golden = Path(__file__).resolve().parents[2] / "engine/tensorfold/zig/tests/server/golden/responses/input-stream.json"
    body = json.loads(golden.read_text())["replies"][0]["body"]
    events = [json.loads(re.sub(r'(?<!")<[a-z0-9_]+>(?!")', "0", m.group(1)))
              for frame in body.split("\n\n") if (m := re.search(r"^data: (.*)$", frame, re.M))]
    assert any(reasoning_event(e) and not e["type"].startswith("response.reasoning") for e in events)
    kept = [project_reasoning(e, "hide") for e in events if not reasoning_event(e)]
    text = json.dumps(kept)
    assert "reasoning_text" not in text and '"type": "reasoning"' not in text
    assert any(e["type"] == "response.output_text.delta" for e in kept)
    assert project_reasoning({"item": {"type": "reasoning", "summary": []}, "type": "x"}, "hide") == {"type": "x"}
