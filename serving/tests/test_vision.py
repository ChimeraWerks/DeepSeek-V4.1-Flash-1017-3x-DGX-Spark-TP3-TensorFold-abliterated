import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
import httpx

from chimera_sparks.api import create_app
from chimera_sparks.config import Config
from chimera_sparks.laky.vision_jobs import Ledger, AdmissionError, repeated
from chimera_sparks.laky.vision_control import confirms_termination
from chimera_sparks.laky.vision_result import Collector


class LedgerTests(unittest.TestCase):
    def test_restart_and_idempotency_do_not_replay(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = Ledger(Path(root) / "jobs.sqlite3", controlled=True)
            body = {"messages": [{"role": "user", "content": "question"}]}
            identity, created = ledger.accept("alice", body, "key")
            self.assertTrue(created)
            self.assertEqual(ledger.accept("alice", {**body, "stream": True}, "key"), (identity, False))
            with self.assertRaises(AdmissionError):
                ledger.accept("alice", {"messages": []}, "key")
            self.assertIsNone(ledger.read(identity, "bob"))
            claimed = ledger.claim()
            self.assertEqual(claimed["id"], identity)
            ledger.dispatch(identity, {"instance": "i", "request_id": "r", "expires": 123})
            ledger.recover()
            self.assertEqual(ledger.read(identity)["state"], "unknown")
            self.assertEqual(ledger.accept("alice", body, "key"), (identity, False))
            self.assertIsNone(ledger.claim())

    def test_receipt_binding_and_numeric_repetition(self):
        dispatch = {"instance": "old", "request_id": "r"}
        receipt = {"schema": "tensorfold.vision-control.v1", "instance": "old", "requested_instance": "old",
                   "request_id": "r", "termination_confirmed": True, "state": "terminated"}
        self.assertTrue(confirms_termination(receipt, dispatch))
        self.assertFalse(confirms_termination({**receipt, "request_id": "other"}, dispatch))
        self.assertFalse(repeated(" ".join("value " + str(i) for i in range(90))))
        self.assertTrue(repeated("repeat this endlessly " * 50))

    def test_repetition_is_drained_and_never_accepted(self):
        collector = Collector()
        def feed(delta, finish=None):
            collector.feed(("data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n").encode())
        feed({"content": "repeat this endlessly " * 50})
        self.assertTrue(collector.repetitive)
        feed({"content": "later answer"}, "stop")
        collector.feed(b"data: [DONE]\n\n")
        self.assertEqual(collector.result("job")["status_code"], 422)


class DurableApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_accept_poll_dedup_and_native_dispatch(self):
        os.environ["VISION_TEST_KEY"] = "vision-unit-test-credential-12345"
        sent = []
        async def upstream(request):
            if request.url.path == "/internal/vision/control":
                return httpx.Response(200, json={"schema": "tensorfold.vision-control.v1", "instance": "native-i"})
            body = json.loads(request.content)
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 20})
            sent.append(body)
            events = [{"id": "completion", "model": "deepseek-v4.1-flash", "choices": [
                {"delta": {"content": "Blue"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
            return httpx.Response(200, content=b"".join(("data: " + json.dumps(v) + "\n\n").encode() for v in events) + b"data: [DONE]\n\n")
        with tempfile.TemporaryDirectory() as root:
            cfg = Config(state_dir=root, clients={"alice": "VISION_TEST_KEY"})
            app = create_app(cfg, transport=httpx.MockTransport(upstream))
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                    headers={"Authorization": "Bearer " + os.environ["VISION_TEST_KEY"]}) as client:
                    body = {"messages": [{"role": "user", "content": [
                        {"type": "text", "text": "color?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
                        "max_tokens": 64, "temperature": 0, "reasoning_effort": "none"}
                    response = await client.post("/v1/vision/jobs", json=body, headers={"Idempotency-Key": "same"})
                    self.assertEqual(response.status_code, 202, response.text)
                    identity = response.json()["job_id"]
                    for _ in range(50):
                        result = (await client.get("/v1/vision/jobs/" + identity)).json()
                        if result["status"] in {"succeeded", "failed", "unknown"}:
                            break
                        await asyncio.sleep(.01)
                    self.assertEqual(result["status"], "succeeded", result)
                    again = await client.post("/v1/vision/jobs", json=body, headers={"Idempotency-Key": "same"})
                    self.assertEqual(again.json()["job_id"], identity)
                    self.assertEqual(len(sent), 1)
                    self.assertEqual(sent[0]["max_tokens"], 64)
                    self.assertEqual(sent[0]["dgx_vision_control"]["instance"], "native-i")
                    self.assertEqual(app.state.engine.gate.active, 0)


if __name__ == "__main__":
    unittest.main()
