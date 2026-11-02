import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import httpx

from chimera_sparks.api import until_disconnect, create_app
from chimera_sparks.assets.rotating_run import Rotator, size_bytes
from chimera_sparks.config import Config
from chimera_sparks.engine import Engine
from chimera_sparks.policy import normalize
from chimera_sparks.prepare import writable_index
from chimera_sparks.supervisor import check_ownership, OwnershipChanged


class FileSafetyTests(unittest.TestCase):
    def test_supervisor_fences_any_replaced_follower_before_group_cleanup(self):
        expected = ["head-a", "follower-b", "follower-c"]
        rows = [{"active": True, "invocation": identity} for identity in expected]
        check_ownership(rows, expected)
        rows[0] = {"active": False, "invocation": ""}
        check_ownership(rows, expected)
        rows[2] = {"active": True, "invocation": "other-owner"}
        with self.assertRaises(OwnershipChanged): check_ownership(rows, expected)
    def test_readonly_hardlinked_index_becomes_independent_without_touching_shards(self):
        with tempfile.TemporaryDirectory() as root:
            snapshot, work = Path(root) / "snapshot", Path(root) / "overlay"
            snapshot.mkdir(); work.mkdir()
            source = snapshot / "model.safetensors.index.json"
            source.write_text('{"weight_map":{"x":"shard.safetensors"}}')
            source.chmod(0o444)
            shard = snapshot / "shard.safetensors"
            shard.write_bytes(b"unchanged")
            os.link(source, work / source.name)
            os.link(shard, work / shard.name)
            result = writable_index(snapshot, work)
            self.assertFalse(os.path.samefile(source, result))
            self.assertTrue(result.stat().st_mode & 0o200)
            self.assertEqual(source.stat().st_mode & 0o777, 0o444)
            self.assertTrue(os.path.samefile(shard, work / shard.name))
            result.write_text('{"overlay":"edited"}')
            self.assertEqual(writable_index(snapshot, work).read_text(), '{"overlay":"edited"}')
            self.assertIn("weight_map", json.loads(source.read_text()))

    def test_logs_keep_only_bounded_latest_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rank.log"
            logger = Rotator(path, 32, 3)
            payload = bytes(range(160))
            logger.write(payload[:17]); logger.write(payload[17:]); logger.close()
            files = [Path(str(path) + ".2"), Path(str(path) + ".1"), path]
            self.assertEqual(b"".join(p.read_bytes() for p in files), payload[-96:])
            self.assertTrue(all(p.stat().st_size <= 32 for p in files))
            self.assertEqual(size_bytes("100m"), 100 * 1024**2)
            for bad in ["", "0m", "1;echo", "1.5m", "-1m"]:
                with self.assertRaises(ValueError): size_bytes(bad)

    def test_logger_preserves_real_child_exit_75(self):
        with tempfile.TemporaryDirectory() as root:
            runner = Path(__file__).resolve().parents[1] / "src/chimera_sparks/assets/rotating_run.py"
            result = subprocess.run([sys.executable, str(runner), "--log", str(Path(root)/"rank.log"),
                "--max-size", "1k", "--max-files", "3", "--", sys.executable, "-c",
                "print('peer failure'); raise SystemExit(75)"], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 75, result.stderr)
            self.assertIn(b"peer failure", (Path(root)/"rank.log").read_bytes())


class DisconnectTests(unittest.IsolatedAsyncioTestCase):
    async def test_incoming_reasoning_guard_rejects_loops_but_keeps_numeric_tables(self):
        os.environ["SAFETY_TEST_KEY"] = "offline-safety-test-key-123456789"
        async def upstream(request):
            if request.url.path == "/tokenize": return httpx.Response(200, json={"count":10})
            return httpx.Response(200, json={"choices":[{"message":{"content":"done"},"finish_reason":"stop"}]})
        with tempfile.TemporaryDirectory() as root:
            app = create_app(Config(state_dir=root, clients={"test":"SAFETY_TEST_KEY"}), transport=httpx.MockTransport(upstream))
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test",
                    headers={"Authorization":"Bearer "+os.environ["SAFETY_TEST_KEY"]}) as client:
                    body = {"messages":[{"role":"assistant","content":"prior","reasoning_content":"repeat this endlessly "*50},
                        {"role":"user","content":"answer"}],"max_tokens":64}
                    bad = await client.post("/v1/chat/completions", json=body)
                    self.assertEqual(bad.status_code,422)
                    self.assertEqual(app.state.engine.requests,0)
                    body["messages"][0]["reasoning_content"] = " ".join("value "+str(i) for i in range(90))
                    good = await client.post("/v1/chat/completions", json=body)
                    self.assertEqual(good.status_code,200,good.text)
                    self.assertEqual(app.state.engine.requests,1)
    async def test_nonstream_disconnect_closes_native_socket_and_releases_admission(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        class Slow(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b"never"
            async def aclose(self): closed.set()
        requests = []
        async def upstream(request):
            requests.append(request.url.path)
            if request.url.path == "/tokenize": return httpx.Response(200, json={"count":10})
            return httpx.Response(200, stream=Slow())
        class Client:
            async def is_disconnected(self):
                await entered.wait()
                return True
        engine = Engine(Config(), httpx.MockTransport(upstream))
        async def operation():
            handle = None
            try:
                body = normalize({"messages":[{"role":"user","content":"hello"}],"max_tokens":64}, Config())
                handle = await engine.begin_stream(body)
                await handle.response.aread()
            finally:
                if handle: await asyncio.shield(engine.close_stream(handle))
        try:
            result = await until_disconnect(Client(), operation())
            self.assertEqual(result.status_code, 499)
            self.assertTrue(closed.is_set())
            self.assertEqual(engine.gate.active, 0)
            self.assertEqual(requests, ["/tokenize", "/v1/chat/completions"])
        finally:
            await engine.close()


if __name__ == "__main__": unittest.main()
