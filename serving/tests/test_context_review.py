"""Offline reproductions of the recipe-side review; no Spark or model traffic."""
import contextlib
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from chimera_sparks import lifecycle
from chimera_sparks.policy import normalize
from chimera_sparks.shortening import Shortener, CompactionError
import test_shortening as shortening_tests
from test_mia_context_tools import module


class ActiveRankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.binary = str(Path(sys.executable).resolve())
        self.data = Path(self.binary).read_bytes()
        self.env = {"TF_DS_POOL_TOKENS": "7004160", "TF_DS_WINDOW": "1048576",
                    "TF_DS_ROUND_ROWS": "48", "TF_DS_KV_SHARD": "1"}
        self.item = {"node": {"native": {"root": self.temp.name},
                             "checksums": {self.binary: hashlib.sha256(self.data).hexdigest()}},
                     "argv": [self.binary], "launch": [self.binary, "-I", "logger.py", "--", self.binary], "unit": "offline-rank", "environment": self.env}
        captured = []
        with patch.object(lifecycle, "remote", lambda node, script, item: captured.append(script)):
            lifecycle.provision_and_check(self.item)
        self.script = captured[0]

    def check_active(self, actual, data=None, item=None):
        wanted = deepcopy(item or self.item)
        raw = b"\0".join((k + "=" + v).encode() for k, v in actual.items()) + b"\0"
        real_read = Path.read_bytes
        real_open = Path.open
        real_text = Path.read_text
        def read_text(path, *args, **kwargs):
            if str(path) == "/proc/424242/task/424242/children":
                return "424243"
            return real_text(path, *args, **kwargs)
        def read(path):
            if str(path) == "/proc/424242/cmdline":
                return b"\0".join(v.encode() for v in wanted["launch"]) + b"\0"
            if str(path) == "/proc/424243/cmdline":
                return self.binary.encode() + b"\0"
            if str(path) == "/proc/424243/environ":
                return raw
            return real_read(path)
        def open_file(path, *args, **kwargs):
            if str(path) == "/proc/424243/exe":
                return io.BytesIO(self.data if data is None else data)
            return real_open(path, *args, **kwargs)
        def run(command, **kwargs):
            if command != ["systemctl", "--user", "is-active", "offline-rank"]:
                raise AssertionError("unexpected command")
            return SimpleNamespace(stdout="active\n")
        def pid(command, **kwargs):
            self.assertEqual(command, ["systemctl", "--user", "show", "offline-rank", "-p", "MainPID", "--value"])
            return b"424242\n"
        with patch.object(Path, "read_text", read_text), patch.object(Path, "read_bytes", read), patch.object(Path, "open", open_file), \
             patch.object(subprocess, "run", run), patch.object(subprocess, "check_output", pid), \
             patch.object(sys, "argv", ["preflight", json.dumps(wanted)]), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(compile(self.script, "offline-preflight", "exec"), {})

    def test_matching_active_unit_is_accepted(self):
        self.check_active(self.env)

    def test_on_to_off_and_extra_managed_setting_are_refused(self):
        item = deepcopy(self.item)
        item["environment"].pop("TF_DS_KV_SHARD")
        with self.assertRaisesRegex(ValueError, "managed settings"):
            self.check_active(self.env, item=item)
        with self.assertRaisesRegex(ValueError, "managed settings"):
            self.check_active({**self.env, "TF_DS_SPILL_GIB": "64"})

    def test_pool_window_round_and_every_persistent_identity_are_compared(self):
        keys = ["TF_DS_POOL_TOKENS", "TF_DS_WINDOW", "TF_DS_ROUND_ROWS",
                "TF_DS_RANK_CACHE_SHA", "TF_DS_ENGINE_ID", "TF_DS_RECIPE_PATCHES",
                "TF_DS_BUILD_FLAGS", "TF_DS_MODEL_REVISION", "TF_DS_ENGRAM_REVISION",
                "TF_DS_ABLATION_ID"]
        for key in keys:
            with self.subTest(key=key):
                item = deepcopy(self.item)
                item["environment"][key] = "new"
                with self.assertRaisesRegex(ValueError, "managed settings"):
                    self.check_active({**self.env, key: "old"}, item=item)

    def test_executable_identity_checked_with_features_off(self):
        item = deepcopy(self.item)
        item["environment"] = {}
        with self.assertRaisesRegex(ValueError, "another executable"):
            self.check_active({}, data=b"old executable", item=item)


class ResponseInstructionTests(unittest.IsolatedAsyncioTestCase):
    cfg = shortening_tests.ShorteningTests.cfg
    history = shortening_tests.ShorteningTests.history
    count = shortening_tests.ShorteningTests.count
    completion = shortening_tests.ShorteningTests.completion

    def body(self):
        items = [{"type": "message", **m} for m in self.history()[:4]]
        items.insert(2, {"type": "message", "role": "developer", "content": "Return JSON only."})
        items.extend([{"type": "message", "role": "user", "content": "Current task"},
            {"type": "function_call", "call_id": "c1", "name": "inspect", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "receipt"}])
        return normalize({"input": items, "instructions": "Top-level contract"}, self.cfg(), api="responses")

    async def test_normalized_input_instructions_keep_roles_content_and_order(self):
        body = self.body()
        protected = [m for m in body["input"] if m.get("role") in {"system", "developer"}]
        result, compacted = await Shortener(self.cfg(), self.count, self.completion).compact(body, 20000, "responses")
        self.assertTrue(compacted)
        self.assertEqual(result["input"][:2], protected)
        self.assertIn("critical_evidence_jsonl", json.dumps(result["input"][2]))
        self.assertEqual(result["instructions"], body["instructions"])
        self.assertEqual(result["input"][-3:], body["input"][-3:])

    async def test_oversized_protected_contract_is_refused(self):
        body = self.body()
        body["input"][0]["content"] *= 2000
        with self.assertRaises(CompactionError):
            await Shortener(self.cfg(), self.count, self.completion).compact(body, 20000, "responses")


class SpillObservationTests(unittest.TestCase):
    def setUp(self):
        self.spill = module("spill_check")
        self.run = object.__new__(self.spill.Run)
        self.run.a = SimpleNamespace(ranks=3, allow_weak=False, max_gap=0, timeout=1,
            spill_min_tokens=1, floor_margin_bytes=0, bytes_per_position=890, small_state_bytes=1, commit_timeout=0)
        self.run.rc = self.spill.C.Receipt("offline-test", {})
        self.run.ranks = {rank: SimpleNamespace(list=lambda: {}, new_files=lambda before: {}) for rank in range(3)}
        self.run.mt = self.run.dt = 4
        self.run.m = lambda name, query: []
        self.strong = {"prompt_sha": "prompt", "token_sha_source": "server", "token_sha": "a"*64,
                       "text_sha": "same", "stream_complete": True}
        self.weak = {**self.strong, "token_sha_source": "text"}
        self.run.base = {"rows": {name: {"hit": self.strong, "seed": self.strong}
            for name in ["D1", "D2", self.spill.DECODER, "Z1", "F1"]}}
        self.run.hit = lambda name: dict(self.strong)
        self.run.its = {"Z1": 3000, "F1": 1000}
        self.run.c = lambda: {"prompt_cache.disk_hits_committed": 10}
        self.run.srv = SimpleNamespace(stream=lambda *args: dict(self.weak))
        self.real_delta = self.spill.C.delta

    def failures(self):
        return self.run.rc.failed()

    def test_streaming_sizes_refuses_weak_and_incomplete_rows(self):
        for row in (self.weak, {**self.strong, "stream_complete": False}):
            with self.subTest(row=row):
                self.run.rc = self.spill.C.Receipt("offline", {})
                self.run.srv.stream = lambda *args: dict(row)
                self.run.sizes()
                self.assertIn("sizes", self.failures())

    def test_dual_restore_refuses_weak_and_incomplete_decoder(self):
        for row in (self.weak, {**self.strong, "stream_complete": False}):
            with self.subTest(row=row):
                self.run.rc = self.spill.C.Receipt("offline", {})
                def stream(*args):
                    args[-1].set()
                    return dict(row)
                self.run.srv.stream = stream
                with patch.object(self.spill.C, "deltas", return_value={"prompt_cache.disk_hits_committed": 2}):
                    self.run.dual()
                self.assertIn("dual-restore", self.failures())

    def test_missing_rank2_or_aggregate_recovery_never_passes(self):
        for counters in ({"spill.recovered_points.rank0": 1, "spill.recovered_points.rank1": 1},
                         {"spill.recovered_points": 3}):
            self.assertFalse(self.run.recovered(counters))

    def test_floor_accepts_complete_three_rank_observations(self):
        before = {f"spill.{key}.rank{rank}": 0 for rank in range(3)
                  for key in ("no_space", "writes_committed", "writes_pending", "disk_free_bytes")}
        after = dict(before)
        for rank in range(3):
            after[f"spill.no_space.rank{rank}"] = 1
            after[f"spill.disk_free_bytes.rank{rank}"] = 1000
        counters = iter((before, after))
        self.run.c = lambda: next(counters, after)
        self.run.srv.chat = lambda *args: dict(self.strong)
        self.run.phase_floor()
        self.assertFalse(self.failures())

    def test_picture_requires_every_rank_not_an_aggregate(self):
        self.run.files = {}
        self.run.wait_commit = lambda before: {0: 1, 1: 1, 2: 1}
        self.run.srv.chat = lambda *args: dict(self.strong)
        self.run.base["rows"]["P1"] = {"seed": self.strong}
        self.run.c = lambda: {"spill.pictures_kept": 3}
        self.run.seed_item("P1")
        self.assertIn("seed P1 kept with its picture", self.failures())

    def test_temporary_cleanup_requires_every_directory(self):
        self.run.ranks.pop(2)
        self.run.temporaries("cleanup")
        self.assertIn("rank directories", self.failures())

    def test_floor_rejects_missing_rank2_and_aggregate_counters(self):
        for ranks in ((0, 1), ()):
            with self.subTest(ranks=ranks):
                before = {}
                after = {}
                for rank in ranks:
                    for key in ("no_space", "writes_committed", "disk_free_bytes"):
                        before[f"spill.{key}.rank{rank}"] = 0
                        after[f"spill.{key}.rank{rank}"] = 1 if key != "writes_committed" else 0
                after.update({"spill.no_space": 3, "spill.writes_committed": 0})
                states = iter((before, after))
                self.run.c = lambda: next(states, after)
                self.run.srv.chat = lambda *args: dict(self.strong)
                self.run.rc = self.spill.C.Receipt("offline", {})
                self.run.phase_floor()
                self.assertIn("floor", self.failures())


class CredentialAndStreamTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.exact = module("exact")
        self.client = module("client")
        self.c = self.exact.C
        self.c._api_key_file = None
        self.addCleanup(setattr, self.c, "_api_key_file", None)

    def test_import_and_defaults_do_not_read_any_credential_file_or_install_opener(self):
        with patch("builtins.open", side_effect=AssertionError("implicit credential read")), \
             patch.object(urllib.request, "install_opener", side_effect=AssertionError("global opener")):
            module("client")
            module("exact")
            os.environ["API_KEY_FILE"] = "/unrelated/private-key"
            self.assertEqual(self.c.api_key(), "")

    def test_explicit_key_file_and_environment_selection(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "selected"
            path.write_text("selected-test-credential\n")
            self.c.use_api_key_args(SimpleNamespace(api_key_file=str(path)))
            self.assertEqual(self.c.api_key(), "selected-test-credential")
            os.environ["API_KEY"] = "runtime-test-credential"
            self.assertEqual(self.c.api_key(), "runtime-test-credential")

    def test_same_origin_headers_redirect_refusal_and_no_cross_origin_client_auth(self):
        os.environ["API_KEY"] = "runtime-test-credential"
        seen = []
        class Opener:
            def open(inner, request, **kwargs):
                seen.append(request)
                return io.BytesIO(b'{"ok":true}')
        with patch.object(self.exact, "OPENER", Opener()):
            self.exact.post("http://chosen.invalid", {}, 1)
        self.assertEqual(seen[0].get_header("Authorization"), "Bearer runtime-test-credential")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.c.SameOriginRedirect().redirect_request(seen[0], None, 302, "", {}, "http://other.invalid/v1")
        caught.exception.close()
        self.client.API_URL = "http://chosen.invalid"
        with self.assertRaises(ValueError):
            self.client.open_url(urllib.request.Request("http://other.invalid/v1"), 1)

    def test_selected_client_origin_receives_headers_without_global_opener(self):
        os.environ["API_KEY"] = "runtime-test-credential"
        self.client.API_URL = "http://chosen.invalid"
        requests = []
        class Opener:
            def open(inner, req, **kwargs):
                requests.append(req)
                return io.BytesIO(b"response")
        with patch.object(self.client.C, "OPENER", Opener()):
            response = self.client.open_url(urllib.request.Request("http://chosen.invalid/v1"), 1)
            response.close()
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer runtime-test-credential")

    def test_server_errors_and_logs_are_redacted(self):
        os.environ["API_KEY"] = "runtime-test-credential"
        error = urllib.error.HTTPError("http://chosen.invalid", 400, "", {},
                                       io.BytesIO(b"echo runtime-test-credential"))
        class Opener:
            def open(inner, *args, **kwargs):
                raise error
        with patch.object(self.exact, "OPENER", Opener()):
            result = self.exact.post("http://chosen.invalid", {}, 1)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.exact.log("echo runtime-test-credential")
        self.assertNotIn("runtime-test-credential", json.dumps(result) + output.getvalue())

    def test_stream_requires_finish_and_done(self):
        complete = [b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n', b"data: [DONE]\n"]
        class Connection:
            def __init__(inner, *args, **kwargs): pass
            def request(inner, *args, **kwargs): pass
            def close(inner): pass
            def getresponse(inner):
                return SimpleNamespace(status=200, readline=lambda: next(lines, b""))
        for events, valid in ((complete, True), (complete[:1], False), (complete[1:], False)):
            lines = iter(events)
            with patch.object(self.c.http.client, "HTTPConnection", Connection):
                row = self.c.Server("http://chosen.invalid", "test", 1, 1).stream([], 1)
            self.assertEqual(row["stream_complete"], valid)
            self.assertEqual("error" not in row, valid)


class SpillFilesystemTests(unittest.TestCase):
    def setUp(self):
        self.c = module("checklib")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.namespace = "native-spill-" + "a"*64
        self.prefix = self.namespace + "/rank2/points/"
        self.parent = self.root / self.namespace / "rank2/points"
        self.parent.mkdir(parents=True)
        self.path = self.parent / "point.man"
        self.path.write_bytes(b"abcdefgh")
        self.outside = self.root.parent / (self.root.name + "-outside")
        self.outside.write_bytes(b"unchanged")
        self.addCleanup(self.outside.unlink)
        (self.root / ".chimera-spill-test").write_text(json.dumps({
            "schema": "chimera.spill-test.v1", "isolated": True,
            "rank": 2, "namespace": self.namespace}))
        self.rank = self.c.RankDir(str(self.root), rank=2)

    def test_valid_isolated_truncate_and_remove(self):
        self.rank.run("truncate", path=self.prefix + "point.man", to=4)
        self.assertEqual(self.path.read_bytes(), b"abcd")
        self.rank.run("remove", path=self.prefix + "point.man")
        self.assertFalse(self.path.exists())
        self.assertEqual(self.outside.read_bytes(), b"unchanged")

    def test_absolute_parent_empty_and_wrong_rank_paths_are_refused(self):
        for path in ("", str(self.outside), "../../" + self.outside.name,
                     self.namespace + "/rank1/points/point.man"):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                self.rank.run("remove", path=path)
        self.assertEqual(self.outside.read_bytes(), b"unchanged")

    def test_symlinks_hardlinks_and_missing_marker_are_refused(self):
        target = self.parent / "link.man"
        target.symlink_to(self.outside)
        with self.assertRaises(RuntimeError):
            self.rank.run("corrupt", path=self.prefix + "link.man")
        with self.assertRaises(RuntimeError):
            self.rank.list()
        target.unlink()
        os.link(self.outside, target)
        with self.assertRaises(RuntimeError):
            self.rank.run("truncate", path=self.prefix + "link.man")
        (self.root / ".chimera-spill-test").unlink()
        with self.assertRaises(RuntimeError):
            self.rank.run("remove", path=self.prefix + "point.man")
        self.assertEqual(self.outside.read_bytes(), b"unchanged")
