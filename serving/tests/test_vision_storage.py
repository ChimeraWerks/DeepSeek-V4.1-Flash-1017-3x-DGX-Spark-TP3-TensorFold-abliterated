# Copyright (c) 2026 LakyFx. MIT; see LICENSES/LakyFx-MIT.txt.
# Adapted from gateway/tests/test_vision_storage.py; see serving/UPSTREAM.json.
"""Storage port regressions, using temporary ledgers and a mocked native API."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx

from chimera_sparks.api import create_app
from chimera_sparks.config import Config
from chimera_sparks.laky import vision_storage as storage
from chimera_sparks.laky.vision_jobs import AdmissionError, Jobs, Ledger


def body(text="what color?", seed=0):
    return {"messages": [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
        "seed": seed, "max_tokens": 64, "reasoning_effort": "none"}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = [100_000.0]
        self.ledger = Ledger(self.root / "vision.sqlite3",
                             clock=lambda: self.clock[0], controlled=True)
        for name, value in (("HOT_LIMIT", 64 * 1024), ("RESULT_RESERVE", 1024),
                            ("REPAIR_RESERVE", 512)):
            self.enterContext(patch.object(storage, name, value))

    def old_job(self, payload=None, state="succeeded", key="old"):
        identity, _ = self.ledger.accept("owner", payload or body(), key)
        with self.ledger.connect() as db:
            db.execute("UPDATE jobs SET state=?,deadline=?,updated=?,result=? WHERE id=?",
                       (state, self.clock[0] - 1800, self.clock[0] - 1800,
                        json.dumps({"status_code": 200, "data": {"answer": "preserved"}}), identity))
        return identity

    def attempts(self, identity, state="unknown"):
        # Optional future/rollback recovery records remain protected even though
        # this port intentionally does not enable recovery or replay.
        with self.ledger.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS vision_attempts (
                job_id TEXT, number INTEGER, state TEXT, body TEXT,
                result TEXT, dispatch TEXT, receipt TEXT)""")
            db.execute("INSERT INTO vision_attempts VALUES (?,1,?,?,NULL,NULL,NULL)",
                       (identity, state, json.dumps(body(text="attempt image"))))

    def test_full_history_archives_losslessly_then_accepts_and_dispatches(self):
        old = self.old_job(body(text="x" * 16000))
        with self.ledger.connect() as db:
            db.execute("UPDATE jobs SET body=? WHERE id=?", (json.dumps(body(text="x" * 64000)), old))
        before = self.ledger.read(old)
        new, created = self.ledger.accept("owner", body(seed=42), "new")
        self.assertTrue(created)
        self.assertEqual(self.ledger.claim()["id"], new)
        self.assertEqual(self.ledger.read(old), before)
        self.assertEqual(self.ledger.status(old, "owner")["result"]["data"]["answer"], "preserved")
        with self.ledger.connect() as db:
            self.assertEqual(db.execute("SELECT body FROM jobs WHERE id=?", (old,)).fetchone()[0], "")
        self.assertEqual(self.ledger.capacity()["archived_jobs"], 1)

    def test_archived_idempotency_and_owner_boundaries_are_preserved(self):
        old = self.old_job(body(seed=43))
        with self.ledger.connect() as db:
            self.assertTrue(self.ledger.storage.archive(db, old))
        self.assertEqual(self.ledger.accept("owner", body(seed=43), "old"), (old, False))
        self.assertIsNone(self.ledger.read(old, "someone else"))
        with self.assertRaisesRegex(AdmissionError, "different input"):
            self.ledger.accept("owner", body(seed=44), "old")

    def test_active_and_unknown_never_archive_or_expire(self):
        for state in ("queued", "running", "unknown"):
            with self.subTest(state=state):
                identity = self.old_job(state=state, key=state)
                # Make UNKNOWN an owned dispatch, so it does not block other admission.
                with self.ledger.connect() as db:
                    db.execute("INSERT INTO vision_dispatch VALUES (?,?,NULL)",
                               (identity, json.dumps({"instance": "i", "request_id": state})))
                self.clock[0] += storage.RESULT_TTL * 2
                self.ledger.storage.maintenance()
                with self.ledger.connect() as db:
                    self.assertNotIn(identity, {r["id"] for r in self.ledger.storage.eligible(db)})
                self.assertTrue(self.ledger.read(identity)["body"])

    def test_failed_job_with_unresolved_physical_attempt_never_expires(self):
        identity = self.old_job(state="failed")
        self.attempts(identity)
        self.clock[0] += storage.RESULT_TTL * 2
        self.ledger.storage.maintenance()
        with self.ledger.connect() as db:
            self.assertNotIn(identity, {r["id"] for r in self.ledger.storage.eligible(db)})
            self.assertTrue(db.execute("SELECT body FROM vision_attempts").fetchone()[0])
        self.assertTrue(self.ledger.read(identity)["body"])

    def test_dedup_window_and_later_deadline_are_protected(self):
        identity = self.old_job()
        with self.ledger.connect() as db:
            db.execute("UPDATE jobs SET deadline=? WHERE id=?", (self.clock[0], identity))
            self.assertFalse(self.ledger.storage.eligible(db))
        self.clock[0] += storage.PAYLOAD_TTL
        self.ledger.storage.maintenance()
        self.assertTrue(self.ledger.read(identity)["body"])
        self.clock[0] += 1
        self.ledger.storage.maintenance()  # interval is deliberately not due yet
        self.assertTrue(self.ledger.read(identity)["body"])
        self.clock[0] += storage.MAINTENANCE_INTERVAL
        self.ledger.storage.maintenance()
        with self.assertRaisesRegex(ValueError, "retention expired"):
            self.ledger.read(identity)

    def test_later_update_extends_payload_and_result_retention(self):
        identity = self.old_job()
        with self.ledger.connect() as db:
            db.execute("UPDATE jobs SET deadline=0,updated=? WHERE id=?", (self.clock[0], identity))
        self.clock[0] += storage.PAYLOAD_TTL
        self.ledger.storage.maintenance()
        self.assertTrue(self.ledger.read(identity)["body"])
        self.clock[0] += storage.RESULT_TTL - storage.PAYLOAD_TTL
        self.ledger.storage.maintenance()
        self.assertIsNotNone(self.ledger.status(identity, "owner"))
        self.clock[0] += storage.MAINTENANCE_INTERVAL + 1
        self.ledger.storage.maintenance()
        self.assertIsNone(self.ledger.status(identity, "owner"))

    def test_utf8_payload_dispatch_receipt_and_attempt_bytes(self):
        identity, _ = self.ledger.accept("owner", body(text="žluťoučký 🐳" * 70))
        with self.ledger.connect() as db:
            raw = db.execute("SELECT body FROM jobs WHERE id=?", (identity,)).fetchone()[0]
            self.assertEqual(self.ledger.storage.usage(db)["charged_bytes"],
                             self.ledger.storage.charge(len(raw.encode())))
        self.ledger.claim()
        self.ledger.dispatch(identity, {"instance": "ž", "request_id": "🐳", "expires": 1})
        self.attempts(identity, "succeeded")
        with self.ledger.connect() as db:
            db.execute("UPDATE vision_dispatch SET receipt=?", ('"🐳"',))
            db.execute("UPDATE jobs SET result=?", ('"ž"',))
            expected = sum(len(value.encode()) for value in (
                raw, '"ž"', db.execute("SELECT body FROM vision_dispatch").fetchone()[0],
                '"🐳"', db.execute("SELECT body FROM vision_attempts").fetchone()[0]))
            self.assertEqual(self.ledger.storage.usage(db)["hot_bytes"], expected)

    def test_rejects_before_accept_when_reservation_exceeds_budget(self):
        with self.assertRaisesRegex(AdmissionError, "no job accepted"):
            self.ledger.accept("owner", body(text="x" * 64000))
        self.assertEqual(self.ledger.counts(), {})

    def test_dispatch_checks_capacity_before_committing_metadata(self):
        identity, _ = self.ledger.accept("owner", body())
        self.ledger.claim()
        with self.assertRaisesRegex(ValueError, "reserved storage"):
            self.ledger.dispatch(identity, {"instance": "x" * 70000, "request_id": "r"})
        self.assertEqual(self.ledger.dispatch_for(identity), {"pending": True})
        self.assertIsNone(self.ledger.read(identity)["backend_id"])

    def test_concurrent_admission_never_oversubscribes_capacity(self):
        def submit(i):
            try:
                return self.ledger.accept("owner", body(text="x" * 8000, seed=i))[1]
            except AdmissionError:
                return False
        with ThreadPoolExecutor(8) as pool:
            accepted = sum(pool.map(submit, range(12)))
        self.assertGreater(accepted, 0)
        self.assertLess(accepted, 12)
        self.assertLessEqual(self.ledger.capacity()["charged_bytes"], storage.HOT_LIMIT)

    def test_archive_and_hot_replacement_are_one_transaction(self):
        old = self.old_job()
        with self.ledger.connect() as db:
            db.execute("CREATE TRIGGER refuse_clear BEFORE UPDATE OF body ON jobs BEGIN SELECT RAISE(ABORT,'test'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            with self.ledger.connect() as db:
                self.ledger.storage.archive(db, old)
        with self.ledger.connect() as db:
            self.assertFalse(db.execute("SELECT 1 FROM vision_payload_archive").fetchone())
            self.assertTrue(db.execute("SELECT body FROM jobs WHERE id=?", (old,)).fetchone()[0])

    def test_archive_preserves_all_attempt_bodies_and_proofs(self):
        old = self.old_job()
        self.attempts(old, "succeeded")
        with self.ledger.connect() as db:
            db.execute("UPDATE vision_attempts SET result=?,dispatch=?,receipt=?",
                       ('{"answer":"complete"}', '{"request_id":"r"}', '{"terminal":true}'))
            before = dict(db.execute("SELECT * FROM vision_attempts").fetchone())
            self.ledger.storage.archive(db, old)
            self.assertEqual(self.ledger.storage.hydrate(db, old)["attempts"]["1"], before["body"])
            after = dict(db.execute("SELECT * FROM vision_attempts").fetchone())
            self.assertEqual(after, {**before, "body": ""})

    def test_corrupted_archive_is_reported_and_never_replayed(self):
        old = self.old_job()
        with self.ledger.connect() as db:
            self.ledger.storage.archive(db, old)
            db.execute("UPDATE vision_payload_archive SET sha256='wrong'")
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.ledger.read(old)
        self.assertEqual(self.ledger.status(old, "owner")["status"], "succeeded")
        self.assertIsNone(self.ledger.claim())

    def test_terminal_payload_expires_but_result_and_identity_survive(self):
        identity = self.old_job()
        self.attempts(identity, "succeeded")
        self.clock[0] += storage.PAYLOAD_TTL
        self.ledger.storage.maintenance()
        self.assertEqual(self.ledger.status(identity, "owner")["result"]["data"]["answer"], "preserved")
        self.assertEqual(self.ledger.accept("owner", body(), "old"), (identity, False))
        with self.assertRaisesRegex(ValueError, "retention expired"):
            self.ledger.read(identity)
        with self.ledger.connect() as db:
            self.assertEqual(db.execute("SELECT length(payload) FROM vision_payload_archive").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT body FROM vision_attempts").fetchone()[0], "")

    def test_old_archive_expires_without_rehydration_then_result_expires(self):
        identity = self.old_job()
        with self.ledger.connect() as db:
            self.ledger.storage.archive(db, identity)
            db.execute("INSERT INTO vision_dispatch VALUES (?,?,?)", (identity, "{}", '{"confirmed":true}'))
        self.clock[0] += storage.PAYLOAD_TTL
        with patch.object(self.ledger.storage, "hydrate", side_effect=AssertionError("must not hydrate")):
            self.ledger.storage.maintenance()
        self.assertEqual(self.ledger.status(identity, "owner")["status"], "succeeded")
        self.clock[0] += storage.RESULT_TTL
        self.ledger.storage.maintenance()
        self.assertIsNone(self.ledger.status(identity, "owner"))
        with self.ledger.connect() as db:
            for table in ("jobs", "vision_dispatch", "vision_payload_archive"):
                self.assertEqual(db.execute("SELECT count(*) FROM " + table).fetchone()[0], 0)

    def test_expiry_rolls_back_payload_and_tombstone_together(self):
        identity = self.old_job()
        self.clock[0] += storage.PAYLOAD_TTL
        with self.ledger.connect() as db:
            db.execute("CREATE TRIGGER fail_expiry BEFORE UPDATE OF body ON jobs BEGIN SELECT RAISE(ABORT,'test'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.ledger.storage.maintenance()
        self.assertIsNone(self.ledger.storage.maintained_at)
        self.assertTrue(self.ledger.read(identity)["body"])
        with self.ledger.connect() as db:
            self.assertFalse(db.execute("SELECT 1 FROM vision_payload_archive").fetchone())

    def test_refusal_is_removed_after_expiry_without_operator_pruning(self):
        self.old_job()
        self.clock[0] += storage.PAYLOAD_TTL
        with patch.object(storage, "HOT_JOBS", 1):
            new, created = self.ledger.accept("owner", body(seed=2), "new")
        self.assertTrue(created)
        self.assertEqual(self.ledger.claim()["id"], new)
        self.assertEqual(self.ledger.capacity()["expired_payload_jobs"], 1)

    def test_maintenance_sweeps_are_bounded_and_rate_limited(self):
        identity = self.old_job()
        with self.ledger.connect() as db:
            row = list(db.execute("SELECT * FROM jobs WHERE id=?", (identity,)).fetchone())
            for i in range(300):
                row[0] = "seed-" + str(i)
                db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)", row)
        self.clock[0] += storage.RESULT_TTL
        self.ledger.storage.maintenance()
        self.assertEqual(sum(self.ledger.counts().values()), 45)
        self.ledger.storage.maintenance()
        self.assertEqual(sum(self.ledger.counts().values()), 45)
        self.clock[0] += storage.MAINTENANCE_INTERVAL
        self.ledger.storage.maintenance()
        self.assertEqual(self.ledger.counts(), {})

    def test_disk_reserve_refuses_without_mutating_history(self):
        old = self.old_job()
        low_disk = type("Disk", (), {"free": 0})()
        with patch.object(storage.shutil, "disk_usage", return_value=low_disk):
            with self.assertRaises(AdmissionError):
                self.ledger.accept("owner", body(seed=3), "new")
            self.assertFalse(self.ledger.capacity()["available"])
        self.assertEqual(self.ledger.counts(), {"succeeded": 1})
        self.assertTrue(self.ledger.read(old)["body"])

    def test_old_database_copy_migrates_additively_with_every_state(self):
        source = self.root / "old.sqlite3"
        copied = self.root / "copied.sqlite3"
        fixture = Path(__file__).parent / "fixtures" / "vision_ledger_14803a3.sql"
        with sqlite3.connect(source) as db:
            db.executescript(fixture.read_text())
            old_jobs = db.execute("SELECT * FROM jobs ORDER BY id").fetchall()
            old_dispatch = db.execute("SELECT * FROM vision_dispatch ORDER BY job_id").fetchall()
            columns = db.execute("PRAGMA table_info(jobs)").fetchall()
            with sqlite3.connect(copied) as destination:
                db.backup(destination)
        migrated = Ledger(copied, clock=lambda: 100000.0, controlled=True)
        # Reopening is idempotent and never implicitly recovers or sweeps.
        Ledger(copied, clock=lambda: 100000.0, controlled=True)
        with migrated.connect() as db:
            self.assertEqual([tuple(r) for r in db.execute("SELECT * FROM jobs ORDER BY id")], old_jobs)
            self.assertEqual([tuple(r) for r in db.execute("SELECT * FROM vision_dispatch ORDER BY job_id")], old_dispatch)
            self.assertEqual([tuple(r) for r in db.execute("PRAGMA table_info(jobs)")], columns)
            self.assertEqual(db.execute("SELECT count(*) FROM vision_payload_archive").fetchone()[0], 0)
            self.assertEqual(db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(set(migrated.counts()), {"queued", "running", "unknown", "succeeded", "failed"})
        with sqlite3.connect(source) as db:
            self.assertFalse(db.execute("SELECT 1 FROM sqlite_master WHERE name='vision_payload_archive'").fetchone())
        # The old SQL remains usable without schema changes or new columns.
        new, created = migrated.accept("owner", body(seed=91), "new")
        self.assertTrue(created)
        self.assertEqual(migrated.read(new)["state"], "queued")


class StorageApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []
        async def upstream(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True, "warming": False})
            if request.url.path == "/internal/vision/control":
                return httpx.Response(200, json={"schema": "tensorfold.vision-control.v1", "instance": "native-i"})
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 20})
            self.calls.append(json.loads(request.content))
            events = [
                {"id": "completion", "model": "deepseek-v4.1-flash", "choices": [
                    {"delta": {"content": "Blue"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
            return httpx.Response(200, content=b"".join(
                ("data: " + json.dumps(v) + "\n\n").encode() for v in events) + b"data: [DONE]\n\n")
        self.enterContext(patch.dict(os.environ, {"VISION_STORAGE_TEST_KEY": "vision-storage-unit-test-only"}))
        cfg = Config(state_dir=self.temp.name, clients={"alice": "VISION_STORAGE_TEST_KEY"})
        self.app = create_app(cfg, transport=httpx.MockTransport(upstream))
        self.life = self.app.router.lifespan_context(self.app)
        await self.life.__aenter__()
        self.addAsyncCleanup(self.life.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test",
            headers={"Authorization": "Bearer " + os.environ["VISION_STORAGE_TEST_KEY"]})
        self.addAsyncCleanup(self.client.aclose)

    async def test_health_capacity_exhaustion_keeps_text_ready(self):
        with patch.object(storage, "HOT_LIMIT", 1):
            self.app.state.jobs.ledger.storage.cached = None
            response = await self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        vision = response.json()["vision_jobs"]
        self.assertFalse(vision["ready"])
        self.assertTrue(vision["blocked"])
        self.assertTrue(vision["degraded"])
        self.assertFalse(vision["storage"]["available"])
        self.assertFalse(vision["automatic_replay"])

    async def test_image_api_delivers_after_history_capacity_reclaimed(self):
        ledger = self.app.state.jobs.ledger
        old, _ = ledger.accept("owner", body(seed=71), "history")
        with ledger.connect() as db:
            db.execute("UPDATE jobs SET state='succeeded',deadline=0,updated=0,result=? WHERE id=?",
                       ('{"status_code":200,"data":{"answer":"old"}}', old))
        ledger.storage.maintained_at = None
        with patch.object(storage, "HOT_JOBS", 1):
            response = await self.client.post("/v1/chat/completions", json=body(seed=72))
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], "Blue")
            self.assertEqual(len(self.calls), 1)
            again = await self.client.post("/v1/chat/completions", json=body(seed=72))
            self.assertEqual(again.status_code, 200, again.text)
            self.assertEqual(len(self.calls), 1)

    async def test_poll_and_reconnect_deliver_result_after_image_expiry(self):
        response = await self.client.post("/v1/chat/completions", json=body(seed=83),
                                          headers={"Idempotency-Key": "retained"})
        self.assertEqual(response.status_code, 200, response.text)
        identity = response.headers["X-DGX-Vision-Job-ID"]
        ledger = self.app.state.jobs.ledger
        with ledger.connect() as db:
            db.execute("UPDATE jobs SET deadline=?,updated=? WHERE id=?",
                       (ledger.clock() - storage.PAYLOAD_TTL - 1,
                        ledger.clock() - storage.PAYLOAD_TTL - 1, identity))
        ledger.storage.maintained_at = None
        await asyncio.to_thread(ledger.storage.maintenance)
        poll = await self.client.get("/v1/vision/jobs/" + identity)
        self.assertEqual(poll.status_code, 200, poll.text)
        self.assertEqual(poll.json()["result"]["data"]["choices"][0]["message"]["content"], "Blue")
        again = await self.client.post("/v1/chat/completions", json=body(seed=83),
                                       headers={"Idempotency-Key": "retained"})
        self.assertEqual(again.status_code, 200, again.text)
        self.assertEqual(len(self.calls), 1)

    async def test_startup_and_periodic_sweep_run_off_event_loop(self):
        jobs = Jobs(Path(self.temp.name) / "thread-test.sqlite3", None)
        threads = []
        original = jobs.ledger.storage.maintenance
        def observe():
            threads.append(threading.get_ident())
            original()
        with patch.object(jobs.ledger.storage, "maintenance", side_effect=observe), \
                patch.object(storage, "MAINTENANCE_INTERVAL", 0.01):
            await jobs.start()
            try:
                async with asyncio.timeout(2):
                    while len(threads) < 2:
                        await asyncio.sleep(0.01)
            finally:
                await jobs.close()
        self.assertGreaterEqual(len(threads), 2)
        self.assertTrue(all(t != threading.get_ident() for t in threads))


if __name__ == "__main__":
    unittest.main()
