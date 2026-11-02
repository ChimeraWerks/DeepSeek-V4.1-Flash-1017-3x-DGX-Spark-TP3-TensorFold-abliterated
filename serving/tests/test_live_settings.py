import asyncio
import json
import os
import tempfile
import unittest

import httpx

from chimera_sparks import live
from chimera_sparks.api import create_app
from chimera_sparks.config import Config
from chimera_sparks.laky.queue import FairPriorityGate


class ConfigLimitTests(unittest.TestCase):
    def test_cap_follows_engine_streams(self):
        Config(max_concurrency=16).validate()
        for bad in [dict(max_concurrency=17), dict(engine_streams=17), dict(engine_streams=8, max_concurrency=12),
                    dict(default_profile="missing"), dict(admin_clients=["nobody"])]:
            with self.assertRaises(ValueError):
                Config(**bad).validate()


class ProfileEffortTests(unittest.TestCase):
    def test_profile_effort_is_a_default_the_request_can_override(self):
        from chimera_sparks.policy import normalize, PolicyError
        cfg = Config(profiles={**Config().profiles, "deep": {"temperature": 1.0, "reasoning_effort": "max"}}).validate()
        base = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(normalize(base, cfg, profile="deep")["reasoning_effort"], "max")
        self.assertEqual(normalize(base, cfg, profile="focused")["reasoning_effort"], "high")  # inherits the default
        self.assertEqual(normalize({**base, "reasoning_effort": "low"}, cfg, profile="deep")["reasoning_effort"], "low")
        self.assertEqual(normalize({**base, "thinking": False}, cfg, profile="standard")["reasoning_effort"], "none")
        self.assertEqual(normalize(base, cfg)["reasoning_effort"], "high")  # no profile: the installation default
        for bad in [{"x": {"reasoning_effort": "turbo"}}, {"x": {"label": 3}}, {"x": {"label": "y" * 121}},
                    {"Has Space": {"temperature": 1.0}}, {"a@b": {"temperature": 1.0}}]:
            with self.assertRaises(ValueError):
                Config(profiles=bad).validate()
        labelled = Config(profiles={"x": {"label": "mine", "temperature": 0.5}}).validate()
        self.assertNotIn("label", normalize(base, labelled, profile="x"))


class GateResizeTests(unittest.IsolatedAsyncioTestCase):
    async def test_lowered_cap_lets_active_finish_and_raised_cap_admits_waiters(self):
        gate = FairPriorityGate(4, 8)
        for _ in range(4):
            await gate.acquire()
        await gate.resize(2, 8)
        waiter = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0)
        await gate.release()
        await gate.release()
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())  # 2 active at a cap of 2: still queued
        await gate.resize(3, 8)
        await asyncio.wait_for(waiter, 1)
        self.assertEqual(gate.active, 3)


class LiveSettingsApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        os.environ["RECIPE_TEST_KEY"] = "test-credential-only-123456789"
        os.environ["RECIPE_ADMIN_KEY"] = "admin-credential-only-12345678"
        self.seen = []
        self.numeric = False

        async def upstream(request):
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True, "warming": False, "numeric_effort": self.numeric})
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"count": 10})
            self.seen.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})
        self.upstream = upstream
        self.app, self.life = await self.start()
        transport = httpx.ASGITransport(app=self.app)
        self.user = httpx.AsyncClient(transport=transport, base_url="http://test",
                                      headers={"Authorization": "Bearer " + os.environ["RECIPE_TEST_KEY"]})
        self.admin = httpx.AsyncClient(transport=transport, base_url="http://test",
                                       headers={"Authorization": "Bearer " + os.environ["RECIPE_ADMIN_KEY"]})

    async def start(self):
        cfg = Config(state_dir=self.temp.name, clients={"test": "RECIPE_TEST_KEY", "ops": "RECIPE_ADMIN_KEY"},
                     admin_clients=["ops"])
        app = create_app(cfg, transport=httpx.MockTransport(self.upstream))
        life = app.router.lifespan_context(app)
        await life.__aenter__()
        return app, life

    async def asyncTearDown(self):
        await self.user.aclose()
        await self.admin.aclose()
        await self.life.__aexit__(None, None, None)
        self.temp.cleanup()

    async def chat(self, **extra):
        answer = await self.user.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                                     "max_tokens": 64, **extra})
        self.assertEqual(answer.status_code, 200, answer.text)
        return self.seen[-1]

    async def test_only_admin_clients_change_settings(self):
        self.assertEqual((await self.user.get("/admin/settings")).status_code, 403)
        self.assertEqual((await self.user.put("/admin/settings", json={"max_concurrency": 4})).status_code, 403)
        view = (await self.admin.get("/admin/settings")).json()
        self.assertEqual((view["settings"]["max_concurrency"], view["engine_streams"]), (16, 16))

    async def test_profiles_and_cap_apply_live_and_persist(self):
        put = await self.admin.put("/admin/settings", json={
            "profiles": {"coding": {"temperature": 0.3, "top_p": 0.95, "top_k": 20}},
            "default_profile": "coding", "max_concurrency": 8})
        self.assertEqual(put.status_code, 200, put.text)
        self.assertEqual(self.app.state.engine.gate.capacity, 8)
        sent = await self.chat()
        self.assertEqual((sent["temperature"], sent["top_p"], sent["top_k"]), (0.3, 0.95, 20))
        sent = await self.chat(temperature=0.9)  # an explicit request value still wins
        self.assertEqual(sent["temperature"], 0.9)
        sent = await self.chat()
        self.assertEqual(sent["temperature"], 0.3)
        saved = json.loads(live.path(self.app.state.engine.cfg).read_text())
        self.assertEqual(set(saved), {"profiles", "default_profile", "max_concurrency"})
        # a restart keeps the live settings
        await self.life.__aexit__(None, None, None)
        self.app, self.life = await self.start()
        self.assertEqual(self.app.state.engine.gate.capacity, 8)
        self.assertEqual(self.app.state.engine.cfg.default_profile, "coding")

    async def test_removed_profile_stays_removed_after_restart(self):
        put = await self.admin.put("/admin/settings", json={"profiles": {"focused": None}})
        self.assertEqual(put.status_code, 200, put.text)
        await self.life.__aexit__(None, None, None)
        self.app, self.life = await self.start()
        self.assertNotIn("focused", self.app.state.engine.cfg.profiles)
        self.assertIn("standard", self.app.state.engine.cfg.profiles)

    async def test_clients_list_and_select_labelled_presets(self):
        put = await self.admin.put("/admin/settings", json={"profiles": {"slot-3": {"label": "Long story", "temperature": 1.2}}})
        self.assertEqual(put.status_code, 200, put.text)
        listed = (await self.user.get("/v1/profiles")).json()
        self.assertEqual(listed["profiles"]["slot-3"]["label"], "Long story")
        self.assertEqual((listed["defaults"]["temperature"], listed["defaults"]["top_p"]), (1.0, 1.0))
        ids = {m["id"]: m for m in (await self.user.get("/v1/models")).json()["data"]}
        self.assertEqual(ids["deepseek-v4.1-flash@slot-3"]["description"], "Long story")
        sent = await self.chat(model="deepseek-v4.1-flash@slot-3")
        self.assertEqual((sent["model"], sent["temperature"]), ("deepseek-v4.1-flash", 1.2))
        self.assertNotIn("label", sent)

    async def test_invalid_updates_change_nothing(self):
        for bad in [{"max_concurrency": 17}, {"default_profile": "missing"}, {"sampling": {"top_p": 0}},
                    {"port": 1}, {"profiles": {"x": {"temperature": 3}}}]:
            answer = await self.admin.put("/admin/settings", json=bad)
            self.assertEqual(answer.status_code, 400, bad)
        self.assertEqual(self.app.state.engine.gate.capacity, 16)
        self.assertFalse(live.path(self.app.state.engine.cfg).exists())

    async def test_reset_returns_to_the_config_file(self):
        await self.admin.put("/admin/settings", json={"profiles": {"live-only": {"temperature": 1.2}},
                                                      "sampling": {"temperature": 0.5}, "max_concurrency": 4})
        reset = await self.admin.post("/admin/settings/reset")
        self.assertEqual(reset.status_code, 200)
        settings = reset.json()["settings"]
        self.assertNotIn("live-only", settings["profiles"])
        self.assertEqual((settings["sampling"]["temperature"], settings["max_concurrency"]), (1.0, 16))
        self.assertEqual(self.app.state.engine.gate.capacity, 16)
        self.assertFalse(reset.json()["saved"])


    async def test_effort_levels_remap_only_on_an_engine_that_takes_numbers(self):
        sent = await self.chat(reasoning_effort="high")
        self.assertEqual((sent["reasoning_effort"], sent["chat_template_kwargs"]["reasoning_effort"]), ("high", "high"))
        refused = await self.admin.put("/admin/settings", json={"effort_map": {"high": 60}})
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(self.app.state.engine.cfg.effort_map["high"], 75)
        self.numeric = True
        view = (await self.admin.get("/admin/settings")).json()
        self.assertEqual((view["numeric_effort"], view["engine_efforts"]["high"]), (True, 75))
        put = await self.admin.put("/admin/settings", json={"effort_map": {"high": 60}})
        self.assertEqual(put.status_code, 200, put.text)
        sent = await self.chat(reasoning_effort="high")
        self.assertEqual((sent["reasoning_effort"], sent["chat_template_kwargs"]["reasoning_effort"]), (60, 60))
        sent = await self.chat(reasoning_effort="low")         # unmoved levels still go by name
        self.assertEqual(sent["reasoning_effort"], "low")
        sent = await self.chat(reasoning_effort="40")          # a request's own number passes as is
        self.assertEqual(sent["reasoning_effort"], 40)
        for bad in [{"effort_map": {"high": 0}}, {"effort_map": {"high": 101}}, {"effort_map": {"high": True}},
                    {"effort_map": {"high": None}}, {"effort_map": {"loud": 50}}]:
            answer = await self.admin.put("/admin/settings", json=bad)
            self.assertEqual(answer.status_code, 400, bad)
        bad = await self.user.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": 0})
        self.assertEqual(bad.status_code, 400)
        reset = await self.admin.post("/admin/settings/reset")
        self.assertEqual(reset.json()["settings"]["effort_map"]["high"], 75)


if __name__ == "__main__":
    unittest.main()


class RequestSeedTest(unittest.TestCase):
    def test_sampled_requests_get_their_own_seed_and_pinned_or_greedy_ones_do_not(self):
        from chimera_sparks.policy import PolicyError, normalize
        cfg = Config()
        chat = {"messages": [{"role": "user", "content": "hi"}]}
        seeds = {normalize({**chat, "temperature": 1.0}, cfg)["seed"] for _ in range(20)}
        self.assertGreater(len(seeds), 1)
        self.assertEqual(normalize({**chat, "temperature": 1.0, "seed": 7}, cfg)["seed"], 7)
        self.assertNotIn("seed", normalize({**chat, "temperature": 0}, cfg))
        for bad in (True, -1, 1.5, "7"):
            with self.assertRaises(PolicyError):
                normalize({**chat, "temperature": 1.0, "seed": bad}, cfg)
