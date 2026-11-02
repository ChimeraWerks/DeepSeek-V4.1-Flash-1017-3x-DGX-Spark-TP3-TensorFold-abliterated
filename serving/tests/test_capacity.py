from copy import deepcopy
import unittest

from chimera_sparks.capacity import plan, REMOTE_CHECK
from chimera_sparks import lifecycle
from unittest.mock import patch


class CapacityTests(unittest.TestCase):
    def ranks(self):
        return [{"node": {"native": {"root": "/srv/engine"}, "capacity_build_receipt": "/srv/engine/BUILD.json",
                "rank_cache_sha256": str(i + 1) * 64},
            "argv": ["/srv/engine/" + ("tensorfold-native" if i == 0 else "tf-dsv41-lanes")]
                + (["serve", "--context", "1048576"] if i == 0 else ["--pool", "10000000"]),
            "environment": {"TF_DS_POOL_TOKENS": "10000000", "TF_DS_ROUND_ROWS": "48"}}
            for i in range(3)]

    def identity(self):
        return {"TF_DS_ENGINE_ID": "engine", "TF_DS_RECIPE_PATCHES": "patches",
                "TF_DS_MODEL_REVISION": "model", "TF_DS_ENGRAM_REVISION": "engram"}

    def test_off_is_a_golden_unchanged_launch(self):
        ranks = self.ranks()
        before = deepcopy(ranks)
        self.assertEqual(plan({}, ranks), before)
        self.assertEqual(plan({"capacity": {"kv_shard": False, "spill_gib": 0}}, ranks), before)

    def test_default_pool_is_refused_and_explicit_whole_tp3_pool_is_shared(self):
        with self.assertRaises(ValueError):
            plan({"capacity": {"kv_shard": True}}, self.ranks())
        ranks = plan({"capacity": {"kv_shard": True, "pool_tokens": 7004160}}, self.ranks())
        for item in ranks:
            self.assertEqual(item["environment"]["TF_DS_KV_SHARD"], "1")
            self.assertEqual(item["environment"]["TF_DS_WINDOW"], "1048576")
        self.assertEqual(ranks[2]["argv"][-1], "7004160")

    def test_spill_flags_match_all_three_environments_and_identity(self):
        raw = {"capacity": {"spill_gib": 64, "identity": self.identity(),
            "spill_paths": ["/srv/spill/rank0", "/srv/spill/rank1", "/srv/spill/rank2"]}}
        ranks = plan(raw, self.ranks())
        self.assertIn("--spill-gib", ranks[0]["argv"])
        self.assertEqual(ranks[0]["environment"]["TF_DS_RANK_CACHE_SHA"], "1" * 64)
        for rank in (1, 2):
            self.assertEqual(ranks[rank]["environment"]["TF_DS_SPILL_GIB"], "64")
            self.assertEqual(ranks[rank]["environment"]["TF_DS_SPILL_MIN_TOKENS"], "8192")
            self.assertEqual(ranks[rank]["environment"]["TF_DS_SPILL_DIR"], f"/srv/spill/rank{rank}")
        compile(REMOTE_CHECK, "capacity-preflight", "exec")

    def test_enabled_invalid_values_and_missing_build_fail_before_remote_work(self):
        for capacity in [{"kv_shard": 1}, {"spill_gib": float("nan")},
            {"kv_shard": True, "pool_tokens": 7004160, "kv_decode": "owner"},
            {"spill_gib": 64}, {"spill_gib": 1, "spill_staging_mib": 8}]:
            with self.assertRaises(ValueError):
                plan({"capacity": capacity}, self.ranks())
        ranks = self.ranks()
        ranks[2]["node"].pop("capacity_build_receipt")
        with self.assertRaises(ValueError):
            plan({"capacity": {"kv_shard": True, "pool_tokens": 7004160}}, ranks)
        ranks = self.ranks()
        ranks[1]["environment"]["TF_DS_KV_SHARD"] = "1"
        with self.assertRaises(ValueError):
            plan({}, ranks)

    def test_launch_preflight_contains_the_real_capacity_gate(self):
        captured = []
        def remote(node, script, item):
            captured.append(script)
            return {"active": False}
        with patch.object(lifecycle, "remote", remote):
            lifecycle.provision_and_check(self.ranks()[0])
        self.assertIn(REMOTE_CHECK, captured[0])
        self.assertIn("spill rank-cache identity does not match", captured[0])
        self.assertLess(captured[0].index("native build lacks requested capacity"),
                        captured[0].index("native startup memory gate blocked"))
        compile(captured[0], "preflight", "exec")
