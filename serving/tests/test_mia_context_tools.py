"""Check that TP3 validation cannot pass on a TP2-only observation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools/mia"


def module(name):
    spec = importlib.util.spec_from_file_location("mia_" + name, TOOLS / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class MiaToolTests(unittest.TestCase):
    def test_spill_requires_three_distinct_commit_counters(self):
        spill = module("spill_check")
        run = object.__new__(spill.Run)
        run.a = SimpleNamespace(ranks=3)
        self.assertFalse(run.committed_everywhere({0: 1, 1: 1}))
        self.assertFalse(run.committed_everywhere({0: 1, 1: 1, 2: 0}))
        self.assertTrue(run.committed_everywhere({0: 1, 1: 1, 2: 1}))
        self.assertFalse(run.committed_everywhere({"all": 3}))

    def test_three_owner_record_and_all_part_idle_gate(self):
        kv = module("kv_shard_check_e2e")
        meta = dict(parts=3, salt="test", seed=1, owner_tokens=10, carry_tokens=10,
            carry_rounds=0, twin_tokens=10, burst=0, move_tokens=0)
        self.assertEqual(set(kv.Plan(meta).owners), {"O1", "O2", "O3"})
        run = object.__new__(kv.Run)
        run.a = SimpleNamespace(settle_wait=0)
        run.p = SimpleNamespace(meta=meta)
        run.c_start = {}
        run.rc = kv.C.Receipt("test", {})
        run.c = lambda: {"kv_shard.parts": 2, "kv_shard.part_rows": 100,
                        "kv_shard.part0.free": 100, "kv_shard.part1.free": 100}
        run.idle()
        self.assertTrue(run.rc.failed())
