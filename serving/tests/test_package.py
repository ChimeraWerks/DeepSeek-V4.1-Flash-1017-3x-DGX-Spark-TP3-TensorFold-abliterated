import json
import tempfile
import unittest
from pathlib import Path
from chimera_sparks.config import Config
from chimera_sparks.lifecycle import document, rank_plan, unit_name, ensure_state


class PackageTests(unittest.TestCase):
    def test_fresh_native_state_is_private_before_any_lock_or_process(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = Config(state_dir=str(Path(root)/"new-state"))
            self.assertEqual(ensure_state(cfg).stat().st_mode & 0o777, 0o700)
    def test_config_relative_state_and_env_references(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "config.json"
            file.write_text(json.dumps({"installation": "node-test", "env_file": "credentials.env",
                "api": {"state_dir": "state", "clients": {"client": "MY_RUNTIME_KEY"}}}))
            path, raw, cfg = document(file)
            self.assertEqual(cfg.state_dir, str(Path(root) / "state"))
            self.assertEqual(unit_name(raw), "node-test-api")
        with self.assertRaises(ValueError):
            Config(clients={"client": "literal secret"}).validate()
        with self.assertRaises(ValueError):
            Config(sampling={"temperature": None, "top_p": 1, "top_k": 0, "presence_penalty": 0}).validate()

    def test_native_plan_is_portable_and_does_not_start(self):
        nodes = [{"rank": i, "host": "node-" + str(i), "native": {
            "rank": i, "root": "/srv/sparks/runtime", "rdma_devices": ["rdma0", "rdma1", "rdma2", "rdma3"],
            "peer_masks": [0 if j == i else 3 for j in range(3)], "gid_index": 3}} for i in range(3)]
        cluster = {"nodes": nodes, "profile": "v1.2", "topology": {
            "master_address": "node-0", "master_port": 29500, "nccl_interface": "fabric0",
            "nccl_hcas": ["rdma0", "rdma1"], "nccl_address_range": "192.0.2.0/24",
            "api_host": "127.0.0.1", "api_port": 8003}}
        plan = rank_plan(cluster)
        self.assertEqual(len(plan), 3)
        self.assertIn("tensorfold-native", plan[0]["argv"][0])
        self.assertIn("tf-dsv41-lanes", plan[2]["argv"][0])
        self.assertTrue(all(p["environment"]["TF_TP_WORLD"] == "3" for p in plan))
        self.assertTrue(all("--max-size" in p["launch"] and "100m" in p["launch"] for p in plan))
        self.assertTrue(all(p["qualification_required"] for p in plan))
        self.assertEqual(rank_plan({k: v for k, v in cluster.items() if k != "profile"}), plan)
        self.assertTrue(all(p["capacity"]["spill"] for p in plan))
        self.assertTrue(all("TF_DS_KV_SHARD" not in p["environment"] for p in plan))

    def test_default_scope_keeps_deferred_features_off_and_benchmark_pinned(self):
        scope = json.loads((Path(__file__).resolve().parents[1] / "engine-scope.json").read_text())
        self.assertNotIn("benchmark_commit", scope)
        self.assertIn("kv_sharding", scope["opt_in_later_default_off"])
        self.assertNotIn("validated_checkpoints", scope["opt_in_later_default_off"])
        self.assertFalse(scope["live_qualified"])
        self.assertEqual(scope["required_native_profile"], "v1.2")
        self.assertEqual(scope["engine_commit"], "678ea1b")


if __name__ == "__main__":
    unittest.main()
