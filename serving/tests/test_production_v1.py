"""Frozen production launch contract; no artifacts, services or network needed."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chimera_sparks.assets import native_launch as N
from chimera_sparks.config import Config
from chimera_sparks.lifecycle import rank_plan


FIXTURE = Path(__file__).parent / "fixtures/production-v1.json"


def topology():
    devices = ["rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1"]
    masks = [[0, 12, 3], [3, 0, 12], [12, 3, 0]]
    return {"ranks": [{"rank": rank, "root": "/srv/v1", "rdma_devices": devices,
                       "peer_masks": masks[rank], "gid_index": 3} for rank in range(3)],
            "master_address": "192.0.2.10", "master_port": 29660,
            "nccl_interface": "enP7s7", "nccl_hcas": devices,
            "nccl_address_range": "192.0.2.0/24", "api_host": "127.0.0.1", "api_port": 8003,
            "library_path": "/opt/nccl/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu"}


def cluster():
    config = topology()
    nodes = [{"rank": rank, "host": f"rank{rank}.invalid", "native": node}
             for rank, node in enumerate(config.pop("ranks"))]
    return {"nodes": nodes, "topology": config}


class ProductionV1Tests(unittest.TestCase):
    def test_all_rank_argv_and_every_environment_value_match_captured_production(self):
        frozen = json.loads(FIXTURE.read_text())
        with patch.dict(os.environ, {}, clear=True), \
             patch("chimera_sparks.lifecycle.remote", side_effect=AssertionError("offline only")):
            managed = rank_plan({**cluster(), "profile": "v1"})  # Historical launch stays explicit.
            for rank in range(3):
                with self.subTest(rank=rank):
                    argv, env = N.command(topology(), "v1", rank)
                    expected_env = {**frozen["environment"], "TF_RDMA_PEER_MASKS": frozen["peer_masks"][rank]}
                    if rank == 0:
                        expected_env.update(frozen["head_environment"])
                    expected_argv = frozen["head_argv"] if rank == 0 else [
                        arg.replace("${RANK}", str(rank)) for arg in frozen["worker_argv"]]
                    self.assertEqual(argv, expected_argv)
                    self.assertEqual(env, expected_env)
                    self.assertEqual(managed[rank]["argv"], argv)
                    self.assertEqual(managed[rank]["environment"], env)
                    self.assertEqual(managed[rank]["launch"][managed[rank]["launch"].index("--") + 1:], argv)
                    self.assertTrue(managed[rank]["capacity"]["spill"])
                    self.assertFalse(managed[rank]["qualification_required"])

    def test_capacity_overrides_do_not_smuggle_or_duplicate_profile_spill(self):
        for key, value in (("TF_DS_SPILL_GIB", "64"), ("TF_DS_SPILL_DIR", "/other"),
                           ("TF_DS_SPILL_STAGING_MIB", "512"), ("TF_DS_KV_SHARD", "1")):
            config = cluster()
            config["environment"] = {key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "cluster.capacity"):
                rank_plan(config)
        config = cluster()
        config["capacity"] = {"spill_gib": 0, "kv_shard": False}
        for item in rank_plan(config):
            self.assertFalse(any(arg.startswith("--spill-") for arg in item["argv"]))
            self.assertNotIn("TF_DS_SPILL_GIB", item["environment"])
            self.assertNotIn("capacity", item)

    def test_v1_direct_execution_fails_closed_without_new_artifact_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = topology()
            config["ranks"][0]["root"] = tmp
            with self.assertRaisesRegex(ValueError, "verified v1 artifact checksum"):
                N.command(config, "v1", 0, validate_files=True)
            checksums = {}
            for name in ("bin/tensorfold-native", "bin/tf-dsv41-lanes", "weights/chat_template.jinja"):
                path = Path(tmp) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"offline artifact")
                checksums[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = Path(tmp) / "bin/BUILD.json"
            receipt.write_text(json.dumps({"schema": "ds41f.native-build.v1", "engine_commit": "7d5b4d8",
                "adaptive_learning": True, "binaries": {name: {"sha256": checksums[str(Path(tmp) / "bin" / name)]}
                for name in ("tensorfold-native", "tf-dsv41-lanes")}}))
            checksums[str(receipt)] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            config["ranks"][0]["checksums"] = checksums
            (Path(tmp) / "bin/tf-dsv41-lanes").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from its verified checksum"):
                N.command(config, "v1", 0, validate_files=True)
            (Path(tmp) / "bin/tf-dsv41-lanes").write_bytes(b"offline artifact")
            data = json.loads(receipt.read_text())
            data["engine_commit"] = "wrong-engine"
            receipt.write_text(json.dumps(data))
            checksums[str(receipt)] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "build receipt for the v1 engine commit"):
                N.command(config, "v1", 0, validate_files=True)
            data["engine_commit"] = "7d5b4d8"
            data["binaries"]["tf-dsv41-lanes"]["sha256"] = "0" * 64
            receipt.write_text(json.dumps(data))
            checksums[str(receipt)] = hashlib.sha256(receipt.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "binary differs from its build receipt"):
                N.command(config, "v1", 0, validate_files=True)

    def test_api_context_default_and_dependent_validation(self):
        cfg = Config().validate()
        self.assertEqual(cfg.context_tokens, 1048576)
        self.assertEqual(cfg.engine_streams, cfg.max_concurrency)
        self.assertEqual(cfg.engine_streams, 16)
        self.assertEqual(cfg.sampling["top_k"], 20)
        replace(cfg, max_output_tokens=cfg.context_tokens - cfg.safety_tokens - 1).validate()
        with self.assertRaisesRegex(ValueError, "output limits must fit"):
            replace(cfg, max_output_tokens=cfg.context_tokens - cfg.safety_tokens).validate()
        replace(cfg, auto_compact=True, compact_trigger_tokens=900000,
                compact_target_tokens=800000).validate()
        with self.assertRaisesRegex(ValueError, "recent < target < trigger < context"):
            replace(cfg, auto_compact=True, compact_trigger_tokens=cfg.context_tokens).validate()


if __name__ == "__main__":
    unittest.main()
