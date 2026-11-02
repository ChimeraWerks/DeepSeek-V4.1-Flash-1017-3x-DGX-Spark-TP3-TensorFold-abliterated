"""Host-only regressions for independently verified recipe review findings."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chimera_sparks import lifecycle as L, supervisor as S
from chimera_sparks.assets import native_launch as N


class RecipeReviewTests(unittest.TestCase):
    def test_provisioning_drift_never_starts_changed_candidate(self):
        for partial, second, consistent in [(False, False, False), (True, False, False),
                                            (True, False, True), (True, True, True)]:
            with self.subTest(partial=partial, second=second, consistent=consistent), tempfile.TemporaryDirectory() as tmp:
                items = [{"node": {"rank": i}, "qualification_required": False} for i in range(3)]
                old = [{"rank": i, "identity": "original"} for i in range(3)]
                changed = [{"rank": i, "identity": "changed" if consistent or i == 1 else "original"} for i in range(3)]
                def gate(plan):
                    if gate.calls == (2 if second else 1):
                        gate.calls += 1
                        if not consistent:
                            raise ValueError("TP3 effective Engram identities differ")
                        return changed
                    gate.calls += 1
                    return old
                gate.calls = 0
                checks = [{"active": x} for x in ([False, True, True] if partial else [False]*3) + [False]*3]
                with patch.object(L, "rank_plan", return_value=items), patch.object(L, "engram_preflight", side_effect=gate), \
                     patch.object(L, "provision_and_check", side_effect=checks), \
                     patch.object(S, "inspect", return_value={"active": False, "invocation": "old"}), \
                     patch.object(S, "cleanup") as cleanup, patch.object(L, "start_rank") as start:
                    with self.assertRaises(ValueError):
                        S.native_up({"cluster": {}}, SimpleNamespace(state_dir=tmp))
                    start.assert_not_called()
                    self.assertEqual(cleanup.call_count, int(second))

    def test_non_utf8_listing_matches_native_byte_contract(self):
        from chimera_sparks.assets import engram_identity as E
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"kit").mkdir()
            (root/"engram").mkdir()
            name=os.fsencode(root/"engram")+b"/\xff"
            with open(name,"wb") as stream:
                stream.write(b"123")
            item={"node":{"rank":0},"argv":["native","serve"],
                  "environment":{"TF_DS_KIT":str(root/"kit"),"TF_DS_ENGRAM":str(root/"engram")}}
            row=E.collect(item)
            self.assertEqual(bytes.fromhex(row["listing_hex"]), b"\xff:3;")
            self.assertEqual(row["kit_engram_sha256"], "0"*64)

    def test_direct_launch_refuses_unbound_but_plan_overrides_explicit_setting(self):
        with patch.dict(os.environ, {"TF_DS_ENGRAM_REVISION": "ambient"}, clear=True):
            with self.assertRaisesRegex(ValueError, "unbound inherited"):
                N.execution_environment({})
            self.assertEqual(N.execution_environment({"TF_DS_ENGRAM_REVISION": "bound"})["TF_DS_ENGRAM_REVISION"], "bound")

    def test_logger_refuses_before_spawn_and_executes_bound_environment(self):
        runner = Path(N.__file__).with_name("rotating_run.py")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/"child"
            command = [sys.executable, str(runner), "--log", str(Path(tmp)/"rank.log")]
            child = [sys.executable, "-c", "import os,pathlib,sys;pathlib.Path(sys.argv[1]).write_text(os.environ['TF_DS_SPILL_GIB']+'|'+os.environ['NCCL_TEST'])", str(target)]
            env = {k:v for k,v in os.environ.items() if not k.startswith(("TF_DS_", "TF_SPILL_"))}
            env.update(TF_DS_SPILL_GIB="999", NCCL_TEST="preserved")
            rejected = subprocess.run(command + ["--"] + child, env=env, capture_output=True, timeout=5)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(b"unbound inherited", rejected.stderr)
            self.assertFalse(target.exists())
            accepted = subprocess.run(command + ["--environment-json", json.dumps({"TF_DS_SPILL_GIB":"64"}), "--"] + child, env=env, capture_output=True, timeout=5)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(target.read_text(), "64|preserved")

    def test_capacity_mutations_are_bound_after_planning(self):
        nodes = [{"rank":i, "native":{"rank":i,"root":"/srv/runtime","rdma_devices":["a","b","c","d"],
                 "peer_masks":[0 if i==j else 3 for j in range(3)],"gid_index":3}} for i in range(3)]
        cluster={"nodes":nodes,"topology":{"master_address":"host","master_port":29500,"nccl_interface":"eth0",
                 "nccl_hcas":["a"],"nccl_address_range":"10.0.0.0/24","api_host":"127.0.0.1","api_port":8003}}
        def mutate(cluster, items, **kwargs):
            for item in items:
                item["environment"]["TF_DS_SPILL_GIB"]="64"
                item["argv"] += ["--spill-gib","64"]
            return items
        with patch("chimera_sparks.capacity.plan", side_effect=mutate):
            items=L.rank_plan(cluster)
        for rank,item in enumerate(items):
            launch=item["launch"]
            self.assertEqual(json.loads(launch[launch.index("--environment-json")+1]),item["environment"])
            self.assertEqual(launch[launch.index("--")+1:],item["argv"])
            self.assertEqual(item["environment"]["TF_DS_STEP_AHEAD"], "0")
        profiles=Path(N.__file__).parent/"profiles"
        self.assertEqual(sorted(p.name for p in profiles.glob("*.json")), ["v1.1.json", "v1.json"])


if __name__ == "__main__":
    unittest.main()
