import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chimera_sparks.assets import engram_identity as E, native_launch
from chimera_sparks import lifecycle as L, supervisor as S


class EngramTests(unittest.TestCase):
    def item(self, root, rank=0):
        return {"node": {"rank": rank}, "argv": ["native", "serve"],
                "environment": {"TF_DS_ENGRAM": str(root / "engram"), "TF_DS_KIT": str(root / "kit"),
                                "TF_DS_ENGRAM_REVISION": "revision"},
                "engram_compatibility_preflight": True}

    def fixture(self, root):
        (root / "engram").mkdir(); (root / "kit").mkdir()
        (root / "engram/z").write_bytes(b"123")
        (root / "engram/a").write_bytes(b"12")
        (root / "engram/subdir").mkdir()
        (root / "engram/link").symlink_to(root / "engram/a")
        (root / "kit/engram.json").write_bytes(b'{}')

    def test_native_regular_listing_and_kit_digest_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root)
            before={str(p):p.lstat() for p in root.rglob('*')}
            row=E.collect(self.item(root))
            digest=hashlib.sha256(b'{}').hexdigest()
            self.assertEqual(bytes.fromhex(row['listing_hex']),b'a:2;z:3;')
            self.assertEqual(row['identity_sha256'],hashlib.sha256(b'revision|'+digest.encode()+b'|a:2;z:3;').hexdigest())
            self.assertEqual(before,{str(p):p.lstat() for p in root.rglob('*')})

    def test_byte_sort_is_of_entire_name_size_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'engram/a0').write_bytes(b'')
            row=E.collect(self.item(root))
            self.assertEqual(bytes.fromhex(row['listing_hex']),b'a0:0;a:2;z:3;')

    def test_missing_kit_metadata_zero_digest_and_missing_rank_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'kit/engram.json').unlink()
            self.assertEqual(E.collect(self.item(root))['kit_engram_sha256'],'0'*64)
            item=self.item(root);item['environment']['TF_DS_ENGRAM']=str(root/'missing')
            with self.assertRaises(FileNotFoundError):E.collect(item)
            with self.assertRaises(ValueError):E.compatible([],[])

    def test_aggregate_revision_listing_kit_and_candidate_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            items=[]
            for rank in range(3):
                item=self.item(root,rank)
                if rank:item['argv']=['lanes','model','cache',str(rank),'3','master','29500',str(root/'kit'),'--engram',str(root/'engram')]
                items.append(item)
            rows=[E.collect(item) for item in items]
            E.compatible(items,rows)
            for key,value in [('TF_DS_ENGRAM_REVISION','other'),('TF_DS_ENGRAM',str(root/'engram/subdir'))]:
                changed=copy.deepcopy(items);changed[2]['environment'][key]=value
                if key=='TF_DS_ENGRAM':changed[2]['argv'][-1]=value
                with self.assertRaises(ValueError):E.compatible(changed,[E.collect(item) for item in changed])
            changed=copy.deepcopy(items);changed[0]['argv'].append('--loop-guard')
            with self.assertRaises(ValueError):E.compatible(changed,rows)
            (root/'kit/engram.json').write_bytes(b'{"changed":1}')
            rows[2]=E.collect(items[2])
            with self.assertRaises(ValueError):E.compatible(items,rows)

    def test_shard_only_without_revision_passes_but_spill_requires_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            items = []
            for rank in range(3):
                item = self.item(root, rank)
                item["environment"].pop("TF_DS_ENGRAM_REVISION")
                item["capacity"] = {"spill": False, "kv_shard": True}
                if rank:
                    item["argv"] = ["lanes", "model", "cache", str(rank), "3", "master", "29500", str(root / "kit")]
                items.append(item)
            rows = [E.collect(item) for item in items]
            self.assertEqual(E.compatible(items, rows), rows)
            self.assertTrue(all(row["engram_revision"] == "" for row in rows))
            changed = copy.deepcopy(items)
            changed[2]["argv"][7] = str(root / "other-kit")
            (root / "other-kit").mkdir()
            with self.assertRaisesRegex(ValueError, "identities differ"):
                E.compatible(changed, [E.collect(item) for item in changed])
            for item in items:
                item["capacity"]["spill"] = True
                with self.assertRaisesRegex(ValueError, "persistent Engram requires"):
                    E.collect(item)

    def test_gate_before_provision_or_partial_world_cleanup(self):
        items=[{'node':{'rank':i},'qualification_required':False} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(L,'rank_plan',return_value=items), \
             patch.object(L,'engram_preflight',side_effect=ValueError('mismatch')), \
             patch.object(L,'provision_and_check') as provision, patch.object(S,'cleanup') as cleanup:
            with self.assertRaisesRegex(ValueError,'mismatch'):S.native_up({'cluster':{}},type('Cfg',(),{'state_dir':tmp})())
            provision.assert_not_called();cleanup.assert_not_called()

    def test_compatible_candidate_is_checked_before_normal_partial_replacement(self):
        items = [{"node": {"rank": i}, "qualification_required": False} for i in range(3)]
        trace = []
        states = [{"active": True, "invocation": str(i)} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp, patch.object(L, "rank_plan", return_value=items), \
             patch.object(L, "engram_preflight", side_effect=lambda plan: trace.append("identity")), \
             patch.object(L, "provision_and_check", side_effect=[{"active": x} for x in (False, True, True, False, False, False)]), \
             patch.object(S, "inspect", side_effect=states + [states[1], states[2], states[0]] + states), \
             patch.object(S, "cleanup", side_effect=lambda *args: trace.append("cleanup")), \
             patch.object(L, "start_rank", side_effect=lambda item: trace.append(item["node"]["rank"])), \
             patch.object(L, "start_guard"), patch.object(L, "wait_native"):
            S.native_up({"cluster": {}}, type("Cfg", (), {"state_dir": tmp})())
        self.assertEqual(trace, ["identity", "identity", "cleanup", "identity", 1, 2, 0])


    def test_remote_failure_prevents_aggregate_success(self):
        items=[{'node':{'rank':i},'engram_compatibility_preflight':True} for i in range(3)]
        with patch.object(L,'remote',side_effect=OSError('unreachable')):
            with self.assertRaises(OSError):L.engram_preflight(items)


    def test_remote_collector_executes_exact_read_only_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            items=[]
            for rank in range(3):
                item=self.item(root,rank)
                if rank:item['argv']=['lanes','model','cache',str(rank),'3','master','29500',str(root/'kit')]
                items.append(item)
            def remote(node,source,item):
                import contextlib
                import io
                import sys
                output=io.StringIO()
                with patch.object(sys,'argv',['collector',json.dumps(item)]),contextlib.redirect_stdout(output):
                    exec(compile(source,'remote-engram','exec'),{})
                return json.loads(output.getvalue())
            with patch.object(L,'remote',side_effect=remote):
                self.assertEqual(len(L.engram_preflight(items)),3)


    def test_loop_guard_only_v1_head_once(self):
        nodes=[{'rank':i,'root':'/srv/runtime','rdma_devices':['a','b','c','d'],
                'peer_masks':[0 if i==j else 3 for j in range(3)],'gid_index':3} for i in range(3)]
        cfg={'ranks':nodes,'master_address':'host','master_port':29500,'nccl_interface':'eth0',
             'nccl_hcas':['a'],'nccl_address_range':'10.0.0.0/24','api_host':'127.0.0.1','api_port':8003}
        for rank in range(3):
            argv,_=native_launch.command(cfg,'v1',rank)
            self.assertEqual(argv.count('--loop-guard'),int(rank==0))


if __name__=='__main__':unittest.main()
