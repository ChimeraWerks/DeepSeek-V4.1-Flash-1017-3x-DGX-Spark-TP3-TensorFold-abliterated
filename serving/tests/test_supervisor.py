import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

from chimera_sparks import __version__, lifecycle as L, supervisor as S


def plan(qualification=False):
    return [{"unit":"rank"+str(i), "node":{"rank":i}, "qualification_required":qualification} for i in range(3)]


class SupervisorTests(unittest.TestCase):
    def test_external_api_controller_does_not_start_a_workstation_api(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=SimpleNamespace(state_dir=root)
            raw={"cluster":{"api_external":True}}
            with patch.object(L,"document",return_value=(Path(root)/"config.json",raw,cfg)), \
                 patch.object(S,"native_up",return_value=(plan(),["a","b","c"])), \
                 patch.object(L,"start_api") as start,patch.object(S.time,"sleep"), \
                 patch.object(S,"inspect",return_value={"active":False,"invocation":"","status":75}), \
                 patch.object(S,"cleanup"):
                self.assertEqual(S.supervise("config"),75)
                start.assert_not_called()
    def test_candidate_guard_refuses_before_remote_or_gpu_work(self):
        with tempfile.TemporaryDirectory() as root, patch.object(L,"rank_plan",return_value=plan(True)), \
             patch.object(L,"provision_and_check") as remote, patch.dict(os.environ,{"CHIMERA_QUALIFICATION_WINDOW":"0"}):
            with self.assertRaisesRegex(ValueError,"qualification window"):
                S.native_up({"cluster":{}},SimpleNamespace(state_dir=root))
            remote.assert_not_called()

    def test_partial_world_is_removed_before_followers_and_head_restart(self):
        with tempfile.TemporaryDirectory() as root:
            items=plan();trace=[]
            cfg=SimpleNamespace(state_dir=root)
            old=[{"active":False,"invocation":""},{"active":True,"invocation":"old1"},{"active":True,"invocation":"old2"}]
            new=[{"active":True,"invocation":"new"+str(i)} for i in range(3)]
            with patch.object(L,"rank_plan",return_value=items), \
                 patch.object(L,"provision_and_check",side_effect=[{"active":x} for x in [False,True,True,False,False,False]]), \
                 patch.object(S,"inspect",side_effect=old+[new[1],new[2],new[0]]+new), \
                 patch.object(S,"cleanup",side_effect=lambda *a:trace.append("cleanup")), \
                 patch.object(L,"start_rank",side_effect=lambda x:trace.append(x["unit"])), \
                 patch.object(L,"start_guard"), patch.object(L,"wait_native"):
                _,identities=S.native_up({"cluster":{}},cfg)
            self.assertEqual(trace,["cleanup","rank1","rank2","rank0"])
            self.assertEqual(identities,["new0","new1","new2"])

    def test_replaced_follower_prevents_all_cleanup(self):
        rows=[{"active":False,"invocation":""},{"active":True,"invocation":"old1"},{"active":True,"invocation":"replacement"}]
        with patch.object(S,"inspect",side_effect=rows),patch.object(L,"stop_rank") as stop:
            with self.assertRaises(S.OwnershipChanged):S.cleanup(plan(),["old0","old1","old2"])
            stop.assert_not_called()

    def test_controller_returns_75_after_cleaning_owned_failed_world(self):
        with tempfile.TemporaryDirectory() as root:
            cfg=SimpleNamespace(state_dir=root)
            raw={"cluster":{}}
            with patch.object(L,"document",return_value=(Path(root)/"config.json",raw,cfg)), \
                 patch.object(S,"native_up",return_value=(plan(),["a","b","c"])), \
                 patch.object(L,"start_api"),patch.object(S.time,"sleep"), \
                 patch.object(S,"inspect",return_value={"active":False,"invocation":"","status":75}), \
                 patch.object(S,"cleanup") as cleanup:
                self.assertEqual(S.supervise("config"),75)
                cleanup.assert_called_once()
            record=json.loads((Path(root)/"native-supervisor.json").read_text())
            self.assertEqual(record["invocations"],["a","b","c"])

    def test_readiness_requires_the_current_controller_invocation(self):
        with tempfile.TemporaryDirectory() as root:
            record=Path(root)/"native-supervisor.json"
            cfg=SimpleNamespace(state_dir=root)
            raw={"installation":"test","cluster":{"profile":"fixture"}}
            record.write_text(json.dumps({"ready":True,"controller_invocation":"old","package_version":__version__}))
            with patch.object(L,"run",return_value="current"), \
                 patch.object(L.subprocess,"run",return_value=subprocess.CompletedProcess([],0)) as status:
                self.assertFalse(L.group_ready(raw,cfg))
                record.write_text(json.dumps({"ready":True,"controller_invocation":"current","package_version":__version__}))
                self.assertTrue(L.group_ready(raw,cfg))
                status.return_value=subprocess.CompletedProcess([],3)
                self.assertFalse(L.group_ready(raw,cfg))

    def test_systemd_restarts_the_controller_and_retains_explicit_window(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/"installation.json").write_text(json.dumps({"snapshot":"/config.json"}))
            cfg=SimpleNamespace(state_dir=root)
            with patch.object(L.subprocess,"run",return_value=subprocess.CompletedProcess([],3)), \
                 patch.object(L,"run") as run,patch.dict(os.environ,{"CHIMERA_QUALIFICATION_WINDOW":"1"}):
                L.start_supervisor({"installation":"test"},cfg)
                command=run.call_args.args[0]
            self.assertIn("--property=Restart=on-failure",command)
            self.assertIn("--setenv=CHIMERA_QUALIFICATION_WINDOW=1",command)
            self.assertIn("supervise",command)


if __name__=="__main__":unittest.main()
