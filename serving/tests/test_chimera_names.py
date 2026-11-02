"""Current v1.1 production contract; v1 remains frozen history."""
import copy
import json
import os
from unittest.mock import patch

import pytest
from chimera_sparks.assets import native_launch as N
from chimera_sparks.lifecycle import rank_plan
from test_production_v1 import cluster, topology


def test_v11_launch_matches_current_production_and_keeps_v1_historical():
    with patch.dict(os.environ, {}, clear=True):
        plan = rank_plan(cluster())
        for rank in range(3):
            old_argv, old_env = N.command(topology(), "v1", rank)
            argv, env = N.command(topology(), "v1.1", rank)
            assert argv == plan[rank]["argv"]
            assert env == plan[rank]["environment"]
            assert old_env["TF_DS_POOL_TOKENS"] == "10000000"
            assert env["TF_DS_POOL_TOKENS"] == "8000000"
            assert env["TF_DS_ENGINE_ID"] == "chimera-tensorfold-tp3-14803a3"
            assert env["TF_DS_RECIPE_PATCHES"] == "chimera-tensorfold-patches-20261102"
            assert env["TF_DS_MODEL_REVISION"].endswith("-chimera-dsv41-flash-1017-abliterated-exl3")
            assert env["TF_DS_PREFILL_COMPUTE_ROWS"] == "8192"
            assert env["TF_DS_PREFILL_SHARED_SPLIT"] == "1"
            assert env["TF_DS_PROMPT_ALIAS"] == "1"
            assert env["TF_DS_PREFILL_RAGGED_CUBIN"] == "/srv/v1/kit/auxiliary/experts_ragged.cubin"
            assert env["TF_DS_SPILL_GIB"] == "256"
            assert env["TF_DS_STREAMS"] == "16"
            assert env["TF_DS_ROUND_ROWS"] == "48"
            assert env["TF_DS_GRAPH_CAP"] == "128"
            if rank == 0:
                assert "--top-k=20" in argv
            launch = plan[rank]["launch"]
            assert json.loads(launch[launch.index("--environment-json") + 1]) == env

def test_v11_retains_exact_artifact_metadata_binding():
    profile = json.loads((N.HERE / "profiles/v1.1.json").read_text())
    old = json.loads((N.HERE / "profiles/v1.json").read_text())
    assert profile["artifact_arm"] == old["arm"]
    assert profile["arm"] == "chimera-dsv41-flash-1017-abliterated-exl3"
    good = {"tensorfold_tp3": {"arm": old["arm"], "world_size": 3}}
    N.validate_model_metadata(good, profile)
    N.validate_model_metadata(good, old)
    for arm in (profile["arm"], "", "wrong-pack"):
        bad = copy.deepcopy(good)
        bad["tensorfold_tp3"]["arm"] = arm
        with pytest.raises(ValueError, match="Wrong model metadata"):
            N.validate_model_metadata(bad, profile)
    good["tensorfold_tp3"]["world_size"] = 2
    with pytest.raises(ValueError, match="Wrong model metadata"):
        N.validate_model_metadata(good, profile)
