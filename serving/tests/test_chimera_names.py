"""Current v1.2 production contract; v1 and v1.1 remain frozen history."""
import copy
import json
import os
from unittest.mock import patch

import pytest
from chimera_sparks.assets import native_launch as N
from chimera_sparks.lifecycle import rank_plan
from test_production_v1 import cluster, topology

OPT_IN_OFF = ("TF_DS_KEYED_DRAFT", "TF_DS_DECODE_GREEDY_HEAD", "TF_DS_DECODE_SAMPLE_HEAD", "TF_SPILL_ROOM_WAIT_MS",
              "TF_DS_GRAPH_TELEMETRY", "TF_DS_FIRST_DRAFT_DEPTH", "TF_DS_DRAFT_CAL", "TF_CACHE_BATCH_DEAD",
              "TF_CACHE_FIRST_CONTINUATION")


def test_v12_launch_matches_current_production_and_keeps_history():
    with patch.dict(os.environ, {}, clear=True):
        plan = rank_plan(cluster())
        for rank in range(3):
            old_argv, old_env = N.command(topology(), "v1", rank)
            _, previous_env = N.command(topology(), "v1.1", rank)
            argv, env = N.command(topology(), "v1.2", rank)
            assert argv == plan[rank]["argv"]
            assert env == plan[rank]["environment"]
            assert old_env["TF_DS_POOL_TOKENS"] == "10000000"
            assert env["TF_DS_POOL_TOKENS"] == "8000000"
            assert previous_env["TF_DS_ENGINE_ID"] == "chimera-tensorfold-tp3-14803a3"
            assert previous_env["TF_DS_RECIPE_PATCHES"] == "chimera-tensorfold-patches-20261102"
            assert "TF_DS_PREFILL_SLICE" not in previous_env
            assert env["TF_DS_ENGINE_ID"] == "chimera-tensorfold-tp3-678ea1b"
            assert env["TF_DS_RECIPE_PATCHES"] == "chimera-tensorfold-patches-20261103"
            assert env["TF_DS_PREFILL_SLICE"] == "5"
            assert env["TF_DS_PREFILL_YIELD"] == "1"
            for flag in OPT_IN_OFF:
                assert env[flag] == "0"
            assert env["TF_DS_REDUCED_HEAD_FATBIN"] == "/srv/v1/kit/optional-kernels/dsv41_reduced_head.final.fatbin"
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

def test_v12_retains_exact_artifact_metadata_binding():
    profile = json.loads((N.HERE / "profiles/v1.2.json").read_text())
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
