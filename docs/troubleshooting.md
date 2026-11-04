# Troubleshooting

Start with `./scripts/status.sh` from the repository root. Check rank and guard units, warm engine health, API health and memory on every Spark. Compare all ranks with the [release manifest](../release/manifest.json).

## Configuration and launch

| Symptom | Check and action |
|---|---|
| `FAIL: missing cluster.env` | Copy `cluster.env.example` to `cluster.env`, edit it and keep it private. Check `CLUSTER_ENV` if using an alternate file. |
| `FAIL: python3 is required` | Make `python3` available on the launch machine. |
| `FAIL: could not read cluster.env` | Check that the file is readable and contains valid, trusted shell assignments. |
| Preflight fails | Correct the reported OS, driver, GPU, RoCE, memory, disk or head-to-worker SSH prerequisite. |
| Preparation cannot find weights | Check `HF_REPO` and `HF_REVISION`. Use the export commit, not the original model commit. |
| Rank preparation fails | Inspect the reported `RANK_CACHE/preparation/.../prepare.log`; check the downloaded safetensors, shipped `tp3/rank<N>.keys`, free disk and any interrupted `.partial` file. Preparation never overwrites rank artifacts. |
| Rank SHA-256 mismatch or extra rank files | Compare the reported digest with `weights.rank_sha256` in the release manifest. Keep exactly one matching `rank<N>of3-*.bin` for this Spark; preserve and resolve mismatched artifacts before retrying. |
| Startup never becomes warm | Check all rank units, binary identities, rank splits and effective settings. Inspect private logs under each node's `SPARK_ROOT`. A listening port alone is insufficient. |
| `KvShardMismatch` | Compare pool and graph settings across every rank. Correct mismatches before restarting the group. |
| API unavailable while engine is healthy | Check API unit status, listener settings, credential configuration and API `/health`. Diagnose the API and engine separately. |
| Peer or Engram read failure stops ranks | Check peer reachability, local storage and rank logs. Restore the failed prerequisite before restarting. Exit 75 signals failure. |

The [reproduction guide](reproduction.md) lists the complete common-helper failure messages and lifecycle commands.

## Weight and build prerequisites

Weights come from [Hugging Face](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated). `cluster.env.example` pins the tested commit in `HF_REVISION`; keep it.

A rank-hash mismatch means the rank file differs from the selected composition. Keep the source record and rank file for diagnosis. Resolve the mismatch without disabling verification or substituting a base-model pack. See [local preparation limits](reproduction.md#prepare-local-weight-inputs).

Build the engine, AOT kernels and auxiliary prefill kernels with the same [build contract](../engine/README.md). Active v1.2 uses 8M/C16/top_k 20, graph cap 128, R48, spill 256 GiB and prefill layer slices. Historical v1.1 uses engine `14803a3`; historical v1 uses a different engine and 10M pool.

`OPTIONAL_CONTROLLER_URL` defaults to empty. Preparation and serving need no separate controller.

## Authentication and requests

| Symptom | Check and action |
|---|---|
| HTTP 401 `invalid_api_key` | Use the correct serving credential in the bearer header. Keep keys out of logs and public reports. |
| HTTP 403 `forbidden` on operator routes | Use the separate operator credential. The recipe does not grant serving clients administrator access. |
| HTTP 400 `invalid_settings` | Correct the live-settings values and cross-field constraints. |
| HTTP 409 `engine_capability` on effort remapping | Use the selected v1.2 engine `591d9dc` with API 0.1.5 and check native `/health` for `numeric_effort`. A non-native effort map requires that capability. |
| Sampled decode is slow | Check request top_k, the selected preset and live defaults. top_k 0 uses whole-vocabulary CPU sampling; the recipe default is 20. |
| HTTP 429 `queue_full` | Wait for capacity or reduce concurrent work. Keep API concurrency at or below the engine's 16 streams. |
| HTTP 400 `admission_oversized` or a context rejection | Reduce prompt or output size. Include output and safety reservations when sizing requests. |
| HTTP 504 `deadline` | Check load, queueing and upstream progress before deciding whether to retry. |
| A stream ends ambiguously | Preserve request state and diagnose before replaying it. For an accepted image job, check its identity first. |
| Short replies are corrupted with thinking off | A system prompt and context below about 50 tokens can trigger an [upstream model issue](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/33). Compare with a longer prompt. |

For endpoint behavior and settings, use the [serving API guide](../serving/README.md). `/health` and `/ready` return HTTP 503 while the native engine is unavailable or warming, or the vision worker is not running.

## Recover without mixing configurations

Run `./scripts/stop.sh` to stop the head API, recipe ranks and memory guards. Weights and spill files stay in place. Save the weight revision, manifest and relevant private logs before changing the configuration.

Fix the cause, then run preflight, start, status and a smoke request in that order. Run one lifecycle controller and one engine at a time. Keep the last working artifacts with their matching configuration; older binaries and newer rank files do not belong together.

## Interpret performance and capacity

Match mode, prompt type, concurrency, actual token count, cache state, warmup, affinity and clocks when comparing benchmark runs. Repeated prompts can lower time to first token through cache reuse.

The release serves text. Vision is not tested with these weights. Configured context is 1,048,576 tokens. The historical serving suite measures through 32K; the November 2 native prefill suite reaches 131,090 tokens. Test full-context stability and high availability separately from the basic smoke request.

Follow [security reporting](../SECURITY.md) for credential exposure or a suspected vulnerability.
