# Run the TP3 recipe

The six lifecycle scripts load your private `cluster.env` using [cluster.env.example](../cluster.env.example)'s settings and the [release manifest](../release/manifest.json). Run them from the repository root.

## Prepare the cluster configuration

Use three DGX Spark systems on DGX OS with [pairwise RoCE links](topology.md) and key-based SSH. Each Spark needs about 440 GB (408 GiB) free: 274 GB for the standard weights, 93 GB for its local rank file, 67.6 GB of Engram rows and 3.6 GB for the kernel kit ([runtime assets](runtime-assets.md)). Budget another 256 GiB for spill on the head.

```bash
cp cluster.env.example cluster.env
chmod 600 cluster.env
```

Edit the shell assignments in `cluster.env`. The scripts source it as shell code; keep it private and out of git. Set `CLUSTER_ENV` to use another configuration file.

| Key | Purpose |
|---|---|
| `SPARK_HOSTS` | Three SSH names in rank order, head first, such as `spark1 spark2 spark3`. |
| `SPARK_USER` | Common SSH user on every Spark. |
| `SSH_KEY` | Local SSH identity used by the launch checkout to reach the Sparks. |
| `MASTER_ADDR` | Head address on the RoCE network. |
| `NCCL_IB_ADDR_RANGE` | RoCE subnet. |
| `SPARK_ROOT` | Per-Spark root for weights, ranks, binaries, kit, spill, logs, guards, service files and API state. Keep a leading tilde quoted for expansion on the Spark. |
| `HF_REPO` | Weights repository; the example names the published release. |
| `HF_REVISION` | Immutable weights commit, not the original DeepSeek model commit; the example pins the tested one. |
| `ENGINE_PORT` | Head engine port, default `8003`. |
| `API_PORT` | Head serving API port, default `18002`. |
| `SPARK_API_KEY` | Serving credential. Empty generates a key on the head or reuses its existing mode-0600 file value. |
| `SPARK_ADMIN_KEY` | Separate operator credential for settings. Empty generates or reuses it on the head. It must differ from the serving key. |

The example pins v1.2: `ENGINE_COMMIT=30a9647`, `STREAMS=16`, `POOL_TOKENS=8000000`, `CONTEXT_TOKENS=1048576`, `GRAPH_CAP=128`, `SPILL_GIB=256`, `MEMORY_RESERVE_GIB=4` and `TOP_K=20`. The launcher checks these values against the manifest and rejects mismatches. To use another release, supply its matching manifest and API configuration rather than tuning these assertions.

Before release, run the offline consistency check from the repository root. It checks the native launcher, API defaults, cluster example, engine scope and tracked text without contacting the Sparks.

```bash
python3 -I tools/check_release_consistency.py
uv run --no-project --with pytest python3 -m pytest tools -q
```

### Optional settings

Use the defaults in `cluster.env.example` unless your topology or installation needs an override.

| Keys | Purpose |
|---|---|
| `HEAD_SSH_KEY`, `SPARK_HOME` | Head-side identity for head-to-worker SSH and remote home-directory expansion. |
| `TP_PORT` | Tensor-parallel rendezvous port, default `29660`. |
| `ENGINE_HOST`, `API_HOST` | Listener addresses; both default to loopback. |
| `SPARK_PYTHON` | Spark-side Python executable, default `python3`. |
| `ENGINE_LABEL`, `API_INSTALLATION` | Installation labels, both default `ds1017`; the API user unit is `ds1017-api`. |
| `NCCL_IB_HCA`, `TF_RDMA_DEVICES` | RDMA device selection. `TF_RDMA_DEVICES` defaults to `NCCL_IB_HCA`; assign it after that key when using a shell reference. |
| `NCCL_SOCKET_IFNAME`, `TF_RDMA_GID_INDEX`, `RDMA_PEER_MASKS` | Network interface, GID and per-rank peer routing. |
| `SPARK_LD_LIBRARY_PATH` | Spark-side CUDA and system library search paths, plus a custom NCCL directory when required by the engine build. |
| `BIN_DIR`, `MODEL_DIR`, `RANK_CACHE`, `KIT_DIR`, `ENGRAM_DIR`, `SPILL_DIR` | Spark-side runtime and model locations beneath `SPARK_ROOT` by default. |
| `ENGRAM_SOURCE` | `ranged` (default) fetches only this rank's Engram rows from DeepSeek's repository; `full` downloads both 101.5 GB shards. See [runtime assets](runtime-assets.md). |
| `GUARD_SCRIPT`, `LOGGER_SCRIPT` | Spark-side memory guard and rotating logger paths. |
| `API_CONFIG`, `API_ENV_FILE` | Spark-side API JSON and private credential file paths. |
| `OUT` | Local build-output prefix, matching the prefix passed to `engine/build/build.sh`. |
| `DEPLOY_RECEIPT` | Local deployment execution record path. |
| `ENGINE_HEALTH_URL` | Optional local tunnel for engine health; normal access uses SSH. |
| `OPTIONAL_CONTROLLER_URL` | Optional existing affinity controller. No controller dependency by default. |

## Prepare local weight inputs

The weights are on [Hugging Face](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated). `prepare.sh` downloads them to each Spark and builds the three rank files. See [weight provenance](../release/weight-provenance.md) and the [model card](model-card.md).

To rebuild the composition yourself instead, select tensors from the six upstream repositories, apply the refusal-removal edit and re-encode; the [historical method](how-v1-was-built.md) describes this, and reconstruction is manual.

`HF_REPO` and `HF_REVISION` in `cluster.env.example` already pin the tested weights commit. Its tokenizer, standard safetensors/index and `tp3/rank<N>.keys` must match the release composition and expected rank hashes. The base model and other community quants are not drop-in replacements.

Build the engine from [engine/README.md](../engine/README.md). Release build hashes are in [build verification](../release/build-verification.json). Preparation stages the kernel kit and Engram tables itself; [runtime assets](runtime-assets.md) lists every file, its pinned source and SHA-256.

## Validate your installation

`./scripts/start.sh` caps each GPU at `GPU_CLOCK_CAP_MHZ` (default 2300, see [tuning](tuning.md#clocks-and-cpu-placement)), deploys the ranks directly and starts the API; its acceptance check is `./scripts/smoke-test.sh`, then `./scripts/status.sh`. Run both before sending other traffic. The bundled native profile's `serving_smoke_qualified=false` flag gates only the separate managed-supervisor path (`--qualification-window`) and direct `native_launch --run`, which these scripts do not use.

## Lifecycle scripts

Each script enables `set -u`, prints its plan, loads `scripts/_common.sh` and dispatches an action to `tools/cluster/recipe.py`. The common helper disables shell tracing, requires `python3`, exports the configuration assignments and passes the release manifest to the action.

### 1 Prepare the hosts and check prerequisites

```bash
./scripts/prep-sparks.sh status
./scripts/prep-sparks.sh apply
./scripts/preflight.sh
```

`prep-sparks.sh` is the exception to the dispatch above: it runs [`tools/spark/spark_prep.py`](../tools/spark/spark_prep.py) on each Spark over SSH. `status` and `plan` stream it and write nothing; `apply` runs under `sudo` with a confirmation per Spark, stages firmware, kernel and boot changes and never reboots. Apply, reboot and repeat until every item reads applied. The kernel step installs the exact `7.0.0-1019-nvidia-64k` package set from the [memory-saver kernel guide](https://github.com/christopherowen/dgx-spark-memory-saver/blob/ec1b17a8ac713c6357b8f22983ded8503a524849/docs/kernel.md) and refuses if apt would change the NVIDIA driver; the memory saver's Secure Boot key enrollment needs a person at each console.

`preflight.sh` checks DGX OS, driver and GPU visibility, RoCE links, available memory, free disk and head-to-worker SSH on all three Sparks. It also checks that each RoCE device's `TF_RDMA_GID_INDEX` slot holds its IPv4 RoCE v2 address inside `NCCL_IB_ADDR_RANGE`; a link flap can move that address to another slot, and when the slot is left empty or holds another kind of GID, preflight names the slot the address moved to. These checks are read-only.

### 2 Prepare local files

```bash
./scripts/prepare.sh
```

Stages the kernel kit and this rank's Engram tables on each Spark, downloads your pinned export, builds and verifies its rank file, then stages engine binaries and guards. Preparation neither starts the engine nor calls a model API.

Each Spark uses system Python 3 and standard-library-only tools for preparation:

1. Stage the small weight and asset tool bundle under `SPARK_ROOT/prepare-tools`.
2. Check free disk for the kit and Engram files, then stage `KIT_DIR` and `ENGRAM_DIR` as [runtime assets](runtime-assets.md) describes. A problem here stops preparation before the weights download.
3. Download `HF_REPO` at `HF_REVISION` into `MODEL_DIR` with the Hugging Face CLI.
4. Build a fresh unpadded checkpoint plan from the standard safetensors headers and index with `tools/weights/make_plan.py`; no export-side `plan.json` is needed.
5. Hash the local source shards with `source_verify.py --plan` and save the immutable source snapshot.
6. Run `rank_cache.py --write` with the shipped `MODEL_DIR/tp3/rank<N>.keys`. Both `--mia` and `--pollard` point at the same downloaded folder; every tensor in this standalone plan uses `mia`.
7. Rehash the complete rank file, including its native index and footer, against `weights.rank_sha256[N]` in the release manifest. A mismatch fails preparation before engine startup.

Preparation produces exactly one `RANK_CACHE/rank<N>of3-<digest-prefix>.bin`, as deployment preflight requires. It skips the split when the sole existing rank file has the correct SHA-256. A wrong digest, extra rank files or symlink rank entries stop preparation without deleting or overwriting files. Resolve the mismatch before retrying.

Plans, source snapshots and `prepare.log` stay under `RANK_CACHE/preparation/`; the rank writer also saves artifact metadata. Its completed `.partial` is a hardlink to the rank, not a second 93 GB copy. Resolve an interrupted writer's `.partial` before retrying; preparation never overwrites it. Keep the source folder unchanged during preparation.

The offline rank0 check for the historical v1 composition took 437.911 s on the local workstation: 1.906 s for the fresh header plan, 180.530 s for source hashing, 200.152 s for rank export and 55.322 s for the full artifact SHA-256 rehash. The 92,889,645,367-byte result matched the release manifest without the export's private `plan.json`. These are historical v1 timings, not a fresh v1.1 preparation run; they exclude download and engine build/staging. Spark-side preparation time has not been measured.

### 3 Start the engine and API

```bash
./scripts/start.sh
```

Runs these steps in order:

1. Install and validate API configuration and private credentials.
2. Validate all three ranks and replace the recipe's rank units.
3. Wait for warm engine health with numeric-effort support, then arm memory guards.
4. Start the serving API and wait for readiness.
5. Send three short greedy warmups directly to the head engine and report tokens per second.

The launcher reads engine flags and sizes from the manifest, calls `tools/cluster/deploy.py`, and uses `chimera_sparks.lifecycle.start_api` for API setup. Keep all ranks on the same configuration.

### 4 Inspect status

```bash
./scripts/status.sh
```

Shows rank and guard user-unit status, warm engine health including `numeric_effort`, API health and available memory on each Spark. Status checks are read-only.

### 5 Send a smoke request

```bash
./scripts/smoke-test.sh
```

Sends one short chat completion through the head API and prints tokens per second. It reads the credential on the head and leaves services, configuration and weights unchanged.

### 6 Stop the deployment

```bash
./scripts/stop.sh
```

Stops the head API, then the recipe's rank units and memory guards. Weights and spill files stay in place.

## Run without the shell entry points

Use the Python action dispatcher directly. First load the common configuration from the repository root:

```bash
source ./scripts/_common.sh
```

Then run the command for the operation you need:

```bash
# Read-only prerequisite checks
python3 -I "$REPO_ROOT/tools/cluster/recipe.py" preflight --manifest "$REPO_ROOT/release/manifest.json"

# Download, split and SHA-256-check each rank; no engine starts
python3 -I "$REPO_ROOT/tools/cluster/recipe.py" prepare --manifest "$REPO_ROOT/release/manifest.json"

# Start ranks and API, wait for readiness, then warm up
python3 -I "$REPO_ROOT/tools/cluster/recipe.py" start --manifest "$REPO_ROOT/release/manifest.json"

# Read-only service and memory status
python3 -I "$REPO_ROOT/tools/cluster/recipe.py" status --manifest "$REPO_ROOT/release/manifest.json"

# One API completion
python3 -I "$REPO_ROOT/tools/cluster/recipe.py" smoke-test --manifest "$REPO_ROOT/release/manifest.json"

# Stop API, ranks and guards
python3 -I "$REPO_ROOT/tools/cluster/recipe.py" stop --manifest "$REPO_ROOT/release/manifest.json"
```

Run one action at a time and stop on failure. The shell entry points print plans; direct commands skip those messages.

## Configuration failures

The common helper prints these errors to standard error and exits with status 1. It replaces `$REPO_ROOT` with the checkout's path in the missing-file message.

```text
FAIL: cannot locate repository root
FAIL: missing cluster.env; copy cluster.env.example to $REPO_ROOT/cluster.env and edit it
FAIL: python3 is required
FAIL: could not read cluster.env
```

Each script returns its Python action's exit status. Fix a failed action before continuing. See [troubleshooting](troubleshooting.md) and the [benchmark method](../benchmarks/README.md).
