# Engine

`tensorfold/` contains the TensorFold native Zig engine with BertholomusAI's
`dsv41` family for DeepSeek V4.1 Flash. Release v1.2 uses engine source
`30a9647`, plus one metrics documentation-comment edit and one test-name wording
edit. Serving logic is unchanged by those edits. `14803a3` is the historical
v1.1 engine and `7d5b4d8` the historical v1 engine.

## Lineage

| Layer | Source |
|---|---|
| Engine | [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and contributors |
| DeepSeek V4.1 family (`dsv41`) | [BertholomusAI](https://github.com/bertholomus/TensorFold) |
| Three-Spark TP3 base | LakyFx DS41F 1.2.5 / K2a ([`b4e476d`](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/tree/b4e476d60f01fda564d760c5906ac1358390c728)), with later 1.2.7 and 1.2.8 ports |
| Ported patches | Mia's AI Lab two- and four-Spark recipes and their contributors ([CREDITS.md](../CREDITS.md)) |

## Our changes

| Area | What changed |
|---|---|
| Prompt prefill | each rank computes its own third of every prompt chunk; replicated attention projections run on owned rows; all-gathers go straight to both peers (`TF_DS_PREFILL_OWNED`, `TF_DS_DIRECT_GATHER`) |
| Sampling | sampled requests draw from the GPU's top-k candidates (`TF_DS_SAMPLE_CAND`), with a full-logits fallback |
| CUDA graphs | ranks vote before capturing or replaying a graph; failed first launches drop their cache entry; caches capped (`TF_DS_GRAPH_CAP`) |
| Spill | finished prompt states spill to NVMe and come back on reuse; the spill layout is agreed across ranks |
| Fault handling | bounded peer-failure recovery, follower exits on a closed lead link, fatal fences, and a rank that cannot read its Engram rows ends the pair instead of hanging |
| Server | numeric `reasoning_effort` (1 to 100), keepalives for long prefill, request-control masking, tool-call prose preservation |

Ported from other recipes and credited in [CREDITS.md](../CREDITS.md): structured
output and required tool calls, image cache identity, idle RDMA parking, prefill
yield and admission, prefix retention, and LakyFx 1.2.8's prefill work.

### Since v1.1 (`14803a3` to `30a9647`)

[CHANGELOG.md](../CHANGELOG.md) lists the checks that were run.

On in production:

| Area | What changed |
|---|---|
| Prompt prefill | yielded prefill runs in layer slices with decode rounds between them (`TF_DS_PREFILL_SLICE=5`); slice time counts as decode occupancy. Our version of Mia's patch 0052 |
| Prompt rendering | `response_format` schema is passed to the template, an empty system block is added when tools or a schema need one, a `thinking` object decides the mode before any effort, Messages tool JSON keeps chat key order, a prefix continuation leaves off the end token. Behavior matched to `deepseek-recipe` |
| Responses | every reasoning item is kept, the custom `apply_patch` tool is accepted and its input streamed, tool namespaces are flattened and restored, `response_format` regex uses the guided path. Behavior matched to `deepseek-recipe` |
| Output parsing and validation | think and DSML markers count only as real tokens, with a spelling fallback for copied reply text; broken tool history and duplicate tool names are refused before inference; strict tool and response schemas are enforced. Behavior matched to `deepseek-recipe` |
| Server fixes | an assistant `reasoning` field is read as `reasoning_content` (Mia's patch 0047, nuk3s); GET and DELETE bodies are drained (Mia's patch 0046, Christopher A. Williamson) |
| Lifetime fixes | a finished job leaves `admitted` before it is freed (found in Mia PR 94 review, hecisaza); a first-token finish releases its lane when the trace fails (TensorFold `ed39acc5`, laudney); the read-ahead thread and sequence start empty |

These match the behaviors listed, not all of `deepseek-recipe`; [CHANGELOG.md](../CHANGELOG.md) names what still differs.

In the source, off in production:

| Flag | What it does | From |
|---|---|---|
| `TF_CACHE_BATCH_DEAD`, `TF_CACHE_FIRST_CONTINUATION` | batched dead-checkpoint release and a first-continuation hint ([port notes](PORT-laky-4.0.2-cache-batch.md)) | LakyFx 4.0.2 |
| `TF_DS_DRAFT_CAL`, `TF_DS_FIRST_DRAFT_DEPTH` | per-stream draft calibration and full first head depth | Mia's patches 0050 and 0049 (lukaszraczylo) |
| `TF_DS_KEYED_DRAFT` | sampled drafts keyed to target draws; needs a `KEYED_DRAFT=true` build | Mia's patch 0054 (lukaszraczylo) |
| `TF_DS_DECODE_GREEDY_HEAD`, `TF_DS_DECODE_SAMPLE_HEAD` | exact reduced output heads; need `TF_DS_REDUCED_HEAD_FATBIN` | LakyFx 4.0.2, adapted to TP3 |
| `TF_SPILL_ROOM_WAIT_MS` | bounded wait for room before a restore starts | Mia's patch 0040 (nuk3s) |
| `TF_DS_GRAPH_TELEMETRY` | CUDA graph hit, capture and pressure counters | ChimeraWerks |
| `TF_STREAM_TOOL_CALLS` | streams DSML tool calls as they decode; rank 0 only | Mia's patch 0048 (nuk3s) |

### After `30a9647`, not yet deployed

| Area | What changed |
|---|---|
| Lifetime fixes | the lane thread makes its CUDA context current as it starts (`LaneHost.enter`), so the first spill restore after a restart no longer runs on an unbound thread. Mia's patch 0059 (issue #102, NguyenPhan2810) |
| PDF input | `--vision-pdf`: PDF parts (Messages `document`, Chat `file`, Responses `input_file`) become page images, rendered by PDFium `chromium/7725` in a limited worker process (`zig/src/server/pdf.zig`, `pdf_worker.zig`) as PNG pages the dsv41 decoder reads without Pillow; the count routes span each page from its size alone. TensorFold PRs 1820 (joernchen), 2387, 2393 and 2599 (ashhart) |

## CPU-only cross-build

Run from the repository root on Linux. Install Bash, Python 3, GNU coreutils,
`ionice` and the complete Zig toolchain matching
[`tensorfold/.zig-version`](tensorfold/.zig-version) (currently `0.17.0`).
Supply a compatible CUDA fatbin directory. Keep the compiler executable hash
and complete distribution for build comparisons; the version string alone
cannot identify the distribution.

Supply every non-opt-in kernel listed in
[`tensorfold/zig/build/cuda.zig`](tensorfold/zig/build/cuda.zig), plus
`dsv41_sample_cand.fatbin`. `dsv41_keyed.fatbin` is needed only for
`KEYED_DRAFT=true`. The script neither ships nor fetches fatbins.
Generate them separately with a compatible CUDA toolchain and headers using
the vendored `fatbins` build target and `-Dnvcc` option. This cross-build
embeds existing fatbins without CUDA compilation or GPU use.

```bash
ZIG=/path/to/zig/zig \
FATBINS=/path/to/fatbins-samplecand \
OUT="$PWD/engine/build/out" \
C="$PWD/engine/build/zig-cache" \
JOBS=4 bash engine/build/build.sh
```

The fixed build is `native dsv41-tools -Dtarget=aarch64-linux-gnu
-Dsample_cand=true`, with vendored modes: native `safe`, tools `fast`.
Optionally set `W` to another TensorFold source tree. Relative overrides resolve
against the caller's directory. `EXPECTED_ZIG_SHA256` requires an exact compiler
executable hash; `FATBINS_SHA256` verifies a trusted SHA-256 manifest relative
to the fatbin directory. Both checks are optional.

Outputs:

- `OUT/native/bin/tensorfold-native`
- `OUT/bin/tf-dsv41-lanes`
- `OUT/bin/tf-dsv41-check`
- `OUT/SHA256SUMS` and `OUT/BUILD_RECEIPT.json`

Build metadata records compiler identity, arguments, source hashes (the `zig/`
tree and root build/version files), supplied fatbin hashes and output hashes.
`status: succeeded` marks a completed build. Redact local paths before sharing
this metadata. Use separate `OUT` and `C` directories for independent rebuilds.
Compare output hashes with the deployed binaries. Compiler distribution, paths,
debug information and toolchain nondeterminism can change binary bytes even
with identical source and options; different hashes mean a mismatch.

### Focused host tests

Run these checks without model weights or a GPU:

```bash
cd engine/tensorfold
/path/to/zig/zig build test-prefill128 test-prefill-owned \
  test-prefill-admission test-native-stepped test-context-capacity \
  test-engram-fault test-direct-gather -j4 --summary all
```

These cover selected host code, not full-engine, inference or multi-rank hardware tests.

### Runtime dependencies are separate

The executables also load a kernel kit (base AOT kernels with the LakyFx 1.2.8
prefill supplement from `tensorfold/zig/kernels/prefill_128/`, EXL3 cubins,
Engram constants and token map, RoPE tables) and each rank's Engram tables.
`scripts/prepare.sh` fetches, checks and stages both; [runtime
assets](../docs/runtime-assets.md) lists every file with its pinned source and
SHA-256, and the manual commands. The optional grammar library
(`libtfgrammar.so`) is staged with the binaries when the build output has one.
PDF input (`--vision-pdf`) loads PDFium's `libpdfium.so` from `TENSORFOLD_LIBPDFIUM`
or `lib/` beside `tensorfold-native` (its directory's `../lib`, then the directory
itself); `prepare.sh` stages the pinned arm64 build into `BIN_DIR`. The server never
loads it: each PDF's render worker does.

`zig build test-pdf-worker -Dpdfium=/path/to/x64/libpdfium.so` spawns that worker
from a host `tensorfold-native` and renders real PDFs; `test-image-count` includes
its exact page count. Without the library both skip.

## Weight tools

`tensorfold/tools/dsv41-tp3/rank_cache.py` cuts a standard EXL3 folder into the
three per-Spark rank files the engine loads.
[`../tools/weights/`](../tools/weights/) reverses the split and checks the round
trip byte for byte.
