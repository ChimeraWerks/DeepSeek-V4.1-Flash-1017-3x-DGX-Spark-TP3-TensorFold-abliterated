# Engine

`tensorfold/` contains the TensorFold native Zig engine with BertholomusAI's
`dsv41` family for DeepSeek V4.1 Flash. Release v1.1 uses engine source
`14803a3`: the Laky 1.2.8 prefill port and our patches, plus one metrics
documentation-comment edit and one test-name wording edit. Serving logic is
unchanged. `7d5b4d8` is the historical v1 engine.

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

## CPU-only cross-build

Run from the repository root on Linux. Install Bash, Python 3, GNU coreutils,
`ionice` and the complete Zig toolchain matching
[`tensorfold/.zig-version`](tensorfold/.zig-version) (currently `0.17.0`).
Supply a compatible CUDA fatbin directory. Keep the compiler executable hash
and complete distribution for build comparisons; the version string alone
cannot identify the distribution.

Supply every non-opt-in kernel listed in
[`tensorfold/zig/build/cuda.zig`](tensorfold/zig/build/cuda.zig), plus
`dsv41_sample_cand.fatbin`. The script neither ships nor fetches fatbins.
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

## Weight tools

`tensorfold/tools/dsv41-tp3/rank_cache.py` cuts a standard EXL3 folder into the
three per-Spark rank files the engine loads.
[`../tools/weights/`](../tools/weights/) reverses the split and checks the round
trip byte for byte.
