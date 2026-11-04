# Changelog

DeepSeek V4.1 Flash 1017 abliterated EXL3 3.84 bpw runs across three NVIDIA DGX Spark systems with TensorFold tensor parallelism 3 (TP3).

v1.2 uses engine base `591d9dc` with exactly two non-runtime edits in `zig/src/server/metrics.zig`. A documentation comment and a test name describe external metrics samplers generically. Serving logic is unchanged by them. Rebuild comparison results: `release/build-verification.json`.

## 2026-11-04 v1.2

v1.2 has not been benchmarked. Every published speed, fidelity, quality and refusal result was measured on v1.1 or earlier. The weights, KV pool, stream count, spill and sampling defaults are unchanged from v1.1.

Checks on the production engine, 2026-11-04, against the previous production engine `8f4faf6` (an unreleased build between v1.1 and v1.2):

- Terminal-Bench prompts: 13 of 13 token-identical.
- Decode: 60.9 tok/s single stream and 158.4 tok/s at 8 streams, unchanged within noise. These are spot checks, not comparable with the benchmark tables.
- Output parsing matched DeepSeek's `deepseek-recipe` in 9 of 9 thinking plus tool call cases.

Engine `14803a3` to `591d9dc`:

- Enabled prefill layer slices (`TF_DS_PREFILL_SLICE=5`): running streams decode between slices of a long prompt chunk. The published prefill numbers predate this.
- Matched `deepseek-recipe` prompt rendering: the `response_format` schema reaches the template, tools or a schema add an empty system block when needed, a `thinking` object decides the mode, Messages tool JSON keeps chat key order, and a prefix continuation leaves off the end token.
- Matched `deepseek-recipe` on Responses: every reasoning item is kept, the custom `apply_patch` tool and `response_format` regex work, and tool namespaces are flattened and restored.
- Matched `deepseek-recipe` on output and validation: think and DSML markers count only as real tokens, broken tool history and duplicate tool names are refused before inference, and strict tool and response schemas are enforced.
- These match the listed behaviors, not all of `deepseek-recipe`. What still differs includes effort numbers, default thinking, default output caps, sampling clamps, deferred tools, system images and logprobs.
- Fixed replayed `reasoning` fields (Mia's patch 0047, nuk3s), undrained GET and DELETE bodies (Mia's patch 0046, Christopher A. Williamson), a finished-job race (found in Mia PR 94 review, hecisaza), a first-token finish that skipped its release (TensorFold `ed39acc5`, laudney) and an uninitialized read-ahead start.
- Added opt-in features, all off in production: dead-checkpoint batches, first-continuation hints, draft calibration, first-draft depth, keyed drafts, reduced output heads, restore-room waits, graph telemetry and streamed tool calls. The launch sets all but streamed tool calls to off explicitly; see [engine/README.md](engine/README.md).

Release:

- Selected native profile `v1.2`, recipe patches `chimera-tensorfold-patches-20261103`; kept `v1`, `v1.1` and their manifests for history.
- Added `release/history/v1.1.json` and moved the v1.1 build record to `release/history/v1.1-build-verification.json`.
- `release/build-verification.json` records the production hashes and a rebuild of the public tree: all 22 allocated ELF sections match in each binary.

Serving API, version unchanged at 0.1.5:

- Removed automatic context shortening. Prompts that do not fit `context_tokens` are refused, and clients manage their own context. Delete `auto_compact`, `compact_*` and `checkpoint_*` from existing configurations; the API no longer accepts them.
- Added `/v1/messages` and `/v1/messages/count_tokens`, forwarded to the engine's native Messages routes, and `x-api-key` authentication.
- A reply cut by `max_tokens` while thinking is a normal length finish, not an empty-reply error. Incomplete streams now log why they stopped.
- Responses admission always uses the native tokenizer, the same as Chat Completions.
- `thinking: {type}` now decides thinking mode, and `ultra` effort maps to `max`, matching DeepSeek's `deepseek-recipe`.
- A Messages reply with thinking but no final text or tool call is now an error, as on Chat Completions, and an oversized Messages `max_tokens` is lowered to the configured bound, as DeepSeek does.

## Historical — 2026-11-02 v1.1

v1.1 uses engine base `14803a3` with the same two non-runtime `metrics.zig` edits; its rebuild comparison is in `release/history/v1.1-build-verification.json`.

- Selected engine baseline `14803a3`, including the LakyFx 1.2.8 prefill port, with matching kernel inputs.
- Set an 8,000,000-token pool, C16, graph cap 128, R48, engine/API top_k 20 and 256 GiB spill. Kept 1,048,576-token configured context and 4 GiB startup reserve.
- Kept the existing mrfakename-edited 3.84 bpw composition and rank identities.
- Selected native profile `v1.1`; kept `v1` and its manifest for history.
- Made the optional affinity-controller integration opt-in instead of a launch prerequisite.
- Added a separate [November 2 benchmark record](benchmarks/README.md#v11--2026-11-02), preserving the October 30 speed suite and its timings.
- Added model lineage, refusal-removal limits and component notices.
- Pinned and staged the kernel kit and Engram tables in `prepare.sh`: the kit from a Hugging Face mirror, the Engram rows from DeepSeek's repository ([runtime assets](docs/runtime-assets.md)).

## Historical — 2026-10-31 numeric reasoning effort

- Engine `7d5b4d8` adds numeric `reasoning_effort` from 1 to 100 to `cebcf1e`. Named levels keep their existing values.
- Serving API `chimera-dgx-sparks` 0.1.5 adds live effort remapping. Non-native mapping changes require engine `/health` to report `numeric_effort`.
- Response checks matched 8/8 long cases, cold and cached, and 12/12 short cases. Named efforts matched their numeric equivalents in these checks.

The [speed suite](benchmarks/README.md) ran on `cebcf1e` on 30 October, with no rerun for the numeric-effort update.

## Historical — 2026-10-30 v1 production recipe

- Combined EXL3 weight sources and a rank-2 refusal-removal edit into the [3.84 bpw composition](release/weight-provenance.md).
- Configured 16 streams, CUDA graph cap 128, a 10,000,000-token KV pool, 1,048,576-token context and 256 GiB NVMe spill on the head.
- Enabled owned-row prefill, direct all-gathers and top_k 20. Kept step-ahead disabled.
- Measured greedy prose decode at 84.5 tok/s for one stream and 260.7 tok/s aggregate for 16 streams. Prefill at 32K reached 3,089 tok/s.
- Fixed Engram read-failure exits. Host tests covered the fault path; Spark fault injection was not run. Separate supervisor tests recovered from a follower failure and a follower reboot.

See the [release manifest](release/manifest.json) for the current configuration and [tuning](docs/tuning.md) for measured tradeoffs. Repository code is Apache-2.0; third-party code keeps its own licenses. Weights: [Hugging Face](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated).
