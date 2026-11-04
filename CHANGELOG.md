# Changelog

DeepSeek V4.1 Flash 1017 abliterated EXL3 3.84 bpw runs across three NVIDIA DGX Spark systems with TensorFold tensor parallelism 3 (TP3).

v1.1 uses engine base `14803a3` with exactly two non-runtime edits in `zig/src/server/metrics.zig`. A documentation comment and a test name now describe external metrics samplers generically. Serving logic is unchanged. Rebuild comparison results: `release/build-verification.json`.

## Unreleased

- Serving: removed automatic context shortening. Prompts that do not fit `context_tokens` are refused, and clients manage their own context. Delete `auto_compact`, `compact_*` and `checkpoint_*` from existing configurations; the API no longer accepts them.
- Serving: added `/v1/messages` and `/v1/messages/count_tokens`, forwarded to the engine's native Messages routes, and `x-api-key` authentication.
- Serving: a reply cut by `max_tokens` while thinking is a normal length finish, not an empty-reply error. Incomplete streams now log why they stopped.
- Serving: Responses admission always uses the native tokenizer, the same as Chat Completions.
- Serving: `thinking: {type}` now decides thinking mode, and `ultra` effort maps to `max`, matching DeepSeek's `deepseek-recipe`.
- Serving: a Messages reply with thinking but no final text or tool call is now an error, as on Chat Completions, and an oversized Messages `max_tokens` is lowered to the configured bound, as DeepSeek does.

## 2026-11-02 v1.1

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
