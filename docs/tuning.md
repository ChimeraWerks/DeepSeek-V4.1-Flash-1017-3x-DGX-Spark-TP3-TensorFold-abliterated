# Production tuning and measured effects

The [release manifest](../release/manifest.json) defines the v1.2 configuration selected on 2026-11-04. The historical comparisons below explain the settings. Speedups are not additive.

## Production configuration

| Setting | Value |
|---|---|
| Engine and API | TensorFold `591d9dc`, serving API 0.1.5 |
| Decode streams | 16 |
| CUDA graph cap / round rows | 128 / R48 |
| KV pool | 8,000,000 tokens |
| Context | 1,048,576 tokens configured; historical suite through 32K; November 2 native prefill through 131,090 tokens |
| NVMe spill | 256 GiB on the head |
| Startup memory reserve | 4 GiB |
| Sampling | temperature 1, top_p 1, top_k 20, no min_p, thinking `high` |

The [v1.1 benchmark record](../benchmarks/README.md#v11--2026-11-02) holds the latest full results; the engine there is `14803a3`. The comparisons below predate v1.1.

The manifest lists the full engine environment and rank-0 options. Set these variables in the engine environment, not as `cluster.env` keys.

```text
TF_DS_SAMPLE_CAND=1
TF_DS_GRAPH_VOTE_SPILL=1
TF_DS_PREFILL_YIELD=1
TF_DS_PREFILL_OWNED=1
TF_DS_DIRECT_GATHER=1
TF_DS_ROUND_ROWS=48
TF_DS_PREFILL_AE=1
TF_DS_PREFILL_W=1
TF_DS_PREFILL_K=1
TF_DS_PREFILL_AC=1
TF_DS_PREFILL_AF=1
TF_DS_PREFILL_D=1
TF_DS_PREFILL_E=1
TF_DS_PREFILL_F=1
TF_DS_PREFILL_COMPUTE_ROWS=8192
TF_DS_PREFILL_SHARED_SPLIT=1
TF_DS_PROMPT_ALIAS=1
TF_DS_PREFILL_SLICE=5
TF_DS_STEP_AHEAD=0
NCCL_BUFFSIZE=1048576
NCCL_MAX_CTAS=8
TF_CACHE_ADAPTIVE_PREFIX=1
TF_CACHE_BALANCED=1
TF_CACHE_NATIVE_PLAN=1
TF_CACHE_REUSE_EVICTION=1
TF_GRAPH_STORAGE_GUARD=1
TF_KV_CONTINUATION_DONOR=1
TF_KV_PRESSURE_PLANNER=1
TF_KV_RELOCATE_MAX_BYTES=335544320
TF_KV_SHORT_REUSE=1
TF_CACHE_X1_TRACE=1
TF_DS_SPILL_GIB=256
```

`TF_DS_PREFILL_SLICE=5` runs each yielded prompt chunk five layers at a time, with a decode round between slices. It needs `TF_DS_PREFILL_YIELD=1`. See [layer slices](../engine/build/PREFILL_LAYER_SLICES.md).

v1.2 also sets the engine's newer opt-in features to off, so a launch does not pick them up by accident:

```text
TF_DS_KEYED_DRAFT=0
TF_DS_DECODE_GREEDY_HEAD=0
TF_DS_DECODE_SAMPLE_HEAD=0
TF_SPILL_ROOM_WAIT_MS=0
TF_DS_GRAPH_TELEMETRY=0
TF_DS_FIRST_DRAFT_DEPTH=0
TF_DS_DRAFT_CAL=0
TF_CACHE_BATCH_DEAD=0
TF_CACHE_FIRST_CONTINUATION=0
```

The manifest also sets `TF_DS_REDUCED_HEAD_FATBIN`, as production does. The engine reads that file only when `TF_DS_DECODE_GREEDY_HEAD` or `TF_DS_DECODE_SAMPLE_HEAD` is on, and `prepare.sh` does not stage it. To try those heads, compile `engine/tensorfold/zig/kernels/cuda/dsv41_reduced_head.cu` yourself and put the result at that path.

The LakyFx 1.2.8 prefill port needs matching AOT and auxiliary kernels from the pinned [build contract](../engine/README.md). `TENSORFOLD_CUDA_KERNELS` selects the kit's `aot` directory; `TF_DS_PREFILL_RAGGED_CUBIN` selects `auxiliary/experts_ragged.cubin`.

`TF_DS_SPILL_DIR` selects the installation's spill directory. Rank 0 also receives `--spill-dir`, `--spill-gib 256` and `--top-k 20`. Rank 0 starts with `--reasoning-effort max`; the API defaults client requests to `high`.

## Historical measurements — 2026-10-27 through 2026-10-30

| Change | Measured comparison | Interpretation |
|---|---|---|
| Owned-row prefill, `7dd3526`, 2026-10-29 | 4K/8K/16K/32K prefill 2,935/3,010/3,027/3,023 tok/s versus 2,447-2,480 on `ee903f5`; +20-22%. Long 8/8 and short 12/12 parity. | Production settings, spill off, cold prompts. Later checks covered cold and cached text with spill on. |
| Extended owned-row work, `46ae23c`, 2026-10-29 | Prefill 2,983/3,083/3,066/3,055 tok/s; about +1% over the earlier owned-row runs. | Small incremental gain after the earlier +20-22%. |
| Direct all-gathers, `93d0b4c`, 2026-10-30 | Prefill 3,043/3,111/3,103/3,077 tok/s versus 2,983/3,083/3,066/3,055; +0.7-2.0%. | Enabled. A later decode comparison with the setting off was within about 3% without a consistent direction. |
| top_k 20, 2026-10-29 | Sampled C16, all cores: 184.1/184.1 tok/s without top_k versus 196.4/193.3 with 20. Efficiency cores: 172.7/172.4 versus 193.5/198.2. | Keeps sampled rows on GPU-prefiltered candidates instead of full-vocabulary CPU sampling. Changes the sampling distribution and timing. |
| Step-ahead, 2026-10-29 | A cold short request arriving after a reply took 0.41-0.48 s off versus 1.17-1.21 s on. Follow-up cache reuse was unchanged. | Disabled. The background pass delayed new work without improving reuse. |
| Spill, 2026-10-27 | TTFT 974 ms after eviction and 1,039 ms after restart versus 6,099 ms cold. | Restore timings from an earlier engine and pack, before v1. |

Cache, storage and memory flags were measured together, with no per-flag speedup breakdown.

## Host memory and firmware

The 2026-10-27 host audit recorded DGX OS 7.6.0, Ubuntu 24.04.5, kernel `7.0.0-1019-nvidia-64k`, NVIDIA driver 580.178.04 and CUDA 13.0. Hosts used 64 KiB pages with memory-saver packing, October firmware and Display Reserved Memory set to Auto.

| Per-node measurement, 2026-10-26 | Before combined changes | After combined changes |
|---|---:|---:|
| Kernel-reserved memory at boot | 6.04 GiB | 2.44 GiB |
| MemTotal | 121.7 GiB | 125.3 GiB |
| Loaded MemAvailable, head | 18.3 GiB | 19.0 GiB |
| Loaded MemAvailable, follower 1 | 19.8 GiB | 20.7 GiB |
| Loaded MemAvailable, follower 2 | 19.8 GiB | 20.9 GiB |
| Display carveout | 2,046 MiB | 512 MiB |

These figures cover combined kernel, memory-saver and firmware/display changes on the older deployment. Component contributions were not measured separately. Paired checks found 285/285 MMLU outputs identical and GSM8K 195/200 correct on both kernels.

Update firmware and kernels separately from the lifecycle scripts. Keep security controls and host networking unchanged.

## Clocks and CPU placement

The 2026-10-30 v1 speed suite used efficiency cores and stock GPU clocks. The capped-clock tests restored stock clocks afterward.

| Trial on `cebcf1e` | Observed effect | Status |
|---|---|---|
| Efficiency cores with GPU capped at 2300 MHz | GPU power 40-43 W in decode and 54.6 W in prefill, versus 56-60 W and 75.8 W stock; prefill at 8K and above was 4-5% lower. | Tested; benchmark uses stock clocks. Small positive decode differences partly reflected prompt-cache effects. |
| All cores with GPU capped at 2200 MHz | Prefill at 8K and above about 5% lower; head P-core sensor reached 89.5 C. | Tested, restored afterward. Affinity and clocks changed together. |
| Efficiency cores capped at 1400 MHz | Prefill 13-17% lower and single-stream decode 2-7% lower; batched decode unchanged. | Tested, stock restored. |
| Deep idle disabled | Flat to 4% lower in the comparison. | No measured gain; defaults restored. |

The earlier core-placement comparison used sampling without top_k 20. With top_k set in the tested cases, the core-mode difference disappeared.

## Sampling and effort

`standard` uses temperature 1 and top_p 1. `focused` uses temperature 1, top_p 0.95 and min_p 0.05. Both use top_k 20. Explicit valid request values override profiles and installation defaults.

| Effort name | Numeric value |
|---|---:|
| minimal | 25 |
| low | 50 |
| medium | 62 |
| high | 75 |
| xhigh | 88 |
| max | 100 |

Use `reasoning_effort: none` to disable thinking. Explicit whole-number efforts from 1 to 100 bypass named-level remapping. API 0.1.5 supports live remapping when the engine advertises `numeric_effort`.

Preset selection checks `X-Recipe-Profile`, the model's `@preset` suffix, the client's assigned profile and the default profile, in that order. Per-field request values override the selected profile, which overrides installation defaults. Request nulls leave defaults unchanged.

The engine has no presence or frequency penalty. The API requires the neutral default `presence_penalty: 0`. Benchmark greedy and thinking modes override ordinary sampling defaults; see [method](../benchmarks/README.md).

## Live settings and capacity

Use the operator credential at `/admin/settings` to update `sampling`, `profiles`, `client_profiles`, `default_profile`, `reasoning_default`, `max_concurrency`, `max_queue` and `effort_map`. Changes persist in `live-settings.json`. Other settings require an API restart. See the [serving guide](../serving/README.md).

Keep `max_concurrency` at or below the configured 16 engine streams. Lowering API concurrency leaves native decode-stream capacity unchanged. API `context_tokens` defaults to 1,048,576, matching the bundled `v1.2` native profile and release manifest.

Spill and optional KV sharding are separate features. TP3 KV sharding requires fetch decode, space in each owner part for the full window plus round reserve, and a pool divisible by 6,144. The production 8,000,000-token pool needs resizing for KV sharding. Spill is incompatible with `cache_salt`.
