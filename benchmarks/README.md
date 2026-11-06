# TP3 benchmark method and results

Results are grouped by runtime and date. Compare runs with matched settings.

## v1.2 — 2026-11-05

The [v1.2 numeric summary](v1.2/summary.json) records cold prefill, the 755-item standard quality set and a
near-empty-pool decode, all measured 5 November 2026 on the production engine `30a9647` on three DGX Spark GB10
systems in TP3, an 8M-token KV pool, 16 streams, all CPU cores and performance cores capped at 3,600 MHz. Prefill used
stock GPU clocks; decode and the quality set capped GPUs at 2,350 MHz. The summary retains the source hashes and all per-stream timings.

### Decode

Aggregate tok/s across concurrent requests; each request generated 400 tokens. One run per point, with cached short
prompts and zero failed streams across all 48 points. Greedy has thinking off and temperature 0; thinking uses temperature
0.7. Both use top_p 1.

| Prompt, greedy | C1 | C2 | C4 | C6 | C8 | C10 | C12 | C16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prose | 81.96 | 111.15 | 161.36 | 210.65 | 219.17 | 239.03 | 251.73 | 270.27 |
| Code | 124.37 | 167.23 | 191.85 | 214.28 | 219.95 | 238.38 | 245.21 | 253.40 |
| Structured | 154.89 | 261.29 | 389.72 | 449.87 | 443.14 | 460.66 | 500.52 | 502.05 |

| Prompt, thinking | C1 | C2 | C4 | C6 | C8 | C10 | C12 | C16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prose | 68.11 | 94.63 | 130.93 | 161.74 | 173.29 | 187.73 | 196.45 | 215.39 |
| Code | 82.03 | 119.26 | 141.91 | 165.10 | 175.10 | 182.19 | 196.31 | 211.29 |
| Structured | 141.53 | 179.62 | 239.88 | 272.96 | 261.64 | 281.40 | 305.44 | 313.27 |

### Prefill and standard quality set

Cold prompts, zero cached tokens, one request at a time. The collector subtracts 171.55 ms of request overhead from
TTFT-derived prefill throughput. Three samples through 32K, two at 64K and 128K.

| Target | Prompt tokens | Samples | Prefill tok/s | TTFT (s) |
|---|---:|---:|---:|---:|
| 4K | 4,116 | 3 | 3616.68 | 1.310 |
| 8K | 8,214 | 3 | 3732.41 | 2.372 |
| 16K | 16,403 | 3 | 3801.73 | 4.486 |
| 32K | 32,786 | 3 | 3815.85 | 8.764 |
| 64K | 65,555 | 2 | 3788.88 | 17.474 |
| 128K | 131,088 | 2 | 3725.14 | 35.362 |

The standard 755-item set scored **97.1%** (720 passed): arithmetic, reasoning and tracking each 100%, GSM8K 96.5%,
MMLU 94.4% and QA 92.0%. The AIME 2025, GPQA-Diamond, IFEval, LiveCodeBench and MMLU-Pro results below are separate
evaluations of the same weights.

### Earlier regression checks

Spot checks on v1.2 (engine `591d9dc`) on 2026-11-04, against the previous production engine `8f4faf6`, an unreleased build between v1.1 and v1.2:

| Check | Result |
|---|---|
| Terminal-Bench prompts | 13 of 13 token-identical |
| Decode | 60.9 tok/s single stream and 158.4 tok/s at 8 streams, unchanged within noise |
| Output parsing against `deepseek-recipe` | Matched in 9 of 9 thinking plus tool call cases |

These are quick regression checks against that engine, not comparable with the tables below.

## v1.1 — 2026-11-02

Production results from **10:59–11:19 p.m. Eastern on 1 November 2026**. The [sanitized numeric summary](v1.1/summary.json) contains all 48 decode points, 6 prefill points, per-stream timing and token counts, request settings and validation notes.

### Conditions

- Three DGX Spark GB10 systems, 128 GB unified memory each, pairwise RoCE, TP3.
- Engine `14803a3`, selected mrfakename-edited 3.84 bpw composition; 8,000,000-token KV pool, 16 streams, graph cap 128, R48, 256 GiB spill enabled, top_k 20.
- All CPU cores on every rank (`0-19`); performance cores capped at 3,700 MHz; stock GPU clocks.
- [sparkDash](https://github.com/MiaAI-Lab/sparkDash) collector `693fa8016ea3a24b29fe36a98367669fd316393b`; native TensorFold responses to OpenAI-style chat requests.
- Greedy: temperature 0, thinking off. Thinking: temperature 0.7, thinking on. Both use top_p 1.
- Decode: 400 output tokens per request at concurrency 1, 2, 4, 6, 8, 10, 12 and 16; one 32-token warmup per section. The collector counts 399 decode tokens after first delivery.
- Prefill: 4K–128K target prompts, greedy mode, 8-token request budget, 2 completed tokens per reported point; 3 samples through 32K and 2 at 64K/128K. All reported prefill points have zero cached tokens.
- Engine identity stayed unchanged; all sections completed with zero manifest errors and zero rank-log error deltas.

### Greedy decode

Aggregate tokens per second across concurrent requests.

| Prompt | C1 | C2 | C4 | C6 | C8 | C10 | C12 | C16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prose | 84.32 | 112.41 | 165.82 | 213.09 | 234.39 | 237.98 | 248.93 | 262.56 |
| Code | 125.70 | 168.71 | 192.05 | 212.78 | 219.37 | 238.29 | 249.74 | 253.94 |
| Structured | 152.85 | 260.68 | 387.71 | 451.95 | 510.91 | 469.27 | 505.15 | 487.60 |

### Thinking decode

Aggregate tokens per second across concurrent requests.

| Prompt | C1 | C2 | C4 | C6 | C8 | C10 | C12 | C16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prose | 68.64 | 94.71 | 135.33 | 157.42 | 174.48 | 190.86 | 197.38 | 216.78 |
| Code | 86.40 | 122.26 | 147.25 | 166.08 | 176.43 | 186.58 | 192.54 | 211.49 |
| Structured | 140.84 | 183.36 | 236.83 | 273.71 | 288.06 | 286.44 | 300.27 | 318.47 |

### Prefill

Collector TTFT-derived throughput, with 165.82 ms of request overhead subtracted. TTFT is time to first token.

| Target | Actual prompt tokens | Samples | Prefill tok/s | TTFT (s) |
|---|---:|---:|---:|---:|
| 4K | 4,115 | 3 | 3560.15 | 1.322 |
| 8K | 8,215 | 3 | 3707.75 | 2.381 |
| 16K | 16,404 | 3 | 3810.75 | 4.470 |
| 32K | 32,790 | 3 | 3832.91 | 8.721 |
| 64K | 65,556 | 2 | 3800.11 | 17.417 |
| 128K | 131,090 | 2 | 3720.85 | 35.397 |

Largest measured prompt: **131,090 tokens** of the configured 1,048,576-token context.

### Quality and refusal screens

Same weights through the serving API at 16 requests at once, finished 2026-11-02 unless noted. Engine `14803a3` with full-vocabulary sampling (top_k 0) unless noted: GPQA-Diamond used the production sampler (top_k 20) and ran mostly on the earlier engine `7d5b4d8`, whose greedy replies match, with its last 131 items on `14803a3`; AIME 2025 ran on `17800d4`, `14803a3` plus one later engine port. Data: [quality.csv](v1.1/quality.csv).

| Suite | Setting | Result |
|---|---|---:|
| GPQA-Diamond | 198 questions × 8 samples, thinking on, effort 50, temperature 1, top_p 1, 32,768-token cap | 90.53% (1,434 of 1,584) |
| AIME 2025 | 30 problems × 4 samples, thinking on, effort 50, temperature 1, top_p 1, 32,768-token cap, rule-based answer check; finished 2026-11-03 | 97.50% (117 of 120) |
| IFEval | 541 prompts, greedy, thinking off, 2,048-token cap, official checker; prompt-level strict | 88.72% (480 of 541) |
| LiveCodeBench v6 | 175 problems, one pass, thinking on, effort 50, temperature 1, 32,768-token cap | 88.00% (154 of 175) |
| MMLU-Pro | 500-item stratified subset of the 12,032 test items, thinking on, effort 50, temperature 1, top_p 1, 32,768-token cap; finished 2026-11-03 | 88.00% (440 of 500) |
| HarmBench, official classifier | The 100 chat answers in [refusal.csv](v1/refusal.csv), rated by `cais/HarmBench-Llama-2-13b-cls` | 97 of 100 complied |
| XSTest safe, all 250 | Local Qwen2.5-7B-Instruct classifier with the published three-way rubric | 0 full refusals, 61 partial (24.4%) |

DeepSeek's model card reports 90.9% GPQA-Diamond for DeepSeek-V4.1-Flash at maximum effort (100), temperature 1 and top_p 0.95; these runs use effort 50. On IFEval, a third-party run of the official DeepSeek-V4.1-Flash-1017 API with the same 541 prompts and settings scored 89.8% prompt-level strict. 95% margins from the normal approximation over all scored answers (repeats are not clustered by question): GPQA-Diamond ±1.4 points, MMLU-Pro ±2.85; one-pass results carry several points of sampling noise. Answers that hit the token cap count as wrong: GPQA-Diamond 7 of 1,584, IFEval 10 of 541, LiveCodeBench 12 of 175, MMLU-Pro 1 of 500, AIME 2025 none.

The XSTest partials sit almost entirely in three prompt types whose right answer is to reject a nonsense premise (nonsense-group discrimination 21 of 25, privacy of fictional characters 20 of 25, real-group nonsense discrimination 14 of 25). XSTest's rubric counts rejecting such a premise as full compliance, so most of these are the 7B grader misapplying it. The other seven prompt types had 6 partials in 175 (3.4%) and no full refusals. The sparkDash set and the original refusal grading remain in the historical sections below.

### Development measurements

[development.csv](dev/development.csv) records every development number the README quotes: the prefill steps from
`ee903f5` to `14803a3`, the spill, prefix-wait, sampling and idle-backoff results, host memory and clock changes, the
graph-capture and qualification fixes, and the arms that were tried and dropped. Each row gives the measured values,
its baseline, engine, date and conditions. `record` says whether it comes from the run log or from a dated
engineering note written during the run. Parity means 8 long and 12 short greedy replies matched the reference in
text and reasoning; token IDs were not recorded.

## Historical v1 — 2026-10-30

v1 measures DeepSeek V4.1 Flash 1017 abliterated EXL3 3.84 bpw on three NVIDIA DGX Spark GB10 systems with [Mia's AI Lab sparkDash](https://github.com/MiaAI-Lab/sparkDash), revision `693fa80`, version 2.75.4.

Numeric summaries: [decode.csv](v1/decode.csv), [prefill.csv](v1/prefill.csv) and [quality.csv](v1/quality.csv).

### Historical measured configuration

| Setting | Value |
|---|---|
| Speed suite date | 2026-10-30 |
| Engine | `cebcf1e` |
| Hardware and network | Three DGX Spark systems, 128 GB unified memory each, pairwise RoCE, TP3 |
| CPU placement | Efficiency cores, masks `0-4,10-14` |
| GPU clocks | Stock |
| Engine capacity | 16 streams, graph cap 128, KV pool 10,000,000 tokens |
| Context | 1,048,576 tokens configured; prefill measured through 32K |
| Spill and reserve | 256 GiB NVMe spill on the head; 4 GiB startup memory reserve |
| Decode output budget | 400 tokens per request |
| Greedy mode | Temperature 0, thinking off |
| Think mode | Temperature 0.7, thinking on |
| Sampling baseline | Temperature 1, top_p 1, top_k 20, no min_p, thinking high; benchmark mode overrides apply |
| Repetition | One run per decode point |

Each section used a 55 C SoC cooldown gate. Engine identity, settings and core masks stayed unchanged, with no rank errors. Six decode sections covered greedy and think modes on prose, code and structured prompts.

The October 31 runtime used engine `7d5b4d8` and API 0.1.5. Its numeric-effort update passed response-parity checks; the full speed suite was not rerun. The [historical manifest](../release/history/v1.json) records v1 separately; [v1.1](../release/history/v1.1.json) is archived the same way.

## Historical decode throughput

Aggregate tokens per second across concurrent requests.

| Mode and prompt | 1 stream | 4 streams | 8 streams | 12 streams | 16 streams |
|---|---:|---:|---:|---:|---:|
| Greedy prose | 84.5 | 162.5 | 209.3 | 235.9 | 260.7 |
| Greedy code | 127.8 | 184.7 | 215.7 | 244.5 | 247.7 |
| Greedy structured | 156.6 | 364.5 | 496.6 | 491.5 | 470.8 |
| Think prose | 71.8 | 132.6 | 171.3 | 196.2 | 208.3 |
| Think code | 86.1 | 140.8 | 174.3 | 194.6 | 206.4 |
| Think structured | 142.6 | 228.1 | 261.7 | 298.6 | 306.1 |

The [decode CSV](v1/decode.csv) has one row per mode, prompt category and stream count. Throughput column: `aggregate_tok_s`.

## Historical prefill

| Prompt tokens | Tokens per second |
|---|---:|
| 4,096 | 2,885 |
| 8,192 | 3,086 |
| 16,384 | 3,120 |
| 32,768 | 3,089 |

The [prefill CSV](v1/prefill.csv) records `prompt_tokens` and `tok_s`. The measured range ends at 32K.

## Historical quality

The sparkDash standard set contains 755 items. The [quality CSV](v1/quality.csv) reports category percentages.

| Category | Score |
|---|---:|
| Arithmetic | 100% |
| Reasoning | 100% |
| Tracking | 100% |
| GSM8K | 97.5% |
| MMLU | 94% |
| QA | 92% |
| Overall | 97.3% |

## Fidelity and refusal behavior

Fidelity is mean per-token KL(official model || pack) against the published
[official-model teacher logits](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits): 65 balanced
cuts of 2,048 tokens, 133,120 positions. Lower KL means closer agreement. It covers quantization plus the refusal-removal
edit, separately from task accuracy. The difference column is paired per cut with a bootstrap 95% interval. Data:
[fidelity.csv](v1/fidelity.csv).

| Weights | Mean KL | Top-1 agreement | p99 KL | This composition lower by (95% CI) | Cuts lower |
|---|---:|---:|---:|---:|---:|
| This composition + mrfakename | 0.0784 | 92.0% | 1.463 | | |
| LakyFx 3.5 bpw, no edit | 0.0872 | 91.2% | 1.517 | 0.0088 [0.0035, 0.0131] | 59 of 65 |
| LakyFx 3.5 bpw + mrfakename | 0.0937 | 90.8% | 1.620 | 0.0153 [0.0117, 0.0189] | 64 of 65 |
| LakyFx 3.5 bpw + pqhaz heretic | 0.1002 | 90.8% | 1.746 | 0.0218 [0.0175, 0.0265] | 64 of 65 |
| This composition + Keys | 0.1111 | 90.7% | 1.953 | 0.0326 [0.0247, 0.0427] | 64 of 65 |
| LakyFx 3.5 bpw + Keys | 0.1225 | 89.7% | 2.119 | 0.0441 [0.0341, 0.0556] | 65 of 65 |
| LakyFx 3.5 bpw + Mia's Keys K5 | 0.1234 | 89.6% | 2.097 | 0.0450 [0.0349, 0.0566] | 65 of 65 |
| LibertAIDAI 3.5 bpw + Keys | 0.1268 | 90.1% | 2.031 | 0.0484 [0.0386, 0.0593] | 64 of 65 |

Refusals use 100 HarmBench `standard` behaviors (stratified by category, copyright excluded) and 50 XSTest safe prompts,
greedy, through the API. Chat has thinking off and a 768-token budget. Thinking runs the first 50 harmful and 25 safe
prompts with a 3,072-token budget. Answers are graded answered, partial or refused with a fixed rubric plus a keyword
screen. Every arm: [refusal.csv](v1/refusal.csv).

| Subset | Answered | Partial | Refused |
|---|---:|---:|---:|
| Harmful, chat | 98 | 2 | 0 |
| Harmful, thinking | 18 | 31 | 1 |
| Safe, chat | 49 | 1 | 0 |
| Safe, thinking | 18 | 6 | 1 |

35 of the 50 harmful thinking replies hit the 3,072-token cap before a final answer; most partials are those.

## Run a comparison

Use the historical sparkDash revision and version for v1 comparisons. Record the collector revision for new runs. Start with the [lifecycle scripts](../docs/reproduction.md), then match modes, prompt categories, stream counts and output budgets. Match engine `cebcf1e` for v1; runs on `7d5b4d8`, `14803a3`, `678ea1b` or `30a9647` are separate measurements.

Record weight identity, engine/API/collector versions, settings, topology, clocks and affinity. Stop unrelated traffic, warm the engine and apply the cooldown gate before each section. Keep prompts, actual token counts, cache state, timing boundaries and errors in private run records. Publish numeric summaries without credentials or host details.

Compare matched cells. Repeated prompts can lower time to first token through cache reuse. Repeat runs to assess a few-percent speed difference. Compare decode, prefill, end-to-end latency, task quality and fidelity separately.
