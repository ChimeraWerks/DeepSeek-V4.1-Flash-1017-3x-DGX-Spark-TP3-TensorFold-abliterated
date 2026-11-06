---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
- deepseek
- deepseek-v4
- exl3
- exllamav3
- abliterated
- uncensored
- mixture-of-experts
- dgx-spark
- tensorfold
- tensor-parallelism
---

# DeepSeek V4.1 Flash on 3× DGX Spark: 3,816 tok/s prefill, 124 tok/s code decode

Abliterated EXL3 weights of DeepSeek-V4.1-Flash-1017. **[Serving recipe on GitHub](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated)**

<p align="center">
  <img src="assets/scorecard.png" width="100%" alt="DeepSeek V4.1 Flash abliterated on 3x DGX Spark, 3.84 bpw EXL3, with a glowing mesh whale. 3,816 tok/s prefill at a 32K prompt, 82 tok/s prose decode and 124 tok/s code decode at 1 request, 270 tok/s combined at 16 requests, 1M token context with an 8M KV pool." />
  <img src="assets/speed-grid.png" width="100%" alt="Prefill tok/s and first token: 4K 3,617 in 1.31 s, 8K 3,732 in 2.37 s, 16K 3,802 in 4.49 s, 32K 3,816 in 8.76 s, 64K 3,789 in 17.47 s, 128K 3,725 in 35.36 s. Combined decode tok/s at C1, C4, C8 and C16. Prose greedy 82.0, 161.4, 219.2, 270.3; prose thinking 68.1, 130.9, 173.3, 215.4; code greedy 124.4, 191.9, 220.0, 253.4; code thinking 82.0, 141.9, 175.1, 211.3. Quality: standard set 97.1%, AIME 2025 97.5%, GPQA-Diamond 90.5%, LiveCodeBench v6 88.0%, KL to official 0.0784, HarmBench refusals 0 of 100." />
</p>

**3× DGX Spark** · TP3 · 3.84 bpw EXL3 · about 274 GB · 1M context configured

<details>
<summary><b>Full specs: v1.2, measured 5 November 2026</b></summary>

| | |
| --- | --- |
| Prefill | **3,816 tok/s** at 32K; a 128K prompt reaches its first token in 35 s |
| Decode | **82 tok/s** prose and **124 tok/s** code at 1 request; **270 tok/s** combined at 16 requests |
| Context | 1,048,576 tokens configured; largest measured prompt 131,088 tokens |
| Hardware | 3 × DGX Spark (GB10, 128 GB), TP3, every pair cabled directly over RoCE |
| Refusals | 0 of 100 HarmBench prompts refused (98 answered, 2 partial); 0 of 250 XSTest safe prompts fully refused |
| Quality | **97.1%** on the 755-item standard set; **97.5%** AIME 2025; **90.5%** GPQA-Diamond (1,434 of 1,584); 88.7% IFEval; 88.0% LiveCodeBench v6; 88.0% MMLU-Pro (500-item subset) |
| Fidelity | Mean KL **0.0784** and 92.0% top-1 agreement with official logits; the lowest KL of the eight packs we scored |
| API | Chat Completions, Responses and Messages routes, tool calls, enforced JSON schemas, 16 streams |
| Quantization | EXL3, 3.8361 bpw over quantized linear layers (output head excluded, 6-bit) |
| Edit | [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated)'s rank-2 refusal-removal edit, applied in FP32 and re-encoded at 6 bits |
| Size | About 274 GB: 35 safetensors shards |

</details>

## Performance

Measured 5 November 2026 on production engine `30a9647` with [sparkDash](https://github.com/MiaAI-Lab/sparkDash).
All CPU cores and performance cores capped at 3,600 MHz. Prefill used stock GPU clocks; decode capped GPUs at 2,350 MHz.

### Prefill

| Prompt | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill tok/s | 3,617 | 3,732 | 3,802 | **3,816** | 3,789 | 3,725 |
| Time to first token | 1.31 s | 2.37 s | 4.49 s | 8.76 s | 17.47 s | 35.36 s |
| Prompt tokens | 4,116 | 8,214 | 16,403 | 32,786 | 65,555 | 131,088 |

<p align="center"><img src="assets/chart-prefill.png" width="100%" alt="Two line charts against prompt length from 4K to 128K tokens. Prefill tok/s: 3,617, 3,732, 3,802, 3,816, 3,789 and 3,725. Time to first token: 1.31, 2.37, 4.49, 8.76, 17.47 and 35.36 seconds" /></p>

Cold prompts, zero cached tokens, one request at a time; tok/s comes from time to first token with the collector's
171.55 ms request overhead subtracted.

### Decode

Combined tok/s across all requests at once:

| Requests at once | 1 | 2 | 4 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose, greedy | **82.0** | 111.2 | 161.4 | 219.2 | 251.7 | **270.3** |
| Prose, thinking | 68.1 | 94.6 | 130.9 | 173.3 | 196.5 | 215.4 |
| Code, greedy | **124.4** | 167.2 | 191.9 | 220.0 | 245.2 | 253.4 |
| Code, thinking | 82.0 | 119.3 | 141.9 | 175.1 | 196.3 | 211.3 |

What each request gets, mean tok/s per request:

| Requests at once | 1 | 2 | 4 | 8 | 12 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose, greedy | **82.0** | 57.2 | 41.0 | 28.1 | 21.7 | 17.5 |
| Prose, thinking | 68.1 | 47.8 | 33.8 | 22.6 | 16.8 | 14.0 |
| Code, greedy | **124.4** | 85.1 | 49.0 | 29.5 | 22.0 | 17.2 |
| Code, thinking | 82.0 | 59.6 | 37.7 | 24.5 | 17.9 | 14.7 |

<p align="center"><img src="assets/chart-decode.png" width="100%" alt="Two line charts of combined decode tok/s against requests at once. Greedy, prose and code: C1 82 and 124, C2 111 and 167, C4 161 and 192, C8 219 and 220, C12 252 and 245, C16 270 and 253. Thinking, prose and code: C1 68 and 82, C2 95 and 119, C4 131 and 142, C8 173 and 175, C12 196 and 196, C16 215 and 211" /></p>

Near-empty KV pool, short cached prompts, 400 output tokens per request, one run per point, timed at the engine. Greedy
is temperature 0 with thinking off; thinking is temperature 0.7 with thinking on. All 354 decode streams completed. Every point and
per-stream timing: [v1.2 benchmarks](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/benchmarks/v1.2/summary.json).

## Run it on three DGX Sparks

The tested setup is exactly three DGX Spark GB10 systems (128 GB each), every pair cabled directly over RoCE. Each
rank file is about 93 GB, so the model needs all three. Budget about 440 GB of free disk per Spark (weights, rank file, Engram rows
and kernel kit) and another 256 GiB on the head for the spill tier. Build the engine first, then follow the
[GitHub recipe](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated#quick-start):

```bash
cp cluster.env.example cluster.env && chmod 600 cluster.env   # HF_REPO and HF_REVISION already pin this repo
./scripts/prep-sparks.sh status   # firmware, 64K kernel, memory saver; apply, reboot, repeat
./scripts/preflight.sh     # check all three Sparks; changes nothing
./scripts/prepare.sh       # download this repo on each Spark, build and verify its rank file
./scripts/start.sh         # start the three ranks and the OpenAI-compatible API
./scripts/smoke-test.sh    # one API request
```

`prepare.sh` downloads this repository to every Spark, builds that Spark's rank file from `tp3/rank<N>.keys` and
checks it against the SHA-256 values in the
[release manifest](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/release/manifest.json).
Hardware, network and memory prerequisites:
[DGX Spark prerequisites](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated#dgx-spark-prerequisites).

The folder is standard EXL3, but it has only been tested with the TensorFold `dsv41` engine on this three-node setup.

Prefer to let an agent do it? The GitHub README has a
[copy-paste setup prompt](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated#set-up-with-your-coding-agent).

### Call it

The API speaks OpenAI Chat Completions and Responses, with streaming, tool calls and per-request thinking control. It
listens on the head's loopback; call it there or through an SSH tunnel to port 18002.

```bash
set -a; . ~/ds1017/api/credentials.env; set +a   # on the head: loads SPARK_API_KEY without printing it
curl -s http://127.0.0.1:18002/v1/chat/completions \
  -H "Authorization: Bearer $SPARK_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"high"}'
```

| Setting | Default | Notes |
| --- | --- | --- |
| Sampling | temperature 1, top_p 1, top_k 20 | Request values win |
| Thinking | on, effort `high` | `"thinking": false` or `"reasoning_effort": "none"` turns it off |
| Effort levels | `minimal` 25, `low` 50, `medium` 62, `high` 75, `xhigh` 88, `max` 100 | Or any integer 1–100 |
| Context | 1,048,576 tokens per request | 8M-token KV pool shared by 16 concurrent streams |

The thinking evals below ran at effort 50 (`low`); IFEval and the chat refusal tests ran with thinking off. Full reference: [serving API](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/serving/README.md).

## Release v1.2

The current production recipe runs engine `30a9647`, native profile `v1.2`, prefill layer slices and the serving API
with native Messages routes. It keeps the same weights as v1.1. Long chat histories use far less memory when building a
prompt, and a returning chat keeps its saved state while it waits in line, so it no longer recomputes from zero. See the
[changelog](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/CHANGELOG.md).

## Fidelity

<p align="center"><img src="assets/chart-kl.png" width="100%" alt="Lollipop chart of mean KL to the official model, with a line at the unedited 3.5 bpw quant: this release 0.0784; LakyFx 3.5 bpw unedited 0.0872, with mrfakename 0.0937, with heretic 0.1002; Keys overlay on our 3.84 bpw 0.1111, on LakyFx 3.5 bpw 0.1225, Mia's Keys K5 on LakyFx 3.5 bpw 0.1234, Keys overlay on LibertAIDAI 3.5 bpw 0.1268" /></p>

KL is mean per-token KL(official ‖ this model) against the published official-model logits
([brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits)):
65 cuts of 2,048 tokens, 133,120 positions, balanced across agentic, chat, code, SWE, math, science, legal,
multilingual, long-context, safety and pretraining text.

- Lower KL than the unedited 3.5 bpw quant on 59 of 65 cuts (paired mean 0.0088 lower, 95% CI 0.0035 to 0.0131).
- The same mrfakename edit scores 0.0937 on a 3.5 bpw base and 0.0784 on this 3.84 bpw mix.
- p99 KL 1.463. Every arm:
  [fidelity.csv](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/benchmarks/v1/fidelity.csv).

## Quality and refusals

The production engine scored **97.1%** (720 of 755) on the standard set measured 5 November 2026. Arithmetic, reasoning
and tracking each reached 100%; GSM8K scored 96.5%, MMLU 94.4% and QA 92.0%. The task evaluations below measure the same
weights on their own suites.

<p align="center"><img src="assets/chart-quality.png" width="100%" alt="Quality and refusals. GPQA-Diamond 90.5% against DeepSeek's 90.9% at max effort, AIME 2025 97.5%, IFEval 88.7% against the official API's 89.8%, LiveCodeBench v6 88.0%, MMLU-Pro 88.0%. 0 of 100 harmful prompts refused, 97 of 100 complied per HarmBench's classifier, 0 of 250 XSTest safe prompts fully refused with 61 partial" /></p>

| Suite | Setting | Score |
| --- | --- | ---: |
| GPQA-Diamond ([EvalScope](https://github.com/modelscope/evalscope)) | 198 questions × 8 repeats, thinking on (effort 50), temperature 1, top_k 20, 32K cap | **90.5%** (1,434 of 1,584) |
| AIME 2025 | 30 problems × 4 repeats, thinking on (effort 50), temperature 1, 32K cap, rule-checked answers | **97.5%** (117 of 120) |
| IFEval | 541 prompts, greedy, thinking off; prompt-level strict (official API, same settings: 89.8%) | **88.7%** (480 of 541) |
| LiveCodeBench v6 | 175 problems, one pass, thinking on (effort 50), temperature 1, 32K cap | **88.0%** (154 of 175) |
| MMLU-Pro | 500-item stratified subset, thinking on (effort 50), temperature 1, 32K cap | **88.0%** (440 of 500) |

DeepSeek reports 90.9% GPQA-Diamond for DeepSeek-V4.1-Flash at maximum effort (100); these runs use effort 50. Quality runs used full-vocabulary sampling (top_k 0) except GPQA-Diamond (top_k 20, the serving default). GPQA ran mostly on engine `7d5b4d8` (greedy output identical) and AIME 2025 on `17800d4`; the rest on `14803a3`.

| Refusal test, greedy | Answered | Partial | Refused |
| --- | ---: | ---: | ---: |
| HarmBench harmful, chat (100) | 98 | 2 | 0 |
| HarmBench harmful, thinking (50) | 18 | 31 | 1 |
| XSTest safe, chat (50) | 49 | 1 | 0 |
| XSTest safe, thinking (25) | 18 | 6 | 1 |

Chat runs use thinking off and 768 output tokens; 50 of the 100 harmful chat replies reached that cap. Thinking runs use 3,072 tokens; 35 of the 50 harmful thinking
replies hit that cap, which is where most partials come from. HarmBench's own classifier
(`cais/HarmBench-Llama-2-13b-cls`) rates 97 of the 100 chat answers as complied. On all 250 XSTest safe prompts, a local
Qwen2.5-7B-Instruct judge with the published rubric found 0 full refusals and 61 partial ones (24.4%); 55 of those are
nonsense-premise prompts, where XSTest's rubric counts rejecting the premise as compliance. Data:
[quality.csv](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/benchmarks/v1.1/quality.csv).

## Files

| Path | Contents |
| --- | --- |
| `model-000NN-of-00035.safetensors`, `model.safetensors.index.json` | Standard EXL3 weights and index |
| `config.json`, `quantization_config.json` | Model and per-tensor EXL3 settings |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `encoding/` | Tokenizer, chat template and DeepSeek's message encoder |
| `tp3/rank0.keys` … `rank2.keys` | Tensor lists for each of the three TP3 ranks |
| `runtime/` | Kernel kit and RoPE tables that `prepare.sh` stages; Engram rows come from DeepSeek's repository |
| `LICENSE` | DeepSeek's MIT license |

## How it was built

The pack starts from easthambuilds' 3.75 bpw EXL3 base and replaces selected tensors with LibertAIDAI, bertholomus and
LakyFx quantizations. Attention, shared experts and selected gate/up components use higher precision; the
least-used quarter of experts in layers 10–37 uses 3-bit components.

mrfakename's rank-2 refusal-removal edit changes attention `wo_b` in layers 0–31 and shared-expert `w2` in layers
10–39. It was applied to the official FP8 weights in FP32, then the affected tensors were re-encoded at 6 bits with
EXL3 mul1 K6 and an identity Hessian. The edit is baked in; there is no runtime switch.

| Source | Revision | Contribution |
| --- | --- | --- |
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `1bae0e1` | Original model |
| [easthambuilds](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw) | `3c39fdf9` | Base EXL3 3.75 bpw pack |
| [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) | `04a492fc` | 6-bit attention and shared experts, 4-bit late-layer gate/up, selected 3-bit experts |
| [bertholomus](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) | `f3657818` | 4-bit gate/up for the most-used half of experts in eight middle layers |
| [LakyFx](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold) | `8621801c` | Selected 3-bit experts and TP3 export lineage |
| [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) | `517efab9` | Refusal-removal edit |

Tensor ranges and the bitrate method:
[weight provenance](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/release/weight-provenance.md).

## Intended use and risks

This model has had its refusal behavior largely removed; the refusal tables above show what remains. It will answer requests the official model declines, including
harmful ones, and it can be wrong, biased or unsafe. Use it for research into quantization fidelity, serving and
refusal behavior, and for applications where you add your own safeguards. You are responsible for how you deploy it
and for complying with the laws that apply to you. Keep the API behind access controls, keep credentials separate and
have people review consequential outputs; see the
[security guidance](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/SECURITY.md).

Tested scope is text on exactly three DGX Sparks. The base model's image input has not been tested with these weights.

## License and credits

MIT, following [DeepSeek's base-model license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE).
Every source above publishes under MIT. Provided as is, without warranty of any kind.

Thanks to [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017);
[easthambuilds](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw),
[LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw),
[bertholomus](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) and
[LakyFx](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold) for the EXL3 tensors;
**[mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated)** for the edit;
[turboderp](https://github.com/turboderp-org/exllamav3) for EXL3; [Ash Hart](https://github.com/ashhart/TensorFold) and
[BertholomusAI](https://github.com/bertholomus/TensorFold) for TensorFold; [Mia's AI Lab](https://github.com/MiaAI-Lab/sparkDash)
for sparkDash; and [brandonmusic](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits) for the
official-model logits. Full list:
[CREDITS.md](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated/blob/main/CREDITS.md).

Built by [ChimeraWerks](https://github.com/ChimeraWerks): weight composition, TP3 engine work, serving and every measurement of this model on this card.
