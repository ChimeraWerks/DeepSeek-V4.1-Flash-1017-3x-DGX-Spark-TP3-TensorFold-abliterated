# Model card: selected TP3 composition

## Artifact and release scope

The v1.2 TP3 recipe runs **DeepSeek V4.1 Flash 1017, EXL3 3.84 bpw, mrfakename-edited composition**, a community derivative of DeepSeek's base model.

**Weights:** [ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated). This repository provides code, configuration and measurements.

| Property | Identity |
|---|---|
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| Base revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Quantization | EXL3, 3.8361 bpw for quantized linear layers excluding output head; title rounds to 3.84 |
| Output head | 6-bit, excluded from headline bitrate |
| Selected edit | [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated), revision `517efab9` |
| Recipe | v1.2, engine `678ea1b`, native profile `v1.2`, serving API 0.1.5 |
| Runtime | Exactly three DGX Spark GB10 systems, TP3; 8M KV / C16 / graph 128 / R48 / top_k 20 / spill 256 GiB |
| Model label suffix | `chimera-dsv41-flash-1017-abliterated-exl3` |
| Weights | [ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated), commit `2fa62d7` |
| Artifact identities | Three expected rank SHA-256 values in [release/manifest.json](../release/manifest.json) |

## Lineage and preparation

The pack starts with easthambuilds' 3.75 bpw EXL3 base and replaces selected tensors with LibertAIDAI, bertholomus and LakyFx quantizations. Attention, shared experts and selected gate/up components use higher precision. Selected low-use expert components use 3 bits. See [weight provenance](../release/weight-provenance.md) for revisions and tensor ranges.

The rank-2 refusal-removal edit changes attention `wo_b` in layers 0–31 and shared-expert `w2` in layers 10–39. Apply it to official weights in FP32, then re-encode the affected tensors at 6 bits with EXL3 mul1 K6 and an identity Hessian. The edit is baked into the weights. The upstream mrfakename card describes its FP8 artifact; this recipe re-encodes the edit for EXL3.

`prepare.sh` downloads the published folder, builds each Spark's rank file from its TP3 key list and verifies the rank hashes. Rebuilding the composition yourself is optional. See [local preparation boundary](reproduction.md#prepare-local-weight-inputs).

## Intended scope and changed behavior

Text serving is tested on this three-node configuration. Use it to study quantization fidelity, serving behavior and refusal changes. See [security guidance](../SECURITY.md) for access controls, credential separation, logging and human review of consequential outputs.

## Evaluation

Against official DeepSeek-V4.1-Flash-1017 logits, the composition measures mean KL 0.0784 and top-1 agreement 92.0% over
65 cuts of 2,048 tokens. That is 0.0088 lower than an unedited 3.5 bpw EXL3 quant (95% CI 0.0035 to 0.0131) and
0.0153 lower than the same mrfakename edit on that quant. Every arm is in [fidelity](../benchmarks/README.md#fidelity-and-refusal-behavior).

Refusals, greedy:

- HarmBench harmful, chat: 98 answered, 2 partial, 0 refused of 100 at 768 output tokens.
- HarmBench harmful, thinking: 18 answered, 31 partial, 1 refused of 50 at 3,072 output tokens; 35 replies hit the cap.
- XSTest safe, chat: 49 answered, 1 partial, 0 refused of 50.

The same weights score 97.5% on AIME 2025 (117 of 120, 4 repeats, thinking on), 90.5% on GPQA-Diamond (1,434 of 1,584, 8 repeats, thinking on), 88.7% on IFEval (prompt-level strict), 88.0% on LiveCodeBench v6 and 88.0% on a 500-item MMLU-Pro subset ([quality.csv](../benchmarks/v1.1/quality.csv)). The configured context is 1,048,576 tokens; the
largest measured prompt is 131,090 tokens. Tested scope is text on exactly three nodes.

## Licenses

Repository code is Apache-2.0; third-party code and weights keep their own licenses. The [base-model license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE) is MIT.

Thanks to the upstream contributors listed in [Credits](../CREDITS.md), [NOTICE](../NOTICE) and [third-party notices](../THIRD_PARTY_NOTICES.md).
