# Weight provenance

DeepSeek V4.1 Flash 1017 abliterated EXL3 3.84 bpw combines community quantizations of the same original model in a standard EXL3 safetensors folder. TP3 rank splitting runs locally.

Weights: [Hugging Face](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated). See the [model card](../docs/model-card.md) and [preparation](../docs/reproduction.md#prepare-local-weight-inputs).

## Sources and selected components

DeepSeek's base license is MIT.

| Source | Revision | Contribution |
|---|---|---|
| [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `1bae0e1088993b9b82de08f59de5be5a34e42170` | Original model; official weights receive the edit before re-encoding. |
| [easthambuilds](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw) | `3c39fdf9` | Base EXL3 3.75 bpw pack. |
| [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) | `04a492fc` | 6-bit attention and shared experts, 4-bit late-layer gate/up, and selected 3-bit expert components. |
| [bertholomus](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) | `f3657818` | 4-bit gate/up for the most-used half of experts in layers 19, 20, 22, 24, 25, 26, 28 and 29. |
| [LakyFx](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold) | `8621801c1fc74aece1fa83e67c6ade3fde1e7894` | Selected 3-bit expert components and TP3 export lineage. |
| [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) | `517efab9` | Rank-2 refusal-removal edit to attention `wo_b` in layers 0-31 and shared-expert `w2` in layers 10-39. |

The least-used quarter of experts in layers 10-37 uses selected 3-bit components from LakyFx and LibertAIDAI. Attention, shared experts and selected gate/up components get higher precision.

The refusal-removal edit runs in FP32 on official FP8 weights. The affected tensors are re-encoded at 6 bits with EXL3 mul1 K6 encoding and an identity Hessian. The edit is baked into the weights; no serving-time switch is needed.

## Bits per weight

ExLlamaV3 `get_storage_info` divides stored bits by weight count across quantized linear layers, excluding the output head.

| Scope | Result |
|---|---:|
| Quantized linear layers, output head excluded | 3.8361 bpw |
| Routed experts | 3.8017 bpw |
| Output head precision | 6 bits |
| Same counter on LakyFx's source pack | 3.5101 bpw |

The title rounds 3.8361 to 3.84: quantized-layer bitrate rather than download size per parameter.

## Historical fidelity evaluation

The composition measured mean KL 0.0784 and top-1 agreement 92.0% against the official model across 65 cuts of 2,048 tokens: 133,120 positions. KL runs from the official model to the pack. The comparison covers quantization and the refusal-removal edit.

See [benchmarks](../benchmarks/README.md) for evaluation scope and [how v1 was built](../docs/how-v1-was-built.md) for selection history. Runtime licenses: [credits](../CREDITS.md), [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
