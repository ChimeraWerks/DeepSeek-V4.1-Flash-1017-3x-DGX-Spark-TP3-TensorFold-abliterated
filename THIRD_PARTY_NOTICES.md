# Third-party notices

This release is an integration snapshot, not a transfer of ownership or a blanket relicensing of its dependencies. Preserve the original license texts, copyright notices, upstream NOTICE files and notices identifying modified files. [CREDITS.md](CREDITS.md) maps contributors and revisions; this index does not replace the component-level notices.

## Code and runtime components

| Component | Terms and retained notices | Scope |
|---|---|---|
| Original ChimeraWerks integration and packaging | Apache-2.0, [LICENSE](LICENSE), [NOTICE](NOTICE) | Repository-original work only |
| TensorFold and BertholomusAI `dsv41` family | Apache-2.0 with earlier MIT portions; [engine NOTICE](engine/tensorfold/NOTICE), [component notices](engine/tensorfold/THIRD_PARTY_NOTICES.md), [license directory](engine/tensorfold/LICENSES/) | Vendored native engine and credited adaptations |
| LakyFx TP3 and serving work | MIT portions retain [serving/LICENSES/LakyFx-MIT.txt](serving/LICENSES/LakyFx-MIT.txt); import revisions/files in [serving/UPSTREAM.json](serving/UPSTREAM.json); engine lineage in [CREDITS.md](CREDITS.md) | Imported and adapted code, including TP3 and prefill lineage |
| Mia's AI Lab and individual patch contributors | Apache-2.0 portions and upstream modification notices retained in the engine; contributor mapping in [CREDITS.md](CREDITS.md) | Ported frontend, memory, cache, spill and failure-handling ideas/code as individually attributed |
| DeepSeek `deepseek-recipe` | MIT, Copyright (c) 2026 DeepSeek; [license](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/LICENSE); revision in [CREDITS.md](CREDITS.md) | Reference for the request and reply behavior matched in the engine server |
| ExLlamaV3 / turboderp | MIT; component attribution and applicable license text in the [engine notices](engine/tensorfold/THIRD_PARTY_NOTICES.md) | EXL3 format, codec lineage and quantization tools; not sole Chimera authorship |
| PyTorch, CUTLASS, Triton, Pillow and other engine dependencies | Individual BSD/MIT/other terms as recorded in the [engine notices](engine/tensorfold/THIRD_PARTY_NOTICES.md) and [license directory](engine/tensorfold/LICENSES/) | Some are separately installed or build-time dependencies, not all bundled |
| NVIDIA CUDA/NCCL, drivers and platform software | Their respective provider terms | Separately obtained runtime/build prerequisites; the top-level Apache license does not license NVIDIA software |
| Zig toolchain | Its upstream terms; exact build input belongs to the [engine build contract](engine/README.md) | Build-time prerequisite |

The vendored engine contains other model families and corresponding notices. Those legitimate upstream credits must remain even when this recipe serves only DeepSeek.

## Model weights

The weights are published separately at [ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated), not in this repository. Source identities and selected tensors are recorded in [release/weight-provenance.md](release/weight-provenance.md).

The [pinned DeepSeek base license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE) is MIT, and the weights ship under it. The selected [mrfakename edit](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) carries MIT metadata. The [Keys/drowzeys overlay](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated) appears only in the historical fidelity comparison and is not part of the published weights.

Keep DeepSeek's license notice with any copy of the weights. See [docs/model-card.md](docs/model-card.md) for behavior, lineage and use limitations.

## Trademarks and endorsement

DeepSeek, DGX Spark, GB10 and other names identify upstream models or products. Attribution does not imply affiliation, endorsement, or permission beyond the applicable terms.
