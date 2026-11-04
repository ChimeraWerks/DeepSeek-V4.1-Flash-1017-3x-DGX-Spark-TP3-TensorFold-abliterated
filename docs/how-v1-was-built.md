# How v1 was built — historical, October 2026

This page covers the v1 build and October 2026 measurements. Active v1.2 uses `678ea1b` and an 8M pool; see the [current manifest](../release/manifest.json). Weights: [Hugging Face](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated).

v1 combines existing EXL3 quantizations of DeepSeek V4.1 Flash 1017 to improve agreement with the official model within the memory budget of three DGX Spark systems. We built it through weight composition and runtime tuning rather than base-model training.

## Select the component mix

The base is easthambuilds' 3.75 bpw pack. LibertAIDAI's quant supplies 6-bit attention and shared experts, plus 4-bit late-layer gate/up components. Bertholomus's 4-bit gate/up components cover the most-used half of experts in layers 19, 20, 22, 24, 25, 26, 28 and 29.

The least-used quarter of experts in layers 10-37 uses selected 3-bit components from LakyFx and LibertAIDAI. This mix fits higher precision where needed within the cluster's memory budget. See [weight provenance](../release/weight-provenance.md) for source revisions and bitrate accounting.

## Choose and apply the refusal-removal edit

We compared edits on a common LakyFx 3.5 bpw base and chose mrfakename's edit. It makes rank-2 changes to attention `wo_b` in layers 0-31 and shared-expert `w2` in layers 10-39. We applied it to the official weights and re-encoded at 6 bits.

On the common base, mean KL was 0.0937 with this edit and 0.0872 without it. The final mix measured KL 0.0784 and top-1 agreement 92.0%. Both metrics compare each pack with the original official model, combining quantization and editing effects.

The fidelity evaluation used 65 balanced cuts of 2,048 tokens, totaling 133,120 positions. The harmful chat subset returned 98 answers, 2 partial answers and no refusals. The thinking subset returned 18 answers, 31 partial answers and 1 refusal. Most thinking partials exhausted the 3,072-token budget before a final answer. See [benchmarks](../benchmarks/README.md) for all subsets and limits.

## Keep the measured runtime choices

Replacement trials showed no benefit on this TP3 mix, so we kept the native 4-bit DSpark speculative model.

Owned-row prefill and direct all-gathers improved prefill throughput. top_k 20 kept sampled requests on GPU-prefiltered candidates instead of whole-vocabulary CPU sampling. Step-ahead stayed disabled: it delayed new work without improving follow-up cache reuse. The [tuning guide](tuning.md) covers the comparisons and test conditions.

## Measure the production configuration

The 2026-10-30 suite used engine `cebcf1e`, efficiency cores and stock GPU clocks. Greedy prose decode reached 84.5 tok/s for one stream and 260.7 tok/s aggregate for 16 streams. Prefill at 32K reached 3,089 tok/s. Each decode point was a single run.

Engine `7d5b4d8` and serving API 0.1.5 followed on 2026-10-31 with numeric reasoning effort and live effort remapping. Checks matched 8/8 long responses, cold and cached, and 12/12 short responses. The speed suite was not rerun for that update.

The [changelog](../CHANGELOG.md) tracks runtime changes. Thanks to the model authors, engine contributors and earlier work listed in [Credits](../CREDITS.md).
