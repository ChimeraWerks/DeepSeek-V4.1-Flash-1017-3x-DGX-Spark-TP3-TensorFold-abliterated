# Credits

Thanks to the model, engine and community contributors behind this deployment.
Reviewed 24 October 2026. [UPSTREAM.json](UPSTREAM.json) lists the serving
component's imported files; research pins are tracked separately from shipped artifacts.

The served v1 pack is our `V4-MRF-1017`, published as
`ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated` (3.84 bpw).
LakyFx's pack contributed source components.

Repository code is Apache-2.0; third-party code keeps its own licenses.

## Model, tools and engine

| Contributor | Contribution | Details |
|---|---|---|
| DeepSeek | Trained DeepSeek-V4.1-Flash-1017; model architecture, reference components, DSpark and Engram. [Official revision/license][DS] | MIT at the pinned source revision. |
| LakyFx | Independent 1017 EXL3 export, custom TP3 layout/recipe, bounded adaptive-controller integration, K2/K2a cache changes and testing/measurement work. [Source ledger][LS], [TP3][TP], [1.2.5][K2] | MIT integration, docs and benchmark tooling. [License][LL] |
| Ashley “Ash” Hart and TensorFold contributors | Native engine, serving/lane infrastructure, CUDA integration, EXL3 module and speculative machinery. [Engine ancestry][BA] | TensorFold 1.0.2, Apache-2.0. |
| BertholomusAI / Albert Lee and contributors | Community `dsv41` implementation on TensorFold: prompt/decode, DSpark, Engram, RDMA and vision. [Attribution][BA] | Apache-2.0 engine with component-specific notices. |
| turboderp / ExLlamaV3 | EXL3 format and quantization tooling. [Laky acknowledgements][HF], [engine usage account][BA] | MIT tooling; TensorFold runs serving. |
| bot-lab-21 and WestWaters / Pollard Weights | Earlier expert export, precision allocation, calibration work and methodology. [Current export's ancestry][HF] | The publisher credits earlier methods and layouts; the current 1017 export uses new model bytes. |
| Mia's AI Lab | Earlier EXL3 export/layout experience and the DeepSeek recipe patches and tools below. [Detailed credits][MC] | Apache-2.0 recipe work; model artifacts have separate terms. Mia's 2.9 bpw pack and LakyFx's source pack are separate from our served v1 composition. |

## Patch and method lineage

| Contributor | Contribution | Details |
|---|---|---|
| Coinupbtc | Prefill/decode interleave in Mia PR #5. [Credits][MC] | Original patch author. |
| Mia's AI Lab and TensorFold contributors | Mia's Zig required/named tool-choice gate ports TensorFold's Python call gate and wires it into DeepSeek. Mia adds image prompt caching atop the interleave. [Credits][MC] | Tool-choice gating, without full JSON-schema grammar enforcement. Applied patches are tracked in our release manifest. |
| William J Riley (`willjriley`) | Original RDMA idle-backoff implementation, merged in TensorFold PR #27. [Original PR][BN] | Publisher's CPU/thermal measurements use TP2. |
| Mykhailo Tamarin | Idle-backoff port to the Zig ring in Mia PR #26. [Port PR][MN] | Closed/unmerged PR; Mia's later integration is credited separately where used. |
| sayyidfareed, Manrique (`hashd1ve`), Nathan McBride, Jay Leaton and other named branch contributors | GID/MTU handling, retained-prefix fixes, direct-output template work, append-buffer fixes and UTF-8 handling. [Complete inventory][MC] | Our applied-patch manifest tracks included changes. |
| Liquan Liu, Yifan Zhang and Bowei Xu / AdaSpark, and imparo | Inspiration for Laky's online cost/acceptance feedback and pricing. [Adaptation boundary][AD] | Bounded controller extending existing speculation. |
| DeepSeek / DeepInfer and NVIDIA / FLARE | Ideas behind Laky's scheduling, cost timing, overlap and exploration experiments. [Source ledger][LS] | Research inspiration. |
| Christopher Owen | Earlier Spark networking/tuning and adapted benchmark methodology credited by Laky. [Source ledger][LS] | Historical networking and benchmark work. |
| tonyd2wild, Kai and earlier serving contributors | Spark serving/build, loading and Engram groundwork. [Acknowledgements][HF] | Earlier vLLM stack. |

## Platform, measurements and our work

NVIDIA supplies DGX Spark/GB10, CUDA and NCCL. Our deployment tunes NCCL buffers.
sparkDash by Mia's AI Lab and its contributors supplies the historical benchmark
tooling, including version 2.70.1. Comparable benches retain their recorded
historical revisions and labels. Third-party dashboard code is not included.

Thanks also to the PyTorch/CUTLASS vision kernels, Triton, Pillow, Zig and
container components listed in the [engine attribution][BA]. The release manifest
lists distributed components; LICENSE, NOTICE and third-party notices cover them.

Our work covers deployment packaging, adaptations and operational fixes,
tracked in our commits and diffs. Memory guards, clock checks, metrics and
launchers build on upstream work. The `1.2.5-mia` label identifies an integration;
the release inventory records authors, changes and binaries.

Mia's GLM page/cards and diffbot's card informed presentation and benchmark
reporting. Their model weights, DFlash2 and vLLM configurations are separate
projects. See the repository [credits](../CREDITS.md).

Each release updates the ordered patch/source/artifact inventory, separates used
versions from reviewed versions and carries earlier credits forward.

[DS]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE
[HF]: https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold/blob/8621801c1fc74aece1fa83e67c6ade3fde1e7894/README.md
[LS]: https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/b4e476d60f01fda564d760c5906ac1358390c728/docs/SOURCES.md
[TP]: https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/b4e476d60f01fda564d760c5906ac1358390c728/docs/TENSORFOLD-TP3.md
[K2]: https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/b4e476d60f01fda564d760c5906ac1358390c728/docs/DS41F-1.2.5.md
[LL]: https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/b4e476d60f01fda564d760c5906ac1358390c728/LICENSE
[AD]: https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/b4e476d60f01fda564d760c5906ac1358390c728/docs/ADAPTIVE-SPECULATION.md
[BA]: https://github.com/bertholomus/TensorFold/blob/6a1af414e561a5a01ca53e2c4a05fed177dbb7ca/tools/dsv41-zig/ATTRIBUTION.md
[MC]: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/370daf0cc38f348516a98ebab38f6053858eab1f/CREDITS.md
[BN]: https://github.com/bertholomus/TensorFold/pull/27
[MN]: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/26

## Serving API ports

LakyFx ds41f-gateway 1.2.6 contributed the durable vision ledger, native
cancellation records, final-only collector, private diagnostics, media admission,
fair queue, request body bounds and stream validators. These ports use revision
21f9a280803b7e34f5b7abad3c479d9c6077ec0d (MIT), with its license in
LICENSES/LakyFx-MIT.txt. UPSTREAM.json records imported files and hashes alongside
our API/configuration/packaging adaptations.

## Native frontend staging history

Coinupbtc's Mia patches 0027/0028 (`ac81e85`, `550a94a`) and Maciej Kowalik's
0029 (`aa3e4d1`) were staged separately in the native fork. They preserve tool-call
prose, mask request-controlled token spellings and accept omitted Anthropic
thinking budgets. They are CPU-tested and ARM64-built, separate from this API wheel.

Robert Wojciechowski's ABLIT writable-index fix stays with the integrator.
His Spill/KV-sharding series is carried in the optional context-capacity source
candidate, with CPU tests and builds completed; live testing is pending.
MiaAI-Lab's maintainers authored the reviewed container-log rotation change.
The repository's launch CREDITS and native import ledger track included changes.

## Operational safety and optional-capacity lineage

Robert Wojciechowski's Mia 0023 supplies the prompt-preparation memory changes.
Our TP3 watchdog/group supervisor adapts his 0020 failure-exit idea separately
from spill/rank-wire. His 534c96b writable-index fix informs our overlay helper.
MiaAI-Lab maintainers' 7ed7988 informs bounded per-engine logs. LakyFx's native
disconnect and numeric-aware repetition behavior is retained and tested in the
API integration. Coinupbtc's 0027/0028 and Maciej Kowalik's 0029 are included.
These native changes and the package were staged offline before the current
production release.

The v1 production manifest enables 256 GiB spill, graph cap 128 and 16 streams.
Optional KV sharding and disk chat continuation are off by default.

Thanks to **Robert Wojciechowski** (GitHub `wojo`, X `rwojo`) for porting GLM
spill, continuation, KV owner sharding and ring exchange into DeepSeek 1017,
and **BadAd84** for designing the spill v2 block tier. **Mia / MiaAI-Lab**
published and validated the TP2 recipe at `2fd7809`.
Our TP3 adaptations and testing status are in
`docs/deploy/ds1017-context-storage-design.md`.
