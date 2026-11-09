# Credits

Everyone whose code, weights or data ships in this release or backs a number it publishes.

Repository code is Apache-2.0 ([LICENSE](LICENSE)); third-party code keeps its own licenses ([NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Weights: [Hugging Face](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated).

## Model and weights

| Author and source | Revision | What was used |
|---|---|---|
| [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) ([MIT license][DS]) | `1bae0e1088993b9b82de08f59de5be5a34e42170` | The model, DSpark drafter and Engram tables; official tensors receive the edit. |
| [easthambuilds](https://huggingface.co/easthambuilds/DeepSeek-V4.1-Flash-1017-EXL3-3.75bpw) | `3c39fdf9` | 3.75 bpw EXL3 base of the composition. |
| [LibertAIDAI](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) | `04a492fc` | 6-bit attention and shared experts, 4-bit late-layer gate/up, selected 3-bit experts. |
| [bertholomus](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw) | `f3657818` | 4-bit gate/up for the most-used experts. |
| [LakyFx](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold) | `8621801c1fc74aece1fa83e67c6ade3fde1e7894` | Selected 3-bit experts and the source EXL3 export. |
| [mrfakename](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) | `517efab9` | Rank-2 refusal-removal edit, applied to official weights and re-encoded at 6 bits. |
| [turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3) | | MIT EXL3 format, quantizer and bitrate accounting. |

## Engine and serving

| Contributor and project | Revision or version | Contribution |
|---|---|---|
| ChimeraWerks, [this repository](https://github.com/ChimeraWerks/DeepSeek-V4.1-Flash-1017-3x-DGX-Spark-TP3-TensorFold-abliterated) | Engine `30a9647`; API 0.1.5 | Weight composition and evaluation, owned-row prefill, direct gathers, shared-expert split, prompt aliasing, 8K prefill windows, yielded prefill layer slices, graph vote and capture fallback, TP3 ports of the TP2/TP4 patches below, lifecycle scripts and release. |
| Ash Hart and TensorFold contributors, [TensorFold attribution][BA] | TensorFold 1.0.2 license reference | Native engine, serving lanes, CUDA integration, EXL3 module and speculative generation (Apache-2.0, earlier MIT code identified by component notices). |
| BertholomusAI / Albert Lee, [TensorFold `dsv41`][BA] | Attribution `6a1af414e561a5a01ca53e2c4a05fed177dbb7ca` | DeepSeek-V4.1 prompt and decode paths, DSpark, Engram, RDMA and vision. |
| LakyFx, [TP3 recipe][TP], [MIT license][LL] | Source `b4e476d60f01fda564d760c5906ac1358390c728`; API imports `21f9a280803b7e34f5b7abad3c479d9c6077ec0d`; 1.2.8 prefill port in `14803a3`; 4.0.2 ports in `591d9dc` | TP3 layout and export, K2/K2a caches, adaptive speculation, 1.2.8 prefill work, the imported serving files in `serving/UPSTREAM.json`, and the 4.0.2 dead-checkpoint release, first-continuation hint and rank-local head reduction (opt-in, off in production). |
| Mia's AI Lab, [TP2 recipe credits][MC], [TP4 recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold) | Credits `370daf0cc38f348516a98ebab38f6053858eab1f` | Apache-2.0 TP2/TP4 patches, image prompt caching and the required/named tool-call gate. |
| DeepSeek, [`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f), [MIT license](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/LICENSE) | 0.2.0 at `935817c66f5d6e64838f419692545f06787e485f` | Reference for the v1.2 prompt rendering, Responses, tool validation and output parsing behavior, and for lowering an omitted output cap to the context left (`resolve_max_tokens`). Behavior was matched. |
| xgrammar authors, [engine attribution][BA] | v0.2.8 | Structured-output library. |
| PDFium authors; Benoît Blanchon, [pdfium-binaries](https://github.com/bblanchon/pdfium-binaries/releases/tag/chromium%2F7725) | `chromium/7725` (147.0.7725.0) | PDF rendering for PDF input: the pinned prebuilt library `prepare.sh` stages (PDFium BSD-3-Clause, packaging MIT; [runtime assets](docs/runtime-assets.md#pdfium-for-pdf-input)). |

## Patch authors

Their TP2/TP4 patches ship in this engine, ported to TP3.

| Contributor | Locator | Contribution |
|---|---|---|
| Robert Wojciechowski (wojo), [Mia credits][MC] | `534c96b`; series 0011-0023 | Spill, continuation, memory preparation, KV owner sharding, ring exchange, failure exit. |
| BadAd84, [Mia credits][MC] | `2fd7809` | Spill v2 block tier. |
| Coinupbtc, [Mia credits][MC] | PR 5; patches 0027/0028 | Prefill/decode interleave, tool-call prose preservation, plain encoding of caller token spellings. |
| Maciej Kowalik (koniaik), [Mia credits][MC] | Patch 0029 | Anthropic thinking-budget handling. |
| hecisaza, [Mia TP4 recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold) | Patch 0035 | Engram read-failure exit. |
| nuk3s, [Mia PR 100](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/100), [PR 101](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/101) | Patches 0040, 0047, 0048 | Restore-room wait, replayed reasoning, streamed tool calls. The wait and streaming are opt-in and off in production. |
| Christopher A. Williamson (cawilliamson), [Mia credits][MC] | Patch 0046 | GET and DELETE body discard. |
| lukaszraczylo, [Mia PR 104](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/104), [PR 105](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/105), [PR 107](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/107), [PR 109](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/109) | Patches 0049, 0050, 0052, 0054 | First-draft depth, draft calibration and keyed drafts (opt-in, off in production); the design our layer slices follow. |
| hecisaza, [Mia PR 94](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/94) | PR 94 review | Found the finished-job race fixed in `7d95481`. |
| hecisaza, [Mia 4x PR 49](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/49) | Patch 0052 | DSML tool calls stay whole when a parameter value holds the block's closer. |
| Keenni, [Mia 4x PR 41](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/41) | PR 41 | Template strings grow in place, reducing head memory on long histories. |
| marknx, [Mia 2x PR 112](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/112) | Patch 0056 | Each picture is prepared in its own scratch memory. |
| Mia's AI Lab, [Mia 2x `48d14d2`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/commit/48d14d23f732d84cf12712c4c0102b5372eff625); reported by Unitea ([unitea1992](https://github.com/unitea1992)), issue 99 | Patch 0058 | `/v1/responses` accepts and ignores `include: ["reasoning.encrypted_content"]`. |
| MiaAI-Lab, [Mia 2x `8f19f56`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/commit/8f19f56ac12eaa538a430cfbbc98aa93e3710e7d); reported by Phan Quý Nguyên ([NguyenPhan2810](https://github.com/NguyenPhan2810)), [issue 102](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/102) | Patch 0059 | The lane thread makes its CUDA context current as it starts, so the first spill restore after a start finds it bound. |
| Hector Isaza (hecisaza), [Mia 4x PR 50](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/50) (`a472fea`), [Mia 2x `36f2dcd`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/commit/36f2dcd07c8c270d112c35174a0135ec1458a68c) | Patches 0053 (4x), 0064 (2x) | A tool-call block too short to hold the DSML block it opens with stays text instead of a slice past its start. |
| [SvangenStudios](https://github.com/SvangenStudios), [Mia 4x PR 31](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/31) | [Patch 0042](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/blob/6a4d55e8b7a7b492747ccccf6d67389f26c7d5dd/zig/patches/0042-json-answer-channel.patch) at `6a4d55e` | A JSON answer beside offered tools stays the answer, not a tool call. |
| hecisaza, [Mia 2x PR 126](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/126) | Patch 0065 (`4217bdb`) | Keepalive ping on Responses streams. |
| Chad Hurley (chadhurley25075-png), [TensorFold](https://github.com/ashhart/TensorFold) | `ceb6f4d3` | Control bytes in access-log lines written as `\xNN`. |
| Christopher Owen, [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f) | [`1e9f851`](https://github.com/christopherowen/spark-ds41f/commit/1e9f851ad1c23daa5ef3b9cf490f9c832402cfe9), [`6a4d398`](https://github.com/christopherowen/spark-ds41f/commit/6a4d398e6d3e30df5799e6d13de18330f8756c1c) | RoCE GID slot check in `preflight.sh`, from its doctor's fixed-slot check. |
| Jürgen Schmied (jschmied) and ashhart, [TensorFold](https://github.com/ashhart/TensorFold) | `460a2868`, `46ce6dfd` | Safetensors header length bounded before allocating; the pattern for our Engram table check. |
| laudney, [TensorFold](https://github.com/ashhart/TensorFold) | `ed39acc5` (#1945) | Release of a first-token finish when its trace fails. |
| joernchen, [TensorFold PR 1820](https://github.com/ashhart/TensorFold/pull/1820) | [`d9440612`](https://github.com/ashhart/TensorFold/commit/d944061253565c09fa83a01b3026e5663b7a1c8e), [`c6c8d71b`](https://github.com/ashhart/TensorFold/commit/c6c8d71b0d54c7ad79ed25b11cd982a1c3b3d85a), [`349fbc26`](https://github.com/ashhart/TensorFold/commit/349fbc26128a0a8bf9858362a233c508fde998d1), [`0c393118`](https://github.com/ashhart/TensorFold/commit/0c393118033671a79744e79f8f9fd9cb99089e29), [`b0ef8da6`](https://github.com/ashhart/TensorFold/commit/b0ef8da6a2e6526f706a810e613d95eb2c9088f5) | PDF document input: the Messages, Chat and Responses PDF parts, PDFium rendering in a limited child worker behind `--vision-pdf`, request-wide page and pixel limits, URL refusal and PDF redaction in the request log. |
| Ash Hart (ashhart), [TensorFold PR 2387](https://github.com/ashhart/TensorFold/pull/2387), [PR 2393](https://github.com/ashhart/TensorFold/pull/2393), [PR 2599](https://github.com/ashhart/TensorFold/pull/2599) | [`4caf162a`](https://github.com/ashhart/TensorFold/commit/4caf162a437d20b2b58353c673ae9ccc0b63b3b6), [`ff1809c4`](https://github.com/ashhart/TensorFold/commit/ff1809c429c77ef9bc3300661b88886121c2c5f8), [`dbff64c4`](https://github.com/ashhart/TensorFold/commit/dbff64c41f05a6aa5431c97377142c89e7f8e34a), [`55ab8533`](https://github.com/ashhart/TensorFold/commit/55ab8533a4a1c12f8cb5d7166251b1c3da6dbab3), [`420a3447`](https://github.com/ashhart/TensorFold/commit/420a3447934772d2ae8d1e204e068b6c19757aa7), [`ca854daf`](https://github.com/ashhart/TensorFold/commit/ca854daf31eff56c0412654d9c4e7e4246001d61), [`3aa47c43`](https://github.com/ashhart/TensorFold/commit/3aa47c4347ad6b553f95df45edd7e977200f5d24) | PDFium render flags without byte-order reversal, the worker as a fresh process of the server's own binary with named crashes, and its Linux host test. Our port makes the pages PNG for the engine's own decoder. |
| William J Riley (willjriley), [TensorFold PR 27](https://github.com/bertholomus/TensorFold/pull/27) | PR 27 | RDMA idle backoff. |
| Mykhailo Tamarin, [Mia PR 26](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/26) | PR 26 | Zig-ring port of the idle backoff. |
| sayyidfareed, Manrique (hashd1ve), Nathan McBride, Jay Leaton, [Mia credits][MC] | Credits `370daf0` | GID/MTU handling, retained prefixes, direct-output templates, append buffers, UTF-8 handling. |

## Measurement

| Source | Revision | Used for |
|---|---|---|
| Mia's AI Lab, [sparkDash](https://github.com/MiaAI-Lab/sparkDash) | `693fa80` / 2.75.4 | Every speed number and the 755-item quality set. |
| brandonmusic, [official-model teacher logits](https://huggingface.co/brandonmusic/DeepSeek-V4.1-Flash-1017-Teacher-Logits) | `cuts-2048`, 65 cuts | Every KL number. |
| [EvalScope](https://github.com/modelscope/evalscope) | `9d298b8` | GPQA-Diamond. |
| HarmBench and XSTest authors | HarmBench `standard`; XSTest safe prompts | Refusal and over-refusal results. |
| drowzeys ([Keys](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated), `de412d22`) and pqhaz ([heretic](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-1017-heretic), `12d58ad0`) | | The other refusal-removal edits in the fidelity comparison. |
| `dgx-spark-memory-saver` contributors | 0.4.1 at `ec1b17a8ac713c6357b8f22983ded8503a524849` | UVM packing in the recommended host setup. |

## License notices

TensorFold's notices: `engine/tensorfold/NOTICE`, `engine/tensorfold/THIRD_PARTY_NOTICES.md` and
`engine/tensorfold/LICENSES/` (including PyTorch/CUTLASS, Pillow and Zig components).

LakyFx serving imports: `serving/UPSTREAM.json`. MIT notice: `serving/LICENSES/LakyFx-MIT.txt`.

[DS]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE
[BA]: https://github.com/bertholomus/TensorFold/blob/6a1af414e561a5a01ca53e2c4a05fed177dbb7ca/tools/dsv41-zig/ATTRIBUTION.md
[TP]: https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/b4e476d60f01fda564d760c5906ac1358390c728/docs/TENSORFOLD-TP3.md
[LL]: https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/b4e476d60f01fda564d760c5906ac1358390c728/LICENSE
[MC]: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/blob/370daf0cc38f348516a98ebab38f6053858eab1f/CREDITS.md
