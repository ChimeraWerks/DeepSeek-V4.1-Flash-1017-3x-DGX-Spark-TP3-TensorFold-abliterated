# dsgx10 / pico-dgx-ds4 port spike

Question: should we adopt the dsgx10 engine (github.com/Picom-code/pico-dgx-ds4, fork of
blaesus/dsgx10, pinned f2117916) for 3 DGX Sparks in a direct-cabled triangle, or port pieces into
TensorFold? This file holds only facts with sources; nothing here is an estimate unless marked
"derived". Their file paths are relative to `ds41f-4xgx10-switchless/` at commit f2117916 unless
they start with `root/`. Read on 2026-11-13: a full source audit of prefill, decode, memory, the
4-rank assumptions and their published numbers, with spot checks against the files cited.

## Prefill facts

- Their prefill runs CED: encoder layers 0-19 over every prompt token, decoder layers 20-39 replayed
  only over the last 128 tokens (DeepSeek tech report section 2.2 and 3.2.2; the model was
  post-trained with the replay simulated). Source: their `docs/ced.md`, `docs/engine.md:365-392`.
- Our TensorFold engine already does the same bounded replay: `Model.fillChunk` /
  `prompt.replayCut` (`engine/tensorfold/zig/src/families/dsv41/model.zig:1686`,
  `prompt.zig:1862`; an encoder-only chunk stops at the decoder's first layer). So CED is not the
  source of the gap.
- Their measured warm 8192-row CED chunk: 42.9 ms per encoder layer per rank on 4 ranks (attention
  14.2, FFN 26.8), rank 0 nsys, upstream `3c36d66`, switchless ring (`docs/engine.md:1554-1569`).
- Ours: about 95 ms of GPU work per layer for an 8192-row chunk on 3 ranks (CHANGELOG.md,
  2026-11-10 trace entry; layer 14 161 ms, layer 1 136 ms). We have no per-kernel split yet.
- Their routed experts run native FP8 activations x FP4 (e2m1) weights with ue8m0 block scales via
  `mma.sync ... kind::mxf8f6f4.block_scale ... m16n8k32` (`engine/kernels/common.cuh:12-48`); no
  dequantized weight buffer. Standalone routed+shared MoE at M=8192: 18.99 ms, 53.4 TFLOPS
  (`engine/kernels/README.md:285-301`). Dense FP8 GEMMs 154-160 TFLOPS at M=8192.
- Their prefill is sequence parallel: residual rows are owned per rank, inputs all-gathered, FFN
  output reduce-scattered with the expert combine fused into the send buffers, attention output
  sent all-to-all; overlap took a 4096-row profile from 1.94 s to 1.12 s (`docs/engine.md:319-356`).
- Their own words: the long-chunk MoE is limited by DRAM traffic (~2.5 GB per layer at ~190 GB/s),
  not by the wire (`docs/engine.md:1562-1569`).
- Hard-coded 4 ranks in prefill: `engine/model/config.h` `kRanks=4` with 16 heads and 2 output
  groups per rank as literals, `prefill.h` `q_rows`, `sum4`, four combine quarters, `comm::kRanks=4`
  ring with opposite-rank relay.

## Memory facts

- Their rank file is 72.38 GiB with 1,609 tensors at TP4; routed experts 67.236 GiB of it, native
  MXFP4, every rank holds all 384 experts at 576 of 2304 intermediate columns (tensor split, not
  expert split). Embedding 1.233 GiB replicated. Checked against `docs/layout.md:18-40`.
- KV is compressed latent, replicated on every rank: 890 bytes per context position per rank.
  16 slots (4 at 1M, 12 at 128K) allocate 5.03 GiB per rank up front.
- Engram tables (188.8 GiB) stay on NVMe, a quarter per rank, with a 4 GiB host row cache.
- Prefill buffers and gathered weights: 4.89 GiB per rank at 8192 rows.
- Derived for TP3 (our arithmetic on their table): the split tensors (71.08 GiB) times 4/3 plus the
  replicated ones (1.31 GiB) is about 96.1 GiB of weights per rank. With slots, prefill buffers,
  Engram cache, gathered router/mHC and scale codes that is about 111 GiB, against the 125.3 GiB
  MemTotal our memory-saver setup gives (README prerequisites). Tight, not impossible.

## Decode facts

- One-row plain decode step at TP4: 18.56 ms (`docs/engine.md:1680-1714`); latest stack5 drafted
  one-row step 18.65 ms (`docs/stack5.md:74-80`).
- Weight bytes a rank reads for one row: 3,358.6 MB in 14.32 ms at 234.6 GB/s, about 77% of the
  step (`docs/engine.md:3727-3759`). Their practical streaming ceiling is about 235 GB/s against a
  273 GB/s peak.
- Drafting: the checkpoint's own 3 DSpark blocks, 5 drafts per pass, exact accept (draft kept only
  if it equals the target's pick); chat accepts about 1.24 drafts a step, code about 2.9.
- Derived for TP3 (our arithmetic): each rank would read about 4/3 of 3,358.6 MB = 4,478 MB, about
  19.1 ms at 234.6 GB/s, so a one-row step near 23 ms against 18.6 ms at TP4. That is roughly 20%
  slower decode than their 4-box numbers; their 85 chat would land near 69, under our 82.

## Claims audit

- No measurement in either repo is strictly comparable: different prompts, output length (256 or 512
  against our 400), cache state, timing definition and aggregation. Every side-by-side below is
  indicative only.
- Their recommended config is not exact-output: `DSGB10_OEA=4,12,9` deliberately changes expert
  choice in batches, and `DSGB10_WIDE=64` differs by about 1 bf16 ulp from the 16-row path (35 of
  768 teacher-forced argmaxes changed). The learned gate can regroup steps (374/374 identical at
  the shipped alpha).
- Use their sparkDash numbers, not their own bench, against ours: every speed number in our README
  is sparkDash (README "Credits"). Their own bench (85 chat / 144 code) runs much lower than their
  sparkDash run (102.0 prose / 164.6 code, sparkDash 1.8.8 at 754f40a, 256 tokens; root README).
  Remaining mismatch: they decode 256 tokens per request, we decode 400, and our sparkDash revision
  is not pinned in summary.json.

| sparkDash, 1 request | Theirs, 4 boxes | Ours, 3 boxes | Gap |
| --- | ---: | ---: | ---: |
| Prose | 102.0 | 82 | they are +24% |
| Code | 164.6 | 124 | they are +33% |

- Single-stream speed does not divide by box count, so per-box division is not used for decode.
  Derived from their decode step (18.6 ms at TP4, about 77% of it weight reads): at TP3 a step near
  23 ms, so about 0.81 of their speed: roughly 82 prose and 133 code on sparkDash on our 3 boxes,
  against our 82 and 124. A port would about tie us on prose and gain about 7% on code.
- 16 users: 376 (theirs, distinct code, own bench) against 253 code / 270 prose (ours, sparkDash);
  not comparable, no sparkDash 16-user figure of theirs exists.
- Prefill 32K: 10,200 (their gate) against 3,816 (ours, sparkDash): about 2,550 against 1,272 a
  box. This is the clear per-box gap.

## TP3 port facts

- No path in dsgx10 runs at TP3 by a flag, a fleet file or `kRanks` alone (`ring_config.h:12`,
  `config.h:43-59`, `weights/plan.cpp:213-287`).
- 64 heads, 8 output groups, vocab 129,280, q-rank 1,280, KV 512, output projection 8,192 and the
  Engram segment 5,120 do not divide by 3. Kernels hard-code 16 heads per rank; the converter only
  makes equal tile-aligned cuts. A whole-group 24/24/16 head split is a design idea, not a feature.
- The relay ring's opposite rank `(r+2)%4` aliases the previous peer at N=3; the direct mode keeps
  the 4-node layout. A triangle needs a new 3-peer transport with uneven counts and offsets.
- The model's 4 mHC residual streams are an architecture fact and stay 4 on 3 boxes, but the code
  maps one stream per rank, which also has to change.
- Files to rewrite or revalidate: the transport (about 3,550 lines with tests), weights and Engram
  pipeline (about 3,950 lines), plus the model, attention kernels and drafter geometry.
- TensorFold already runs this model at TP3 in production, so a port would redo partition work our
  engine has already done.

## Verdict

**Do not port the dsgx10 engine to 3 boxes.** Port pieces into TensorFold instead.

| Reason | Fact |
| --- | --- |
| Decode would not improve | On sparkDash a 3-box port projects to about 82 prose / 133 code against our 82 / 124 (derived above). |
| The port is a rewrite | Partitioning, packing, transport, attention kernels and drafter geometry all assume 4 ranks. |
| Memory is tight | About 111 of 125.3 GiB per box at native precision (derived above). |
| Outputs | Their recommended config changes outputs (OEA, wide step); dropping those gives back part of their 16-user lead. |
| Abliteration | Native weights would need the refusal edit re-applied to native tensors. |

Ranked pieces worth taking into TensorFold:

1. **Prefill, the one clear per-box gap (about 2x).** CED is already ours, so the gap is in kernels
   and data movement: native FP8 x FP4 block-scaled tensor-core expert GEMMs with no dequant pass,
   sequence-parallel residual rows with the expert combine fused into the reduce-scatter (their
   4096-row profile 1.94 s to 1.12 s), and wo_b's all-to-all. Which applies depends on where our
   95 ms per layer goes, so the next step is a per-kernel profile of one of our 8192-row chunks.
   If our EXL3 expert decode dominates, the lever is a weight-format decision, not a kernel port.
2. **Certified LM head** (`DSGB10_CERT`): +1.5% chat on one user, same tokens, proven on 1.34 M rows.
3. **Learned verify gate** (`DSGB10_CF_MLP`): +1.1% chat, same outputs at the shipped alpha.
4. **Lossless exponent-coded LM head**: 0.35-0.38 ms a step at 1-16 rows.

## Our prefill split

Measured 2026-11-13 on the three Sparks: production build `77e26ab` with production flags, rank 0
under `nsys launch` (CUDA trace only), one cold 32,768-token prompt after warmup and one uncached 8K
prompt. Under nsys the 32K first token took 8.93 s (about 3,670 tok/s against 3,816 on sparkDash).
The engine cut the prompt into chunks of 8,192, 8,192, 8,192, 6,144 and 1,920 rows (from the
sparse-attention launch grids), then replayed 128 rows through the decoder. The table is the mean of
the 60 encoder layers (layers 0-19 of the first three chunks) that ran 8,192 rows. Kernels are
grouped with the stage regexes of the 2026-10-29 profile (`~/AI/quant-lab/prefill-profile-20261029/
classify.py`). Script, trace and log: `~/AI/quant-lab/prof-prefill-20261113/` (`classify2.py`,
`split.txt`).

- Every kernel runs on one CUDA stream, so nothing overlaps. GPU busy is 97.7% of the 8.87 s window
  (kernel sum 8.66 s). Host gaps are not the problem.
- Per 8,192-row layer: 104.8 ms mean, 98.7 ms median. The three chunks agree (104.7, 104.3,
  105.3 ms).
- Layers 1 and 14 (the Engram layers) cost 140 and 164 ms: each adds about 25 ms of all-gather and
  14 ms of dense GEMM. Engram's own kernels are 0.2 ms.

| Group (per 8,192-row layer, rank 0) | ms | Share | Main kernels (ms) |
| --- | ---: | ---: | --- |
| EXL3 experts | 45.4 | 43.4% | routed gate-up 15.5, routed down 8.5, routed input rotation 7.5, down combine 6.4, shared expert 5.2, gate-up epilogue 2.0 |
| Attention projections (dense EXL3) | 18.2 | 17.4% | `_gemm` 14.5, input rotation + unpack + glinear 3.7 |
| Sparse attention and indexer | 14.0 | 13.3% | sparse attention 11.8, indexer score + top-k 1.3, rope and KV norm 0.8 |
| Collectives (NCCL send/recv all-gather, `TF_DS_DIRECT_GATHER`) | 13.2 | 12.6% | all exposed; about 10 ms on a plain layer, 35 ms on layers 1 and 14 |
| Engram | 0.2 | 0.2% | gate and pool; its extra all-gather and GEMM are counted above |
| Other | 13.8 | 13.1% | hyper-connections 7.7, fp32/bf16 casts 4.1, fp32 SIMT sgemm 1.5, norms 0.5 |
| Total | 104.8 | 100% | |

Two numbers for the transfer-overlap port:

- (a) Exposed NCCL all-gather: 12.6% of GPU time, 13.2 ms per 8,192-row layer, with 0% overlap
  (one stream).
- (b) Routed experts plus dense EXL3: 58.4 ms, 55.8% (routed 40.2 ms + dense 18.2 ms). With the
  shared expert, 63.6 ms, 60.7%.

Beside the two earlier splits:

| Split | Ours 2026-11-13 (`77e26ab`, 8,192 rows) | Ours 2026-10-29 (`ee903f5`, 2,048 rows) | dsgx10 (4 ranks, 8,192 rows) |
| --- | ---: | ---: | ---: |
| Routed + shared experts (old regex) | 43.4% | 33% | FFN 26.8 ms: routed up 8.45, down + combine reduce-scatter 13.2, shared 1.26 |
| Dense EXL3 linears / attention projections | 17.4% | 14% | wq_b 1.15, wo_a 1.80, wo_b 2.59 ms |
| Attention + indexer | 13.3% | 12% | sparse attention 4.79 ms (attention total 14.2) |
| NCCL all-gather, exposed | 12.6% (bf16 wire) | 24% (fp32) | 4.11 ms exposed collectives |
| Hyper-connections | 7.3% | 13% | not itemised |
| Per layer per rank | 104.8 ms | not per layer | 42.9 ms |

Derived (our arithmetic): at TP3 their per-rank work would be 4/3 larger, 42.9 x 4/3 = 57.2 ms per
layer, against our 104.8 ms. That is 1.83x, which matches the measured 2.0x per-box prefill gap.
Scaling each of their groups by 4/3 splits our extra 47.6 ms like this:

| Where our extra time goes (derived, per layer) | Ours ms | Theirs x 4/3 ms | Extra ms |
| --- | ---: | ---: | ---: |
| Routed gate-up incl. input rotation and epilogue | 25.1 | 11.3 | 13.8 |
| Routed down + combine (theirs includes the reduce-scatter) | 14.9 | 17.6 | -2.7 |
| Shared expert | 5.2 | 1.7 | 3.5 |
| Attention projections + sparse attention + indexer | 32.2 | 18.9 | 13.3 |
| Exposed collectives | 13.2 | 5.5 | 7.7 |
| Hyper-connections, casts, sgemm, norms | 13.8 | 2.5 (their unitemised remainder) | 11.3 |

The columns do not sum exactly to 104.8 and 57.2: their FFN has 3.9 ms they do not itemise, and their
4.11 ms of exposed collectives may sit inside their attention and FFN rows. Inside the attention row,
dense projections are about +5.7 ms (ours 18.2 against their 12.5 attention outside sparse
attention) and sparse attention plus indexer about +7.6 ms (14.0 against 6.4).

**Verdict: the gap is mostly kernels and data movement, with the EXL3 format as the largest single
part.** The EXL3-bound kernels (expert and dense GEMMs with in-kernel decode, plus their input
rotations) account for about 20 of the 47.6 extra ms (experts +14.6, dense projections +5.7). The
other ~27 ms is format-independent: exposed all-gather +7.7, sparse attention and indexer +7.6, and
hyper-connections, casts and fp32 sgemm +11.3.

## Next step

Port the transfer overlap first: 13.2 ms per layer of all-gather runs on the compute stream with
nothing beside it. Then the routed gate-up path (15.5 ms GEMM + 7.5 ms rotation against their
11.3 ms) decides whether a native FP4 expert format is worth the abliteration rework.
