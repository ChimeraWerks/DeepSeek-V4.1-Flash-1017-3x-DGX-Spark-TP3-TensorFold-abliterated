# Frontend port credits

Built on the K2a/Mia/nucleus baseline. Thanks to TensorFold, BertholomusAI,
LakyFx, Mia AI Lab and the component authors. Licenses and copyrights are in
LICENSE, NOTICE, THIRD_PARTY_NOTICES.md and LICENSES.

- Coinupbtc authored Mia patches 0027 and 0028: preserving prose before DeepSeek
  tool calls and encoding request-supplied control spellings as plain text.
  Sources: Mia commits ac81e85cb953edbb1b4a8bf791b40d33d82deb51 and
  550a94a792e22eb6df685e572ba7cece79544eb6, PRs #44 and #43.
- Maciej Kowalik (koniaik) authored the thinking-without-budget change carried
  as Mia patch 0029, commit aa3e4d1966c1ce2b136b13fed030c3ef73f2a7b6, PR #53.
  Mia still validates explicitly invalid budgets.
- Patch 0028 also covers rendered JSON member names and decoded tool arguments
  on the pre-0023 tokenizer. TF_DS_LEGACY_TOOL_SEPARATOR_ROUNDTRIP controls old
  input newline repair. New replies already have correct prose boundaries;
  this build ignores TF_DS_TOOL_SEPARATOR_ROUNDTRIP.

These frontend changes leave the nucleus sampler, TP3 layout, native transport,
model weights, cache placement and CUDA kernels unchanged. Added-token masking
prevents supported control-token spellings from becoming control tokens;
instruction-following attacks remain possible.

## Prompt preparation memory

Robert Wojciechowski (wojo/rwojo) authored Mia patch 0023, from commit 2fd7809.
This branch adapts its page-backed tokenizer/template scratch, per-piece walk,
concurrent preparation floor and bounded connection arenas. It keeps the later
control-token masking and decoded-tool-argument protection.

## TP3 failure recovery

Robert Wojciechowski's Mia 0020 inspired the bounded failure exit. Our independent
TP3 adapter monitors both follower links without consuming frames, tracks active-work
progress, stops admissions/rounds after fatal collective errors and exits 75.
Recipe supervision replaces all ranks, avoiding CUDA teardown from the monitor thread.

For optional DeepSeek context capacity, Robert Wojciechowski (wojo / rwojo)
ported GLM spill, chat continuation, owner KV sharding and ring exchange.
BadAd84 designed spill v2 blocks. Mia / MiaAI-Lab published the DeepSeek TP2
recipe at 2fd7809571e4d34f46a055dabf11caed9b145988. This fork adapts it to three
ranks on LakyFx's K2a base, keeping existing credits. All new controls default
off. TP3 owner/auto decode refuses startup; fetch, spill and restart recovery
still need live testing.

## LakyFx 1.2.8 prefill adaptations

LakyFx published the measured TP3 D/E/F, K, R/U/W, AC, AE and AF work
(source recipe 8e6d177c096a175eeabafacbbdfeb97e6042d3e9). This fork adapts it
behind independent test flags and keeps its owned-row transport, spill layout,
graph voting and numeric effort.

Thanks to Ash Hart and TensorFold contributors for the attention source, adapted
with LakyFx's three HB16 head-tail masks. EXL3 prompt MMA follows TensorFold
612332f60668744b32a82eca8a91afa2b2929d69, BobClawblaw #976 and MiaAI-Lab patches
0004/0009. Thanks to ExLlamaV3 / Turboderp (MIT, Copyright 2025) and to vcruz305
for trellis helpers. Their notices are in the vendored CUDA source,
THIRD_PARTY_NOTICES.md and LICENSES. Original sources, license files and measured
kernel hashes are preserved.

The isolated asset builder adapts LakyFx's prefill_assets.py. The pinned manifest
and compile records in zig/kernels/prefill_128 identify the measured SM121 assets.
Locally rebuilt cubins are separate and need GPU testing.
