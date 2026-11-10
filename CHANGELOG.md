# Changelog

DeepSeek V4.1 Flash 1017 abliterated EXL3 3.84 bpw runs across three NVIDIA DGX Spark systems with TensorFold tensor parallelism 3 (TP3).

v1.2 uses engine base `30a9647` with exactly two non-runtime edits in `zig/src/server/metrics.zig`. A documentation comment and a test name describe external metrics samplers generically. Serving logic is unchanged by them. Rebuild comparison results: `release/build-verification.json`.

## 2026-11-10 cancelled prompts resume, and warm turns no longer wait behind cold ones

In production since 2026-11-10: main `635f57e` with `TF_DS_PREFILL_CHECKPOINTS=1` and `TF_FILL_UNCACHED=1`, the same kernel images and the other flags unchanged. Checked live after the restart: 24 short and 2 long fixed replies byte-identical to the previous production's, identical seeded replies at `top_k` 0 and 20, the mixed-load probe, decode at one and four requests within 1% of the same session's baseline, and both scenarios below.

- With `TF_DS_PREFILL_CHECKPOINTS=1` a cold prompt keeps a prompt-cache rung every 65,536 tokens while it loads. A 110,025-token prompt cancelled at 81,920 tokens resumed from the 65,536 rung: the resend took 13.0 s instead of 30.1 s. Carried in startup agreement word 6 bit 56 (atom2ueki's ladder checkpoints, Mia 4x PR 53).
- With `TF_FILL_UNCACHED=1` an admitted prompt that has not started is ranked by what its cache hit leaves to compute, where it was ranked as fully cold. A warm 40,038-token second turn arriving one second into a cold 30,012-token prompt got its first token in 1.34 s instead of 7.25 s and finished first; the cold prompt finished in 8.3 s against 8.1 s (devopsfortunato's TensorFold PR 2276, backported by laidick in Mia's GLM-5.3 recipe PR 20).
- Sampling penalties are in the build and off in production. With `TF_DS_PENALTIES=1` and a `-Dpenalties=true` build, `presence_penalty`, `frequency_penalty` and `repetition_penalty` are applied to the full head before any filter, per row with that row's own history, so a reply is the same with drafts on and off. `repetition_penalty` alone covers the whole reply; `repetition_penalty_window` and `penalty_include_prompt` narrow or widen it. Measured on the three Sparks: 20 of 20 checks pass (greedy, seeded and prompt-inclusive replies identical with drafts on and off, four concurrent requests with different penalties identical to each alone, repeats on a repetitive prompt from 97.7% to 0%). A penalized request decodes at 50.2 tok/s against 62.4 (a round with a penalized row runs without CUDA graphs), four at 103 combined against 125, and three plain requests beside one penalized one keep 31.0 tok/s each against 32.0. With the switch off the fields are accepted and ignored as before.
- Reply logprobs are in the build and off in production. With `TF_DS_REPLY_LOGPROBS=1` and a `-Dreply_logprobs=true` build, Chat Completions and Responses return `logprobs` with up to 20 `top_logprobs`: each rank summarises its third of the output head for a committed token and rank 0 merges them, normalized over the whole vocabulary, from the raw row before penalties, masks, temperature and filters. Measured: 20 of 20 checks pass (reply unchanged by asking, streaming equal to non-streaming, drafts off equal, raw rows under penalties, a scored reply beside others equal to alone). A scored request decodes at 58.8 tok/s against 62.4, four at 110 combined against 125, and neighbours are unaffected.
- Rank 0's CUDA host wrapper copied the backend's hooks one by one and left out three: the logprobs capability, so the server refused logprobs with the switch on, and the two round-timing hooks, so `TF_DS_ROUND_TIMING` never ran on rank 0. All three are forwarded now, with a host test that reads what the server reads.
- `TF_KV_CACHE_GROW=1` (aaa4xu, Mia 2x PR 115) and `TF_FILL_LEND=1` (after TensorFold PR 2224) are in the build and off: lending measured no difference here, and the grow-in-place path needs a full pool that the test did not reach.

## 2026-11-10 running replies keep a share of time while a prompt loads

In production since 2026-11-10: main `cd18925` with `TENSORFOLD_PREFILL_SHARE=0.5`, the same kernel images and the other flags unchanged. Measured on the three Sparks with one sampled reply streaming while a cold prompt loads: beside a 30,012-token prompt the reply runs at 30.2 tok/s instead of 6.2 (64 alone), four replies at 16.2 tok/s each instead of 6.5 (33 alone), and the prompt's first token takes 17.0 s instead of 9.6 s (8.2 s alone). At a share of 0.7 the reply runs at 18.6 tok/s and the prompt's first token takes 12.5 s. A prompt loading with nothing else running is unchanged. Checked live after the restart: the mixed-load probe, and identical seeded replies at `top_k` 0 and 20.

- `TENSORFOLD_PREFILL_SHARE` had no effect on this engine. A prompt slice only queues its GPU work, so the decode round at the next park waited for it and was billed its time, which cancelled the debt the share is meant to build. With the share on, rank 0 now synchronizes the stream before a park and at a chunk's end. Share off adds no synchronize.
- A park between prompt slices ran exactly one decode round. With the share on it now runs rounds until the preceding slice's debt is paid, up to `TF_DS_PARK_MAX_MS` (250 ms by default, 0 for no cap); what a park cannot pay is paid between chunks as before. Share off keeps one round per park.
- The reply still pauses about 0.7 s once per prompt chunk: before a park the engine waits for the next chunk's Engram read.
- With `tool_choice` required and thinking on, the first token could be the backend's draw handle instead of the drawn token. The handle is now read before the feed is reassigned (markyu0401's fix, Mia's 2x recipe PR 118).
- Smaller slices help without any code change (`TF_DS_PREFILL_SLICE=2`: 11.6 tok/s beside the same prompt, first token 11.2 s) but less than the share does; production keeps 5.

## 2026-11-10 keyed drafts while a prompt loads

In production since 2026-11-10: main `53dedad`, the same kernel images and flags as the entry below. Checked live after the restart: a sampled 2,500-token reply kept streaming while cold 8,012-token and 30,012-token prompts loaded beside it, every rank stayed up, and the seeded `top_k` 0 and `top_k` 20 replies were still identical.

- The keyed top_k build stopped all three ranks when a new prompt arrived during a sampled reply. With `TF_DS_PREFILL_SLICE` on, one decode round runs between a prompt's layer slices, and followers accept only the next slice, a verify frame and a draft pass while a chunk is open. The keyed draft pass has its own frame kind, which that check did not list, so followers returned `OpenPrefillChunk` and the rank group exited 75. It happened twice in production on 2026-11-10 (5:07 AM and 8:04 AM Eastern); the supervisor restored service both times.
- Followers now accept the keyed draft pass in that window as they accept the plain one. It runs the same drafter pass on the drafter's own buffers, with graph capture off while the chunk is open. The check moved into `openChunkAllows`, and a host test covers every frame kind.
- While a cold prompt loads, a reply already streaming runs at about 6 to 7 tok/s instead of about 60 (one decode round between slices of five layers). That is the existing `TF_DS_PREFILL_SLICE=5` behavior, measured here for the first time.

## 2026-11-09 keyed drafts for top_k requests

In production since 2026-11-09, with the owner's approval: main `551d77e` built with `-Dkeyed_draft=true`, its keyed kernel image compiled on the head Spark, and `TF_DS_KEYED_FILTER` left unset (mirror). The other flags are unchanged. Checked live after the restart: all three ranks run the new binaries, two seeded `top_k` 20 thinking replies were identical, as were two seeded `top_k` 0 replies, and a `top_k` 20 reply decoded at 72.6 and 70.4 tok/s against 65.7 after the afternoon restart.

- Keyed drafts reached only requests that sample without `top_k`, `top_p` or `min_p`, so API-default requests (`top_k` 20) still drafted greedily against a sampled target. Every sampled request now gets keyed draft noise, and a request with `top_k` has it mirrored in the draft: each rank keeps at least its `top_k` best columns of the draft logits plus the Markov bias before the noise is added, so a tail token the target cannot pick is never proposed. A `top_k` of 1,024 or more is not mirrored.
- Target sampling and acceptance are unchanged, so a reply still depends only on its seed, never on the draft. Grammar-constrained requests still draft greedily. `top_p` and `min_p` are not mirrored.
- `TF_DS_KEYED_FILTER` on rank 0 selects the rule (`2` mirror, the default; `1` noise without the mirror; `0` the previous rule). The draft pass frame carries each stream's `top_k`, and the rank protocol version is 3, checked at startup with CUDA graphs on or off, so old and new ranks refuse to start together.
- Two new kernels live in the opt-in `dsv41_keyed` image: `tf_ds_keyed_select_kernel` finds each row's cut in one pass (the `top_k`-th largest of 1,024 strided maxima) and `tf_ds_keyed_apply_kernel` masks below it and adds the noise across the whole grid. The row grew from 24 to 32 bytes: a `KEYED_DRAFT=true` build needs that fatbin rebuilt.
- `tf-dsv41-lanes` requests take `min_p`.
- A first Spark run on 2026-11-09 with a one-kernel version confirmed the mechanism at one request with `top_k` 20: 2.16 to 2.46 tokens a round, 65.3 to 68.9 tok/s, replies identical. That kernel did its noise and a 32-pass search on one block per row, about 2.9 ms a round, and slowed `top_k` 0 requests by about 10%; the two kernels above replace it.
- Measured on the three Sparks with the two-kernel build, one binary with the previous rule and the new one, thinking on at temperature 1.0: `top_k` 20 decode went from 66.1 to 71.4 tok/s at one request (+8%) and from 115 to 126 tok/s combined at four (+9%), and the `focused` profile (`top_p` 0.95, `min_p` 0.05) from 64.5 to 68.9 tok/s (+7%). Tokens a round rose from 2.15 to 2.42. `top_k` 0 speed did not change (67.5 and 67.1 tok/s). One-request replies were identical in both arms.
- Checked on the host: the kernels matched a CPU reference on an RTX 5090 (31,557,392 checks: masked sets exact, the true `top_k` always kept, noise within 2e-6, `top_k` 0 bit-equal to the old kernel), where one call costs 94 µs against 1,004 µs before; the keyed-draft, structured-output, graph-vote and drafter-fault test steps pass.

## 2026-11-09 measured round costs

In production since 2026-11-09, with the owner's approval, in the bundled restart: `TF_DS_ROUND_MS` set on every rank to a round-cost table measured on this cluster, in place of the served `ROUND_MS`.

- The draft planner weighs a longer draft against the round it costs, by rows. The served table was 9% high at 8 rows and 9 to 13% low at 32 to 48 rows on this cluster.
- The new table is a straight fit of fixed-draft sweeps with 2, 4, 6, 9 and 12 requests in flight: 22.79 ms plus 3.823 ms a row, rows 1 to 48, within 4.9% of every measured point.
- Checked live: sampled decode at one and two requests matched the keyed draft deploy within noise (67 to 70, 45 and 63.5 tok/s against 66 to 70, 45 and 63.2).

## 2026-11-09 PDF page cache

In production since 2026-11-09, with the owner's approval, in the engine of the bundled restart (main `90d6c84`, keyed drafts still on). The API is unchanged. Checked live: the same two-page PDF read twice was counted and read back correctly both times, the second in 3.4 s against 3.9 s.

- The engine keeps the pages it rendered from a PDF, so a document sent again is not rendered again. An agent client resends a PDF it read with every turn, and the count route measures it before each generation; each time, a worker was started and every page measured or drawn again (about 80 ms an A4 page on the x64 host), one document at a time across the server.
- Pages are keyed by the SHA-256 of the document's bytes, whether they were drawn or only measured, the page limit, the render scale and limits, and the PDFium file's path, inode, size and modification time. Only pages that passed the server's checks of the worker's reply are kept.
- A hit starts no worker and takes no render lock: its pages are copied into the request and count against its page and pixel limits as a worker's reply does, with the same refusals. A count is also answered from the rendered pages' PNG headers. Two requests waiting on the same document render it once.
- Refusals (encrypted, corrupt, too many pages, timed out) are not kept: some depend on what else the request holds.
- The cache holds at most 256 MiB, counting the pages, their bookkeeping and the map; `TENSORFOLD_PDF_CACHE_MIB` sets it, and `0` turns it off. The least recently used documents are dropped first; a document larger than the whole cache is not kept.
- Checked on the host with the x64 library: a twelve-page A4 document took 1.0 to 1.1 s a turn to count and render, and 0.3 ms on a hit. Hits match a fresh render byte for byte and start no worker; different modes, limits, documents and libraries miss; the byte bound holds through evictions; eight threads hitting and filling together start at most two workers per document.

## 2026-11-09 omitted output caps fit the context left

In production since 2026-11-09, with the owner's approval, as API `0.1.5.post8` in the bundled restart. The engine is unchanged. Checked live: a Chat request with no output cap and a Messages tool result holding a PDF both answered through the API.

- A request that leaves out its output cap (`max_tokens` or `max_completion_tokens` on Chat, `max_output_tokens` on Responses, `max_tokens` on Messages) gets `output_tokens` (32,768), lowered after the prompt is counted to the context left: `context_tokens` minus the prompt and `safety_tokens`. Before, the API filled 32,768 and refused a near-full conversation with 400 `context_length_exceeded`, although it could still answer. With no token left, it is still refused. The engine gets the lowered cap, and admission weighs it.
- A cap the client gave is kept as before: one that does not fit beside the prompt is still refused with 400. A client cannot mark its own cap for lowering.
- This follows DeepSeek's reference server: `resolve_max_tokens` in [`deepseek-recipe` `request/mod.rs`](https://github.com/deepseek-ai/deepseek-recipe/blob/935817c66f5d6e64838f419692545f06787e485f/deepseek-recipe/src/request/mod.rs#L155-L234). Its omitted default depends on the mode (8,192 without thinking, 65,536 with it, 131,072 at `max` effort); the API keeps its one documented `output_tokens` default.
- The engine's own Chat and Responses routes already lower an omitted cap (its `--max-tokens`, 4,096 by default) to its context window and refuse a given one that does not fit; its Messages route requires `max_tokens`. Behind the API it always gets a cap, so its behavior is unchanged.
- Checked on the host: new serving tests cover Chat, Responses and Messages, streamed and whole, near a full context, with an omitted cap, a given one and no room. The lowering test fails without the change.

## 2026-11-09 keyed drafts

In production since 2026-11-09, with the owner's approval: the PDF input engine plus one fix, built with `-Dkeyed_draft=true` and run with `TF_DS_KEYED_DRAFT=1` on every rank.

- Keyed drafts crashed TP3 on the first sampled request: the graph vote codec refused a draft key carrying the keyed bit, so rank 0 could not encode its start frame and every rank exited with `GraphVoteFatal`. Draft keys now carry `c` 0 or 1, other values are still refused, and the vote protocol version is 2, so old and new ranks refuse to start together.
- Keyed drafts add keyed Gumbel noise to the draft proposals of unfiltered sampled rows (`top_k` 0, `top_p` 1, `min_p` 0). Target sampling and acceptance are unchanged, so replies are the same token for token. Rows with `top_k` 20, the API default, or greedy rows are not touched.
- Measured on the three Sparks, one binary with the flag off, on, and off again, thinking on at temperature 1.0: `top_k` 0 decode went from 59.7 to 67.4 tok/s at one request (+13%) and from 28.9 to 31.6 tok/s a stream at four (+9%). Draft acceptance rose from 0.47 to 0.58 at one request. `top_k` 20 and greedy speeds did not change. One-request replies were identical in every arm; at four requests the off-again arm differed from the first by the same rows as the keyed arm (batch order, not keyed drafts).
- Checked live after the deploy: two seeded `top_k` 0 thinking replies through the engine API were identical, two concurrent `top_k` 0 replies and a `top_k` 20 reply succeeded, and the PDF and upstream fix checks passed.

## 2026-11-09 PDF input

The engine part is in production since 2026-11-09, with the owner's approval, on top of the bf16 prefill wire build; `libpdfium.so` sits beside `tensorfold-native` ([runtime assets](docs/runtime-assets.md#pdfium-for-pdf-input)). The API part (counting and gating documents as media) followed the same day. PDF input is on by default; `--no-vision-pdf` turns it off. Checked live: a two-page PDF was counted exactly and read back correctly, through the engine and through the API as a Messages tool result, and a streamed reply running alongside showed no decode stall (longest gap 0.1 s).

- The engine reads PDF documents: Messages `document` blocks (base64 `application/pdf`, also inside a `tool_result`, which is how agent clients send a PDF they read), Chat `file` parts with `file_data`, Responses `input_file` parts and `data:application/pdf` image URLs. Each page becomes one image for the vision tower. Remote URLs and file IDs are refused. Ported from TensorFold: joernchen's [PR 1820](https://github.com/ashhart/TensorFold/pull/1820) and ashhart's [PR 2387](https://github.com/ashhart/TensorFold/pull/2387), [PR 2393](https://github.com/ashhart/TensorFold/pull/2393) and [PR 2599](https://github.com/ashhart/TensorFold/pull/2599). Upstream turns it on with `--vision-pdf`, which this engine also accepts.
- PDFium `chromium/7725` renders each document in a fresh worker process: the server's own binary, with CPU, core-file and file-size limits, an address-space limit of its starting size plus 1 GiB, its other file descriptors closed, and a 10-second deadline. The server reads the worker's reply as untrusted: each page must be a PNG of at most 4,096 pixels a side, and the page and pixel limits are recomputed from the pages' headers. A request takes at most 16 pages (`TENSORFOLD_MAX_PDF_PAGES` sets 1 to 64) and 32 megapixels. Pages render at 144 DPI, within 2,048 pixels a side and 3.6 megapixels.
- Pages reach the vision path as PNG files, which the engine decodes itself, so PDF input needs no Pillow on the Sparks. Upstream sends JPEG pages through libjpeg-turbo.
- `/tokenize` and Messages `count_tokens` count each page exactly as generation holds it. The worker reads each page's size at the render scale without drawing it, and the count spans the page from the PNG header the rendered page starts with.
- With `--no-vision-pdf`, a PDF part gets a 400 that names the flag. Without the library, a 501 names `libpdfium.so`. The request log replaces the bytes of every PDF part (`file_data`, `data`, `url` or `file_url`, in Chat `file` and `input_file` parts) with `<redacted>`.
- The API sends documents to the engine's count unchanged and treats them as media: they take the media gate, and a single PDF question goes to the durable vision jobs, as an image question does. With an engine that cannot count them, each document reserves 16 pages of `image_token_reserve`.
- `prepare.sh` fetches the pinned arm64 PDFium archive, checks its SHA-256 and the library's, and stages `libpdfium.so` into `BIN_DIR`. A file that differs from its pin is never staged. The direct launcher (`native_launch.py --run`) requires the SHA-256 of the `libpdfium.so` the head would load, as it does for the binaries.
- Checked on the host with the x64 library: the spawned worker renders text and colour that the engine's own PNG decoder reads back (red stays red), the page and pixel limits hold across a request's PDFs, and a count matches generation for nine pages in eight shapes and an image. Server, tool parser, `engine_api`, serving and cluster tests pass, and the aarch64 `native` and `dsv41-tools` build succeeds.

## 2026-11-09 eight upstream fixes

Production moved to `30a9647` plus the exact image count change and these fixes, with the owner's approval. The weights, KV pool, stream count, spill and sampling defaults are unchanged. The rebuilt binaries embed the production fatbins byte for byte. Checked live after the restart: the Responses `include` request returned 200, and a JSON answer shaped like a call to an offered tool came back as content with finish reason `stop`.

- The engine's `/v1/responses` route accepts `include: ["reasoning.encrypted_content"]` and ignores it. Pi and other Responses API clients send it with every reasoning request, and each such request was refused with a 400. Reasoning already comes as text in its reasoning item, and nothing is encrypted. Any other `include` value is still refused and named. Ported from Mia's AI Lab patch 0058, reported by Unitea (unitea1992).
- The lane thread makes its CUDA context current as it starts. With the spill tier on, the first request after a restart could resume a saved state from disk before any model call had bound the context on that thread. The restore's `cuMemAlloc` was then refused with 709, read as a lost context, and broke the rank group. The warmups `start.sh` sends after a start hid this; a supervisor restart sends none. Port of Mia's patch 0059 (issue #102, reported by NguyenPhan2810).
- Checked on the host: a new lane host test in `lane_host_tier_test.zig` fails without the change and passes with it, the other tier tests pass, and `tensorfold-native` builds.
- When a request offers tools and asks for a JSON answer, the answer stays the answer on Chat, Responses and Messages, streamed or whole. These fields take this path: `response_format` `json_object` and `json_schema`, `guided_json`, and `structured_outputs` `json` and `json_object`; Responses `text.format` and Messages `output_config.format` arrive as `response_format`. Before, an answer shaped like `{"name": <offered tool>, "arguments": {...}}` became a tool call with finish reason `tool_calls` and its content was dropped. Thinking replies still split their reasoning. Ported from SvangenStudios's patch 0042 ([Mia 4x PR 31](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/31), [patch file](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/blob/6a4d55e8b7a7b492747ccccf6d67389f26c7d5dd/zig/patches/0042-json-answer-channel.patch)).
- `/v1/responses` streams now send a `: ping` comment after 25 s of silence, as chat and Messages streams already do, so idle proxies keep them open through a long prefill or decoding that sends no deltas. Ported from Mia 2x patch 0065 (hecisaza).
- The engine writes C0, DEL and C1 control bytes in its `[tensorfold]` log lines as `\xNN`, so a request line can no longer drive the terminal of someone reading the rank logs or forge a log line. Ported from TensorFold `ceb6f4d3` (Chad Hurley).
- `preflight.sh` checks that each RoCE device's `TF_RDMA_GID_INDEX` slot holds its IPv4 RoCE v2 address inside `NCCL_IB_ADDR_RANGE`. When that slot is empty or holds another kind of GID, as after a link flap, it names the slot where the IPv4 RoCE v2 GID now sits; when the slot holds an IPv4 address outside the range, it names that address. A malformed `TF_RDMA_GID_INDEX` or `NCCL_IB_ADDR_RANGE` is listed as a problem. Before, a moved GID showed up only as a failed ring setup at start. Ported from christopherowen/spark-ds41f (Christopher Owen), [`1e9f851`](https://github.com/christopherowen/spark-ds41f/commit/1e9f851ad1c23daa5ef3b9cf490f9c832402cfe9) and [`6a4d398`](https://github.com/christopherowen/spark-ds41f/commit/6a4d398e6d3e30df5799e6d13de18330f8756c1c).
- Engram table loading checks each file's header length against the file size and a 100 MB cap before it allocates, so a corrupt header fails as `BadSafetensors`. Before, a damaged length could allocate up to the whole table file. Ported from upstream TensorFold `460a2868` and `46ce6dfd`.

Hardening only. Production decoding cannot reach this path: the served tokenizer has no `<tool_call>` token, so a spelled `<tool_call>` is never an envelope.

- A `<tool_call>` block (or `<|tool_call>`, `<x:tool_call>`) that opens with a DSML opener and is shorter than that opener and its closer together now stays text. The parser cut both from it, a slice past its start that panics in the ReleaseSafe server (Mia 4x patch 0053 and 2x patch 0064, Hector Isaza).
- Every payload long enough parses exactly as before.

## 2026-11-08 exact image counts

Production moved from `30a9647` to `30a9647` plus this change, with the owner's approval. The weights, KV pool, stream count, spill and sampling defaults are unchanged. The rebuilt binaries embed the production fatbins byte for byte.

- The engine's `/tokenize` and Messages `count_tokens` routes count each image as the span generation gives it, read from the image's header without decoding it. Before, they counted one token per image. Replies to requests with images add `image_tokens`.
- The API sends images to those routes as they are. Before, it charged 16,384 tokens per image, while an image on this model takes 180 to 1,024. Chats with about 50 images were refused as over context while holding about 140K tokens.
- With an engine that cannot count images, the API falls back to `[image]` text plus `image_token_reserve` per image, now 1,024 by default (the model's `max_image_tokens`).
- An image the engine refuses while counting now returns that 400 to the client, not a 503.
- Checked live: a Messages request with 60 screenshot-size images and `max_tokens` 64,000 succeeded with 58,157 input tokens. The flat reserve refused it.

## 2026-11-06 GPU clock cap

- Added `GPU_CLOCK_CAP_MHZ` (default 2300, `0` for stock clocks): `start.sh` applies it to every Spark, matching the cap production serves with.
- Added `./scripts/clocks.sh` to read or change the cap on running Sparks without a restart.
- Added clock, temperature and power results for four GPU caps on `30a9647` in `benchmarks/v1.2/clocks.csv`.
- Rebuilt the scorecard, charts and link preview from the v1.2 data in one design.

## 2026-11-05 engine 30a9647

Production moved from `678ea1b` to `30a9647` with the owner's approval. The weights, KV pool, stream count, spill and sampling defaults are unchanged.

- Long chat histories use far less memory when building a prompt.
- A returning chat keeps its saved state while it waits in line, so it no longer recomputes from zero.
- Production suite on `30a9647`: 48 decode points, six cold-prefill points and 97.1% (720 of 755) on the standard quality set; peak prefill 3,816 tok/s at 32K on stock GPU clocks.

## 2026-11-04 v1.2

The weights, KV pool, stream count, spill and sampling defaults are unchanged from v1.1.

Checks on engine `591d9dc`, 2026-11-04, against the previous production engine `8f4faf6` (an unreleased build between v1.1 and v1.2):

- Terminal-Bench prompts: 13 of 13 token-identical.
- Decode: 60.9 tok/s single stream and 158.4 tok/s at 8 streams, unchanged within noise. These are quick regression checks, not comparable with the benchmark tables.
- Output parsing matched DeepSeek's `deepseek-recipe` in 9 of 9 thinking plus tool call cases.
- Engine `678ea1b` against `591d9dc`: Terminal-Bench prompts 13 of 13 token-identical.

Engine `14803a3` to `30a9647`:

- Enabled prefill layer slices (`TF_DS_PREFILL_SLICE=5`): running streams decode between slices of a long prompt chunk.
- Matched `deepseek-recipe` prompt rendering: the `response_format` schema reaches the template, tools or a schema add an empty system block when needed, a `thinking` object decides the mode, Messages tool JSON keeps chat key order, and a prefix continuation leaves off the end token.
- Matched `deepseek-recipe` on Responses: every reasoning item is kept, the custom `apply_patch` tool and `response_format` regex work, and tool namespaces are flattened and restored.
- Matched `deepseek-recipe` on output and validation: think and DSML markers count only as real tokens, broken tool history and duplicate tool names are refused before inference, and strict tool and response schemas are enforced.
- These match the listed behaviors, not all of `deepseek-recipe`. What still differs includes effort numbers, default thinking, default output caps, sampling clamps, deferred tools, system images and logprobs.
- Kept a DSML tool call whole when a parameter value holds its block's closer (Mia 4x patch 0052, hecisaza).
- Reduced head memory: template strings grow in place (Mia 4x PR 41, Keenni), and each picture is prepared in its own scratch (Mia 2x PR 112, marknx).
- Made prompt rendering linear for long chat histories, and kept a queued request's matched state in place while it waits.
- Fixed replayed `reasoning` fields (Mia's patch 0047, nuk3s), undrained GET and DELETE bodies (Mia's patch 0046, Christopher A. Williamson), a finished-job race (found in Mia PR 94 review, hecisaza), a first-token finish that skipped its release (TensorFold `ed39acc5`, laudney) and an uninitialized read-ahead start.
- Added opt-in features, all off in production: dead-checkpoint batches, first-continuation hints, draft calibration, first-draft depth, keyed drafts, reduced output heads, restore-room waits, graph telemetry and streamed tool calls. The launch sets all but streamed tool calls to off explicitly; see [engine/README.md](engine/README.md).

Release:

- Added vision job retention from LakyFx 4.0.3: finished image bodies expire after one hour and results after seven days, so storage no longer fills up and refuses new jobs.
- Selected native profile `v1.2`, recipe patches `chimera-tensorfold-patches-20261103`; kept `v1`, `v1.1` and their manifests for history.
- Added `release/history/v1.1.json` and moved the v1.1 build record to `release/history/v1.1-build-verification.json`.
- `release/build-verification.json` records the production hashes and a rebuild of the public tree: all 22 allocated ELF sections match in each binary.

Serving API, version unchanged at 0.1.5:

- Removed automatic context shortening. Prompts that do not fit `context_tokens` are refused, and clients manage their own context. Delete `auto_compact`, `compact_*` and `checkpoint_*` from existing configurations; the API no longer accepts them.
- Added `/v1/messages` and `/v1/messages/count_tokens`, forwarded to the engine's native Messages routes, and `x-api-key` authentication.
- A reply cut by `max_tokens` while thinking is a normal length finish, not an empty-reply error. Incomplete streams now log why they stopped.
- Responses admission always uses the native tokenizer, the same as Chat Completions.
- `thinking: {type}` now decides thinking mode, and `ultra` effort maps to `max`, matching DeepSeek's `deepseek-recipe`.
- A Messages reply with thinking but no final text or tool call is now an error, as on Chat Completions, and an oversized Messages `max_tokens` is lowered to the configured bound, as DeepSeek does.

## Historical — 2026-11-02 v1.1

v1.1 uses engine base `14803a3` with the same two non-runtime `metrics.zig` edits; its rebuild comparison is in `release/history/v1.1-build-verification.json`.

- Selected engine baseline `14803a3`, including the LakyFx 1.2.8 prefill port, with matching kernel inputs.
- Set an 8,000,000-token pool, C16, graph cap 128, R48, engine/API top_k 20 and 256 GiB spill. Kept 1,048,576-token configured context and 4 GiB startup reserve.
- Kept the existing mrfakename-edited 3.84 bpw composition and rank identities.
- Selected native profile `v1.1`; kept `v1` and its manifest for history.
- Made the optional affinity-controller integration opt-in instead of a launch prerequisite.
- Added a separate [November 2 benchmark record](benchmarks/README.md#v11--2026-11-02), preserving the October 30 speed suite and its timings.
- Added model lineage, refusal-removal limits and component notices.
- Pinned and staged the kernel kit and Engram tables in `prepare.sh`: the kit from a Hugging Face mirror, the Engram rows from DeepSeek's repository ([runtime assets](docs/runtime-assets.md)).

## Historical — 2026-10-31 numeric reasoning effort

- Engine `7d5b4d8` adds numeric `reasoning_effort` from 1 to 100 to `cebcf1e`. Named levels keep their existing values.
- Serving API `chimera-dgx-sparks` 0.1.5 adds live effort remapping. Non-native mapping changes require engine `/health` to report `numeric_effort`.
- Response checks matched 8/8 long cases, cold and cached, and 12/12 short cases. Named efforts matched their numeric equivalents in these checks.

The [speed suite](benchmarks/README.md) ran on `cebcf1e` on 30 October, with no rerun for the numeric-effort update.

## Historical — 2026-10-30 v1 production recipe

- Combined EXL3 weight sources and a rank-2 refusal-removal edit into the [3.84 bpw composition](release/weight-provenance.md).
- Configured 16 streams, CUDA graph cap 128, a 10,000,000-token KV pool, 1,048,576-token context and 256 GiB NVMe spill on the head.
- Enabled owned-row prefill, direct all-gathers and top_k 20. Kept step-ahead disabled.
- Measured greedy prose decode at 84.5 tok/s for one stream and 260.7 tok/s aggregate for 16 streams. Prefill at 32K reached 3,089 tok/s.
- Fixed Engram read-failure exits. Host tests covered the fault path; Spark fault injection was not run. Separate supervisor tests recovered from a follower failure and a follower reboot.

See the [release manifest](release/manifest.json) for the current configuration and [tuning](docs/tuning.md) for measured tradeoffs. Repository code is Apache-2.0; third-party code keeps its own licenses. Weights: [Hugging Face](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated).
