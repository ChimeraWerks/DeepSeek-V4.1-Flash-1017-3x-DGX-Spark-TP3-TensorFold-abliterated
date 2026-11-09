# Changelog

DeepSeek V4.1 Flash 1017 abliterated EXL3 3.84 bpw runs across three NVIDIA DGX Spark systems with TensorFold tensor parallelism 3 (TP3).

v1.2 uses engine base `30a9647` with exactly two non-runtime edits in `zig/src/server/metrics.zig`. A documentation comment and a test name describe external metrics samplers generically. Serving logic is unchanged by them. Rebuild comparison results: `release/build-verification.json`.

## 2026-11-09 PDF input

The engine part is in production since 2026-11-09, with the owner's approval, on top of the bf16 prefill wire build; `libpdfium.so` sits beside `tensorfold-native` ([runtime assets](docs/runtime-assets.md#pdfium-for-pdf-input)). The API part (counting and gating documents as media) is not deployed yet. PDF input is on by default; `--no-vision-pdf` turns it off. Checked live: a two-page PDF was counted exactly and read back correctly, through the engine and through the API as a Messages tool result, and a streamed reply running alongside showed no decode stall (longest gap 0.1 s).

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
