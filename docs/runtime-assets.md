# Runtime assets

Besides the weights and its rank file, each Spark needs a kernel kit in `KIT_DIR` and its rank's Engram tables in `ENGRAM_DIR`. `./scripts/prepare.sh` fetches, checks and stages both before it downloads the weights. Every file below is pinned by commit and SHA-256 in [release/runtime-assets.json](../release/runtime-assets.json), and [tools/assets/runtime_assets.py](../tools/assets/runtime_assets.py) checks them all.

## What each Spark holds

| Path | Contents | Size | Source |
|---|---|---|---|
| `KIT_DIR` (default `SPARK_ROOT/kit`) | Triton AOT kernels with the prefill 1.2.8 supplement, EXL3 cubins, `engram.json`, `engram-token-map.json`, vision kernels, four RoPE tables | 589 files, 1.88 GB | Kit mirror on Hugging Face plus this repository |
| `ENGRAM_DIR` (default `SPARK_ROOT/weights/engram-tp3`) | DeepSeek's two Engram shards, holding this rank's third of the rows, plus `SOURCE.json` | 67.6 GB allocated (`ranged`); 203.1 GB (`full`) | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) |
| `SPARK_ROOT/downloads/runtime-5a612f9b090c` | The downloaded kit inputs, kept after staging; delete them once the kit checks out | 1.73 GB | Kit mirror |

With the default `ranged` Engram source, each Spark needs about 440 GB (408 GiB) free: 274 GB of weights, a 93 GB rank file, 67.6 GB of Engram rows and 3.6 GB for the kit. `full` raises that to about 574 GB.

## Pinned sources

Kit mirror: folder `runtime/` of [ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated at `5a612f9`](https://huggingface.co/ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated/tree/5a612f9b090c8d841fcf2c97b22e166ba27cab25/runtime). It is a separate pin from the tested weights commit `2fa62d7`.

| File | Bytes | SHA-256 |
|---|---|---|
| `runtime/tensorfold-kernels.tar.gz` | 39,844,643 | `3e94bac39a953165b05930fc9ef91c905f6238e92640fab3dd984973d761f341` |
| `runtime/rope/rope-plain-cos.f32` | 422,576,128 | `66a47d6f185bdb2962ca0ffe13952ab70409dc15ea52f721a8e09189e5f26fe6` |
| `runtime/rope/rope-plain-sin.f32` | 422,576,128 | `4ca39e36077fced64beab67b879470deb8f641a1cc26b5d227835c8c09812ecb` |
| `runtime/rope/rope-compressed-cos.f32` | 422,576,128 | `e02d894c119fc61a26d2aea51488b87d35e4b33f4e266b8335e75aca5155b921` |
| `runtime/rope/rope-compressed-sin.f32` | 422,576,128 | `d3a39e5b47a1d36ba9c20bfacbdf8803f3df4b7fcf7a0f947e6ef008f0a0aa2b` |

From this repository: [`prefill_128/payload.tar.gz`](../engine/tensorfold/zig/kernels/prefill_128/payload.tar.gz), the 21-file LakyFx 1.2.8 supplement, which its [MANIFEST.json](../engine/tensorfold/zig/kernels/prefill_128/MANIFEST.json) pins.

Engram: [deepseek-ai/DeepSeek-V4.1-Flash-1017 at `1bae0e1`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170), MIT, not gated.

| File | Bytes | SHA-256 | Tensors |
|---|---|---|---|
| `model-00047-of-00048.safetensors` | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` | `layers.1.engram.embed.weight`, `.scale` |
| `model-00048-of-00048.safetensors` | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` | `layers.14.engram.embed.weight`, `.scale` |

Each rank reads one third of each table's rows. `release/runtime-assets.json` records each rank's header bytes, byte ranges and their SHA-256s, and the `SOURCE.json` to write. That file is 618 bytes on every rank, because the engine compares the file names and sizes in `ENGRAM_DIR` across ranks and refuses to start if they differ. Keep only the two shards and `SOURCE.json` in that folder; `check` reports anything else.

## What prepare does

On each Spark in rank order, before the weights download:

1. Copies the asset tool, its pins and the prefill supplement to `SPARK_ROOT/prepare-tools`.
2. Checks free disk for whatever is still missing and stops with the shortfall before downloading anything.
3. If `KIT_DIR` does not exist, downloads `runtime/` at `5a612f9`, then runs `runtime_assets.py kit`. The tool checks every download, unpacks the archive, adds the RoPE tables, applies the prefill supplement and checks all 589 files against their pins. It stages everything in `KIT_DIR.partial-<pid>` and renames that to `KIT_DIR` only after the check passes. It never overwrites an existing `KIT_DIR`; a kit that fails its check must be moved aside first.
4. Fetches the Engram tables as `ENGRAM_SOURCE` selects, then runs `runtime_assets.py check`.

`ENGRAM_SOURCE` in `cluster.env` picks the Engram path:

| Value | What it downloads | Disk | Notes |
|---|---|---|---|
| `ranged` (default) | The two shard headers and this rank's rows, using HTTP range requests on DeepSeek's files | 67.6 GB allocated, 203.1 GB apparent | Writes sparse files at the original offsets. Resumes after an interruption: rerun `prepare.sh`. Checks each range's SHA-256. |
| `full` | Both complete shards with `hf download` | 203.1 GB | Plain download, no range requests. Checks this rank's ranges before writing `SOURCE.json`. |

Both give the engine the same bytes for every row it reads. `SOURCE.json` is written last, so a directory without it is incomplete. Copy a `ranged` directory only with a tool that keeps the sparse holes, such as `rsync --sparse` or `cp --sparse=always`.

## Manual staging

Run these on each Spark from a checkout of this repository, with the Hugging Face CLI installed. Set `RANK` to 0, 1 or 2.

```bash
hf download ChimeraWerks/DeepSeek-V4.1-Flash-1017-EXL3-3.84bpw-abliterated --revision 5a612f9b090c8d841fcf2c97b22e166ba27cab25 --include 'runtime/*' --local-dir ~/ds1017/downloads/runtime-5a612f9b090c
```

```bash
python3 -I tools/assets/runtime_assets.py kit --runtime ~/ds1017/downloads/runtime-5a612f9b090c --out ~/ds1017/kit
```

```bash
python3 -I tools/assets/runtime_assets.py engram --rank "$RANK" --dir ~/ds1017/weights/engram-tp3
```

For the `full` path, download the two shards first, then check them:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --include model-00047-of-00048.safetensors model-00048-of-00048.safetensors --local-dir ~/ds1017/weights/engram-tp3
```

```bash
python3 -I tools/assets/runtime_assets.py engram --rank "$RANK" --dir ~/ds1017/weights/engram-tp3 --mode full
```

Check a Spark at any time. The default checks sizes and small files in seconds; `--full` hashes every kernel, table and Engram range.

```bash
python3 -I tools/assets/runtime_assets.py check --kit ~/ds1017/kit --engram ~/ds1017/weights/engram-tp3 --rank "$RANK" --full
```

`./scripts/start.sh` also stops before changing any rank if a kit file or `SOURCE.json` is missing, and names the file.

## Where the files come from

- **Kernel kit archive.** A byte-for-byte copy of [LakyFx `tensorfold/artifacts/tensorfold-kernels.tar.gz` at `b4e476d`](https://github.com/LakyFx/deepseek-v41-flash-exl3-tp3-spark/blob/b4e476d60f01fda564d760c5906ac1358390c728/tensorfold/artifacts/tensorfold-kernels.tar.gz). It holds the Bertholomus AI `dsv41` kit with the LakyFx TP3 additions. Its 512 `aot/` files match `base_files` in the prefill MANIFEST, so the supplement applies to it unchanged. `KIT-LICENSES.md` inside lists each file's license: Apache-2.0, MIT and BSD-3-Clause.
- **RoPE tables.** 3,301,376 fp32 rows, the hashes the archive's `rope.json` records. [`prepare_rope.py`](../engine/tensorfold/tools/dsv41-tp3/prepare_rope.py) with `--pool 3300000` makes them on the CPU, from the 1,114,112-row tables in [bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10 `kit/` at `19a4422`](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-TensorFold-TP2-2xGB10/tree/19a442293f4d724439e9d725c83f0ea2affb9908/kit) and the RoPE arithmetic of [TensorFold `8b1fe210`](https://github.com/bertholomus/TensorFold/tree/8b1fe210d5e0d68e0cdb3b14755c135465c9ffa8). A rerun with torch 2.15.0 on the CPU gave all four pinned hashes; `runtime/rope/RECEIPT.json` in the mirror records it. Apache-2.0.
- **Token map.** `engram-token-map.json` (`de40bcdd…`) is derived from `tokenizer.json`. [`tools/assets/token_map.py`](../tools/assets/token_map.py) rebuilds it byte for byte from the published tokenizer.
- **Engram tables.** DeepSeek's own tensors at the model revision, unchanged; the refusal-removal edit does not touch them. The sparse per-rank layout and byte ranges follow the [LakyFx TP3 Engram release](https://huggingface.co/LakyFx/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw-TensorFold/tree/8621801c1fc74aece1fa83e67c6ade3fde1e7894/engram-tp3). MIT.

The vendored [build_kit.sh](../engine/tensorfold/tools/dsv41-zig/build_kit.sh) is upstream's from-source kit builder. It needs that engine's container image, GPU launch recordings and helpers this repository does not carry, so it is reference only. The served kit is the pinned archive above.

The kit does not include the optional grammar library (`libtfgrammar.so`). `prepare.sh` stages it with the engine binaries when the build output has one.

## PDFium for PDF input

PDF input (on in `tensorfold-native serve` unless `--no-vision-pdf`) renders pages with PDFium. The build is [bblanchon/pdfium-binaries `chromium/7725`](https://github.com/bblanchon/pdfium-binaries/releases/tag/chromium%2F7725) (PDFium 147.0.7725.0), pinned under `pdfium` in [release/runtime-assets.json](../release/runtime-assets.json). The library is not in this repository.

| Asset | Archive bytes | Archive SHA-256 | `lib/libpdfium.so` SHA-256 |
|---|---|---|---|
| `pdfium-linux-arm64.tgz` (the Sparks) | 3,555,478 | `a0dab89cdbee57832292034eaad27e010f84dea1c367e1e0535730933ead2315` | `5cfe82b0f16d33c0117d9df10790ba83b47d0b10deb74b1e7f540e607452a752` (7,752,328 bytes) |
| `pdfium-linux-x64.tgz` (host tests only) | 3,606,567 | `4f16da3f9c8e5790b9d59959f789bceff6eb8fdc7980e800d93b83392eb9cee4` | `fccddaf6adda2b435074dd88fad5398d7221cbc369d994cf561e7bfa84d7cf7b` (7,551,304 bytes) |

The engine looks for the library in this order and never on the loader's search path:

1. `TENSORFOLD_LIBPDFIUM`, when set; an empty value means none.
2. `lib/libpdfium.so` in the parent of the server's directory.
3. `libpdfium.so` beside `tensorfold-native`.

Only rank 0's `tensorfold-native` reads PDFs, and the server process never loads the library: each document's render worker loads it after it sets its limits. At startup with PDF input on, the server logs the file it found and its SHA-256, or that it is missing. A missing library makes PDF parts return HTTP 501 that names `libpdfium.so`. The direct launcher (`native_launch.py --run`) finds the same file and refuses to start rank 0 unless the rank's `checksums` pin its SHA-256, as they pin the binaries.

`prepare.sh` stages it with the engine binaries: it reuses `OUT/lib/libpdfium.so` when that is the pinned file, or downloads the pinned archive into `OUT/downloads`. It checks the archive's and the library's SHA-256 and copies the library into `BIN_DIR` on each Spark, beside `tensorfold-native`. A file that differs from its pin stops `prepare.sh`; without network it warns and stages the engine without the library. To stage it by hand on the head:

```bash
sha256sum pdfium-linux-arm64.tgz   # a0dab89c...2315
tar -xzf pdfium-linux-arm64.tgz lib/libpdfium.so
sha256sum lib/libpdfium.so         # 5cfe82b0...a752
install -m 755 lib/libpdfium.so ~/ds1017/bin/libpdfium.so
```

PDFium is BSD-3-Clause. The libraries it bundles (FreeType, libjpeg-turbo, OpenJPEG, libpng, zlib, lcms, ICU, abseil and others) keep their licenses in the archive's `licenses/`. The packaging is MIT.

## Prompt MoE kernel source

`prefill_128/SOURCE.json` declares `fdf5fc13…` for `experts_ragged.cu`, the hash recorded when the LakyFx 1.2.8 assets were imported. The committed file (`6d051e5a…`) adds an opt-in `TF_EXL3_W_RANGE16` compile path that is off by default. Built with nvcc 13.4.92 without that define, both files give the same cubin (`6a17bde3…`). The release loads the shipped LakyFx cubin `401d6629…` from the payload; a local build of either source differs from it. `SOURCE.json` keeps the declared value because it is part of the vendored tree that [build verification](../release/build-verification.json) hashes. [release/prefill-source-delta.json](../release/prefill-source-delta.json) records the hashes, compile flags and the change. The release consistency check verifies all of them.
