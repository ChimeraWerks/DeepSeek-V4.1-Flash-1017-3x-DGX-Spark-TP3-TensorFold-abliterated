# Yielded prefill layer slices

TF_DS_PREFILL_SLICE is default 0 (the existing whole-chunk frame and arithmetic).
Release v1.2 sets it to 5. Values 1 through 40 cap the number of layers per slice. This path requires
TF_DS_PREFILL_YIELD=1 and a yielded host callback. KV-sharded fills deliberately
retain whole chunks because their planner/twin lifecycle has not been qualified
for mid-chunk decode. Images are supported through the existing per-slot retained
spans/rows; their tower preparation is not sliced.

## Protocol and agreement

Opcode 18 is fill_slice. Opcode 13 remains fill_resume; 16/17 are reserved for
other ports. Each slice carries start, rows, replay, layer bounds and the AE
next-row prediction. The same ordered frame is broadcast to both TP3 followers.
Continuation must match the open chunk and its next layer exactly. The abort
variant closes a canceled chunk on all ranks before ordinary cleanup resumes.
Other outer operations are rejected while a chunk is open.

The six-bit count occupies prefill128.bits 24–29, which becomes actual startup
agreement word 6 bits 26–31 after the existing shift by two. Existing prefill
bits16/17 already occupy actual bits18/19 and remain intact. The count also
enters native.rankExchange through the existing prefill128 feature string.
Malformed/out-of-range values encode 63 and are rejected after agreement.
Different counts refuse with PrefillSettingMismatch. All-invalid ranks refuse
BadPrefillSlice. Off retains the existing agreement value and feature string.
No additional agreement words, CUDA modules, kernels or checkpoint formats.

## Custody

Only one decode round runs at a park. The host does not admit requests, drain
cancellations, pump restores, place rows, snapshot or finish jobs there. Tokens
are streamed, but terminal backend releases and host completion are deferred to
the completed chunk or canceled-chunk boundary. Mid-chunk decode time is
excluded from prefill occupancy and pays the optional share debt. The backend
may safely retain pointers into its fills/streams tables during that interval.

The chunk's h/pre/post/comb/gm, owned-third descriptor, replay floor, shared
attention state, image spans and DSpark taps stay in their existing buffers.
Decode uses Round buffers; its shared Chunk scratch (xsd, invalid, gxh/gz and
group counters) is transient at layer boundaries and is not prompt-alias storage.
DSpark decode absorption uses per-slot rings, leaving the filling slot alone.

Both current and next-chunk Engram threads join before decode. Their ahead rows
are separate, but hashing storage is shared. Verification rehashes the filling
sequence before a subsequent prompt layer can schedule read-ahead from it.
Both verification graphs and drafter pass graphs run eagerly while a chunk is
open; no graph capture or graph vote can enter that interval.

Any non-cancellation error with an open chunk takes the fatal TP-world path
before joining cleanup. Cancellation of the filling request is checked even
inside its first chunk and broadcasts the abort. No partial chunk is offered as
a completed cache mark.

## Qualification

CPU: zig build test-prefill-slices test-prefill128 test-native-stepped
test-prefill-admission test-prefill-owned test-drafter-pairfault test-engram-fault
-j8 --summary all. Follow the repository toolchain/cache instructions.

Parent-only Spark plan: identical TP3 binaries, production fatbins, YIELD=1,
AC=1, COMPUTE_ROWS=8192, C16/R48, same DSpark/sample settings. Compare slice 0,
5, 10 with one 128K cold prompt while four fixed decoder streams run. Record
per-token timestamps, max/p99 inter-token gaps for each decoder, long-prompt
TTFT and total prefill wall time. Repeat three runs per setting. Compare exact
greedy token IDs for every request. Also check images, replay-boundary tails,
cancel during first/middle chunks, Engram faults and rank-count mismatch.
Run graph-on/off and DSpark-on/off controls. KV-shard-on must remain whole-chunk.
Seeded sampled traffic is a separate state-safety check, not claimed bit parity
across arbitrary scheduling. Budget 30–60 minutes plus a sustained mixed run.

CPU tests and byte preservation do not establish GPU arithmetic parity or a
latency improvement. Release v1.2 runs with slices on.
