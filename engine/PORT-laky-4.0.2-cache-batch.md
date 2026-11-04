# Laky 4.0.2 bounded checkpoint work on TP3

Both switches are opt-in and default off:

- TF_CACHE_BATCH_DEAD=1: the existing keep scan releases contiguous groups
  of 2-16 already-dead, unborrowed entries.
- TF_CACHE_FIRST_CONTINUATION=1: successful, fresh, unshared terminal
  checkpoints receive a weak metadata hint. This uses the existing
  TF_CACHE_ADAPTIVE_PREFIX=1 history and TF_KV_PRESSURE_PLANNER=1 path.

## Ownership and protocol

The core retains every entry and all charged bytes until the backend returns.
Scoped borrowers split a batch and retain their existing retirement handling.
Durable-tier release runs before any buffer free; lineage and media allocations
are freed exactly once with the other host fields.

Our outer opcode 11 already means RDMA idle. Batch release therefore uses
opcode 16, not Laky's 11. The coordinated keyed-draft port reserves opcode 17.
A payload is count:u32 followed by 2-16 unique nonzero id:u64 values, after
the existing kind:u8, step:u64 header. The follower validates the whole
payload and every kept ID before freeing anything.

The head first preserves existing spill capture guards and sharded-planner
metadata cleanup. Those prerequisites can send their own existing metadata
frames; they are not removed or folded into unsafe local frees. The actual
buffer-release transaction sends one frame to each of both TP3 followers,
frees locally, then requires both exact (kind, step, count) receipts before
retiring snapshot objects. No per-ID release frames, deferred reclamation,
new allocation pool, or new CUDA kernels are introduced. A send/receipt
failure ends the rank group through the existing fatal custody path.
Idle RDMA parking remains available after a completed release.

## First-continuation policy

The hint never increments real restore evidence. It expires after 32
save/restore events and ignores odd endpoints, stale snapshots and loading
states. It runs after the existing production placement stages, scans at most
32 existing boundary candidates, and shares the planner's evaluation cap and
2 ms monotonic deadline, including any no-copy retry.

A candidate cannot increase lost rows, add a new proven-reuse victim, or
increase learned reconstruction cost by more than 5%. Existing protected,
borrowed, pinned and loading checks remain in the shared evaluator. A plan
that already moves storage is not refined, and this stage never adds a move.
Timestamped checkpoint events and numeric first-continuation/drop timing
diagnostics support the parent's mixed-workload comparison.

## Agreement and compatibility

Both cache flags occupy bits 48-49 of word 6 in Model.agree, on every rank
before dependent allocation. Their mask does not conflict with the
coordinated keyed-draft port's word-6 bit 63. Existing prefill bits remain
unchanged.

Persistent feature bytes gain cache_batch=1;, cache_batch=2;, or
cache_batch=3; only when enabled. Off adds no bytes and preserves existing
copied-record fingerprints. Tests retain three copied production-config
identities with zero reprocessing and distinguish all three experimental
arms. No live ledger, stored row, cache directory or schema is deleted.

## CPU checks and parent qualification

zig build test-cache-batch covers the upstream dead-batch/first-hint tests,
planner safety, both TP3 follower sockets, malformed/failing receipts,
borrowed accounting, startup bits and copied fingerprints.
Also run test-context-capacity, test-native-custody,
test-native-placement, test-prefill128 and test.

The cross-build output is engine/build/out-cache-batch. It uses the existing
SM121 fatbins unchanged. No new kernels need staging.

Only the owner parent runs Spark qualification: compare the original build
and this build with flags off, then batch-only, first-only and both flags.
Keep all other model, cache, graph, C16/R48 and sampling settings fixed.
Use separate experimental namespaces without deleting existing caches.
Require exact greedy token/output parity, then compare short one-off prompts
mixed with returning 8K-64K conversations. Record cached_tokens, dead_at_keep,
per-batch send/free/receipt latency, decode tok/s, TTFT, CPU placement time and
memory/accounting. Sampled temperature-1.0 traffic is a separate seeded arm,
not a greedy-parity substitute.

The brief names profiles/v1.1.json, but this checkout contains
serving/src/chimera_sparks/assets/profiles/v1.json; it enables the adaptive
cache and pressure planner. The owner must verify the actual serving profile
when staging. This port has not deployed, restarted, or tested on any Spark.

Source: supplied Laky v402 diffs for prompt_cache, checkpoint_policy,
kv_pressure, native/cuda and dsv41 native/lanes; see DS41F-4.0.2.md,
Bounded checkpoint work. Greedy-head changes in the shared upstream
lanes diff are intentionally outside this port.
