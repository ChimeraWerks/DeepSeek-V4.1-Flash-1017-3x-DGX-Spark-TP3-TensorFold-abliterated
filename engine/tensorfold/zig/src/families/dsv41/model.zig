//! One TP rank of DeepSeek-V4.1-Flash loaded for serving: the weights, kernels, caches and buffers the layer gate sets
//! up (layer_main.zig), for a pool of draft.max_streams streams (MultiDecoder's slots: each stream's window rings,
//! positional stores and drafter rings in its slot, its compressed rows in its extent of one shared plane), and the
//! primitives a lane round runs on them. Both ranks run the same primitives in the same order (lanes.zig sends them
//! to rank 1): a prompt's chunks into its stream's slot and extent (fill), one round over every stream's window and
//! the windows' taps absorbed (verify), and the drafter's batched pass (pass). Every primitive is the gates' own code
//! path (prompt.zig, round.zig, draft.zig) with no checks.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const rank_cache = @import("rank_cache.zig");
const weights = @import("weights.zig");
const prompt = @import("prompt.zig");
pub const prefill128 = @import("prefill_128.zig");
const prompt2d = @import("prompt2d.zig");
const tri_attn = @import("tri_attn.zig");
const round = @import("round.zig");
const sampling = @import("sampling.zig");
const graph_vote = @import("graph_vote.zig");
const greedy_head = @import("decode_greedy_head.zig");
const draft = @import("draft.zig");
const engram = @import("engram.zig");
const engram_io = @import("engram_io.zig");
const engram_aio = @import("engram_aio.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const ops_mod = @import("ops.zig");
const keyed_draft = @import("keyed_draft.zig");
const exact = @import("exact.zig");
const cublas = @import("cublas.zig");
const comm_mod = @import("comm.zig");
const link = @import("link.zig");
const ring2d = @import("ring2d.zig");
const rdma = @import("rdma.zig");
const pool_mod = @import("pool.zig");
const shard = @import("shard.zig");
const shard_ops = @import("shard_ops.zig");

/// Host scheduling policy is runtime-only: never change the durable spill namespace.
fn roomWaitAgreement(leader: u64, follower: u64) !void {
    if (leader != follower) return error.RestoreRoomSettingMismatch;
}

test "restore room agreement checks both TP3 followers and exact wait bounds" {
    for ([_][3]u64{ .{ 0, 0, 0 }, .{ 30000, 30000, 30000 } }) |values| {
        for (values[1..]) |v| try roomWaitAgreement(values[0], v);
    }
    for ([_][3]u64{ .{ 30000, 0, 30000 }, .{ 30000, 30000, 0 }, .{ 30000, 30000, 29999 } }) |values| {
        var mismatches: usize = 0;
        for (values[1..]) |v| roomWaitAgreement(values[0], v) catch { mismatches += 1; };
        try std.testing.expectEqual(@as(usize, 1), mismatches);
    }
}

/// Historical cache/snapshot quantum. Compute is independently selected by TF_DS_PREFILL_AC.
pub const cache_quantum = prefill128.cache_quantum;
pub const chunk_rows = cache_quantum;

pub const SliceArgs = struct { start: usize, n: usize, replay: usize };
pub const SliceEnd = enum { more, head, encoder_only };
pub const SliceResume = struct {
    live: bool = false,
    next: usize = 0,
    args: SliceArgs = .{ .start = 0, .n = 0, .replay = 0 },
    shared: prompt.Shared = .{},
    floor: usize = 0,
    kv_done: ?usize = null,
    ahead: ?std.Thread = null,

    pub fn validate(rs: *const SliceResume, args: SliceArgs, lo: usize, hi: usize, layers: usize, seq_len: usize) !void {
        if (hi <= lo or hi > layers or args.n == 0 or args.start > seq_len or args.n != seq_len - args.start) return error.BadSlice;
        if (lo == 0) {
            if (rs.live) return error.OpenPrefillChunk;
        } else if (!rs.live or rs.next != lo or !std.meta.eql(args, rs.args)) return error.BadSlice;
    }
};

pub fn sliceHi(layers: usize, per: usize, lo: usize) usize {
    return if (per == 0) layers else lo + @min(per, layers - lo);
}
/// The pool's streams (--parallel 4).
pub const max_streams = draft.max_streams;
/// The pool's streams when unset (--parallel 4).
pub const max_streams_default = draft.default_streams;
/// The RDMA ring's slot (the served 4.25 MiB: a 16-row round's logits half fits).
pub const ring_bytes = 4456448;

/// TP3 remains opt-in until its GPU/numerical campaign passes. Both HTTP and
/// follower/lane entrypoints share this policy, before weight loading.
pub fn checkWorldMode(world: u32, experimental: ?[]const u8) !void {
    if (world == 2 or world == 4) return;
    if (world != 3) return error.NotPortedYet;
    if (!std.mem.eql(u8, experimental orelse "", "1")) return error.ExperimentalTP3Required;
}

fn experimentalTP3() ?[]const u8 {
    return if (std.c.getenv("TF_DS_EXPERIMENTAL_TP3")) |v| std.mem.span(v) else null;
}

test "experimental TP3 requires exact opt-in while legacy worlds remain available" {
    try checkWorldMode(2, null);
    try checkWorldMode(4, null);
    try checkWorldMode(3, "1");
    for ([_]?[]const u8{ null, "", "0", "true", "01", "1 " }) |value|
        try std.testing.expectError(error.ExperimentalTP3Required, checkWorldMode(3, value));
    for ([_]u32{ 0, 1, 5, 6 }) |world|
        try std.testing.expectError(error.NotPortedYet, checkWorldMode(world, "1"));
}

/// Avoid silently falling back to NCCL for a full TP3 verification window's
/// head logits. Pinned host send/receive buffers scale with slots*(world+1).
pub fn tp3RingBytes(local_vocab: usize, rows: usize) !usize {
    if (local_vocab == 0 or local_vocab % 128 != 0 or rows < round.default_rows or rows > round.max_rows) return error.BadRoundRows;
    const values = try std.math.mul(usize, local_vocab, rows);
    const bytes = try std.math.mul(usize, values, 4);
    return @max(ring_bytes, bytes); // divisible by64 because vocab is128-aligned
}

test "TP3 RDMA capacity includes C8 K5 head without reducing legacy floor" {
    try std.testing.expectEqual(ring_bytes, try tp3RingBytes(43136, 16));
    const bytes = try tp3RingBytes(43136, 48);
    try std.testing.expectEqual(@as(usize, 8282112), bytes);
    try std.testing.expect(bytes >= 48 * 43136 * 4);
    try std.testing.expectEqual(@as(usize, 0), bytes % 64);
    try std.testing.expectEqual(@as(usize, 61210624), (bytes - ring_bytes) * 4 * (3 + 1));
    try std.testing.expectError(error.BadRoundRows, tp3RingBytes(43136, 49));
    try std.testing.expectError(error.BadRoundRows, tp3RingBytes(43135, 48));
}

pub const Options = struct {
    model_dir: []const u8, // config.json
    cache_dir: []const u8, // the lane's per-rank weight file (TF_DS_RANK_CACHE)
    kit_dir: []const u8, // a gate dir's layout: aot/, cubins/, rope-{plain,compressed}-{cos,sin}.f32, engram.json
    aot_dir: ?[]const u8 = null, // the recorded Triton set (aot.json and its cubins; null: kit_dir/aot)
    rank: u32,
    world: u32,
    master: [4]u8,
    port: u16,
    engram_dir: ?[]const u8 = null, // the Engram tables (none: a checkpoint without Engram layers)
    token_map: ?[]const u8 = null, // the compressed token map (the lane's JSON cache)
    pool: usize = 1 << 18, // positions of the streams' shared plane (their extents together)
    // the positions one conversation may reach (its prompt and reply: the RoPE table's rows, the prompt chunks' and
    // rounds' buckets); null: rank 0's (a follower adopts it at open), or the pool's on rank 0 (today's engine)
    window: ?usize = null,
    // TF_DS_KV_SHARD (shard.zig): owner-only compressed rows over the TP2 ranks; null: read from the environment on
    // every rank alike (TF_DS_KV_SHARD, TF_DS_KV_SHARD_DECODE), as TF_DS_IDX_SPLIT is
    kv_shard: ?shard.Mode = null,
    drafts: bool = true, // load the DSpark drafter
    rdma_devices: ?[]const u8 = null, // the decode gathers over RDMA on these devices (comma separated), else NCCL: TP2's
    // ring (rdma.zig), or at world 4 the 2D split's decode-size exchanges over its rings (ring2d.zig)
    rdma_kernels: ?[]const u8 = null, // world 4: the rings' kernels' image (rdma_gather.cu's fatbin; null: the embedded one)
    graphs: bool = false, // the rounds' stretches as CUDA graphs (round.Graphs)
    graph_cap: usize = 0, // round and drafter keys kept per cache (TF_DS_GRAPH_CAP / --graph-cap; 0: all)
    side: bool = false, // the mixes' side work on a stream of its own (round.Round.useSide)
    prefetch: bool = false, // the paced L2 prefetch (round.Round.usePrefetch)
    engram_aio: bool = false, // a round's Engram reads by Linux AIO on O_DIRECT (else the reader pool)
    arena_bytes: usize = 0, // the device arena's first block (0: it grows from 256 MiB blocks as the buffers ask)
    round_rows: usize = round.default_rows, // a round's rows at most (TF_DS_ROUND_ROWS; the four-node lane's 48)
    streams: usize = draft.default_streams, // the pool's streams (--parallel; the four-node lane's 16)
    round_ms: ?[]const f64 = null, // the lane core's round costs by rows (TF_DS_ROUND_MS; null: the served ROUND_MS)
    vision_bias: ?[]const u8 = null, // gate_bias_vl.safetensors (null: the kit's vision/): the image-span routing bias
};

pub const Model = struct {
    draft_settings: @import("draft_settings.zig").Settings = .{},
    gpa: std.mem.Allocator,
    host: std.heap.ArenaAllocator, // allocations that live as long as the model
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    cfg: Config,
    rank: u32,
    world: u32,
    streams: usize, // the pool's streams (Options.streams)
    round_ms: []const f64, // the lane core's round costs by rows (Options.round_ms, else draft.round_ms)
    pool_cap: usize,
    cache_dir: std.Io.Dir,
    cache_file: std.Io.File,
    cache_stamp: std.Io.File.Stat,
    ix: rank_cache.Index,
    w: weights.Weights,
    fan: prompt2d.Fan,
    comm: comm_mod.Comm,
    rdma_mod: ?cuda.Module = null, // the RDMA kernels (TP2's ring or world 4's rings)
    ring: ?*rdma.Ring = null,
    set: cuda.aot.Set,
    pf: exl3_prefill.Kernels,
    lg: exl3_linear.Kernels,
    exk: exl3_experts.Kernels,
    ops: ops_mod.Ops,
    ex: exact.Exact,
    rope_buf: cuda.DeviceBuffer,
    arena: prompt.Arena,
    eng: prompt.Engine,
    ch: prompt.Chunk,
    caches: prompt.Caches,
    blas: cublas.Blas,
    rings: []u64, // each layer's window rings, max_streams slots
    ring_view: []u64, // a filling stream's slot of them
    rd: round.Round,
    graphs: ?round.Graphs = null,
    pgraphs: ?draft.PassGraphs = null, // the drafter pass as CUDA graphs (with graphs)
    gvote: ?graph_vote.Control = null, // both graph caches' cross-rank vote on a miss (graph_vote.zig)
    dpool: ?draft.Pool = null,
    dr: ?draft.Drafter = null,
    tables: ?engram_io.Tables = null,
    epool: ?*engram_io.Pool = null,
    aio: ?*engram_aio.Aio = null,
    paio: ?*engram_aio.Aio = null, // the prompt chunks' Engram reads, on a ring of their own
    eh: ?prompt.EngramHost = null,
    host_pos: []i64,
    ids64: []i64,
    vocab: usize,
    logits: []f32, // round.max_rows rows of logits on the host
    amax: u64, // u32 [round.max_rows]: a round's greedy tokens on the device (roundArgmax)
    amax_host: [round.max_rows]u32 = undefined,
    decode_greedy_head: bool = false,
    decode_sample_head: bool = false,
    sample_output: usize = 0,
    greedy_output: greedy_head.Output = .{}, // published only after verifyHead fences
    cand: u64, // optional GPU top candidates: [vals f32, ids i64, flags i32]
    cand_vals: [round.max_rows * sampling.max_candidates]f32 = undefined,
    cand_ids: [round.max_rows * sampling.max_candidates]i64 = undefined,
    cand_flags: [round.max_rows]i32 = undefined,
    // the stream a fill is running: its slot and its view of the caches (its extent from base)
    fill_slot: usize = 0,
    // the prompt a fill's chunks come from (the caller's, set for the fill) and the thread reading the next chunk's
    // Engram rows during this one (fillChunk), joined before the prompt can change (joinAhead)
    fill_seq: []const i32 = &.{},
    ahead: ?std.Thread = null,
    prefill: prefill128.Flags = .{},
    rs: SliceResume = .{},
    compute_rows: usize = chunk_rows,
    prompt_cap: usize = 0,
    // Rank zero sends the next exact chunk size, including its cache marks.
    next_engram_rows: usize = 0,
    engram_audit: bool = false,
    view: prompt.Caches = undefined,
    // the filling prompt's image spans and their rows (setImages; cleared by fillBegin)
    spans: std.ArrayList(prompt.Span) = .empty,
    img_buf: ?cuda.DeviceBuffer = null,
    // profiling (lanes' --profile): the last round's forward and absorb, synchronized apart (ns)
    prof: bool = false,
    timer: ?*round.PhaseTimer = null, // a profile's GPU time by phase (eager rounds)
    ptimer: ?*round.PhaseTimer = null, // the same for prompt chunks (fills), a chunk a 'round'
    t_enqueue: u64 = 0, // the host's time to issue the forward (the GPU idle before it)
    // Model.useClock: the rounds' GPU clock gaps summed (ns; gap i = clock i + 1 - clock i: the forward's start to
    // the first stretch, then each stretch, then the wait before the next), and the rounds counted
    clk_ph: [256]u64 = @splat(0), // by the phase of the clock that ends it
    clk_cnt: [256]u64 = @splat(0),
    clk_rounds: u64 = 0,
    t_forward: u64 = 0,
    t_absorb: u64 = 0,
    // world 4: the 2D split's exchanges (prompt2d.zig, TP4's), with the RDMA rings when asked
    two: ?prompt2d.Two = null,
    rdma_rings: ?ring2d.Rings = null,
    // the pool / window split and TF_DS_KV_SHARD (open): the positions one stream's extent may take at most (a part
    // sharded; the request check's bound), the single-conversation limit reported (shard.plan), the shard's kernels
    request_cap: usize = 0,
    window_cap: usize = 0,
    kvs: ?shard_ops.KvShard = null,

    /// Loads rank o.rank's weights from the lane's rank cache, links the ranks (rank 0 listens on o.port), opens NCCL
    /// and sets up every buffer. Needs `ctx` current on this thread.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, o: Options) !*Model {
        const keyed_image = try keyed_draft.selectImage(keyed_draft.enabled(), cuda.kernels.dsv41_keyed);
        try checkWorldMode(o.world, experimentalTP3());
        if (o.rank >= o.world) return error.UnsupportedRank;
        if (o.round_rows < round.default_rows or o.round_rows > round.max_rows) return error.BadRoundRows;
        if (o.world == 3) std.debug.print("EXPERIMENTAL TP3: GPU/numerical qualification pending; rank {d}\n", .{o.rank});
        const m = try gpa.create(Model);
        const custody = cuda.context_health.Custody.init(m) catch |err| { gpa.destroy(m); return err; };
        const cleanup = cuda.context_health.cleanupMark();
        errdefer if (cuda.context_health.cleanupReleased(cleanup)) custody.release() else custody.retain();

        errdefer if (cuda.context_health.cleanupReleased(cleanup)) gpa.destroy(m);
        m.* = undefined;
        m.gpa = gpa;
        // m.* = undefined skips field defaults: the read-ahead thread and its sequence must start empty, since
        // fillChunk and joinAhead read them before any prompt pass sets them
        m.ahead = null;
        m.fill_seq = &.{};
        if (std.c.getenv("TF_DS_REPLAY_FLOOR")) |v| prompt.replay_floor = std.mem.eql(u8, std.mem.span(v), "1"); // every rank alike
        prompt.short_continuation_gather = if (std.c.getenv("TF_DS_CACHE_SHORT_CONTINUATION")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false; // same opt-in on every rank
        prompt.owned_on = if (std.c.getenv("TF_DS_PREFILL_OWNED")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false; // every rank alike (its collectives pair up)
        prompt.bf16_wire_on = prompt.owned_on and if (std.c.getenv("TF_DS_PREFILL_BF16_WIRE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false; // every rank alike
        prompt.owned_router_on = prompt.owned_on and if (std.c.getenv("TF_DS_PREFILL_OWNED_ROUTER")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false; // every rank alike
        m.rs = .{};
        m.prefill = prefill128.Flags.fromEnv();
        try m.prefill.validate();
        m.compute_rows = m.prefill.computeRows();
        m.next_engram_rows = 0;
        m.engram_audit = prefill128.enabled("TF_DS_PREFILL_ENGRAM_AUDIT");
        m.spans = .empty;
        m.img_buf = null;
        m.host = std.heap.ArenaAllocator.init(gpa);
        errdefer if (cuda.context_health.cleanupReleased(cleanup)) m.host.deinit();
        const a = m.host.allocator();
        m.io = io;
        m.ctx = ctx;
        m.rank = o.rank;
        m.world = o.world;
        if (o.streams == 0 or o.streams > max_streams) return error.BadStreams;
        m.streams = o.streams;
        m.round_ms = o.round_ms orelse &draft.round_ms;
        if (m.round_ms.len == 0) return error.BadRoundMs;
        m.pool_cap = o.pool; // (agree, below, compares it with the other rank's)
        m.fill_slot = 0;
        m.prof = false;
        m.timer = null;
        m.ptimer = null;
        m.t_enqueue = 0;
        m.t_forward = 0;
        m.t_absorb = 0;
        m.dpool = null;
        m.dr = null;
        m.tables = null;
        m.epool = null;
        m.aio = null;
        m.paio = null;
        m.eh = null;
        m.rdma_mod = null;
        m.ring = null;
        m.graphs = null;
        m.pgraphs = null;
        m.gvote = null;
        m.cand = 0;
        m.decode_greedy_head = greedy_head.enabled(envOf("TF_DS_DECODE_GREEDY_HEAD"));
        m.decode_sample_head = greedy_head.enabled(envOf("TF_DS_DECODE_SAMPLE_HEAD"));
        m.sample_output = 0;
        m.greedy_output = .{};
        m.two = null;
        m.rdma_rings = null;
        m.kvs = null;
        const d = ctx.d;
        m.cfg = try Config.read(a, io, o.model_dir);
        try m.cfg.checkWorld(o.world);
        const c = &m.cfg;
        m.stream = try cuda.Stream.init(d, true);
        errdefer if (cleanupAllowed(m, cleanup)) m.stream.deinit();
        // Later acquisitions install their own cleanup above this defer. Each
        // failure must first establish quiescence, including partial uploads.


        // the weights (and the DSpark blocks) from the lane's rank cache
        m.cache_dir = try std.Io.Dir.cwd().openDir(io, o.cache_dir, .{ .iterate = true });
        errdefer if (cleanupAllowed(m, cleanup)) m.cache_dir.close(io);
        const cache_name = try rank_cache.find(a, io, m.cache_dir, o.rank, o.world);
        m.cache_file = try m.cache_dir.openFile(io, cache_name, .{});
        errdefer if (cleanupAllowed(m, cleanup)) m.cache_file.close(io);
        m.cache_stamp = try m.cache_file.stat(io);
        m.ix = try rank_cache.openFile(gpa, io, m.cache_file);
        errdefer if (cleanupAllowed(m, cleanup)) m.ix.deinit();
        const sp = try prompt2d.split(o.rank, o.world);
        m.w = try weights.load(gpa, d, .{ .cache = .{ .file = m.cache_file, .index = &m.ix, .io = io } }, c.*, sp, o.drafts);
        errdefer if (cleanupAllowed(m, cleanup)) m.w.deinit();
        if (o.drafts and m.w.dspark == null) return error.NoDrafter;

        // the ranks' link and NCCL (rank 0 listens: rank 1 retries its connect while rank 0 loads)
        m.fan = try prompt2d.Fan.open(io, o.master, o.port, o.rank, o.world);
        errdefer if (cleanupAllowed(m, cleanup)) m.fan.close();
        m.comm = try m.fan.comm(o.rank, o.world);
        var comm_ready = true;
        errdefer if (cleanupAllowed(m, cleanup) and comm_ready) m.comm.deinit();

        // the pool, the window and the shard: every rank's compared first thing after NCCL, before anything rank-local
        // can fail and before any buffer depends on them (a follower without a window takes rank 0's). A setting this
        // rank cannot parse is refused after the gather, so the other rank refuses with it instead of waiting. Then the
        // plan (shard.plan: the effective single-conversation limit), the same on every rank from the same numbers.
        const want_mode: anyerror!shard.Mode = if (o.kv_shard) |k| k else shard.Mode.parse(envOf("TF_DS_KV_SHARD"), envOf("TF_DS_KV_SHARD_DECODE"));
        // the pool and window are 0008's (pool.zig): rank 0's pool is TF_DS_POOL over --context (native.open), a
        // follower's its --pool; a window is --context on rank 0 and TF_DS_WINDOW on a follower (else rank 0's)
        const want_pool: anyerror!usize = if (o.pool == 0) error.BadPool else o.pool;
        const want_window: anyerror!?usize = windowOf(o);
        const ag = try agree(m, o, want_mode, want_pool, want_window);
        if (m.prefill.bits() != 0 and o.world != 3) return error.Prefill128RequiresTP3;
        try m.prefill.validateRagged();
        tri_attn.configureMasked(m.prefill.k, o.rank);
        prompt.embed_rows_on = m.prefill.d;
        prompt.index_rows_on = m.prefill.e;
        prompt.front_rows_on = m.prefill.f;
        exl3_experts.prompt_ragged_mma = m.prefill.w;
        exl3_experts.prompt_ragged_range16 = m.prefill.w_range16;
        exl3_experts.prompt_shared_split = m.prefill.shared_split;
        exl3_experts.prompt_ragged_rank = o.rank;
        // after the agreement (itself an NCCL all-gather every rank makes alike)
        m.comm.direct = directGather();
        if (m.comm.direct) std.log.info("dsv41: rank {d}: TF_DS_DIRECT_GATHER on: all-gathers as direct sends to every peer", .{o.rank});
        const mode = ag.mode;
        const pool = ag.pool;
        const window = ag.window;
        m.pool_cap = pool;
        if (mode != .off and ((o.world != 2 and o.world != 3) or sp.pair != null)) return error.KvShardNeedsTwoRanks;
        if (o.world == 3 and mode != .off and mode != .fetch) return error.Tp3FetchOnly;
        const pl = try shard.plan(pool, window, o.round_rows, o.world, mode != .off);
        // (with the spill tier: spillLayout is owner_sharded with the part size, so its compat digest and block lineage
        // carry the owner layout and the part salt; spill_state packs a sharded plane's part-local rows)
        m.request_cap = pl.extent;
        m.window_cap = pl.window;
        m.prompt_cap = if (m.prefill.af) try prefill128.workspaceTokens(m.request_cap, pl.window, round.max_rows) else m.request_cap;
        // TP2: the decode gathers' RDMA ring (both ranks connect before either returns)
        if (sp.pair == null and o.rdma_devices != null) {
            const list = o.rdma_devices.?;
            if (!cuda.kernels.available) return error.NoKernelImages;
            m.rdma_mod = try cuda.Module.load(d, cuda.kernels.dsv41_rdma);
            var devices: std.ArrayList([]const u8) = .empty;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |dev| try devices.append(a, dev);
            const bytes = if (o.world == 3) try tp3RingBytes(m.w.head.n, o.round_rows) else ring_bytes;
            m.ring = try openRingSized(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, &m.comm, m.stream, bytes);
        }
        errdefer if (cleanupAllowed(m, cleanup)) if (m.rdma_mod) |*x| x.unload();
        errdefer if (cleanupAllowed(m, cleanup)) if (m.ring) |r| r.destroy(gpa);

        // kernels: the recorded Triton set, the served extension cubins, the torch-op images
        m.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, o.aot_dir orelse (if (m.prefill.k) envOf("TENSORFOLD_CUDA_KERNELS") else null) orelse try std.fs.path.join(a, &.{ o.kit_dir, "aot" }));
        errdefer if (cleanupAllowed(m, cleanup)) m.set.deinit();
        m.pf = try exl3_prefill.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear.cubin"));
        errdefer if (cleanupAllowed(m, cleanup)) m.pf.unload();
        m.lg = try exl3_linear.Kernels.load(d, try readKit(a, io, o.kit_dir, "linear_grouped.cubin"));
        errdefer if (cleanupAllowed(m, cleanup)) m.lg.unload();
        m.exk = try exl3_experts.Kernels.load(d, try readKit(a, io, o.kit_dir, "experts.cubin"), try readKit(a, io, o.kit_dir, "experts_cb.cubin"), exl3_linear.codebook_mul1);
        errdefer if (cleanupAllowed(m, cleanup)) m.exk.unload();
        if (m.prefill.w) {
            const path = std.c.getenv("TF_DS_PREFILL_RAGGED_CUBIN") orelse return error.MissingRaggedCubinPath;
            const blob = try std.Io.Dir.cwd().readFileAllocOptions(io, std.mem.span(path), a, .limited(1 << 28), .@"16", null);
            try m.exk.loadRagged(d, blob);
        }
        if (!cuda.kernels.available) return error.NoKernelImages;
        const sample_cand = std.mem.eql(u8, envOf("TF_DS_SAMPLE_CAND") orelse "", "1");
        if (sample_cand and cuda.kernels.dsv41_sample_cand.len == 0) return error.NoSampleCandidateKernel;
        m.ops = try ops_mod.Ops.load(d, cuda.kernels.torch_pointwise, cuda.kernels.torch_movement, cuda.kernels.dsv41_ops);
        errdefer if (cleanupAllowed(m, cleanup)) m.ops.unload();
        if (m.decode_sample_head and (!sample_cand or o.world != 3 or m.two != null)) m.pairFault(error.GreedyHeadRankFault);
        if (m.decode_greedy_head or m.decode_sample_head) {
            const path = envOf("TF_DS_REDUCED_HEAD_FATBIN") orelse m.pairFault(error.NoGreedyHeadKernel);
            const blob = std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 28), .@"16", null) catch |err| m.pairFault(err);
            m.ops.enableGreedyHead(d, blob) catch |err| m.pairFault(err);
        }
        if (sample_cand) try m.ops.enableCandidates(d, cuda.kernels.dsv41_sample_cand);
        if (keyed_image) |image| try m.ops.enableKeyed(d, image);
        m.ex = try exact.Exact.load(d, cuda.kernels.dsv41_torch);
        errdefer if (cleanupAllowed(m, cleanup)) m.ex.unload();

        // the RoPE tables (the served build's bits: torch CPU pow and polar), only the agreed conversation window plus decode reserve
        const half = c.rope_dim / 2;
        const rope_rows = pool_mod.ropeRows(window, round.max_rows);
        m.rope_buf = try cuda.DeviceBuffer.alloc(d, 4 * rope_rows * half * 4);
        errdefer if (cleanupAllowed(m, cleanup)) m.rope_buf.free();
        {
            const part = try gpa.alloc(u8, rope_rows * half * 4);
            defer gpa.free(part);
            for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
                var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ o.kit_dir, name }), .{});
                defer f.close(io);
                if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // a table shorter than the conversation prefix
                try m.rope_buf.upload(j * part.len, part);
            }
        }
        const tbl = rope_rows * half * 4;
        const rope: prompt.Rope = .{ .cos = m.rope_buf.ptr, .sin = m.rope_buf.ptr + tbl };
        const rope_c: prompt.Rope = .{ .cos = m.rope_buf.ptr + 2 * tbl, .sin = m.rope_buf.ptr + 3 * tbl };

        m.arena = try prompt.Arena.init(d, o.arena_bytes);
        errdefer if (cleanupAllowed(m, cleanup)) m.arena.deinit();
        // NCCL may retain pointers into the pool. Release it before any pool
        // block even on construction failure; the earlier defer covers pre-pool.
        errdefer if (cleanupAllowed(m, cleanup)) {
            m.comm.deinit();
            if (cuda.context_health.cleanupReleased(cleanup)) comm_ready = false;
        };
        // world 4 (layer_main's setup): the 2D exchanges' buffers, and with rdma_devices their rings (whose infos go
        // round over NCCL world 4)
        errdefer if (cleanupAllowed(m, cleanup)) if (m.rdma_rings) |*x| x.close(gpa);
        if (sp.pair != null) {
            m.two = try prompt2d.Two.init(c, &m.w, &m.arena, sp, chunk_rows);
            if (o.rdma_devices) |devs| {
                const image: []const u8 = if (o.rdma_kernels) |path|
                    try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(1 << 26), .@"16", null)
                else if (cuda.kernels.available) cuda.kernels.dsv41_rdma else return error.NoRdmaKernels;
                m.rdma_mod = try cuda.Module.load(d, image);
                var devices: std.ArrayList([]const u8) = .empty;
                var dit = std.mem.splitScalar(u8, devs, ',');
                while (dit.next()) |x| try devices.append(a, x);
                m.rdma_rings = try ring2d.Rings.open(gpa, d, try rdma.Kernels.load(m.rdma_mod.?), devices.items, o.rank, .{ .max_bytes = ring_bytes, .gid_index = try rdma.servedGidIndex() }, &m.comm, m.stream);
                m.two.?.rings = &m.rdma_rings.?;
            }
        }
        m.eng = .{ .d = d, .s = m.stream, .t = .{ .set = &m.set, .stream = m.stream }, .blas = undefined, .comm = &m.comm, .pf = &m.pf, .lin = &m.lg, .ex = &m.exk, .ops = &m.ops, .exact = &m.ex, .c = c, .w = &m.w, .world = sp.world, .plain = rope, .compressed = rope_c, .ring = m.ring, .two = if (m.two) |*t| t else null };
        m.eng.round_rows = o.round_rows;
        m.eng.prompt_alias = m.prefill.prompt_alias;
        m.ch = try prompt.Chunk.init(&m.eng, &m.arena, m.compute_rows, m.prompt_cap);
        if (m.prefill.ac or m.prefill.af) std.debug.print("prefill_workspace_setup: rank={d} pool_tokens={d} request_tokens={d} prompt_tokens={d} chunk_rows={d} kv_quantum={d} mask_bytes={d} contract=PASS\n", .{ m.rank, m.pool_cap, m.request_cap, m.prompt_cap, m.compute_rows, cache_quantum, m.compute_rows * m.ch.pool_ms });
        errdefer if (cleanupAllowed(m, cleanup)) if (m.kvs) |*x| x.k.unload(); // (registered before the load: nothing between them leaks it)
        if (mode != .off) {
            // TF_DS_KV_SHARD: the fetch kernels, the parts (a trash row a round row), owner-only planes
            if (!cuda.kernels.available) return error.NoKernelImages;
            const sh = try shard.Shard.init(o.rank, o.world, pool, round.max_rows);
            m.kvs = .{ .sh = sh, .mode = mode, .k = try shard_ops.Kernels.load(d, cuda.kernels.dsv41_shard) };
            m.eng.kv = &m.kvs.?;
            m.caches = try prompt.Caches.initShard(&m.eng, &m.arena, pool, m.streams, sh);
            try m.ch.useShard(&m.eng, &m.arena, prompt.prompt_fetch_rows, m.request_cap);
        } else m.caches = try prompt.Caches.init(&m.eng, &m.arena, pool, m.streams);
        m.blas = try cublas.Blas.open(m.stream, m.ch.blas_ws);
        errdefer if (cleanupAllowed(m, cleanup)) m.blas.close();
        m.eng.blas = &m.blas;
        m.rings = try a.alloc(u64, c.layers);
        m.ring_view = try a.alloc(u64, c.layers);
        for (m.rings) |*r| {
            r.* = try m.arena.take(m.streams * m.eng.ringBytes());
            try d.check(d.api.cuMemsetD8Async(r.*, 0, m.streams * m.eng.ringBytes(), m.stream.handle), "cuMemsetD8Async");
        }
        m.rd = try round.Round.init(&m.eng, &m.arena, a, m.request_cap);
        if (m.kvs != null) try m.rd.useShard(&m.eng, &m.arena, a);
        m.amax = try m.arena.take(round.max_rows * 4);
        if (m.decode_greedy_head) {
            m.rd.greedy_local = m.arena.take(round.max_rows * 8) catch |err| m.pairFault(err);
            m.rd.greedy_all = m.arena.take(o.world * round.max_rows * 8) catch |err| m.pairFault(err);
            m.rd.greedy_reference = m.arena.take(round.max_rows * 4) catch |err| m.pairFault(err);
        }
        if (sample_cand) m.cand = try m.arena.take(round.max_rows * (sampling.max_candidates * 12 + 4));
        if (m.decode_sample_head) {
            m.rd.sample_local = m.arena.take(round.max_rows * (sampling.max_candidates + 1) * 8) catch |err| m.pairFault(err);
            m.rd.sample_all = m.arena.take(o.world * round.max_rows * (sampling.max_candidates + 1) * 8) catch |err| m.pairFault(err);
            m.rd.sample_reference = m.arena.take(round.max_rows * (sampling.max_candidates * 12 + 4)) catch |err| m.pairFault(err);
            m.rd.sample_vals = m.cand;
            m.rd.sample_ids = m.cand + round.max_rows * sampling.max_candidates * 4;
            m.rd.sample_flags = m.rd.sample_ids + round.max_rows * sampling.max_candidates * 8;
        }
        if (o.graphs) m.graphs = round.Graphs.init(gpa, o.graph_cap);
        if (o.side) try m.rd.useSide(&m.eng);
        if (o.prefetch) try m.rd.usePrefetch(&m.eng);
        if (o.drafts) {
            m.dpool = try draft.Pool.init(&m.eng, &m.arena, m.streams);
            m.dr = try draft.Drafter.init(&m.eng, &m.arena, sp);
            if (o.graphs) {
                m.pgraphs = draft.PassGraphs.init(gpa, o.graph_cap);
                m.dr.?.graphs = &m.pgraphs.?;
            }
        }
        // Every rank must agree before recording a graph and before its first launch (graph_vote.zig). KV shard
        // keeps the vote off; a spill-only pair opts in at pairing with TF_DS_GRAPH_VOTE_SPILL=1.
        if (o.graphs and m.world > 1 and m.kvs == null) {
            const fault_text = envOf("TF_DS_GRAPH_FAULT");
            const fault = if (fault_text) |t| graph_vote.parseFault(t) else null;
            if (fault_text != null and fault == null) std.log.warn("dsv41: TF_DS_GRAPH_FAULT ignored: expected RANK:reserve|ready|launch-first[:COUNT]", .{});
            m.gvote = .{
                .rank = @intCast(m.rank),
                .world = @intCast(m.world),
                .timeout_ns = graph_vote.timeoutNs(envOf("TF_DS_GRAPH_VOTE_MS")),
                .transport = .{ .ctx = m, .send = voteSend, .recv = voteRecv },
                .clock = link.Link.monotonicNs,
                .healthy_ctx = m,
                .healthy = voteHealthy,
                .fault = fault,
            };
            if (m.graphs) |*gr| gr.voter = &m.gvote.?;
            if (m.pgraphs) |*pg| pg.voter = &m.gvote.?;
        } else if (o.graphs and m.world > 1) {
            std.log.warn("dsv41: rank {d}: CUDA graph capture is not cross-rank protected while KV shard is on", .{m.rank});
        }
        try m.stream.synchronize();
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize"); // the setup's legacy-stream copies too

        // Engram on the host: the compressed token map, the multipliers, the tables and 64 readers
        if (o.engram_dir) |edir| {
            const map_text = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
            const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
            const EngramJson = struct { multipliers: [][]i64 };
            const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit_dir, "engram.json" }), a, .limited(1 << 26));
            const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
            var mult: [engram.max_layers][engram.max_ngram]i64 = @splat(@splat(0));
            for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
                mult[l][k] = v;
            };
            m.tables = try engram_io.Tables.open(gpa, io, edir);
            m.epool = try engram_io.Pool.init(gpa, io, 64);
            m.eh = try prompt.EngramHost.init(a, c, engram.Hasher.init(c.*, map, mult), &m.tables.?, m.epool.?, sp.rank, sp.world, m.compute_rows);
            if (o.engram_aio) {
                // a round's reads, 2 x 12 x its rows at most: 1,024 slots for each 16 rows of the round-row setting
                m.aio = try engram_aio.Aio.init(gpa, io, 1024 * ((o.round_rows + 15) / 16));
                m.eh.?.aio = m.aio;
                // a prompt chunk's reads (2 x 12 x 2,048 at most): a ring of 4,096 slots (32 MiB), 1,024 rows a batch
                m.paio = try engram_aio.Aio.init(gpa, io, 4096);
                m.eh.?.paio = m.paio;
            }
        } else if (c.engram_layers.slice().len > 0) return error.NoEngramTables;

        // a vision checkpoint's image routing bias (without it a prompt with images is refused; text runs as ever)
        if (c.vision) {
            const path = o.vision_bias orelse try std.fs.path.join(a, &.{ o.kit_dir, "vision", "gate_bias_vl.safetensors" });
            m.loadVisionBias(path) catch |err| if (o.vision_bias != null or err != error.FileNotFound) return err;
        }

        m.host_pos = try a.alloc(i64, m.compute_rows);
        m.ids64 = try a.alloc(i64, m.compute_rows);
        m.vocab = sp.world * (if (m.two) |t| t.hw[0] + t.hw[1] else m.w.head.n); // 2D: both pairs' vocabulary parts
        m.logits = try a.alloc(f32, round.max_rows * m.vocab);
        if (mode != .off or window != pool) m.notePlan(mode, pl);
        custody.release();
        return m;
    }

    /// The environment variable `name` (null when unset).
    fn envOf(name: [:0]const u8) ?[]const u8 {
        return std.mem.span(std.c.getenv(name) orelse return null);
    }

    /// The window this rank was given: `o.window` (rank 0: --context), else TF_DS_WINDOW (0008: the recipe gives it to
    /// every rank, = CONTEXT; never past the pool), else null (agree takes rank 0's). When both are given they must be one
    /// number, so the RoPE tables and the request bound cannot disagree on a rank.
    fn windowOf(o: Options) !?usize {
        return pickWindow(o.window, o.pool, envOf("TF_DS_WINDOW"));
    }
    fn pickWindow(given: ?usize, pool: usize, env: ?[]const u8) !?usize {
        const set = env != null and std.mem.trim(u8, env.?, " \t\r\n").len != 0;
        const from_env: ?usize = if (set) try pool_mod.window(pool, env) else null;
        if (given) |w| {
            if (from_env) |e| if (e != @min(w, pool)) return error.BadWindow;
            return w;
        }
        return from_env;
    }

    /// What every rank serves once agree has compared them.
    const Agreed = struct { mode: shard.Mode, pool: usize, window: usize };

    /// Every rank's (mode, pool, window, round rows, graphs, their vote protocol and cache cap) gathered and compared
    /// before any buffer depends on them (a graph setting mismatch refuses on every rank). A rank
    /// whose shard switch or pool did not parse sends a sentinel (bad_setting) instead of failing before the gather,
    /// then refuses with its own error after it; the other rank sees the sentinel and refuses too (KvShardMismatch),
    /// so neither waits on the other. A rank whose switch, pool or round rows differ from rank 0's refuses
    /// (KvShardMismatch, on both at once), a window given on both must match, and a rank without one takes rank 0's
    /// (the follower lane has no --context).
    fn directGather() bool {
        return if (std.c.getenv("TF_DS_DIRECT_GATHER")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    }

    fn agree(m: *Model, o: Options, want_mode: anyerror!shard.Mode, want_pool: anyerror!usize, want_window: anyerror!?usize) !Agreed {
        const n: usize = 10;
        const draft_config = @import("draft_settings.zig");
        const want_draft = draft_config.Settings.parse(envOf("TF_DS_DRAFT_CAL"), envOf("TF_DS_FIRST_DRAFT_DEPTH"));
        const draft_words = if (want_draft) |v| v.words() else |_| draft_config.invalid;
        const bad_setting: u64 = 0xff;
        const mine = [n]u64{
            if (want_mode) |x| @backingInt(x) else |_| bad_setting,
            if (want_pool) |x| x else |_| 0,
            if (want_window) |x| x orelse 0 else |_| bad_setting,
            o.round_rows,
            // CUDA graphs and their vote's protocol: a rank voting while another runs eagerly would hang or fail closed
            if (o.graphs) 1 | (@as(u64, graph_vote.version) << 8) else 0,
            if (o.graphs) @intCast(o.graph_cap) else 0, // rank 0 and every follower must use one LRU cap
            // the prompt's collectives pair up only when every rank runs the same kind
            @import("cache_batch.zig").Flags.fromEnv().agreement(@as(u64, @intFromBool(prompt.owned_on)) | @as(u64, @intFromBool(directGather())) << 1 | (m.prefill.bits() << 2) |
                (if (@import("graph_telemetry.zig").enabled()) @import("graph_telemetry.zig").agreement_bit else 0) | keyed_draft.settingBit(keyed_draft.enabled()) | greedy_head.agreement(m.decode_greedy_head) | (if (m.decode_sample_head) greedy_head.sample_agreement_bit else @as(u64, 0)) |
                (@as(u64, @intFromBool(prompt.bf16_wire_on)) << 52) | (@as(u64, @intFromBool(prompt.owned_router_on)) << 53)),
            draft_words[0], draft_words[1],
            @import("engine_api").Spill.roomWaitMs(if (std.c.getenv("TF_SPILL_ROOM_WAIT_MS")) |v| std.mem.span(v) else null),
        };
        var dev = try cuda.DeviceBuffer.alloc(m.ctx.d, (1 + o.world) * n * 8);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try m.comm.allGather(dev.ptr, dev.ptr + n * 8, n, .u64, m.stream);
        try m.stream.synchronize();
        var all: [4][n]u64 = undefined;
        if (o.world > all.len) return error.BadWorld;
        try dev.download(n * 8, std.mem.sliceAsBytes(all[0..o.world]));
        // this rank's own parse errors first (each rank names what it could not read), then any other rank's
        m.draft_settings = try want_draft;
        const mode = try want_mode;
        const pool = try want_pool;
        _ = try want_window;
        var firsts: [4][4]u64 = undefined;
        for (all[0..o.world], firsts[0..o.world]) |r, *f| {
            if (r[0] == bad_setting or r[1] == 0 or r[2] == bad_setting) return error.KvShardMismatch;
            if (r[0] != all[0][0] or r[1] != all[0][1] or r[3] != all[0][3]) return error.KvShardMismatch;
            if (r[4] != all[0][4] or r[5] != all[0][5]) return error.GraphSettingMismatch;
            try greedy_head.agree(r[6], all[0][6]);
            try keyed_draft.checkAgreement(r[6], all[0][6]);
            try prefillAgreement(all[0][6], r[6]);
            try draft_config.check(all[0][7..9].*, r[7..9].*);
            try roomWaitAgreement(all[0][9], r[9]);
            f.* = r[0..4].*;
        }
        const window = try pool_mod.agreedWindow(firsts[0..o.world]);
        return .{ .mode = mode, .pool = pool, .window = window };
    }

    /// The start-up's note of what it serves when the pool is not the window or the shard is on: the pool, its parts,
    /// the single-conversation limit (below the window asked: said so, with the numbers), the compressed planes'
    /// bytes a rank (calculated: rows by the formula; the arena's figure is the measured one).
    fn notePlan(m: *const Model, mode: shard.Mode, pl: shard.Plan) void {
        var buf: [16]shard.Source = undefined;
        const src = m.kvSources(&buf);
        const sharded = mode != .off;
        const trash: usize = if (m.kvs) |k| k.sh.trash else 0;
        const planes = shard.poolBytes(src, pl.pool, m.world, sharded, trash);
        const flat = shard.poolBytes(src, pl.pool, m.world, false, 0);
        std.log.info("dsv41: pool {d} positions, {s} {d} a part; a conversation may use up to {d} tokens{s}; compressed planes {d} MiB a rank ({d} MiB replicated); arena {d} MiB used", .{
            pl.pool, @tagName(mode), pl.part, pl.window, if (pl.shrunk) " (less than the window asked: a part holds no more)" else "", planes >> 20, flat >> 20, m.arena.used >> 20,
        });
    }

    /// Every kv-source layer's packed row bytes (shard.Source: the capacity arithmetic's and the carry cost's), the
    /// first `buf.len` of them.
    pub fn kvSources(m: *const Model, buf: *[16]shard.Source) []shard.Source {
        var ns: usize = 0;
        const c = m.eng.c;
        for (m.w.layers) |lay| {
            if (lay.comp_wkv == null or ns == buf.len) continue;
            buf[ns] = .{ .ratio = lay.ratio, .comp_row = c.head_dim / 2 + c.head_dim / 16, .idx_row = if (lay.idx_wk != null) c.index_head_dim / 2 + c.index_head_dim / 32 else 0 };
            ns += 1;
        }
        return buf[0..ns];
    }

    fn cleanupAllowed(m: *Model, mark: usize) bool {
        if (!cuda.context_health.cleanupReleased(mark)) return false;
        m.stream.synchronize() catch { cuda.context_health.cleanupRetained(); return false; };
        return true;
    }

    pub fn close(m: *Model) void { if (!m.closeOutcome()) cuda.context_health.cleanupRetained(); }

    pub fn closeOutcome(m: *Model) bool {
        const cleanup = cuda.context_health.cleanupMark();
        m.joinAhead();
        m.stream.synchronize() catch { cuda.context_health.cleanupRetained(); return false; };
        if (m.img_buf) |*b| b.free();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.spans.deinit(m.gpa);
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.graphs) |*g| g.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.pgraphs) |*g| g.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.timer) |t| {
            t.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
            m.gpa.destroy(t);
        }
        if (m.ptimer) |t| {
            t.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
            m.gpa.destroy(t);
        }
        m.rd.dropSide(&m.eng);
        m.rd.dropPrefetch();
        m.rd.dropShard();
        if (m.aio) |x| x.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.paio) |x| x.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.epool) |p| p.deinit(m.gpa);
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.tables) |*t| t.close();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.rdma_rings) |*r| r.close(m.gpa);
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.ring) |r| {
            r.stop();
            r.destroy(m.gpa);
        }
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.comm.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.blas.close();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.kvs) |*x| x.k.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.arena.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.rope_buf.free();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.ex.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.ops.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.exk.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.lg.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.pf.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.set.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        if (m.rdma_mod) |*x| x.unload();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.fan.close();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.w.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.cache_file.close(m.io);
        m.ix.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.cache_dir.close(m.io);
        m.stream.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.host.deinit();
        if (!cuda.context_health.cleanupReleased(cleanup)) return false;
        m.gpa.destroy(m);
        return true;
    }


    /// TP2's RDMA ring for the decode gathers: created on both ranks, the queue-pair infos exchanged over NCCL,
    /// connected, and started once both are (an all-gather as the barrier).
    pub fn openRing(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream) !*rdma.Ring {
        return openRingSized(gpa, d, k, devices, rank, comm, stream, ring_bytes);
    }

    pub fn openRingSized(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, devices: []const []const u8, rank: u32, comm: *const comm_mod.Comm, stream: cuda.Stream, bytes: usize) !*rdma.Ring {
        const world: usize = comm.world;
        if (world < 2 or world > rdma.max_ranks or rank >= world) return error.BadRdmaWorld;
        const mask_text = if (std.c.getenv("TF_RDMA_PEER_MASKS")) |v| std.mem.span(v) else null;
        const peer_masks = try rdma.parsePeerMasks(mask_text, rank, comm.world, @intCast(devices.len));
        const r = try rdma.Ring.create(gpa, d, k, devices, rank, comm.world, .{ .max_bytes = bytes, .gid_index = try rdma.servedGidIndex(), .peer_masks = peer_masks });
        errdefer r.destroy(gpa);
        const per = @sizeOf(rdma.Info);
        var mine: [rdma.max_ranks]rdma.Info = undefined;
        for (0..world) |peer_rank| mine[peer_rank] = r.info(@intCast(peer_rank));
        const send_bytes = world * per;
        var dev = try cuda.DeviceBuffer.alloc(d, (world + 1) * send_bytes);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(mine[0..world]));
        try comm.allGather(dev.ptr, dev.ptr + send_bytes, send_bytes, .u8, stream);
        try stream.synchronize();
        var all: [rdma.max_ranks * rdma.max_ranks]rdma.Info = undefined;
        try dev.download(send_bytes, std.mem.sliceAsBytes(all[0 .. world * world]));
        var peers: [rdma.max_ranks]rdma.Info = undefined;
        for (0..world) |peer_rank| peers[peer_rank] = all[peer_rank * world + rank];
        try r.connect(peers[0..world]);
        // every queue pair ready on both ranks before either sends
        try comm.allGather(dev.ptr, dev.ptr + send_bytes, 1, .u8, stream);
        try stream.synchronize();
        try r.start();
        return r;
    }

    fn readKit(a: std.mem.Allocator, io: std.Io, kit: []const u8, name: []const u8) ![]const u8 {
        return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ kit, "cubins", name }), a, .limited(1 << 28), .@"16", null);
    }

    /// The link to the other rank (rank 0's to rank 1, rank 1's to rank 0).
    pub fn peer(m: *const Model) link.Link {
        return m.fan.links[0].?;
    }

    /// Rank 0's links to its followers: rank 1's, or ranks 1-3's on the four-node split.
    pub fn followers(m: *const Model) []const ?link.Link {
        return m.fan.links[0 .. m.world - 1];
    }

    /// The graph vote's frames ride the command links. Only the engine thread writes them or rank_wire frames;
    /// a graph miss happens after the outer verify/pass frame is sent and consumed, and its CUDA-only body does
    /// not send rank_wire traffic. Each operation has a deadline; a failed frame ends the pair, never retries.
    fn voteLink(m: *const Model, other: u16) !link.Link {
        if (m.rank == 0) {
            if (other == 0 or other >= m.world) return error.BadRank;
            return m.fan.links[other - 1] orelse error.LinkClosed;
        }
        if (other != 0) return error.BadRank;
        return m.fan.links[0] orelse error.LinkClosed;
    }

    fn voteSend(ctx: *anyopaque, other: u16, bytes: []const u8, deadline: u64) anyerror!void {
        const m: *const Model = @ptrCast(@alignCast(ctx));
        return (try m.voteLink(other)).sendWithin(bytes, deadline);
    }

    fn voteRecv(ctx: *anyopaque, other: u16, buf: []u8, deadline: u64) anyerror![]u8 {
        const m: *const Model = @ptrCast(@alignCast(ctx));
        return (try m.voteLink(other)).recvBy(buf, deadline);
    }

    /// Host-only: never a CUDA call, so a poisoned rank can still vote (fatal).
    fn voteHealthy(ctx: *anyopaque) bool {
        const m: *const Model = @ptrCast(@alignCast(ctx));
        return m.ctx.healthy();
    }

    /// The spill tier's vote remains off unless explicitly enabled on every rank. KV shard is always off.
    pub fn keepGraphVote(m: *const Model, with_spill: bool) bool {
        return graph_vote.enabled(m.gvote != null, m.world, with_spill, m.kvs != null,
            graph_vote.spillOptIn(envOf("TF_DS_GRAPH_VOTE_SPILL")));
    }

    pub fn disableGraphVote(m: *Model) void {
        if (m.gvote == null) return;
        if (m.graphs) |*gr| gr.voter = null;
        if (m.pgraphs) |*pg| pg.voter = null;
        m.gvote = null;
        std.log.warn("dsv41: rank {d}: CUDA graph capture is not cross-rank protected while the spill tier shares the rank links", .{m.rank});
    }

    pub fn drafting(m: *const Model) bool {
        return m.dr != null;
    }

    // -- the primitives (both ranks, in the same order) ----------------------------------------------------------

    /// A stream's prompt is about to fill pool slot `slot`, its compressed rows in its extent from position `base`.
    /// Wait for the next chunk's Engram read-ahead, if one runs (before the prompt it reads changes or goes).
    /// An error after which the ranks cannot go on together (round.pair_ending) met on rank 0 outside a round: the
    /// followers may wait in a collective, so the ring is aborted, the links closed and the process exits 75 (the
    /// restart-both status).
    pub fn pairFault(m: *Model, err: anyerror) noreturn {
        std.log.err("dsv41: rank {d}: {s} parts the ranks; exiting 75", .{ m.rank, @errorName(err) });
        if (m.ring) |ring| ring.abort("pair fault");
        m.fan.close();
        std.process.exit(75);
    }

    pub fn joinAhead(m: *Model) void {
        if (m.rs.ahead) |th| th.join();
        m.rs.ahead = null;
        if (m.ahead) |th| th.join();
        m.ahead = null;
    }

    /// A kept prompt state's own bytes (the prompt cache's snapshots; the compressed plane stays in the stream's
    /// extent): one slot's window ring of every layer and positional stores of every ratio-2 layer.
    pub fn snapBytes(m: *const Model) usize {
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var n: usize = m.rings.len * m.eng.ringBytes();
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            if (k != 0) n += pb;
            if (sc != 0) n += pb;
        }
        // A complete prompt also resumes its exact head distribution and drafter rings.
        if (m.dpool) |p| n += p.stages * p.ring * p.hd * 2;
        n += m.vocab * 4;
        return n;
    }

    /// Slot `slot`'s own state into `buf` (save: true) or back from it, in the stream's order (no wait): with the
    /// extent's compressed rows, the state a prompt pass stands in at a chunk end.
    pub fn snapCopy(m: *Model, slot: usize, buf: u64, save: bool) !void {
        const rb = m.eng.ringBytes();
        const pb = prompt.raw_rows * m.eng.c.head_dim * 4;
        var at = buf;
        for (m.rings) |r| {
            try m.copyState(r + slot * rb, at, rb, save);
            at += rb;
        }
        for (m.caches.raw_kv[0..m.eng.c.layers], m.caches.raw_score[0..m.eng.c.layers]) |k, sc| {
            for ([_]u64{ k, sc }) |live| if (live != 0) {
                try m.copyState(live + slot * pb, at, pb, save);
                at += pb;
            };
        }
        if (m.dpool) |p| {
            const db = p.ring * p.hd * 2;
            for (p.rings[0..p.stages]) |r| {
                try m.copyState(r + slot * db, at, db, save);
                at += db;
            }
        }
        try m.copyState(m.promptLogits(), at, m.vocab * 4, save);
    }

    /// The compressed rows of positions [0, at) of the extent at `from` copied into the extent at `to` (a kept prompt
    /// resumed in another extent while a live stream holds its own), in the stream's order: every kv-source layer's
    /// codes and scales and its index keys, the rows view() places at those positions.
    /// TF_DS_KV_SHARD: the extents lie in one part each (else CrossesPart, nothing copied); the index keys copy on
    /// every rank as replicated, the compressed rows by shard.Shard.extentMove: within a part its owner copies them
    /// (the other rank holds none), across parts (a carry) the source's owner sends them straight from its plane and the
    /// destination's owner receives them into its own, both ranks in this same order (lanes.zig runs copyExtent on
    /// every rank in its frame order). The source's rows stay; nothing is freed.
    pub fn copyExtent(m: *Model, from: usize, to: usize, at: usize) !void {
        // PORT(round3): copyExtent may call checkExtent first so a refusal never follows a partial copy; kept as it
        // was here so the switch-off path is unchanged (kv_lanes calls checkExtent before every transfer)
        const c = m.eng.c;
        const cs = &m.caches;
        const mv: ?shard.Move = if (cs.shard) |sh| try sh.extentMove(from, to, at) else null;
        var spans: [2 * 64]comm_mod.Span = undefined;
        var ns: usize = 0;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (from % r != 0 or to % r != 0) return error.BadView;
            const n = at / r;
            const cb = c.head_dim / 2;
            const sb = c.head_dim / 16;
            if (mv) |x| switch (x) {
                .none => {},
                .local => |l| {
                    try m.copySpan(cs.comp_codes[i], l.from / r, l.to / r, n, cb);
                    try m.copySpan(cs.comp_scales[i], l.from / r, l.to / r, n, sb);
                },
                .send => |t| {
                    spans[ns] = .{ .ptr = cs.comp_codes[i] + t.from / r * cb, .bytes = n * cb };
                    spans[ns + 1] = .{ .ptr = cs.comp_scales[i] + t.from / r * sb, .bytes = n * sb };
                    ns += 2;
                },
                .recv => |t| {
                    spans[ns] = .{ .ptr = cs.comp_codes[i] + t.to / r * cb, .bytes = n * cb };
                    spans[ns + 1] = .{ .ptr = cs.comp_scales[i] + t.to / r * sb, .bytes = n * sb };
                    ns += 2;
                },
            } else {
                try m.copySpan(cs.comp_codes[i], from / r, to / r, n, cb);
                try m.copySpan(cs.comp_scales[i], from / r, to / r, n, sb);
            }
            if (lay.idx_wk != null) {
                try m.copySpan(cs.idx_codes[i], from / r, to / r, n, c.index_head_dim / 2);
                try m.copySpan(cs.idx_scales[i], from / r, to / r, n, c.index_head_dim / 32);
            }
        }
        // a carry or a live move across parts (kv_lanes.zig runs parts.zig's ops in frame order on every rank, rank 0
        // decides the transfer's ok and every rank finishes with it; the generation is published at finish)
        const x = mv orelse return;
        var bytes: usize = 0;
        for (spans[0..ns]) |sp| bytes += sp.bytes;
        switch (x) {
            .send => |t| try m.comm.sendSpans(spans[0..ns], t.peer, m.stream),
            .recv => |t| try m.comm.recvSpans(spans[0..ns], t.peer, m.stream),
            else => return,
        }
        if (m.kvs) |*k| {
            k.stats.carries += 1;
            k.stats.carry_bytes += bytes;
        }
    }

    /// copyExtent's refusals with nothing enqueued: the extents' parts (shard.Shard.extentMove: CrossesPart) and every
    /// kv source's ratio (BadView). Deterministic: every rank refuses the same arguments alike under the agreed
    /// layout (kv_lanes.zig checks before any rank enqueues its side, so a refused carry leaves no NCCL half waiting).
    pub fn checkExtent(m: *const Model, from: usize, to: usize, at: usize) !void {
        if (m.caches.shard) |sh| _ = try sh.extentMove(from, to, at);
        for (m.eng.w.layers) |lay| {
            if (lay.comp_wkv == null) continue;
            const r: usize = lay.ratio;
            if (from % r != 0 or to % r != 0 or at % r != 0) return error.BadView;
        }
    }

    /// Exact replicated-plane bytes for an idle relocation, checked before enqueue.
    pub fn extentBytes(m: *const Model, from: usize, to: usize, at: usize) !u64 {
        if (m.kvs != null or at == 0 or from > m.pool_cap or to > m.pool_cap or
            at > m.pool_cap - from or at > m.pool_cap - to or
            (from < to + at and to < from + at)) return error.BadView;
        try m.checkExtent(from, to, at);
        var total: u64 = 0;
        for (m.eng.w.layers) |lay| {
            if (lay.comp_wkv == null) continue;
            const rows = at / lay.ratio;
            var width = m.eng.c.head_dim / 2 + m.eng.c.head_dim / 16;
            if (lay.idx_wk != null) width += m.eng.c.index_head_dim / 2 + m.eng.c.index_head_dim / 32;
            total = try std.math.add(u64, total, try std.math.mul(u64, rows, width));
        }
        return total;
    }

    /// Rows [from, from + n) of `plane` (rows of `row` bytes; a sharded plane's rows of this rank's part) to rows
    /// [to, to + n), in the stream's order.
    fn copySpan(m: *Model, plane: u64, from: usize, to: usize, n: usize, row: usize) !void {
        const d = m.ctx.d;
        try d.ensureHealthy();
        try d.check(d.api.cuMemcpyDtoDAsync_v2(plane + to * row, plane + from * row, n * row, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    fn copyState(m: *Model, live: u64, kept: u64, bytes: usize, save: bool) !void {
        const d = m.ctx.d;
        try d.ensureHealthy();
        const dst = if (save) kept else live;
        const src = if (save) live else kept;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, m.stream.handle), "cuMemcpyDtoDAsync");
    }

    // -- spill v2's checkpoint state (port of wojo's GLM 0110, PR #124: spill_state.zig) ------------------------------
    // A kept point's state on this rank as the spill tier stores it: the descriptor derived from the loaded weights and
    // the live allocation, and the planes' addresses and copies for spill.Packer. Nothing here runs unless the tier
    // calls it (round 2's lanes wiring); the served path is unchanged.

    pub const spill = @import("spill_state.zig");

    comptime {
        std.debug.assert(draft.max_streams <= spill.max_slots);
    }

    /// This rank's state planes as spill.Packer reads and writes them: their device addresses (the whole pool's
    /// compressed planes, a slot's rings and raw stores) and copies in the model's stream's order.
    pub const SpillDev = struct {
        m: *Model,

        pub const Event = cuda.Event;

        /// Row 0 of a block plane over the whole pool (the extent at base b starts b / ratio rows in, as view() puts it);
        /// TF_DS_KV_SHARD: an owner-only plane's is row 0 of this rank's part (spill_state's rowAddr adds the local row).
        pub fn blockPlane(dv: SpillDev, p: spill.Plane) u64 {
            const cs = &dv.m.caches;
            return switch (p.kind) {
                .comp_codes => cs.comp_codes[p.layer],
                .comp_scales => cs.comp_scales[p.layer],
                .idx_codes => cs.idx_codes[p.layer],
                .idx_scales => cs.idx_scales[p.layer],
                .ring, .raw_kv, .raw_score, .draft_ring, .logits => 0,
            };
        }

        /// Slot `slot`'s rows of a small plane (snapCopy's addresses).
        pub fn slotPlane(dv: SpillDev, p: spill.Plane, slot: usize) u64 {
            const m = dv.m;
            const base: u64 = switch (p.kind) {
                .ring => m.rings[p.layer],
                .raw_kv => m.caches.raw_kv[p.layer],
                .raw_score => m.caches.raw_score[p.layer],
                .draft_ring => if (m.dpool) |dp| (if (p.layer < dp.stages) dp.rings[p.layer] else 0) else 0,
                // one buffer for every slot, as snapCopy copies it: the logits of the prompt pass that just ended
                .logits => return m.promptLogits(),
                .comp_codes, .comp_scales, .idx_codes, .idx_scales => 0,
            };
            return if (base == 0) 0 else base + slot * p.slotBytes();
        }

        pub fn capacity(dv: SpillDev) usize {
            return dv.m.caches.cap;
        }

        pub fn slots(dv: SpillDev) usize {
            return dv.m.streams;
        }

        pub fn d2h(dv: SpillDev, dst: []u8, src: u64) !void {
            const d = dv.m.ctx.d;
            d.ensureHealthy() catch |err| dv.failed(err);
            d.check(d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, dv.m.stream.handle), "cuMemcpyDtoHAsync") catch |err| dv.failed(err);
        }

        pub fn h2d(dv: SpillDev, dst: u64, src: []const u8) !void {
            const d = dv.m.ctx.d;
            d.ensureHealthy() catch |err| dv.failed(err);
            d.check(d.api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, dv.m.stream.handle), "cuMemcpyHtoDAsync") catch |err| dv.failed(err);
        }

        pub fn fill(dv: SpillDev, dst: u64, value: u8, bytes: usize) !void {
            const d = dv.m.ctx.d;
            d.ensureHealthy() catch |err| dv.failed(err);
            d.check(d.api.cuMemsetD8Async(dst, value, bytes, dv.m.stream.handle), "cuMemsetD8Async") catch |err| dv.failed(err);
        }

        pub fn record(dv: SpillDev, ev: cuda.Event) !void {
            ev.record(dv.m.stream) catch |err| dv.failed(err);
        }

        /// Whether the copies before `ev` completed (spill_tier polls it; errors invalidate the device, never certify the
        /// copy as complete).
        pub fn done(_: SpillDev, ev: cuda.Event) !bool {
            return ev.done();
        }

        /// The copies before `ev` completed (a cancel, a drop, a shutdown: bounded by what was enqueued).
        pub fn wait(_: SpillDev, ev: cuda.Event) !void {
            try ev.synchronize();
        }

        /// Unknown DMA quiescence requires process teardown, not buffer reuse.
        pub fn failed(d: SpillDev, err: anyerror) noreturn {
            std.log.err("dsv41: rank {d} spill device failure ({s}); exiting 75 without recycling DMA buffers", .{ d.m.rank, @errorName(err) });
            if (d.m.ring) |ring| ring.abort("spill device failure");
            d.m.fan.close();
            std.process.exit(75);
        }
    };

    /// The spill state descriptor of this rank, derived from the loaded layers (Caches.init's reading of them) and
    /// checked against the live allocation: snapCopy's bytes, a ring per layer, every plane present. TF_DS_KV_SHARD:
    /// owner_sharded, cut into the shard's parts (its codes and scales owner-only, at their part-local rows).
    pub fn spillLayout(m: *const Model) !spill.StateLayout {
        const layers = m.eng.w.layers;
        if (layers.len > spill.max_layers) return error.SpillLayoutMismatch;
        var shapes: [spill.max_layers]spill.LayerShape = undefined;
        for (layers, shapes[0..layers.len]) |lay, *s| {
            const kv = lay.comp_wkv != null;
            s.* = .{ .ratio = if (kv) lay.ratio else 0, .kv = kv, .idx = kv and lay.idx_wk != null, .raw = kv and lay.ratio > 1 };
        }
        const sh = m.caches.shard;
        var geo = spill.Geometry.of(m.eng.c, chunk_rows, prompt.ring_extra, prompt.raw_rows);
        if (m.dpool) |p| geo = .{ .block_tokens = geo.block_tokens, .window = geo.window, .ring_extra = geo.ring_extra, .raw_rows = geo.raw_rows, .head_dim = geo.head_dim, .index_head_dim = geo.index_head_dim, .draft_stages = p.stages, .draft_ring = p.ring, .draft_hd = p.hd };
        geo.vocab = m.vocab;
        const raw = try spill.StateLayout.init(shapes[0..layers.len], geo, if (sh != null) .owner_sharded else .replicated);
        const l = if (sh) |x| try raw.withParts(x.part) else raw;
        if (l.small_bytes != m.snapBytes() or m.rings.len != layers.len) {
            std.log.err("dsv41 spill layout: small state {d} B vs snapBytes {d} B; rings {d} vs layers {d}", .{ l.small_bytes, m.snapBytes(), m.rings.len, layers.len });
            return error.SpillLayoutMismatch;
        }
        const dv: SpillDev = .{ .m = @constCast(m) };
        for (l.blockPlanes()) |p| if (dv.blockPlane(p) == 0) {
            std.log.err("dsv41 spill layout: block plane {s} of layer {d} has no device plane", .{ @tagName(p.kind), p.layer });
            return error.SpillLayoutMismatch;
        };
        for (l.smallPlanes()) |p| {
            if (dv.slotPlane(p, 0) == 0) {
                std.log.err("dsv41 spill layout: small plane {s} {d} has no device plane", .{ @tagName(p.kind), p.layer });
                return error.SpillLayoutMismatch;
            }
            if (p.kind == .ring and p.slotBytes() != m.eng.ringBytes()) {
                std.log.err("dsv41 spill layout: ring {d} is {d} B, the engine's {d} B", .{ p.layer, p.slotBytes(), m.eng.ringBytes() });
                return error.SpillLayoutMismatch;
            }
        }
        return l;
    }

    /// The pool's extent and slot generations (one counter a chunk of positions), sized once at setup.
    pub fn spillGenerations(m: *const Model, gpa: std.mem.Allocator) !spill.Generations {
        return spill.Generations.init(gpa, m.caches.cap, chunk_rows);
    }

    /// A packer over this rank's planes, `l` from spillLayout and `g` from spillGenerations (both the caller's, alive
    /// while it is used). Its calls enqueue on the model's stream and record the caller's event: no allocation, no wait.
    pub fn spillPacker(m: *Model, l: *const spill.StateLayout, g: *spill.Generations) spill.Packer(SpillDev) {
        return .{ .dev = .{ .m = m }, .layout = l, .gens = g, .rank = m.rank };
    }

    /// The Engram read-ahead joined and its rows dropped, so the next chunk reads its own rows from its prompt's ids.
    /// Baseline hazard (also without spill): fillChunk takes a joined read-ahead whose (ahead_start, ahead_n) match its
    /// chunk, and a pass that ends between chunks (cancel, error) leaves one standing; a later prompt resuming a kept
    /// point at that position (in memory or restored) would consume rows hashed from the other prompt's ids. The fix
    /// is one call at the start of every pass, in fillBegin after its joinAhead (outside this block: routed to the
    /// coordinator, docs/port/dsv41-state.md); until it lands, a restore calls this right before the resumed pass.
    pub fn resetEngramAhead(m: *Model) void {
        m.joinAhead();
        if (m.eh) |*eh| {
            eh.ahead_ready = .{ false, false };
            eh.ahead_n = 0;
            eh.ahead_err = null;
        }
    }

    /// After a restored prompt's fill: the drafter's rings were rebuilt by its replay through the prompt's end
    /// (spill.checkReplayRebuilds' rule, here as the pool records it), so the first draft reads no unwritten row.
    pub fn spillCheckResumed(m: *const Model, slot: usize, len: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        if (m.dpool) |*p| if (p.absorbed[slot] != len) return error.DraftNotRebuilt;
    }

    /// A compressed plane as copyExtent copies it: its address, row bytes (copyExtent's own widths from the config)
    /// and ratio.
    const SpillRefPlane = struct { plane: u64, row: usize, ratio: usize };

    /// copyExtent's planes in its order (every kv layer: codes, scales, then the index keys' codes and scales), read
    /// from the loaded layers and the config, not from the spill layout.
    fn spillRefPlanes(m: *const Model, out: *[4 * spill.max_layers]SpillRefPlane) ![]SpillRefPlane {
        const c = m.eng.c;
        const cs = &m.caches;
        var k: usize = 0;
        for (m.eng.w.layers, 0..) |lay, i| {
            if (lay.comp_wkv == null) continue;
            if (k + 4 > out.len) return error.SpillLayoutMismatch;
            const r: usize = lay.ratio;
            out[k] = .{ .plane = cs.comp_codes[i], .row = c.head_dim / 2, .ratio = r };
            out[k + 1] = .{ .plane = cs.comp_scales[i], .row = c.head_dim / 16, .ratio = r };
            k += 2;
            if (lay.idx_wk != null) {
                out[k] = .{ .plane = cs.idx_codes[i], .row = c.index_head_dim / 2, .ratio = r };
                out[k + 1] = .{ .plane = cs.idx_scales[i], .row = c.index_head_dim / 32, .ratio = r };
                k += 2;
            }
        }
        return out[0..k];
    }

    /// Stage 2's completeness check on one instance (a GPU tool, not a serving path), against the engine's own resume
    /// primitives rather than the layout itself. The point at `at` of slot `from_slot` and the extent at `from`:
    ///   - reference: copyExtent(from, ref, at) into a third extent and snapCopy(from_slot) into a scratch Snap buffer;
    ///   - packed, unpacked into slot `to_slot` and the extent at `to` (both overwritten, as is `ref`), packed again.
    /// Then (round trip) both packs are equal; (blocks) the restored rows of `to` equal copyExtent's rows at `ref`, and
    /// the packed blocks, cut with offsets computed from copyExtent's loop and the config's row widths, equal them;
    /// (small) a snapCopy of the restored slot equals the reference Snap, and the packed small state equals it too.
    /// Every argument is checked before anything is enqueued; its own pinned staging, device scratch and event, none
    /// freed before the stream is idle.
    pub fn spillSelfCheck(m: *Model, gpa: std.mem.Allocator, from_slot: usize, from: usize, to_slot: usize, to: usize, ref: usize, at: usize) !void {
        // (its reference reads the pool-wide planes copyExtent's replicated loop addresses; TF_DS_KV_SHARD's packing is
        // checked by spill_state's two-rank host tests and on the GPU by the restored token_sha)
        if (m.caches.shard != null) return error.SpillSelfCheckSharded;
        if (from_slot >= m.streams or to_slot >= m.streams or from_slot == to_slot) return error.BadSlot;
        if (at == 0) return error.BadRange;
        for ([_]usize{ from, to, ref, at }) |v| if (v % chunk_rows != 0) return error.BadRange;
        const cap = m.caches.cap;
        for ([_]usize{ from, to, ref }) |b| if (b >= cap or at > cap - b) return error.BadRange;
        for ([_][2]usize{ .{ from, to }, .{ from, ref }, .{ to, ref } }) |p| if (p[0] < p[1] + at and p[1] < p[0] + at) return error.BadRange;
        const l = try m.spillLayout();
        var refs_buf: [4 * spill.max_layers]SpillRefPlane = undefined;
        const refs = try m.spillRefPlanes(&refs_buf);
        var ref_bytes: usize = 0;
        for (refs) |rp| ref_bytes += at / rp.ratio * rp.row;
        const nb = try l.blocks(at);
        const blk = try std.math.mul(usize, nb, l.block_bytes);
        if (ref_bytes != blk) { // the layout's block bytes are copyExtent's
            std.log.err("dsv41 spill self-check: copyExtent block {d} B vs layout {d} B", .{ ref_bytes, blk });
            return error.SpillLayoutMismatch;
        }
        const small = l.small_bytes;
        var g = try m.spillGenerations(gpa);
        defer g.deinit(gpa);
        var pk = m.spillPacker(&l, &g);

        var a = try cuda.HostBuffer.alloc(m.ctx.d, blk + small);
        defer a.free();
        var b = try cuda.HostBuffer.alloc(m.ctx.d, blk + small);
        defer b.free();
        var rows = try cuda.HostBuffer.alloc(m.ctx.d, 2 * blk); // copyExtent's rows at ref, then the restored rows at to
        defer rows.free();
        var snaps = try cuda.HostBuffer.alloc(m.ctx.d, 2 * small); // the reference Snap, then the restored slot's
        defer snaps.free();
        var sa = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
        defer sa.free();
        var sb = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
        defer sb.free();
        var ev = try cuda.Event.init(m.ctx.d, false);
        defer ev.deinit();
        defer m.stream.synchronize() catch {}; // (runs first: nothing above is freed under an enqueued copy)

        // the engine's own copies of the point
        try m.snapCopy(from_slot, sa.ptr, true);
        try m.copyExtent(from, ref, at);
        // pack, restore elsewhere, pack again
        _ = try pk.packBlocks(try g.pin(from, from + at), 0, nb, a.bytes[0..blk], ev);
        _ = try pk.packSmall(try g.pinSlot(from_slot), a.bytes[blk..][0..small], ev);
        try ev.synchronize();
        const dst = try g.reserve(to, to + at);
        const ds = try g.reserveSlot(to_slot);
        const ub = try pk.unpackBlocks(dst, 0, nb, a.bytes[0..blk], ev);
        const us = try pk.unpackSmall(ds, a.bytes[blk..][0..small], ev);
        _ = try pk.packBlocks(dst, 0, nb, b.bytes[0..blk], ev);
        _ = try pk.packSmall(ds, b.bytes[blk..][0..small], ev);
        try m.snapCopy(to_slot, sb.ptr, true);
        // plain copies of each plane's rows [ref / r, (ref + at) / r) and [to / r, (to + at) / r), plane after plane
        const d = m.ctx.d;
        var off: usize = 0;
        for ([_]usize{ ref, to }) |base| for (refs) |rp| {
            const n = at / rp.ratio * rp.row;
            try d.check(d.api.cuMemcpyDtoHAsync_v2(rows.bytes[off..].ptr, rp.plane + base / rp.ratio * rp.row, n, m.stream.handle), "cuMemcpyDtoHAsync");
            off += n;
        };
        try d.check(d.api.cuMemcpyDtoHAsync_v2(snaps.bytes.ptr, sa.ptr, small, m.stream.handle), "cuMemcpyDtoHAsync");
        try d.check(d.api.cuMemcpyDtoHAsync_v2(snaps.bytes[small..].ptr, sb.ptr, small, m.stream.handle), "cuMemcpyDtoHAsync");
        try m.stream.synchronize();
        if (!g.valid(ub) or !g.valid(us)) return error.StaleExtent;

        if (!std.mem.eql(u8, a.bytes[0 .. blk + small], b.bytes[0 .. blk + small])) return error.SpillRoundTripMismatch;
        const ref_rows = rows.bytes[0..blk];
        if (!std.mem.eql(u8, ref_rows, rows.bytes[blk..][0..blk])) return error.SpillRestoreMismatch;
        // the packed blocks against copyExtent's rows: block k's plane j is that plane's rows from k * 2048 / ratio
        var in_block: usize = 0; // plane j's offset in a block (copyExtent's order and widths)
        var in_rows: usize = 0; // plane j's offset in ref_rows
        for (refs) |rp| {
            const per = chunk_rows / rp.ratio * rp.row;
            for (0..nb) |k| {
                if (!std.mem.eql(u8, b.bytes[k * l.block_bytes + in_block ..][0..per], ref_rows[in_rows + k * per ..][0..per])) {
                    std.log.err("dsv41 spill self-check: block {d} plane at {d} differs from copyExtent's rows", .{ k, in_block });
                    return error.SpillLayoutMismatch;
                }
            }
            in_block += per;
            in_rows += at / rp.ratio * rp.row;
        }
        if (in_block != l.block_bytes) {
            std.log.err("dsv41 spill self-check: planes cover {d} B of a {d} B block", .{ in_block, l.block_bytes });
            return error.SpillLayoutMismatch;
        }
        const ref_snap = snaps.bytes[0..small];
        if (!std.mem.eql(u8, ref_snap, snaps.bytes[small..][0..small])) return error.SpillRestoreMismatch;
        if (!std.mem.eql(u8, ref_snap, b.bytes[blk..][0..small])) {
            const first = std.mem.indexOfDiff(u8, ref_snap, b.bytes[blk..][0..small]) orelse 0;
            std.log.err("dsv41 spill self-check: packed small state differs from snapCopy at byte {d} of {d}", .{ first, small });
            return error.SpillLayoutMismatch;
        }
    }

    /// Each layer's gate_bias_vl (f32 [experts], the checkpoint's extra file) on the device: Gate.forward's bias at
    /// image-span positions.
    fn loadVisionBias(m: *Model, path: []const u8) !void {
        const core = @import("core");
        var f = try core.safetensors.File.open(m.gpa, m.io, path);
        defer f.close(m.io);
        var nb: [64]u8 = undefined;
        for (m.w.layers, 0..) |*lay, i| {
            const t = f.get(try std.fmt.bufPrint(&nb, "layers.{d}.ffn.gate.bias_vl", .{i})) orelse return error.MissingVisionBias;
            if (!t.is(.f32, &.{m.cfg.experts})) return error.BadVisionBias;
            const dst = try m.arena.take(t.bytes.len);
            const tmp = try m.gpa.dupe(u8, t.bytes); // (a copy out of the file mapping first: see vit.zig)
            defer m.gpa.free(tmp);
            try m.ctx.d.check(m.ctx.d.api.cuMemcpyHtoD_v2(dst, tmp.ptr, tmp.len), "cuMemcpyHtoD");
            lay.gate_b_vl = dst;
        }
    }

    /// The filling prompt's image spans (absolute positions, in order) and their rows bf16 [tokens, hidden] from the
    /// host, after fillBegin and before its chunks.
    pub fn setImages(m: *Model, spans: []const prompt.Span, rows: []const u8) !void {
        if (rows.len > 0) {
            if (m.img_buf == null or m.img_buf.?.len < rows.len) {
                if (m.img_buf) |*b| b.free();
                m.img_buf = null;
                m.img_buf = try cuda.DeviceBuffer.alloc(m.ctx.d, rows.len);
            }
            const d = m.ctx.d;
            try d.check(d.api.cuMemcpyHtoDAsync_v2(m.img_buf.?.ptr, rows.ptr, rows.len, m.stream.handle), "cuMemcpyHtoDAsync");
            try m.stream.synchronize(); // (the caller's rows may go once this returns)
        }
        m.spans.clearRetainingCapacity();
        try m.spans.appendSlice(m.gpa, spans);
        m.ch.spans = m.spans.items;
        m.ch.img_rows = if (m.img_buf) |b| b.ptr else 0;
    }

    pub fn fillBegin(m: *Model, slot: usize, base: usize) !void {
        if (slot >= m.streams) return error.BadSlot;
        // a read-ahead an earlier pass left (cancelled between chunks) never feeds this one: a resume at that pass's
        // next chunk start would otherwise take its rows, hashed from the other prompt's ids
        if (m.rs.live) return error.OpenPrefillChunk;
        m.resetEngramAhead();
        m.spans.clearRetainingCapacity();
        m.ch.spans = &.{};
        m.view = try m.caches.view(&m.eng, slot, base);
        for (m.rings, m.ring_view) |r, *v| v.* = r + slot * m.eng.ringBytes();
        m.fill_slot = slot;
    }

    /// TF_DS_KV_SHARD: the filling prompt's view (fillBegin's, at `base`) with its borrowed twins (`on`; `local`: this
    /// rank's twin when it does not own the extent; prompt.Caches.twinned), or plain again. Both ranks bind at the
    /// same frame (kv_lanes.zig), between chunks only.
    pub fn bindTwin(m: *Model, base: usize, on: bool, local: ?usize) !void {
        const v = try m.caches.view(&m.eng, m.fill_slot, base);
        m.view = if (on) try v.twinned(&m.caches, &m.eng, base, local) else v;
    }

    /// The prompt's chunk of n rows from position `start` (seq: the stream's ids through the chunk's end) through every
    /// layer with the served bounded replay (replay = the prompt's length - the window: an encoder-only chunk stops at
    /// the decoder's first layer). A chunk that reaches the head leaves the logits of its last row in ch.head_g and,
    /// drafting, its DSpark taps absorbed into the slot's drafter rings. True when it reached the head.
    pub fn fillChunk(m: *Model, seq: []const i32, start: usize, n: usize, replay: usize) !bool {
        return (try m.fillSlice(seq, .{ .start = start, .n = n, .replay = replay }, 0, m.eng.c.layers)) == .head;
    }

    pub fn dropChunk(m: *Model) void {
        m.joinAhead();
        m.rs = .{};
    }

    pub fn fillSlice(m: *Model, seq: []const i32, args: SliceArgs, lo: usize, hi: usize) !SliceEnd {
        try m.rs.validate(args, lo, hi, m.eng.c.layers, seq.len);
        if (lo == 0) try m.openChunk(seq, args);
        // Fatal rank-local failures must poison the world before any joining cleanup.
        return m.runSlice(seq, lo, hi);
    }

    fn openChunk(m: *Model, seq: []const i32, args: SliceArgs) !void {
        const start = args.start;
        const n = args.n;
        if (n == 0 or n > m.compute_rows or seq.len != start + n) return error.BadChunk;
        if (m.prefill.af and seq.len > m.prompt_cap) return error.PromptTooLong;
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        // an image position's id is negative in the sequence (its n-grams dead for Engram): its embedding's id is the
        // image token's, which its span row then replaces
        const image_id: i64 = if (c.image_token) |t| t else 0;
        for (0..n) |j| m.ids64[j] = if (seq[start + j] < 0) image_id else seq[start + j];
        const pt = m.ptimer;
        if (pt) |t| try t.mark(m.stream, .start);
        try prompt.begin(e, ch, m.ids64[0..n], start, m.host_pos);
        if (ch.spans.len > 0) try prompt.embedSpans(e, ch);
        if (pt) |t| try t.mark(m.stream, .embed);
        // Engram rows read ahead on a thread while the GPU runs the layers before them (the served reader pool's
        // overlap): this chunk's both layers' during the chunk before it (m.ahead), else the first layer's from the
        // chunk's start and the next's once the first is applied
        var ahead: ?std.Thread = m.ahead;
        m.ahead = null;
        errdefer if (ahead) |th| th.join();
        if (m.eh) |*eh| {
            if (ahead != null and (eh.ahead_start != start or eh.ahead_n != n)) {
                ahead.?.join();
                ahead = null;
            }
            // a read-ahead of this chunk already joined (a round ran between two chunks): its rows stand ready
            const joined = ahead == null and eh.ahead_start == start and eh.ahead_n == n and eh.ahead_ready[0];
            if (ahead == null and !joined) {
                eh.ahead_start = start;
                eh.ahead_n = n;
                eh.ahead_ready = .{ false, false };
                eh.ahead_err = null;
                ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, @as(usize, 0) }) catch null;
            }
        }
        m.rs = .{ .live = true, .args = args, .ahead = ahead };
    }

    fn runSlice(m: *Model, seq: []const i32, lo: usize, hi: usize) !SliceEnd {
        const e = &m.eng;
        const c = e.c;
        const ch = &m.ch;
        const rs = &m.rs;
        const start = rs.args.start;
        const n = rs.args.n;
        const replay = rs.args.replay;
        const pt = m.ptimer;
        const taps = m.dr != null;
        for (lo..hi) |li| {
            if (li == c.layers / 2) {
                if (!try prompt.replayCut(e, ch, &m.view, &rs.shared, li, replay, m.host_pos)) {
                    // encoder-only: no synchronize either (as at the chunk's end below), so the next chunk's host steps
                    // run while this one's last layers do; the profile alone waits
                    if (pt) |t| {
                        try m.stream.synchronize();
                        try t.finish();
                    }
                    if (rs.ahead) |th| th.join();
                    rs.* = .{}; // preserve next-chunk read-ahead exactly as whole chunks did
                    return .encoder_only;
                }
                rs.floor = replay;
                rs.kv_done = li;
            }
            if (e.w.layers[li].engram_wkv != null) {
                const eh = if (m.eh) |*x| x else return error.NoEngramTables;
                if (rs.ahead) |th| {
                    const wait_begin = if (m.engram_audit) m.now() else 0;
                    th.join();
                    if (m.engram_audit) std.debug.print("prefill_engram_wait: rank={d} start={d} rows={d} layer={d} ns={d}\n", .{ m.rank, start, n, li, m.now() - wait_begin });
                    rs.ahead = null;
                }
                if (eh.ahead_err) |err| return err;
                try prompt.engramApply(e, ch, eh, li, seq);
                const j = std.mem.indexOfScalar(u16, c.engram_layers.slice(), @intCast(li)).?;
                const ne = @min(2, c.engram_layers.slice().len);
                if (j + 1 < ne) {
                    if (!eh.ahead_ready[j + 1])
                        rs.ahead = std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, seq, start, n, j + 1 }) catch null;
                } else if (m.fill_seq.len > start + n and m.fill_seq.ptr == seq.ptr) {
                    // the next chunk's rows while this one's layers run
                    const nn = if (m.prefill.ae) m.next_engram_rows else @min(m.compute_rows, m.fill_seq.len - start - n);
                    if (nn == 0 or nn > m.compute_rows or nn > m.fill_seq.len - start - n) return error.BadChunk;
                    eh.ahead_start = start + n;
                    eh.ahead_n = nn;
                    eh.ahead_ready = .{ false, false };
                    eh.ahead_err = null;
                    // AE inherits only the first layer. The second is scheduled
                    // after applying it, rather than joining both at layer one.
                    m.ahead = if (m.prefill.ae)
                        std.Thread.spawn(.{}, prompt.engramReadAhead, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn, @as(usize, 0) }) catch null
                    else std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ eh, c, m.fill_seq[0 .. start + n + nn], start + n, nn }) catch null;
                    if (m.engram_audit) std.debug.print("prefill_engram_plan: rank={d} start={d} rows={d} first_layer_only={d}\n", .{ m.rank, start + n, nn, @intFromBool(m.prefill.ae) });
                }
                if (pt) |t| try t.mark(m.stream, .engram);
            }
            // a DSpark tap reads the streams after the layer's Engram (_forward_k's order)
            if (taps) if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| try prompt.tap(e, ch, j);
            try prompt.attnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_attn);
            try prompt.attention(e, ch, &m.view, &rs.shared, li, m.ring_view[li], rs.floor, rs.kv_done != null and rs.kv_done.? == li);
            if (pt) |t| try t.mark(m.stream, .attn);
            try prompt.gather(e, ch, ch.pa, ch.ga);
            if (pt) |t| try t.mark(m.stream, .gather_a);
            try prompt.ffnMixes(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .mix_ffn);
            try prompt.moe(e, ch, li);
            if (pt) |t| try t.mark(m.stream, .moe);
            try prompt.gather(e, ch, ch.pm, ch.gm);
            if (pt) |t| try t.mark(m.stream, .gather_m);
            prompt.endLayer(ch);
        }
        rs.next = hi;
        if (hi < c.layers) return .more;
        try prompt.head(e, ch);
        if (pt) |t| try t.mark(m.stream, .head);
        if (m.dr) |*dr| {
            // its taps (the tap layers side by side) into the slot's drafter rings
            const dw = c.hidden;
            for (0..c.dspark_taps.slice().len) |j| try e.ops.copyRows(e.s, prompt.tapRows(e, ch, j), dw * 2, dr.at + j * dw * 2, dr.taps_w * 2, dw * 2, ch.n);
            try dr.absorb(e, ch, &m.dpool.?, m.fill_slot, dr.at, ch.n, ch.start);
        }
        // no synchronize: the next chunk's host steps (its first Engram layer's hashing, table reads and decode, ~0.14 s
        // at 2,048 rows) run while this chunk's kernels do; readers of the results (the prompt's logits) synchronize.
        // The profile (--profile 2) times a chunk to its end, so it alone waits.
        if (pt) |t| {
            try m.stream.synchronize();
            try t.finish();
        }
        if (rs.ahead) |th| th.join();
        rs.* = .{}; // next-chunk m.ahead deliberately survives the whole-chunk path
        return .head;
    }

    /// The prompt's logits (its last row's, the ranks' columns in rank order) after a fill's last chunk.
    pub fn promptLogits(m: *const Model) u64 {
        return m.ch.head_g;
    }

    /// One round over every stream's window (rows.windows, each with its stream's slot and extent), then each window
    /// whose stream drafts has its rows' taps absorbed into its slot's drafter rings at its positions (the served
    /// eager absorb: rows past the ones the round keeps sit where the drafter never reads, and the next window
    /// overwrites them). absorb[i]: window i's stream drafts.
    pub fn verify(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        return m.verifyHead(rows, absorb, false);
    }

    /// Explicit agreed Lanes choice only. Direct verify callers keep full logits.
    /// No error-path drain: Lanes owns command custody before sending/parsing.
    pub fn verifyHead(m: *Model, rows: round.Rows, absorb: []const bool, use_head: bool) !void {
        m.greedy_output.begin();
        m.sample_output = 0;
        m.rd.sample_count = 0;
        m.rd.greedy_out = 0;
        if (use_head) {
            if (!m.decode_greedy_head or m.two != null or m.rd.greedy_local == 0 or m.rd.greedy_all == 0 or m.rd.greedy_reference == 0)
                return error.GreedyHeadRankFault;
            m.rd.greedy_out = m.amax;
        }
        defer m.rd.greedy_out = 0;
        try m.verifyRun(rows, absorb);
        if (use_head) m.greedy_output.complete(rows.ids.len);
    }


    /// All ranks enter from the agreed sampled-count verify-frame extension.
    pub fn verifySampleHead(m: *Model, rows: round.Rows, absorb: []const bool, count: usize) !void {
        m.greedy_output.begin();
        m.sample_output = 0;
        m.rd.greedy_out = 0;
        m.rd.sample_count = 0;
        if (!m.decode_sample_head or m.two != null or m.rd.sample_local == 0 or m.rd.sample_all == 0 or m.rd.sample_reference == 0 or m.cand == 0 or count == 0 or count > sampling.max_candidates)
            return error.GreedyHeadRankFault;
        m.rd.sample_count = count;
        defer m.rd.sample_count = 0;
        try m.verifyRun(rows, absorb);
        m.sample_output = count;
        // Reuse output custody to refuse stale logits on a successful reduction.
        if (!m.rd.sample_fallback) m.greedy_output.complete(rows.ids.len);
    }

    fn verifyRun(m: *Model, rows: round.Rows, absorb: []const bool) !void {
        m.joinAhead(); // a prompt chunk's Engram read-ahead shares the host buffers a round's Engram reads use
        const e = &m.eng;
        if (m.prof) try m.stream.synchronize(); // (the GPU idle, so the issue time shows whether the host keeps it fed)
        const t0 = m.now();
        try round.forward(e, &m.rd, &m.ch, &m.caches, m.rings, if (m.eh) |*x| x else null, &.{}, rows, m.request_cap, null, if (m.rs.live) null else if (m.graphs) |*g| g else null);
        if (m.prof) {
            m.t_enqueue = m.now() - t0;
            try m.stream.synchronize();
            m.t_forward = m.now() - t0;
            if (m.timer) |t| try t.finish();
            if (m.rd.clk != 0) {
                var h: [1 + 2 * round.Round.max_clocks]u64 = undefined;
                const d = m.ctx.d;
                try d.check(d.api.cuMemcpyDtoH_v2(&h, m.rd.clk, h.len * 8), "cuMemcpyDtoH");
                const n = @min(h[0], round.Round.max_clocks);
                for (1..n) |i| {
                    const tag = h[2 + 2 * i] & 0xff;
                    m.clk_ph[tag] += h[1 + 2 * i] -% h[1 + 2 * (i - 1)];
                    m.clk_cnt[tag] += 1;
                }
                m.clk_rounds += 1;
            }
        }
        const t1 = m.now();
        if (m.dr) |*dr| {
            const wins = rows.windows orelse return error.BadRound;
            var items: [max_streams]draft.Item = undefined;
            var n: usize = 0;
            for (wins, absorb) |win, yes| {
                if (!yes) continue;
                items[n] = .{ .slot = @intCast(rows.slots.?[win.row]), .taps = m.rd.taps + win.row * dr.taps_w * 2, .n = win.n, .start = @intCast(rows.pos[win.row]) };
                n += 1;
            }
            if (n > 0) try dr.absorbMany(e, &m.ch, &m.dpool.?, items[0..n]);
        }
        try m.stream.synchronize();
        if (m.prof) m.t_absorb = m.now() - t1;
        if (m.rs.live) if (m.eh) |*eh| {
            const args = m.rs.args;
            eh.hasher.hashes(m.fill_seq, args.start, args.n, eh.hashes[0 .. args.n * eh.hasher.layers * eh.hasher.cols()]);
        };
    }

    /// The rounds' GPU clocks (a profile): the forward's start and each stretch's ends, for its gaps.
    pub fn useClock(m: *Model, marks: u8) !void {
        m.rd.clk_marks = marks == 1 or marks == 3;
        m.rd.clk_shard = marks == 4; // the sharded attention's parts alone (round.PhaseTimer.shardMark)
        m.rd.clk_ratio = if (marks == 3) 2 else null;
        m.rd.clk_layers = marks == 2 or marks == 3;
        if (m.rd.clk == 0) m.rd.clk = try m.arena.take((1 + 2 * round.Round.max_clocks) * 8);
    }

    /// The rounds' GPU time by phase (eager rounds; --profile 2): a timer on the round.
    pub fn usePhaseTimer(m: *Model) !void {
        if (m.timer != null) return;
        const t = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(t);
        t.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.timer = t;
        const pt = try m.gpa.create(round.PhaseTimer);
        errdefer m.gpa.destroy(pt);
        pt.* = try round.PhaseTimer.init(m.gpa, m.ctx.d);
        m.ptimer = pt;
        m.rd.timer = t;
    }

    /// The monotonic clock (ns).
    pub fn now(m: *const Model) u64 {
        return @intCast(std.Io.Timestamp.now(m.io, .awake).nanoseconds);
    }

    /// The round's logits, R rows of the vocabulary, on the host (after verify).
    /// The round's greedy tokens, R rows (the device's argmax of the round's logits, as sampling.argmax), on the host.
    pub fn roundArgmax(m: *Model, R: usize) ![]const u32 {
        if (m.sample_output != 0) return error.HeadLogitsUnavailable;
        const selected = try m.greedy_output.selected(R);
        const d = m.ctx.d;
        if (!selected)
            try m.ops.argmaxRows(m.stream, m.rd.logits, m.vocab, m.cfg.tokenVocab(), R, m.amax);
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.amax_host, m.amax, R * 4), "cuMemcpyDtoH");
        return m.amax_host[0..R];
    }

    const CandidateRows = struct { vals: []const f32, ids: []const i64, flags: []const i32 };

    /// Bounded sampled rows read only their sorted candidates unless any row needs CPU fallback.
    fn copyCandidates(m: *Model, logits: u64, R: usize, count: usize) !CandidateRows {
        if (m.cand == 0 or R == 0 or R > round.max_rows or count == 0 or count > sampling.max_candidates) return error.BadCandidateShape;
        const vals = m.cand;
        const ids = vals + round.max_rows * sampling.max_candidates * 4;
        const flags = ids + round.max_rows * sampling.max_candidates * 8;
        try m.ops.topCandRows(m.stream, logits, m.vocab, m.cfg.tokenVocab(), R, count, vals, ids, flags);
        return m.readCandidates(R, count);
    }

    fn readCandidates(m: *Model, R: usize, count: usize) !CandidateRows {
        const d = m.ctx.d;
        const vals = m.cand;
        const ids = vals + round.max_rows * sampling.max_candidates * 4;
        const flags = ids + round.max_rows * sampling.max_candidates * 8;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_flags, flags, R * 4), "cuMemcpyDtoH");
        var good = true;
        for (m.cand_flags[0..R]) |flag| good = good and flag == 0;
        if (good) {
            try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_vals, vals, R * count * 4), "cuMemcpyDtoH");
            try d.check(d.api.cuMemcpyDtoH_v2(&m.cand_ids, ids, R * count * 8), "cuMemcpyDtoH");
        }
        return .{ .vals = m.cand_vals[0 .. R * count], .ids = m.cand_ids[0 .. R * count], .flags = m.cand_flags[0..R] };
    }

    pub fn roundCandidates(m: *Model, R: usize, count: usize) !CandidateRows {
        if (m.sample_output != 0) {
            if (count != m.sample_output or R != m.rd.r) return error.BadCandidateShape;
            return m.readCandidates(R, count);
        }
        try m.greedy_output.requireLogits();
        return m.copyCandidates(m.rd.logits, R, count);
    }

    pub fn promptCandidates(m: *Model, count: usize) !CandidateRows {
        return m.copyCandidates(m.ch.head_g, 1, count);
    }

    pub fn roundLogits(m: *Model, R: usize) ![]const f32 {
        try m.greedy_output.requireLogits();
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.rd.logits, R * m.vocab * 4), "cuMemcpyDtoH");
        m.cfg.maskHostLogits(m.logits[0 .. R * m.vocab]);
        return m.logits[0 .. R * m.vocab];
    }

    /// Rows first .. first + n of the round's logits on the host, at their places in roundLogits' layout (the other
    /// rows are left as they were): the grammar-masked windows' rows when the rest of the round needs none.
    pub fn roundLogitRows(m: *Model, first: usize, n: usize) !void {
        try m.greedy_output.requireLogits();
        const d = m.ctx.d;
        const V = m.vocab;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits[first * V ..].ptr, m.rd.logits + first * V * 4, n * V * 4), "cuMemcpyDtoH");
        m.cfg.maskHostLogits(m.logits[first * V .. (first + n) * V]);
    }

    /// The prompt's logits on the host (after a fill's last chunk).
    pub fn promptLogitsHost(m: *Model) ![]const f32 {
        const d = m.ctx.d;
        try m.stream.synchronize();
        try d.check(d.api.cuMemcpyDtoH_v2(m.logits.ptr, m.ch.head_g, m.vocab * 4), "cuMemcpyDtoH");
        m.cfg.maskHostLogits(m.logits[0..m.vocab]);
        return m.logits[0..m.vocab];
    }

    /// The drafter's batched pass for N streams: each one's pending token at position q0 in its slot (its rings hold
    /// its positions below q0), `steps` drafts each (the drafter's drafts and confidences on the host).
    pub fn pass(m: *Model, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, keyed: ?draft.Keyed) !void {
        if (m.dr == null) return error.NoDrafter;
        const dr = &m.dr.?;
        const pool = &m.dpool.?;
        const graphs = dr.graphs;
        if (m.rs.live) dr.graphs = null;
        defer dr.graphs = graphs;
        for (q0, slots) |q, s| pool.absorbed[@intCast(s)] = @intCast(q);
        try dr.pass(&m.eng, &m.ch, pool, tokens, q0, slots, steps, null, keyed);
    }
};

test "pickWindow: --context wins and must match TF_DS_WINDOW; a follower takes TF_DS_WINDOW, else rank 0's" {
    try std.testing.expectEqual(@as(?usize, 1048576), try Model.pickWindow(1048576, 2101248, null));
    try std.testing.expectEqual(@as(?usize, 1048576), try Model.pickWindow(1048576, 2101248, "1048576"));
    try std.testing.expectEqual(@as(?usize, 1048576), try Model.pickWindow(1048576, 2101248, " "));
    try std.testing.expectError(error.BadWindow, Model.pickWindow(1048576, 2101248, "65536"));
    try std.testing.expectError(error.BadWindow, Model.pickWindow(null, 2101248, "x"));
    try std.testing.expectEqual(@as(?usize, 65536), try Model.pickWindow(null, 2101248, "65536"));
    try std.testing.expectEqual(@as(?usize, 2101248), try Model.pickWindow(null, 2101248, "9999999")); // never past the pool
    try std.testing.expectEqual(@as(?usize, null), try Model.pickWindow(null, 2101248, null));
}

test "decode greedy head actual Model rejects stale logits and invalid row reads before driver access" {
    var m: Model = undefined; // no CUDA driver exists in this test
    m.greedy_output = .{ .rows = 2 };
    m.sample_output = 0;
    try std.testing.expectError(error.HeadLogitsUnavailable, m.roundLogits(2));
    try std.testing.expectError(error.HeadLogitsUnavailable, m.roundLogitRows(0, 1));
    try std.testing.expectError(error.HeadLogitsUnavailable, m.roundCandidates(2, 1));
    try std.testing.expectError(error.BadGreedyHeadRows, m.roundArgmax(1));
}

test "decode greedy head actual Model invalid route resets previous readiness without CUDA" {
    var m: Model = undefined;
    m.greedy_output = .{ .rows = 16 };
    m.rd.greedy_out = 17;
    m.decode_greedy_head = false;
    try std.testing.expectError(error.GreedyHeadRankFault, m.verifyHead(undefined, &.{}, true));
    try std.testing.expectEqual(@as(usize, 0), m.greedy_output.rows);
    try std.testing.expectEqual(@as(u64, 0), m.rd.greedy_out);
}


test "decode greedy head actual Model selected output fences then copies only token ids" {
    const Fake = struct {
        var phase: usize = 0;
        var good: bool = true;
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            good = good and phase % 2 == 0;
            phase += 1;
            return cuda.abi.success;
        }
        fn copy(dst: ?*anyopaque, src: cuda.abi.DevicePtr, bytes: usize) callconv(.c) cuda.abi.Result {
            good = good and phase % 2 == 1 and src == 0x12340 and bytes == 8;
            phase += 1;
            const ids: *[2]u32 = @ptrCast(@alignCast(dst.?));
            ids.* = .{ 43136, 129279 };
            return cuda.abi.success;
        }
    };
    Fake.phase = 0;
    Fake.good = true;
    var driver: cuda.Driver = undefined;
    driver.api.cuStreamSynchronize = Fake.sync;
    driver.api.cuMemcpyDtoH_v2 = Fake.copy;
    // Pure host registry fixture; no Driver.open, cuInit or GPU context creation.
    const health = cuda.context_health;
    const fake_context: cuda.abi.Context = @ptrFromInt(0x7dd85002);
    try health.retain(@intFromPtr(&driver), fake_context);
    defer health.release(@intFromPtr(&driver), fake_context);
    health.setCurrent(@intFromPtr(&driver), fake_context);
    defer health.clearCurrent(@intFromPtr(&driver), fake_context);
    var owner = health.Owner.capture();
    defer owner.release();
    var ctx: cuda.Context = undefined;
    ctx.d = &driver;
    var m: Model = undefined;
    m.ctx = &ctx;
    m.stream = .{ .d = &driver, .owner = owner, .handle = null };
    m.amax = 0x12340;
    m.greedy_output = .{ .rows = 2 };
    m.sample_output = 0;
    // Ops and logits are deliberately undefined: selected output cannot touch them.
    try std.testing.expectEqualSlices(u32, &.{ 43136, 129279 }, try m.roundArgmax(2));
    try std.testing.expectEqualSlices(u32, &.{ 43136, 129279 }, try m.roundArgmax(2));
    try std.testing.expectEqual(@as(usize, 4), Fake.phase);
    try std.testing.expect(Fake.good);
}

test "decode greedy head sampled custody refuses stale logits and resets invalid routes" {
    var m: Model = undefined;
    m.sample_output = 28;
    m.greedy_output = .{ .rows = 2 };
    m.rd.r = 2;
    try std.testing.expectError(error.HeadLogitsUnavailable, m.roundArgmax(2));
    try std.testing.expectError(error.HeadLogitsUnavailable, m.roundLogits(2));
    try std.testing.expectError(error.BadCandidateShape, m.roundCandidates(2, 20));
    try std.testing.expectError(error.BadCandidateShape, m.roundCandidates(1, 28));
    m.decode_sample_head = false;
    try std.testing.expectError(error.GreedyHeadRankFault, m.verifySampleHead(undefined, &.{}, 28));
    try std.testing.expectEqual(@as(usize, 0), m.sample_output);
    try std.testing.expectEqual(@as(usize, 0), m.rd.sample_count);
    try std.testing.expectEqual(@as(usize, 0), m.greedy_output.rows);
}

test "prefill slice bounds preserve complete layer order and whole path" {
    for ([_]usize{ 1, 20, 40 }) |layers| for ([_]usize{ 0, 1, 5, 10, 40 }) |per| {
        var lo: usize = 0;
        while (lo < layers) {
            const hi = sliceHi(layers, per, lo);
            try std.testing.expect(hi > lo and hi <= layers);
            if (per > 0) try std.testing.expect(hi - lo <= per);
            lo = hi;
        }
        try std.testing.expectEqual(layers, lo);
    };
    try std.testing.expectEqual(@as(usize, 40), sliceHi(40, 0, 0));
}

test "prefill slice resume after park validates exact arguments and rejects replay or skipped layers" {
    const a: SliceArgs = .{ .start = 8192, .n = 8192, .replay = 32768 };
    var rs: SliceResume = .{};
    try rs.validate(a, 0, 5, 40, 16384);
    rs = .{ .live = true, .next = 5, .args = a, .shared = .{ .kv_layer = 3, .kk = 4, .pool = true }, .floor = 7 };
    try rs.validate(a, 5, 10, 40, 16384);
    try std.testing.expectError(error.OpenPrefillChunk, rs.validate(a, 0, 5, 40, 16384));
    try std.testing.expectError(error.BadSlice, rs.validate(a, 10, 15, 40, 16384));
    try std.testing.expectError(error.BadSlice, rs.validate(.{ .start = 8192, .n = 8192, .replay = 0 }, 5, 10, 40, 16384));
    try std.testing.expectError(error.BadSlice, rs.validate(a, 5, 41, 40, 16384));
    try std.testing.expectEqual(@as(?usize, 3), rs.shared.kv_layer);
    try std.testing.expectEqual(@as(usize, 7), rs.floor);
    var model_: Model = undefined;
    model_.rs = .{};
    model_.rs = rs;
    model_.rs.ahead = try std.Thread.spawn(.{}, struct {
        fn run() void {}
    }.run, .{});
    model_.ahead = null;
    model_.dropChunk();
    try std.testing.expect(!model_.rs.live and model_.rs.ahead == null);
    try std.testing.expectError(error.BadSlice, model_.rs.validate(a, 5, 10, 40, 16384));
}

fn prefillAgreement(leader: u64, follower: u64) !void {
    if (leader != follower) return error.PrefillSettingMismatch;
}

test "prefill slice mismatch refuses either follower and off word stays exact" {
    const base = @as(u64, 1) | @as(u64, 1) << 1 | (prefill128.Flags{ .ac = true, .rows = 8192, .w = true, .w_range16 = true, .shared_split = true }).bits() << 2;
    for (0..2) |mismatch| {
        var followers = [_]u64{ base | @as(u64, 5) << 26, base | @as(u64, 5) << 26 };
        try prefillAgreement(followers[0], followers[1]);
        followers[mismatch] = base | @as(u64, 10) << 26;
        try std.testing.expectError(error.PrefillSettingMismatch, prefillAgreement(base | @as(u64, 5) << 26, followers[mismatch]));
    }
    try prefillAgreement(base, base | (prefill128.Flags{}).bits() << 2);
}
