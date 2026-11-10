//! One rank's weights on the GPU in the served engine's layouts (weights.py, exl3/linear.py, exl3/experts.py): EXL3
//! linears in strips, each layer's experts in one trellis buffer with pointer, width and scale tables, the rest as the
//! loader keeps it. Every device tensor is listed by the Python engine's name for digests against its weights.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const rank_cache = @import("rank_cache.zig");
const checkpoint = @import("checkpoint.zig");
const dio = @import("core").direct_io;

/// The largest stretch of the rank cache read whole for one layer's experts.
const max_span = 4 << 30;
/// The size of one direct read within such a stretch.
const span_piece = 64 << 20;

/// An EXL3 linear in the strips layout: words int32 [N/128, K/16, 8, 8 * bits], fp16 suh [K] and svh [N].
/// plan_n: the N its decode plan is made for, when not n: a 2D node's column part of a TP2 slice keeps that slice's
/// plan (linear.py pinned plans), so its columns sum in the slice's K order.
pub const Linear = struct { words: u64, suh: u64, svh: u64, k: u32, n: u32, k2: u32, plan_n: u32 = 0 };

/// A layer's routed experts and its shared expert (the last entry), as exl3_experts.prepare lays them out. On a 2D node
/// (exl3/experts2d.py prepare) gate / up hold its intermediate blocks (width of the half's down_k) and down its
/// output columns (down_n of dims) over the half's whole intermediate.
pub const Experts = struct {
    trellis: u64, // gate trellises of every expert, then up, then down (pack_trellises)
    gate_ptr: u64, // int64 [E]
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64, // int32 [E]: half-bits a value
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]: down's input signs on the gate / up blocks here
    svh_d: u64, // fp16 [E, down_n]
    count: u32,
    dims: u32,
    width: u32,
    k2_gu: [2]u32,
    k2_d: [2]u32,
    // Zero means unavailable in manual fixtures. Loader always initializes all
    // four bounds, retaining the original all-expert bounds above verbatim.
    routed_k2_gu: [2]u32 = .{ 0, 0 },
    routed_k2_d: [2]u32 = .{ 0, 0 },
    shared_k2_gu: [2]u32 = .{ 0, 0 },
    shared_k2_d: [2]u32 = .{ 0, 0 },
    down_k: u32, // down's K: the rank's whole intermediate (width but on a 2D node)
    down_n: u32, // down's output columns here (dims but on a 2D node)
    shared_gate: [2]u64 = .{ 0, 0 }, // the shared expert's gate trellis: its first byte and length (the paced L2 prefetch)
    shared_up: [2]u64 = .{ 0, 0 }, // its up and down trellises the same way (the served "moe" fork reads all three)
    shared_down: [2]u64 = .{ 0, 0 },
};

/// Projection-major gate/up/down metadata; shared is the last expert, not
/// necessarily the last active group-place. All-expert bounds remain separate.
pub fn expertK2Bounds(k2s: []const i32, count: usize, shared: bool, gateup: bool) [2]u32 {
    std.debug.assert(count > 0 and k2s.len == 3 * count);
    if (!shared and count == 1) return .{ 0, 0 };
    var bounds: [2]u32 = .{ std.math.maxInt(u32), 0 };
    const first: usize = if (gateup) 0 else 2;
    const end: usize = if (gateup) 2 else 3;
    for (first..end) |projection| {
        const start: usize = if (shared) count - 1 else 0;
        const stop: usize = if (shared) count else count - 1;
        for (start..stop) |e| {
            const v: u32 = @intCast(k2s[projection * count + e]);
            bounds = .{ @min(bounds[0], v), @max(bounds[1], v) };
        }
    }
    return bounds;
}

test "prefill128 shared split K2 ranges match all 43 rank0 metadata records" {
    const Bin = struct { k2: i32, count: usize };
    const Record = struct { layer: []const u8, count: usize, routed: [3][]const Bin, shared: [3]i32 };
    const Fixture = struct { source_sha256: []const u8, records: []const Record };
    const parsed = try std.json.parseFromSlice(Fixture, std.testing.allocator, @embedFile("fixtures/shared_split_k2.json"), .{});
    defer parsed.deinit();
    var counts: [3]usize = @splat(0);
    for (parsed.value.records) |record| {
        const k2s = try std.testing.allocator.alloc(i32, record.count * 3);
        defer std.testing.allocator.free(k2s);
        for (0..3) |projection| {
            var at: usize = 0;
            for (record.routed[projection]) |bin| {
                @memset(k2s[projection * record.count + at ..][0..bin.count], bin.k2);
                at += bin.count;
            }
            try std.testing.expectEqual(record.count - 1, at);
            k2s[(projection + 1) * record.count - 1] = record.shared[projection];
        }
        const gu = expertK2Bounds(k2s, record.count, false, true);
        const down = expertK2Bounds(k2s, record.count, false, false);
        const shared_gu = expertK2Bounds(k2s, record.count, true, true);
        const shared_down = expertK2Bounds(k2s, record.count, true, false);
        const xe = @import("exl3_experts.zig");
        const index = xe.range(gu[0], gu[1]);
        try std.testing.expectEqual(index, xe.range(down[0], down[1]));
        if (std.mem.startsWith(u8, record.layer, "layers.")) {
            const layer = try std.fmt.parseInt(usize, record.layer[7..], 10);
            const expected: usize = if (layer < 10 or layer >= 38) 0 else 1;
            try std.testing.expectEqual(expected, index);
            try std.testing.expectEqual(@as(usize, 2), xe.range(shared_gu[0], shared_gu[1]));
            try std.testing.expectEqual(@as(usize, 2), xe.range(shared_down[0], shared_down[1]));
            counts[index] += 1;
        } else {
            try std.testing.expectEqual(@as(usize, 0), index);
            try std.testing.expectEqual([2]u32{ 8, 8 }, shared_gu);
            counts[2] += 1;
        }
    }
    try std.testing.expectEqual([3]usize{ 12, 28, 3 }, counts);
    // GU bounds must include BOTH projections; down has independent metadata.
    const asymmetric = [_]i32{ 8, 12, 6, 10, 8, 12 };
    try std.testing.expectEqual([2]u32{ 6, 8 }, expertK2Bounds(&asymmetric, 2, false, true));
    try std.testing.expectEqual([2]u32{ 10, 12 }, expertK2Bounds(&asymmetric, 2, true, true));
}

pub const Layer = struct {
    idx: u32,
    ratio: u8,
    hc_attn: [3]u64 = .{ 0, 0, 0 }, // fn [24, hc * d] f32, scale [3], base [24]
    hc_ffn: [3]u64 = .{ 0, 0, 0 },
    attn_norm: u64 = 0,
    ffn_norm: u64 = 0,
    wq_a: Linear = undefined,
    q_norm: u64 = 0,
    wq_b: Linear = undefined, // this rank's heads' columns
    wkv: Linear = undefined,
    kv_norm: u64 = 0,
    sink: u64 = 0, // this rank's heads, f32
    wo_a: [4]Linear = undefined, // this rank's groups
    groups: u32 = 4, // wo_a's entries loaded: the rank's groups, its pair's half of them on a 2D node
    wo_b: Linear = undefined, // this rank's groups' input rows (a 2D node: its pair's output columns of them)
    comp_wkv: ?Linear = null,
    comp_wgate: ?Linear = null,
    comp_norm: u64 = 0,
    idx_wq_b: ?Linear = null,
    idx_proj: u64 = 0, // [idx_heads, d] f32
    idx_proj_h: u64 = 0, // the same in fp16 (Model.__init__)
    idx_wk: ?Linear = null,
    idx_k_norm: u64 = 0,
    gate_w: u64 = 0, // [E, d] f16
    gate_b: u64 = 0, // [E] f32
    gate_b_vl: u64 = 0, // [E] f32: the routing bias of image-span tokens (the checkpoint's extra gate_bias_vl)
    experts: Experts = undefined,
    engram_wkv: ?Linear = null, // this rank's hash columns' input rows
    engram_qk: u64 = 0, // q_weight * k_weight [hc, d] f32
};

pub const DSpark = struct { blocks: []Layer, main_proj: Linear, main_norm: u64, norm: u64, markov_embed: u64, markov_head: u64, conf: u64 };

/// A device tensor by the Python engine's dotted name.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    embed: u64 = 0, // [V, d] bf16
    norm: u64 = 0,
    head: Linear = undefined, // this rank's vocabulary columns
    vocab_lo: usize = 0,
    vocab_hi: usize = 0,
    layers: []Layer = &.{},
    dspark: ?DSpark = null,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,
    bytes: u64 = 0,
    pending: ?cuda.DeviceBuffer = null,

    pub fn deinit(w: *Weights) void {
        const cleanup = cuda.context_health.cleanupMark();
        if (w.pending) |*b| {
            b.free();
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
            w.pending = null;
        }
        while (w.buffers.items.len > 0) {
            w.buffers.items[w.buffers.items.len - 1].free();
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
            w.buffers.items.len -= 1;
        }
        for (w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);
        w.named.deinit(w.gpa);
        if (w.dspark) |d| w.gpa.free(d.blocks);
        w.gpa.free(w.layers);
        w.* = undefined;
    }
};

/// Where entries come from: the lane's rank cache file, or the checkpoint read by the split.
pub const Source = union(enum) {
    cache: struct { file: std.Io.File, index: *const rank_cache.Index, io: std.Io, pf: ?*Prefetch = null },
    shards: *const checkpoint.Shards,

    pub fn size(s: Source, key: []const u8) !usize {
        return switch (s) {
            .cache => |c| (c.index.get(key) orelse return error.MissingEntry).bytes,
            .shards => |sh| (try checkpoint.describe(sh, key)).bytes,
        };
    }

    pub fn read(s: Source, key: []const u8, out: []u8) !void {
        switch (s) {
            .cache => |c| {
                const e = c.index.get(key) orelse return error.MissingEntry;
                if (e.bytes != out.len) return error.BadSize;
                if (try c.file.readPositionalAll(c.io, out, e.offset) != out.len) return error.ShortRead;
                if (c.pf) |pf| _ = pf.pos.fetchMax(e.offset + e.bytes, .monotonic);
            },
            .shards => |sh| try checkpoint.read(sh, key, out),
        }
    }
};

/// Trellis int16 [kt, nt, w16] as int32 words [nt / 8, kt, 8, w16 / 2] (linear.py strips): each 128-column block's
/// tiles in k order. `tile` is one tile's bytes (w16 * 2).
pub fn strips(src: []const u8, out: []u8, kt: usize, nt: usize, tile: usize) void {
    for (0..nt / 8) |j| for (0..kt) |i| {
        const from = (i * nt + j * 8) * tile;
        const to = (j * kt + i) * 8 * tile;
        @memcpy(out[to..][0 .. 8 * tile], src[from..][0 .. 8 * tile]);
    };
}

/// Fills `buf` from `f` at the aligned offset `base`, in direct reads of at most `piece` (aligned) bytes.
fn readWhole(f: dio.File, buf: dio.Buffer, base: u64, piece: usize) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = @min(buf.len - done, piece);
        _ = try f.read(@alignCast(buf[done..][0..n]), base + done, n);
        done += n;
    }
}

/// The rank cache read ahead of the loader (TF_DS_LOAD_PREFETCH=1 enables it; `0046` of the 4-Spark recipe, by
/// shalk3). The loader reads the entries it does not read as a span one at a time, in file order, each read waiting on
/// the disk. A few threads pread those parts of the file in chunks up to `lead` bytes past the loader's furthest read
/// and drop the bytes, so the loader's own reads come from the page cache; the weights are unchanged. A layer's expert
/// entries are skipped: the loader reads them whole with direct reads, which the page cache does not serve. A pread cannot
/// be interrupted, so on a hard-mounted NFS file an outage holds the load's end until the read returns; the recipe reads
/// its weight files from local disk.
const Prefetch = struct {
    io: std.Io,
    fd: std.posix.fd_t,
    chunks: []const [2]u64, // [offset, end) of each chunk to read ahead, in file order
    pos: std.atomic.Value(u64) = .init(0), // the end of the loader's furthest read
    next: std.atomic.Value(usize) = .init(0), // the next chunk to read ahead
    stop: std.atomic.Value(bool) = .init(false),
    lead: u64 = default_lead, // how far past the loader's furthest read the threads may read
    warmed: std.atomic.Value(u64) = .init(0), // the bytes the threads read
    threads: [threads_n]std.Thread = undefined,
    started: usize = 0,

    const threads_n = 2;
    const chunk: u64 = 32 << 20;
    const default_lead: u64 = 512 << 20;
    const page_budget: u64 = 4 << 30;

    fn on() bool {
        return selected(if (std.c.getenv("TF_DS_LOAD_PREFETCH")) |v| std.mem.span(v) else null);
    }
    fn selected(value: ?[]const u8) bool {
        return if (value) |v| std.mem.eql(u8, v, "1") else false;
    }

    /// The chunks of the bytes the loader reads one entry at a time: every entry but a layer's experts, adjacent
    /// entries joined, each run cut into pieces of at most `size` bytes.
    fn plan(gpa: std.mem.Allocator, index: *const rank_cache.Index, size: u64) ![][2]u64 {
        if (size == 0) return error.BadPrefetchChunk;
        var out: std.ArrayList([2]u64) = .empty;
        errdefer out.deinit(gpa);
        var pages: u64 = 0;
        var stretch: ?[2]u64 = null;
        for (index.entries.keys(), index.entries.values()) |key, e| {
            if (e.bytes == 0) continue;
            if (spanned(key)) continue;
            // Bound total requested page-cache warming, not just distance ahead.
            const footprint: u64 = dio.span(e.offset, @intCast(e.bytes));
            if (footprint > page_budget - pages) return error.PrefetchBudget;
            pages += footprint;
            if (stretch) |*r| if (r[1] == e.offset) {
                r[1] = e.offset + e.bytes;
                continue;
            };
            if (stretch) |r| try cut(gpa, &out, r, size);
            stretch = .{ e.offset, e.offset + e.bytes };
        }
        if (stretch) |r| try cut(gpa, &out, r, size);
        return out.toOwnedSlice(gpa);
    }

    fn spanned(key: []const u8) bool {
        return std.mem.indexOf(u8, key, ".ffn.experts.") != null or std.mem.indexOf(u8, key, ".ffn.shared_experts.") != null;
    }

    fn cut(gpa: std.mem.Allocator, out: *std.ArrayList([2]u64), r: [2]u64, size: u64) !void {
        var at = r[0];
        while (at < r[1]) : (at += size) try out.append(gpa, .{ at, @min(at + size, r[1]) });
    }

    fn start(p: *Prefetch) void {
        for (&p.threads) |*th| {
            th.* = std.Thread.spawn(.{}, run, .{p}) catch break;
            p.started += 1;
        }
    }

    fn finish(p: *Prefetch) void {
        p.stop.store(true, .monotonic);
        p.join();
    }

    fn join(p: *Prefetch) void {
        for (p.threads[0..p.started]) |th| th.join();
        p.started = 0;
    }

    fn run(p: *Prefetch) void {
        const buf = std.heap.page_allocator.alloc(u8, chunk) catch return;
        defer std.heap.page_allocator.free(buf);
        while (!p.stop.load(.monotonic)) {
            const i = p.next.fetchAdd(1, .monotonic);
            if (i >= p.chunks.len) return;
            const c = p.chunks[i];
            while (c[0] > p.pos.load(.monotonic) + p.lead) {
                if (p.stop.load(.monotonic)) return;
                std.Io.sleep(p.io, .fromMilliseconds(2), .awake) catch {};
            }
            const n: usize = @intCast(@min(c[1] - c[0], chunk));
            var done: usize = 0;
            while (done < n) {
                const r = std.c.pread(p.fd, buf.ptr + done, n - done, @intCast(c[0] + done));
                if (r < 0) {
                    if (std.c.errno(r) == .INTR) continue;
                    return; // the loader reads it itself
                }
                if (r == 0) return;
                done += @intCast(r);
                _ = p.warmed.fetchAdd(@intCast(r), .monotonic);
            }
        }
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    src: Source,
    w: *Weights,
    host: std.ArrayList(u8) = .empty,
    aux: std.ArrayList(u8) = .empty,
    // A layer's expert entries read whole: experts() visits them out of file order, which defeats kernel readahead.
    ahead: bool = false,
    span: []u8 = &.{},
    span_lo: u64 = 0,
    span_buf: ?dio.Buffer = null, // page_allocator's, so its pages go back to the system before the pool is sized
    direct: ?dio.File = null, // the cache reopened for direct reads (no page cache to fill and then reclaim)
    span_bytes: u64 = 0, // bytes read as spans and the time they took, for the load's log line
    span_ns: u64 = 0,

    fn deinit(L: *Loader) void {
        L.host.deinit(L.gpa);
        L.aux.deinit(L.gpa);
        if (L.span_buf) |b| std.heap.page_allocator.free(b);
        if (L.direct) |*f| f.close();
    }

    fn upload(L: *Loader, name: []const u8, bytes: []const u8) !u64 {
        // Custody precedes upload and every fallible host bookkeeping operation.
        // The enclosing heap Weights owns pending even when append cannot allocate.
        try cuda.DeviceBuffer.allocInto(L.d, bytes.len, &L.w.pending);
        try L.w.pending.?.upload(0, bytes);
        try L.w.buffers.append(L.gpa, L.w.pending.?);
        const ptr = L.w.pending.?.ptr;
        L.w.pending = null; // exactly one ledger owns the device allocation
        const owned_name = try L.gpa.dupe(u8, name);
        errdefer L.gpa.free(owned_name);
        try L.w.named.append(L.gpa, .{ .name = owned_name, .ptr = ptr, .len = bytes.len });
        L.w.bytes += bytes.len;
        return ptr;
    }

    /// The entry's bytes, valid until the next entry() (a view into the span until endSpan).
    fn entry(L: *Loader, key: []const u8) ![]u8 {
        if (L.spanned(key)) |bytes| return bytes;
        try L.host.resize(L.gpa, try L.src.size(key));
        try L.src.read(key, L.host.items);
        return L.host.items;
    }

    /// The entry's bytes in the span, when it lies inside it.
    fn spanned(L: *const Loader, key: []const u8) ?[]u8 {
        if (L.span.len == 0) return null;
        const c = switch (L.src) {
            .cache => |c| c,
            .shards => return null,
        };
        const e = c.index.get(key) orelse return null;
        if (e.offset < L.span_lo or e.offset + e.bytes > L.span_lo + L.span.len) return null;
        return L.span[@intCast(e.offset - L.span_lo)..][0..@intCast(e.bytes)];
    }

    /// The cache bytes [lo, hi) holding every part experts() reads of a layer; null without a cache or an entry.
    fn expertSpan(L: *const Loader, prefixes: []const [3][]u8, pq: anytype) !?[2]u64 {
        const c = switch (L.src) {
            .cache => |c| c,
            .shards => return null,
        };
        var kb: [256]u8 = undefined;
        var lo: u64 = std.math.maxInt(u64);
        var hi: u64 = 0;
        for (prefixes) |pp| for (0..3) |j| for ([_][]const u8{ "tr", "suh", "svh" }) |part| {
            const e = c.index.get(try plan.partKey(&kb, pp[j], pq.cols(j), pq.rows(j), part)) orelse return null;
            lo = @min(lo, e.offset);
            hi = @max(hi, e.offset + e.bytes);
        };
        return if (hi > lo) .{ lo, hi } else null;
    }

    /// Serves the entries in the cache bytes [lo, hi) from one whole read until endSpan; on failure each is read alone.
    fn beginSpan(L: *Loader, lo: u64, hi: u64) void {
        L.readSpan(lo, hi) catch |err| {
            std.log.warn("rank cache: reading bytes {d}..{d} whole failed ({t}); their entries are read one by one", .{ lo, hi, err });
            L.span = &.{};
        };
    }

    fn readSpan(L: *Loader, lo: u64, hi: u64) !void {
        const c = switch (L.src) {
            .cache => |c| c,
            .shards => return,
        };
        if (hi <= lo or hi - lo > max_span) return;
        if (L.direct == null) {
            var path: [64]u8 = undefined;
            L.direct = try dio.File.open(try std.fmt.bufPrintSentinel(&path, "/proc/self/fd/{d}", .{c.file.handle}, 0));
        }
        if (!L.direct.?.direct) return error.DirectReadUnavailable;
        const base = std.mem.alignBackward(u64, lo, dio.alignment);
        const need = dio.span(lo, @intCast(hi - lo));
        if (need > max_span) return;
        if (L.span_buf == null or L.span_buf.?.len < need) {
            if (L.span_buf) |b| std.heap.page_allocator.free(b);
            L.span_buf = null;
            L.span_buf = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(dio.alignment), need);
        }
        const t0 = std.Io.Timestamp.now(c.io, .awake);
        try readWhole(L.direct.?, L.span_buf.?[0..need], base, span_piece);
        L.span_ns += @intCast(t0.durationTo(std.Io.Timestamp.now(c.io, .awake)).nanoseconds);
        L.span_bytes += need;
        L.span = L.span_buf.?[@intCast(lo - base)..][0..@intCast(hi - lo)];
        L.span_lo = lo;
        if (c.pf) |pf| _ = pf.pos.fetchMax(hi, .monotonic);
    }

    fn endSpan(L: *Loader) void {
        L.span = &.{};
    }

    /// plain(name, dtype).cuda(): the entry as it is.
    fn plain(L: *Loader, tensor: []const u8, dtype: ?rank_cache.DType) !u64 {
        var kb: [256]u8 = undefined;
        return L.upload(tensor, try L.entry(try plan.plainKey(&kb, tensor, dtype)));
    }

    /// linear(prefix, cols, rows): Exl3Linear.from_tensors in strips.
    fn linear(L: *Loader, prefix: []const u8, cols: ?[2]usize, rows: ?[2]usize) !Linear {
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        const tr = try L.entry(try plan.partKey(&kb, prefix, cols, rows, "tr"));
        const t = try L.shape(try plan.partKey(&kb, prefix, cols, rows, "tr"));
        const kt = t[0];
        const nt = t[1];
        const w16 = t[2];
        if (w16 % 8 != 0 or nt % 8 != 0) return error.UnexpectedTensor;
        try L.aux.resize(L.gpa, tr.len);
        strips(tr, L.aux.items, kt, nt, w16 * 2);
        const words = try L.upload(try std.fmt.bufPrint(&nb, "{s}.words", .{prefix}), L.aux.items);
        const suh = try L.upload(try std.fmt.bufPrint(&nb, "{s}.suh", .{prefix}), try L.entry(try plan.partKey(&kb, prefix, cols, rows, "suh")));
        const svh = try L.upload(try std.fmt.bufPrint(&nb, "{s}.svh", .{prefix}), try L.entry(try plan.partKey(&kb, prefix, cols, rows, "svh")));
        return .{ .words = words, .suh = suh, .svh = svh, .k = @intCast(kt * 16), .n = @intCast(nt * 16), .k2 = @intCast(w16 / 8) };
    }

    /// A trellis entry's [kt, nt, w16].
    fn shape(L: *Loader, key: []const u8) ![3]usize {
        return switch (L.src) {
            .cache => |c| blk: {
                const e = c.index.get(key) orelse return error.MissingEntry;
                if (e.rank != 3) return error.UnexpectedTensor;
                break :blk .{ e.shape[0], e.shape[1], e.shape[2] };
            },
            .shards => |sh| blk: {
                const s = try checkpoint.describe(sh, key);
                if (s.rank != 3) return error.UnexpectedTensor;
                break :blk .{ s.shape[0], s.shape[1], s.shape[2] };
            },
        };
    }

    /// load_block's experts: exl3_parts of w1 / w3 by output columns and w2 by input rows for every expert and the
    /// shared one, packed as pack_trellises packs them, with prepare's tables. `half`: the rank's intermediate; `gu`:
    /// gate / up's columns of it here (the half, or a 2D node's blocks); `dcols`: down's output columns (null: all).
    fn experts(L: *Loader, p: []const u8, n: usize, half: [2]usize, gu: [2]usize, dcols: ?[2]usize, d: usize) !Experts {
        const e_count = n + 1;
        const width = gu[1] - gu[0];
        const down_k = half[1] - half[0];
        const down_n = if (dcols) |dc| dc[1] - dc[0] else d;
        var kb: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        const prefixes = try L.gpa.alloc([3][]u8, e_count);
        for (prefixes) |*pp| pp.* = .{ &.{}, &.{}, &.{} };
        defer {
            for (prefixes) |pp| for (pp) |x| L.gpa.free(x);
            L.gpa.free(prefixes);
        }
        for (0..e_count) |e| {
            const ep = if (e < n) try std.fmt.bufPrint(&nb, "{s}.ffn.experts.{d}", .{ p, e }) else try std.fmt.bufPrint(&nb, "{s}.ffn.shared_experts", .{p});
            for ([_][]const u8{ "w1", "w3", "w2" }, 0..) |proj, j| prefixes[e][j] = try std.fmt.allocPrint(L.gpa, "{s}.{s}", .{ ep, proj });
        }
        // exl3_parts' (cols, rows) of projection j: gate / up by output columns, down by input rows (and on a 2D
        // node by its output columns too)
        const Parts = struct {
            gu: [2]usize,
            half: [2]usize,
            dcols: ?[2]usize,
            fn cols(q: @This(), j: usize) ?[2]usize {
                return if (j < 2) q.gu else q.dcols;
            }
            fn rows(q: @This(), j: usize) ?[2]usize {
                return if (j == 2) q.half else null;
            }
        };
        const pq: Parts = .{ .gu = gu, .half = half, .dcols = dcols };
        // the parts below sit together in the cache, expert by expert: read them whole, then visit them in memory
        if (L.ahead) if (try L.expertSpan(prefixes, pq)) |extent| L.beginSpan(extent[0], extent[1]);
        defer if (L.ahead) L.endSpan();
        // sizes first: one buffer, gate trellises of every expert, then up, then down
        const sizes = try L.gpa.alloc(usize, 3 * e_count);
        defer L.gpa.free(sizes);
        var total: usize = 0;
        for (0..3) |j| for (0..e_count) |e| {
            const key = try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), "tr");
            sizes[j * e_count + e] = try L.src.size(key);
            total += sizes[j * e_count + e];
        };
        try cuda.DeviceBuffer.allocInto(L.d, total, &L.w.pending);
        const ptrs = try L.gpa.alloc(u64, 3 * e_count);
        defer L.gpa.free(ptrs);
        const k2s = try L.gpa.alloc(i32, 3 * e_count);
        defer L.gpa.free(k2s);
        var at: usize = 0;
        for (0..3) |j| for (0..e_count) |e| {
            const key = try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), "tr");
            const tr = try L.entry(key);
            try L.w.pending.?.upload(at, tr);
            const t = try L.shape(key);
            ptrs[j * e_count + e] = L.w.pending.?.ptr + at;
            k2s[j * e_count + e] = @intCast(t[2] / 8);
            at += sizes[j * e_count + e];
        };
        const trellis = L.w.pending.?.ptr;
        try L.w.buffers.append(L.gpa, L.w.pending.?);
        L.w.pending = null; // ownership transferred exactly once
        const owned_name = try std.fmt.allocPrint(L.gpa, "{s}.ffn.experts.trellis", .{p});
        {
            errdefer L.gpa.free(owned_name);
            try L.w.named.append(L.gpa, .{ .name = owned_name, .ptr = trellis, .len = total });
        }
        L.w.bytes += total;
        var x: Experts = undefined;
        x.trellis = trellis;
        x.shared_gate = .{ ptrs[n], sizes[n] }; // j = 0 (gate), e = n (the shared expert)
        x.shared_up = .{ ptrs[e_count + n], sizes[e_count + n] };
        x.shared_down = .{ ptrs[2 * e_count + n], sizes[2 * e_count + n] };
        x.count = @intCast(e_count);
        x.dims = @intCast(d);
        x.width = @intCast(width);
        x.down_k = @intCast(down_k);
        x.down_n = @intCast(down_n);
        const tables = [_]*u64{ &x.gate_ptr, &x.up_ptr, &x.down_ptr };
        const k2t = [_]*u64{ &x.gate_k2, &x.up_k2, &x.down_k2 };
        for (0..3) |j| {
            tables[j].* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.ptr{d}", .{ p, j }), std.mem.sliceAsBytes(ptrs[j * e_count ..][0..e_count]));
            k2t[j].* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.k2{d}", .{ p, j }), std.mem.sliceAsBytes(k2s[j * e_count ..][0..e_count]));
        }
        x.k2_gu = .{ std.math.maxInt(u32), 0 };
        x.k2_d = .{ std.math.maxInt(u32), 0 };
        for (k2s[0 .. 2 * e_count]) |v| x.k2_gu = .{ @min(x.k2_gu[0], @as(u32, @intCast(v))), @max(x.k2_gu[1], @as(u32, @intCast(v))) };
        for (k2s[2 * e_count ..]) |v| x.k2_d = .{ @min(x.k2_d[0], @as(u32, @intCast(v))), @max(x.k2_d[1], @as(u32, @intCast(v))) };
        x.routed_k2_gu = expertK2Bounds(k2s, e_count, false, true);
        x.routed_k2_d = expertK2Bounds(k2s, e_count, false, false);
        x.shared_k2_gu = expertK2Bounds(k2s, e_count, true, true);
        x.shared_k2_d = expertK2Bounds(k2s, e_count, true, false);
        // stack(mats, j, n): every expert's scale row in order; suh_d is down's input signs on gate / up's blocks
        // (experts2d.prepare: the half's [gu - half), all of it but on a 2D node)
        const off = gu[0] - half[0];
        const stacks = [_]struct { dst: *u64, proj: usize, part: []const u8, len: usize, src_len: usize, off: usize, name: []const u8 }{
            .{ .dst = &x.suh_g, .proj = 0, .part = "suh", .len = d, .src_len = d, .off = 0, .name = "suh_g" },
            .{ .dst = &x.suh_u, .proj = 1, .part = "suh", .len = d, .src_len = d, .off = 0, .name = "suh_u" },
            .{ .dst = &x.svh_g, .proj = 0, .part = "svh", .len = width, .src_len = width, .off = 0, .name = "svh_g" },
            .{ .dst = &x.svh_u, .proj = 1, .part = "svh", .len = width, .src_len = width, .off = 0, .name = "svh_u" },
            .{ .dst = &x.suh_d, .proj = 2, .part = "suh", .len = width, .src_len = down_k, .off = off, .name = "suh_d" },
            .{ .dst = &x.svh_d, .proj = 2, .part = "svh", .len = down_n, .src_len = down_n, .off = 0, .name = "svh_d" },
        };
        for (stacks) |s| {
            try L.aux.resize(L.gpa, e_count * s.len * 2);
            for (0..e_count) |e| {
                const j = s.proj;
                const row = try L.entry(try plan.partKey(&kb, prefixes[e][j], pq.cols(j), pq.rows(j), s.part));
                if (row.len != s.src_len * 2) return error.UnexpectedTensor;
                @memcpy(L.aux.items[e * s.len * 2 ..][0 .. s.len * 2], row[s.off * 2 ..][0 .. s.len * 2]);
            }
            s.dst.* = try L.upload(try std.fmt.bufPrint(&nb, "{s}.ffn.experts.{s}", .{ p, s.name }), L.aux.items);
        }
        return x;
    }

    fn block(L: *Loader, c: Config, s: plan.Split, p: []const u8, i: usize, n_experts: usize) !Layer {
        var nb: [256]u8 = undefined;
        const gl = s.groups(c);
        var lay: Layer = .{ .idx = @intCast(i), .ratio = c.ratios[i] };
        for ([_][]const u8{ "fn", "scale", "base" }, 0..) |part, k| {
            lay.hc_attn[k] = try L.plain(try std.fmt.bufPrint(&nb, "{s}.hc_attn_{s}", .{ p, part }), .f32);
            lay.hc_ffn[k] = try L.plain(try std.fmt.bufPrint(&nb, "{s}.hc_ffn_{s}", .{ p, part }), .f32);
        }
        lay.attn_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn_norm.weight", .{p}), null);
        lay.ffn_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn_norm.weight", .{p}), null);
        lay.wq_a = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wq_a", .{p}), null, null);
        lay.q_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.q_norm.weight", .{p}), null);
        lay.wq_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wq_b", .{p}), s.wqB(c), null);
        lay.wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wkv", .{p}), null, null);
        lay.kv_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.kv_norm.weight", .{p}), null);
        {
            var kb: [256]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{s}.attn.attn_sink", .{p});
            const all = try L.entry(try plan.plainKey(&kb, name, .f32));
            const hs = s.headSpan(c);
            lay.sink = try L.upload(name, all[hs[0] * 4 ..][0 .. hs[1] * 4]);
        }
        const ga = s.woAGroups(c);
        for (0..ga[1]) |g| lay.wo_a[g] = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wo_a.slice.{d}", .{ p, ga[0] + g }), null, null);
        lay.groups = @intCast(ga[1]);
        lay.wo_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.wo_b", .{p}), s.woBCols(c), s.range(gl * c.o_lora));
        if (s.pair != null) {
            // the 2D column parts keep their TP2 slices' decode plans
            lay.wq_b.plan_n = @intCast(s.heads(c) * c.head_dim);
            lay.wo_b.plan_n = @intCast(c.hidden);
        }
        if (c.kv_sources.has(i)) {
            lay.comp_wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.wkv", .{p}), null, null);
            lay.comp_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.norm.weight", .{p}), null);
            if (c.ratios[i] > 1) lay.comp_wgate = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.compressor.wgate", .{p}), null, null);
        }
        if (c.index_sources.has(i)) {
            lay.idx_wq_b = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.wq_b", .{p}), null, null);
            const name = try std.fmt.bufPrint(&nb, "{s}.attn.indexer.weights_proj.weight", .{p});
            var kb: [256]u8 = undefined;
            const f32s = try L.entry(try plan.plainKey(&kb, name, .f32));
            lay.idx_proj = try L.upload(name, f32s);
            // Model.__init__: idx_proj.to(float16), the stored f16 values back (the f32 copy widened them exactly)
            try L.aux.resize(L.gpa, f32s.len / 2);
            for (0..f32s.len / 4) |k| {
                const v: f32 = @bitCast(std.mem.readInt(u32, f32s[4 * k ..][0..4], .little));
                std.mem.writeInt(u16, L.aux.items[2 * k ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little);
            }
            lay.idx_proj_h = try L.upload(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.weights_proj_h", .{p}), L.aux.items);
            if (c.kv_sources.has(i)) {
                lay.idx_wk = try L.linear(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.wk", .{p}), null, null);
                lay.idx_k_norm = try L.plain(try std.fmt.bufPrint(&nb, "{s}.attn.indexer.k_norm.weight", .{p}), null);
            }
        }
        lay.gate_w = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn.gate.weight", .{p}), .f16);
        lay.gate_b = try L.plain(try std.fmt.bufPrint(&nb, "{s}.ffn.gate.bias", .{p}), .f32);
        const xp = s.expertParts(c);
        lay.experts = try L.experts(p, n_experts, xp.half, xp.gu, xp.dcols, c.hidden);
        if (c.engram_layers.has(i)) {
            lay.engram_wkv = try L.linear(try std.fmt.bufPrint(&nb, "{s}.engram.wkv", .{p}), s.engramCols(c), s.range(s.engramRows(c)));
            if (s.pair != null) lay.engram_wkv.?.plan_n = @intCast((c.hc + 1) * c.hidden);
            var kb: [256]u8 = undefined;
            const q = try L.gpa.dupe(u8, try L.entry(try plan.plainKey(&kb, try std.fmt.bufPrint(&nb, "{s}.engram.q_weight", .{p}), .f32)));
            defer L.gpa.free(q);
            const k = try L.entry(try plan.plainKey(&kb, try std.fmt.bufPrint(&nb, "{s}.engram.k_weight", .{p}), .f32));
            if (q.len != k.len) return error.UnexpectedTensor;
            // q * k on the GPU in f32: one rounding a product, the same on the host
            for (0..q.len / 4) |j| {
                const a: f32 = @bitCast(std.mem.readInt(u32, q[4 * j ..][0..4], .little));
                const b: f32 = @bitCast(std.mem.readInt(u32, k[4 * j ..][0..4], .little));
                std.mem.writeInt(u32, q[4 * j ..][0..4], @bitCast(a * b), .little);
            }
            lay.engram_qk = try L.upload(try std.fmt.bufPrint(&nb, "{s}.engram.qk", .{p}), q);
        }
        return lay;
    }
};

/// Every tensor of rank `s` on device `d`, from `src` (weights.py _load, in its order).
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, src: Source, c: Config, s: plan.Split, dspark: bool) !Weights {
    const w = try gpa.create(Weights);
    const custody = cuda.context_health.Custody.init(w) catch |err| { gpa.destroy(w); return err; };
    w.* = .{ .gpa = gpa };
    errdefer {
        const cleanup = cuda.context_health.cleanupMark();
        w.deinit();
        if (cuda.context_health.cleanupReleased(cleanup)) {
            custody.release();
            gpa.destroy(w);
        } else {
            custody.retain();
            if (@import("builtin").is_test) failed_construction = w;
        }
    }
    var L: Loader = .{ .gpa = gpa, .d = d, .src = src, .w = w, .ahead = Prefetch.on() };
    defer L.deinit();
    var pf: Prefetch = undefined;
    var pf_chunks: [][2]u64 = &.{};
    defer if (pf_chunks.len > 0) gpa.free(pf_chunks);
    var pf_on = false;
    defer if (pf_on) pf.finish();
    if (L.ahead and src == .cache) {
        const rc = src.cache;
        pf_chunks = Prefetch.plan(gpa, rc.index, Prefetch.chunk) catch &.{};
        if (pf_chunks.len > 0) {
            pf = .{ .io = rc.io, .fd = rc.file.handle, .chunks = pf_chunks };
            L.src.cache.pf = &pf;
            pf.start();
            pf_on = true;
        }
    }
    w.embed = try L.plain("embed.weight", null);
    w.norm = try L.plain("norm.weight", null);
    const vr = s.headCols(c);
    w.head = try L.linear("head", vr, null);
    if (s.pair != null) w.head.plan_n = @intCast(s.vocab(c)); // the vocabulary half's plan
    w.vocab_lo = vr[0];
    w.vocab_hi = vr[1];
    w.layers = try gpa.alloc(Layer, c.layers);
    var nb: [64]u8 = undefined;
    for (0..c.layers) |i| w.layers[i] = try L.block(c, s, try std.fmt.bufPrint(&nb, "layers.{d}", .{i}), i, c.experts);
    if (dspark and c.dspark_block > 0) {
        const blocks = try gpa.alloc(Layer, c.draft_layers);
        errdefer gpa.free(blocks);
        // DSpark stays TP2 inside the pair on a 2D node (weights.py _load_2d)
        const tp2: plan.Split = .{ .rank = s.rank, .world = s.world };
        for (0..c.draft_layers) |j| blocks[j] = try L.block(c, tp2, try std.fmt.bufPrint(&nb, "mtp.{d}", .{j}), c.layers + j, c.draft_experts);
        var lb: [64]u8 = undefined;
        const last = try std.fmt.bufPrint(&lb, "mtp.{d}", .{c.draft_layers - 1});
        var tb: [128]u8 = undefined;
        w.dspark = .{
            .blocks = blocks,
            .main_proj = try L.linear("mtp.0.main_proj", null, null),
            .main_norm = try L.plain("mtp.0.main_norm.weight", null),
            .norm = try L.plain(try std.fmt.bufPrint(&tb, "{s}.norm.weight", .{last}), null),
            .markov_embed = try L.plain(try std.fmt.bufPrint(&tb, "{s}.markov_head.embed.weight", .{last}), null),
            .markov_head = try L.plain(try std.fmt.bufPrint(&tb, "{s}.markov_head.head.weight", .{last}), null),
            .conf = try L.plain(try std.fmt.bufPrint(&tb, "{s}.confidence_head.proj.weight", .{last}), null),
        };
    }
    if (L.span_bytes > 0) std.log.info("rank cache: {d} expert bytes read whole in {d} ns", .{ L.span_bytes, L.span_ns });
    const result = w.*;
    custody.release();
    gpa.destroy(w);
    return result;
}

test "strips: each 128-column block's tiles in k order" {
    // kt 2, nt 16 (two blocks of 8 tiles), one byte a tile
    var src: [32]u8 = undefined;
    for (&src, 0..) |*x, i| x.* = @intCast(i);
    var out: [32]u8 = undefined;
    strips(&src, &out, 2, 16, 1);
    // block 0: k row 0 tiles 0..7, k row 1 tiles 16..23; block 1: tiles 8..15, then 24..31
    const want = [_]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 };
    try std.testing.expectEqualSlices(u8, &want, &out);
}

var failed_construction: ?*Weights = null;

test "custody stage weights actual load failure retains in progress ledger" {
    const health = cuda.context_health;
    const Mock = struct {
        var frees: usize = 0;
        var fail_free: bool = false;
        var fail_upload: bool = false;
        var fail_alloc: bool = false;
        fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result { p.* = 0x123400; return if (fail_alloc) cuda.abi.error_not_ready else 0; }
        fn upload(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result { return if (fail_upload) cuda.abi.error_not_ready else 0; }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result { frees += 1; return if (fail_free) 1 else 0; }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Mock.alloc;
    d.api.cuMemcpyHtoD_v2 = Mock.upload;
    d.api.cuMemFree_v2 = Mock.free;
    const h: cuda.abi.Context = @ptrFromInt(0xcaa1);
    try health.retain(@intFromPtr(&d), h);
    defer health.release(@intFromPtr(&d), h);
    health.setCurrent(@intFromPtr(&d), h);
    defer health.clearCurrent(@intFromPtr(&d), h);
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "weights", .data = "abcd" });
    const file = try tmp.dir.openFile(io, "weights", .{});
    defer file.close(io);
    var index = try rank_cache.parseIndex(gpa, "[[\"embed.weight|None\",\"uint8\",[4],0,4]]", 4);
    defer index.deinit();
    const src: Source = .{ .cache = .{ .file = file, .index = &index, .io = io } };
    // Upload failure exercises pending; missing norm exercises an appended buffer.
    for (0..3) |stage| for ([_]bool{ false, true }) |free_fail| {
        const upload_fail = stage == 1;
        Mock.frees = 0;
        Mock.fail_alloc = stage == 0;
        Mock.fail_upload = upload_fail;
        Mock.fail_free = free_fail;
        failed_construction = null;
        const before = cuda.usage(false).device;
        const result = load(gpa, &d, src, undefined, undefined, false);
        if (stage < 2) try std.testing.expectError(error.NotReady, result) else try std.testing.expectError(error.MissingEntry, result);
        try std.testing.expectEqual(@as(usize, 1), Mock.frees);
        if (free_fail) {
            const w = failed_construction.?;
            const b = if (stage < 2) w.pending.? else w.buffers.items[0];
            try std.testing.expectEqual(@as(u64, 0x123400), b.ptr);
            try b.owner.ensure();
            try std.testing.expectEqual(before + 4, cuda.usage(false).device);
            Mock.fail_free = false;
            health.releaseTestCustody(w);
            w.deinit();
            gpa.destroy(w);
            try std.testing.expectEqual(@as(usize, 2), Mock.frees);
        } else try std.testing.expect(failed_construction == null);
        try std.testing.expectEqual(before, cuda.usage(false).device);
    };
}

test "custody stage weights preappend allocation failure and exact cleanup" {
    const health = cuda.context_health;
    const Mock = struct {
        var fail: bool = false;
        var frees: usize = 0;
        fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result { p.* = 0xfeed00; return 0; }
        fn upload(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result { return 0; }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result { frees += 1; return if (fail) 1 else 0; }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Mock.alloc; d.api.cuMemcpyHtoD_v2 = Mock.upload; d.api.cuMemFree_v2 = Mock.free;
    const h: cuda.abi.Context = @ptrFromInt(0xcaa3);
    try health.retain(@intFromPtr(&d), h);
    defer health.release(@intFromPtr(&d), h);
    health.setCurrent(@intFromPtr(&d), h);
    defer health.clearCurrent(@intFromPtr(&d), h);
    for (0..3) |stage| for ([_]bool{ false, true }) |fail| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = stage });
        const gpa = failing.allocator();
        const w = try std.testing.allocator.create(Weights);
        defer std.testing.allocator.destroy(w);
        w.* = .{ .gpa = gpa };
        var loader: Loader = .{ .gpa = gpa, .d = &d, .src = undefined, .w = w };
        Mock.frees = 0; Mock.fail = fail;
        try std.testing.expectError(error.OutOfMemory, loader.upload("entry", "abcd"));
        try std.testing.expectEqual(@as(usize, 0), Mock.frees);
        const mark = health.cleanupMark();
        w.deinit();
        try std.testing.expectEqual(!fail, health.cleanupReleased(mark));
        try std.testing.expectEqual(@as(usize, 1), Mock.frees);
        if (fail) {
            const held = if (stage == 0) w.pending.? else w.buffers.items[0];
            try std.testing.expectEqual(@as(u64, 0xfeed00), held.ptr);
            try held.owner.ensure();
            Mock.fail = false;
            w.deinit();
            try std.testing.expectEqual(@as(usize, 2), Mock.frees);
        }
    };
}

test "custody stage weights experts actual constructor allocation and late failure matrix" {
    const health = cuda.context_health;
    const Mock = struct {
        var acquired: usize = 0;
        var released: usize = 0;
        var fail_free: bool = false;
        var fail_upload: bool = false;
        fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
            acquired += 1; p.* = 0xf00000 + acquired * 256; return 0;
        }
        fn upload(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
            return if (fail_upload) cuda.abi.error_not_ready else 0;
        }
        fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
            if (fail_free) return 1;
            released += 1; return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Mock.alloc;
    d.api.cuMemcpyHtoD_v2 = Mock.upload;
    d.api.cuMemFree_v2 = Mock.free;
    const h: cuda.abi.Context = @ptrFromInt(0xcaa4);
    try health.retain(@intFromPtr(&d), h);
    defer health.release(@intFromPtr(&d), h);
    health.setCurrent(@intFromPtr(&d), h);
    defer health.clearCurrent(@intFromPtr(&d), h);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    try tmp.dir.writeFile(io, .{ .sub_path = "experts", .data = "abcdefghabcdefghabcdefgh" });
    const file = try tmp.dir.openFile(io, "experts", .{});
    defer file.close(io);
    var kb: [256]u8 = undefined;
    var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer text.deinit();
    try text.writer.writeAll("[");
    for ([_][]const u8{ "w1", "w3", "w2" }, 0..) |proj, j| {
        var nb: [128]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&nb, "layer.ffn.shared_experts.{s}", .{proj});
        const key = try plan.partKey(&kb, prefix, if (j < 2) .{ 0, 1 } else null, if (j == 2) .{ 0, 1 } else null, "tr");
        try text.writer.print("{s}[\"{s}\",\"uint8\",[1,1,8],{d},8]", .{ if (j == 0) "" else ",", key, j * 8 });
    }
    try text.writer.writeAll("]");
    var index = try rank_cache.parseIndex(std.testing.allocator, text.written(), 24);
    defer index.deinit();
    var saw_pending = false;
    var saw_appended = false;
    // Host allocation failures range across prefixes, tables, name and append.
    // The non-failing allocator reaches the later missing scale source entry.
    for (0..40) |stage| for ([_]bool{ false, true }) |fail| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = stage });
        const gpa = failing.allocator();
        var weights: Weights = .{ .gpa = gpa };
        var loader: Loader = .{ .gpa = gpa, .d = &d, .src = .{ .cache = .{ .file = file, .index = &index, .io = io } }, .w = &weights };
        defer loader.host.deinit(gpa);
        defer loader.aux.deinit(gpa);
        Mock.acquired = 0; Mock.released = 0; Mock.fail_free = fail; Mock.fail_upload = stage == 39;
        const before = cuda.usage(false).device;
        if (loader.experts("layer", 0, .{ 0, 1 }, .{ 0, 1 }, null, 1)) |_| return error.ExpectedFailure else |err| {
            try std.testing.expect(err == error.OutOfMemory or err == error.MissingEntry or err == error.NotReady);
        }
        try std.testing.expectEqual(@as(usize, 0), Mock.released);
        saw_pending = saw_pending or weights.pending != null;
        saw_appended = saw_appended or weights.buffers.items.len > 0;
        const mark = health.cleanupMark();
        weights.deinit();
        if (!health.cleanupReleased(mark)) {
            const held = if (weights.pending) |b| b else weights.buffers.items[weights.buffers.items.len - 1];
            try held.owner.ensure();
            Mock.fail_free = false;
            weights.deinit();
        }
        try std.testing.expectEqual(Mock.acquired, Mock.released);
        try std.testing.expectEqual(before, cuda.usage(false).device);
    };
    try std.testing.expect(saw_pending and saw_appended);
}

test "rank cache span: entries read whole come back byte for byte, the rest one by one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // every boundary off the 4 KiB grid; "after" keeps the aligned end of [first, last] inside the file
    const name = "rank0of1-span.bin";
    try rank_cache.writeFake(a, io, tmp.dir, name, &.{
        .{ .key = "before", .dtype = "uint8", .shape = &.{5000}, .fill = 1 },
        .{ .key = "first", .dtype = "uint8", .shape = &.{3000}, .fill = 2 },
        .{ .key = "middle", .dtype = "uint8", .shape = &.{70001}, .fill = 3 },
        .{ .key = "last", .dtype = "uint8", .shape = &.{4096}, .fill = 4 },
        .{ .key = "after", .dtype = "uint8", .shape = &.{9000}, .fill = 5 },
    });
    var ix = try rank_cache.open(std.testing.allocator, io, tmp.dir, name);
    defer ix.deinit();
    var file = try tmp.dir.openFile(io, name, .{});
    defer file.close(io);
    var L: Loader = .{ .gpa = std.testing.allocator, .d = undefined, .src = .{ .cache = .{ .file = file, .index = &ix, .io = io } }, .w = undefined };
    defer L.deinit();

    const first = ix.get("first").?;
    const last = ix.get("last").?;
    L.beginSpan(first.offset, last.offset + last.bytes);
    try std.testing.expect(L.span.len > 0);
    for ([_]struct { key: []const u8, len: usize, fill: u8 }{
        .{ .key = "first", .len = 3000, .fill = 2 },
        .{ .key = "middle", .len = 70001, .fill = 3 },
        .{ .key = "last", .len = 4096, .fill = 4 },
    }) |want| {
        const got = L.spanned(want.key).?;
        try std.testing.expectEqual(want.len, got.len);
        for (got) |b| try std.testing.expectEqual(want.fill, b);
    }
    try std.testing.expect(L.spanned("before") == null);
    try std.testing.expect(L.spanned("after") == null);
    const before = try L.entry("before");
    try std.testing.expectEqual(@as(usize, 5000), before.len);
    for (before) |b| try std.testing.expectEqual(@as(u8, 1), b);
    L.endSpan();
    try std.testing.expect(L.spanned("first") == null);

    // reads in pieces smaller than the span: the file's bytes, as a buffered read sees them
    const pieces = try std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(dio.alignment), 12 * dio.alignment);
    defer std.heap.page_allocator.free(pieces);
    try readWhole(L.direct.?, pieces, 0, dio.alignment);
    var buffered: [12 * dio.alignment]u8 = undefined;
    try std.testing.expectEqual(@as(usize, buffered.len), try file.readPositionalAll(io, &buffered, 0));
    try std.testing.expectEqualSlices(u8, &buffered, pieces);

    // a span that runs past the end of the file is given up, and its entries are still read one by one
    const after = ix.get("after").?;
    L.beginSpan(after.offset, after.offset + after.bytes + (1 << 20));
    try std.testing.expectEqual(@as(usize, 0), L.span.len);
    const tail = try L.entry("after");
    try std.testing.expectEqual(@as(usize, 9000), tail.len);
    for (tail) |b| try std.testing.expectEqual(@as(u8, 5), b);
}

test "load read-ahead: the entries read one at a time in file order, without a layer's experts, in chunks" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const name = "rank0of1-prefetch.bin";
    try rank_cache.writeFake(a, io, tmp.dir, name, &.{
        .{ .key = "embed.weight|None", .dtype = "uint8", .shape = &.{100}, .fill = 1 },
        .{ .key = "layers.0.attn.wq|None", .dtype = "uint8", .shape = &.{50}, .fill = 2 },
        .{ .key = "layers.0.ffn.experts.0.w1|None|None|tr", .dtype = "uint8", .shape = &.{300}, .fill = 3 },
        .{ .key = "layers.0.ffn.shared_experts.w2|None|None|suh", .dtype = "uint8", .shape = &.{20}, .fill = 4 },
        .{ .key = "layers.1.attn.wq|None", .dtype = "uint8", .shape = &.{70}, .fill = 5 },
        .{ .key = "norm.weight|None", .dtype = "uint8", .shape = &.{30}, .fill = 6 },
    });
    var ix = try rank_cache.open(std.testing.allocator, io, tmp.dir, name);
    defer ix.deinit();
    // [0, 150) before the experts, [470, 570) after them, in chunks of at most 64 bytes
    const chunks = try Prefetch.plan(std.testing.allocator, &ix, 64);
    defer std.testing.allocator.free(chunks);
    try std.testing.expectEqualSlices([2]u64, &.{ .{ 0, 64 }, .{ 64, 128 }, .{ 128, 150 }, .{ 470, 534 }, .{ 534, 570 } }, chunks);
}

test "load read-ahead: the loader reads the same bytes with the threads running, and they stop" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const name = "rank0of1-prefetch-read.bin";
    try rank_cache.writeFake(a, io, tmp.dir, name, &.{
        .{ .key = "a|None", .dtype = "uint8", .shape = &.{300000}, .fill = 7 },
        .{ .key = "layers.0.ffn.experts.0.w1|None|None|tr", .dtype = "uint8", .shape = &.{200000}, .fill = 8 },
        .{ .key = "b|None", .dtype = "uint8", .shape = &.{123457}, .fill = 9 },
    });
    var ix = try rank_cache.open(std.testing.allocator, io, tmp.dir, name);
    defer ix.deinit();
    var file = try tmp.dir.openFile(io, name, .{});
    defer file.close(io);
    const chunks = try Prefetch.plan(std.testing.allocator, &ix, 4096);
    defer std.testing.allocator.free(chunks);
    var pf: Prefetch = .{ .io = io, .fd = file.handle, .chunks = chunks };
    const src: Source = .{ .cache = .{ .file = file, .index = &ix, .io = io, .pf = &pf } };
    pf.start();
    defer pf.finish();
    // the threads read every planned byte before the loader starts (the lead allows it all): they did the reading
    try waitWarmed(&pf, planned(chunks));
    for ([_]struct { key: []const u8, len: usize, fill: u8 }{
        .{ .key = "a|None", .len = 300000, .fill = 7 },
        .{ .key = "layers.0.ffn.experts.0.w1|None|None|tr", .len = 200000, .fill = 8 },
        .{ .key = "b|None", .len = 123457, .fill = 9 },
    }) |want| {
        const out = try std.testing.allocator.alloc(u8, want.len);
        defer std.testing.allocator.free(out);
        try src.read(want.key, out);
        for (out) |x| try std.testing.expectEqual(want.fill, x);
    }
    try std.testing.expectEqual(ix.get("b|None").?.offset + 123457, pf.pos.load(.monotonic));
    pf.finish();
    try std.testing.expectEqual(@as(usize, 0), pf.started);
}

fn planned(chunks: []const [2]u64) u64 {
    var n: u64 = 0;
    for (chunks) |c| n += c[1] - c[0];
    return n;
}

/// Waits (up to 10 s) until the threads have read `want` bytes.
fn waitWarmed(pf: *Prefetch, want: u64) !void {
    var waited: usize = 0;
    while (pf.warmed.load(.monotonic) < want) : (waited += 1) {
        if (waited == 10_000) return error.TestTimedOut;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
}

fn prefetchFile(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !void {
    try rank_cache.writeFake(a, io, dir, name, &.{.{ .key = "a|None", .dtype = "uint8", .shape = &.{40000}, .fill = 7 }});
}

test "load read-ahead: the lead holds the threads back until the loader's reads move it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try prefetchFile(arena.allocator(), io, tmp.dir, "rank0of1-lead.bin");
    var ix = try rank_cache.open(std.testing.allocator, io, tmp.dir, "rank0of1-lead.bin");
    defer ix.deinit();
    var file = try tmp.dir.openFile(io, "rank0of1-lead.bin", .{});
    defer file.close(io);
    const chunks = try Prefetch.plan(std.testing.allocator, &ix, 4096);
    defer std.testing.allocator.free(chunks);
    const base = chunks[0][0];
    // the loader has read nothing past the entry's start: only the chunks that start within the lead are read
    var pf: Prefetch = .{ .io = io, .fd = file.handle, .chunks = chunks, .lead = 4096 };
    pf.pos.store(base, .monotonic);
    pf.start();
    defer pf.finish();
    try waitWarmed(&pf, 8192);
    std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    try std.testing.expectEqual(@as(u64, 8192), pf.warmed.load(.monotonic));
    // the loader's read moves past the entry: the rest is read
    _ = pf.pos.fetchMax(chunks[chunks.len - 1][1], .monotonic);
    try waitWarmed(&pf, planned(chunks));
    try std.testing.expectEqual(planned(chunks), pf.warmed.load(.monotonic));
}

test "load read-ahead: finish stops threads held back by the lead" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try prefetchFile(arena.allocator(), io, tmp.dir, "rank0of1-park.bin");
    var ix = try rank_cache.open(std.testing.allocator, io, tmp.dir, "rank0of1-park.bin");
    defer ix.deinit();
    var file = try tmp.dir.openFile(io, "rank0of1-park.bin", .{});
    defer file.close(io);
    const chunks = try Prefetch.plan(std.testing.allocator, &ix, 4096);
    defer std.testing.allocator.free(chunks);
    var pf: Prefetch = .{ .io = io, .fd = file.handle, .chunks = chunks, .lead = 4096 };
    pf.pos.store(chunks[0][0], .monotonic);
    pf.start();
    defer pf.finish(); // a failed check below still stops the threads before chunks and pf go away
    try std.testing.expect(pf.started > 0);
    try waitWarmed(&pf, 8192);
    pf.finish();
    try std.testing.expectEqual(@as(usize, 0), pf.started);
    try std.testing.expect(pf.warmed.load(.monotonic) < planned(chunks));
}

test "load read-ahead: a chunk past the file's end is read as far as the file goes, and an unreadable file is left to the loader" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try prefetchFile(arena.allocator(), io, tmp.dir, "rank0of1-short.bin");
    var ix = try rank_cache.open(std.testing.allocator, io, tmp.dir, "rank0of1-short.bin");
    defer ix.deinit();
    var file = try tmp.dir.openFile(io, "rank0of1-short.bin", .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    // a short read (the chunk runs 4000 bytes past the end) and a chunk wholly past the end (end of file at once)
    const past = [_][2]u64{ .{ size - 100, size + 4000 }, .{ size + 10, size + 20 } };
    var pf: Prefetch = .{ .io = io, .fd = file.handle, .chunks = &past };
    pf.start();
    pf.join();
    try std.testing.expectEqual(@as(u64, 100), pf.warmed.load(.monotonic));
    // a read that fails (here the descriptor is a directory's): the threads stop, and the loader's own read is unchanged
    var bad: Prefetch = .{ .io = io, .fd = tmp.dir.handle, .chunks = &past };
    bad.start();
    bad.join();
    try std.testing.expectEqual(@as(u64, 0), bad.warmed.load(.monotonic));
    const src: Source = .{ .cache = .{ .file = file, .index = &ix, .io = io, .pf = &bad } };
    const out = try std.testing.allocator.alloc(u8, 40000);
    defer std.testing.allocator.free(out);
    try src.read("a|None", out);
    for (out) |x| try std.testing.expectEqual(@as(u8, 7), x);
}

test "rank cache load opt in and page budget stay bounded" {
    for ([_]?[]const u8{ null, "", "0", "true", "2" }) |v| try std.testing.expect(!Prefetch.selected(v));
    try std.testing.expect(Prefetch.selected("1"));
    var ix = try rank_cache.parseIndex(std.testing.allocator,
        "[[\"big|None\",\"uint8\",[4294967297],0,4294967297]]", 4294967297);
    defer ix.deinit();
    try std.testing.expectError(error.PrefetchBudget, Prefetch.plan(std.testing.allocator, &ix, Prefetch.chunk));
    try std.testing.expectError(error.BadPrefetchChunk, Prefetch.plan(std.testing.allocator, &ix, 0));
    try std.testing.expectEqual(@as(usize, 2), Prefetch.threads_n);
    try std.testing.expectEqual(@as(u64, 512 << 20), Prefetch.default_lead);
}
