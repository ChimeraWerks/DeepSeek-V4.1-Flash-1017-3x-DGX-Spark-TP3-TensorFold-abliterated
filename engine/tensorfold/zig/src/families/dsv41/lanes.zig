//! DeepSeek-V4.1-Flash behind TensorFold 1.0's lane core (core/lanes): the round loop picks each round's streams and
//! windows and keeps the accepted rows; this backend runs them on the port's model (model.zig) with the served lane's
//! arithmetic. Each stream has a pool slot (its window rings, positional stores and drafter rings) and an extent of
//! the shared compressed plane (first fit, chunk-aligned, as large as its prompt, its budget and a round's rows), and
//! its ids by position on both ranks (Engram's n-grams). verify is one round over every window (round.zig) with each
//! row drawn by the served sampler (sampling.zig) at its keyed position, and the drafting streams' window rows
//! absorbed into their drafter rings at once (the served eager absorb: the rows past the kept ones sit where the
//! drafter never reads, and the next window writes over them); draft is one batched drafter pass (draft.zig) for
//! every stream asked, the drafts held for the next round with their confidences, whose sigmoids are each draft's
//! chance of landing once the earlier ones did (probabilities: their running products, the chance that a draft and
//! every earlier one land; the round loop's allocation trims the windows by them). The caches are
//! positional, so keep only checks that a path is a prefix. Rank 0 runs the core; before each primitive it sends rank
//! 1 the primitive's inputs over the ranks' link, and rank 1 (follow) runs the same primitive: both ranks issue the
//! same kernels and collectives in the same order. The drafted reply is the serial one: every row is drawn with its
//! stream's key at its own position, whatever the round's composition.
const std = @import("std");
const lanes = @import("lanes");
const model = @import("model.zig");
const round = @import("round.zig");
const draft = @import("draft.zig");
const sampling = @import("sampling.zig");
const keyed_draft = @import("keyed_draft.zig");
const penalty_pool = @import("penalty_pool.zig");
const greedy_head = @import("decode_greedy_head.zig");
const link = @import("link.zig");
const graph_vote = @import("graph_vote.zig");
const cuda = @import("cuda");
const spill_tier = @import("spill_tier.zig");
const parts = @import("parts.zig");
const kv_lanes = @import("kv_lanes.zig");
const shard = @import("shard.zig");
const rank_wire = @import("rank_wire.zig");
const fingerprint = @import("fingerprint.zig");
const pressure = @import("kv_pressure.zig");
const move_budget = @import("kv_move_budget.zig");
const idle_relocation = @import("idle_relocation.zig");
const idle_protocol = @import("idle_protocol.zig");
const cache_batch = @import("cache_batch.zig");

const be = lanes.backend;
const Model = model.Model;
const max_streams = model.max_streams;
const max_rows = round.max_rows;

/// The spill tier on rank 0 (the prompt cache's Tier) and on a follower, over this rank's planes (spill_tier.zig).
pub const SpillTier = spill_tier.Leader(Model.SpillDev);
pub const SpillFollower = spill_tier.Follower(Model.SpillDev);

comptime {
    // parts.zig keeps lanes' block and growth slack (it is self-contained so `zig test` runs it alone)
    std.debug.assert(parts.chunk_rows == model.chunk_rows and parts.growth_rows == growth_rows);
}

/// A window of the pending row and the drafter's block: the widest one the gates checked row for row.
pub const window_rows = 6;

/// The image rows a fill_images frame carries at most.
pub const image_piece = 32 << 10;

/// A gate's diagnostic (the tools' --logits-sha 1): each prompt's logits' sha256 on stderr.
pub var logits_sha = false;

/// The primitives rank 0 sends rank 1 (each frame: kind u8, step u64, its fields little-endian).
/// With the spill tier on: rank_wire's frames (spill_tier: restores' agreement, guards, and the notices of rank 0's
/// write at keep and point metadata: keep/share/touch/forget, tracked like every other command). TF_DS_KV_SHARD adds
/// rank_wire too (rank_wire.zig's frames: the parts' agreement, each plan's ops, each transfer's
/// shard_xfer/shard_finish, the digest checks). Both on together share one rank_wire Leader (Lanes.rw, rank 0) and
/// one Follower a follower (PairFollower): one Tracker sees every frame of the pair. Kind 11 is the
/// coordinated RDMA idle boundary (not the old, untracked spill notes); all ranks need this version.
pub const Kind = enum(u8) { fill_begin = 1, fill_chunk = 2, verify = 3, pass = 4, snap_save = 5, snap_restore = 6, snap_drop = 7, fill_images = 8, done = 9, rank_wire = 10, idle_move = 12, fill_resume = 13, graph_vote_agree = 15, fill_slice = 18, rdma_idle = 11, snap_drop_many = cache_batch.kind, pass_keyed = 17 };

// Shared with the real follower loop; reserved/unknown outer kinds fail closed.
fn decodeOuterKind(raw: u8) !Kind {
    return switch (raw) {
            1 => .fill_begin,
            2 => .fill_chunk,
            3 => .verify,
            4 => .pass,
            5 => .snap_save,
            6 => .snap_restore,
            7 => .snap_drop,
            8 => .fill_images,
            9 => .done,
            10 => .rank_wire,
            11 => .rdma_idle,
            12 => .idle_move,
            13 => .fill_resume,
            15 => .graph_vote_agree,
            cache_batch.kind => .snap_drop_many,
            17 => .pass_keyed,
            18 => .fill_slice,
            else => return error.BadFrame,
        };
}

/// What a follower takes while a sliced prompt chunk is open: the next slice, and decode rounds that run
/// between slices (each round's verify and draft pass, plain or keyed). Any other outer operation would run inside the chunk.
fn openChunkAllows(kind: Kind) bool {
    return switch (kind) {
        .fill_slice, .verify, .pass, .pass_keyed => true,
        else => false,
    };
}

test "keyed draft pass frame is admitted while a prefill slice chunk is open" {
    for ([_]Kind{ .fill_slice, .verify, .pass, .pass_keyed }) |k| try std.testing.expect(openChunkAllows(k));
    for ([_]Kind{ .fill_begin, .fill_chunk, .snap_save, .snap_restore, .snap_drop, .fill_images, .done, .rank_wire, .rdma_idle, .idle_move, .fill_resume, .graph_vote_agree, .snap_drop_many }) |k| try std.testing.expect(!openChunkAllows(k));
}

test "outer follower decoder accepts fill_resume 13 and rejects reserved kinds" {
    try std.testing.expectEqual(Kind.fill_resume, try decodeOuterKind(13));
    try std.testing.expectEqual(Kind.rdma_idle, try decodeOuterKind(11));
    for ([_]u8{ 0, 14, 255 }) |raw| try std.testing.expectError(error.BadFrame, decodeOuterKind(raw));
    for ([_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12 }) |raw| try std.testing.expectEqual(raw, @intFromEnum(try decodeOuterKind(raw)));
}

comptime {
    std.debug.assert(@backingInt(Kind.rank_wire) == @import("rank_wire.zig").lanes_kind);
}

const SliceFrame = struct {
    abort: bool = false,
    lo: u32 = 0,
    hi: u32 = 0,
    args: model.SliceArgs = .{ .start = 0, .n = 0, .replay = 0 },
    next_rows: u32 = 0,
};
fn putSlice(w: *Wire, gpa: std.mem.Allocator, f: SliceFrame) !void {
    try w.int(gpa, u8, @intFromBool(f.abort));
    if (f.abort) return;
    if (f.hi <= f.lo or f.args.n == 0) return error.BadSlice;
    try w.int(gpa, u32, f.lo);
    try w.int(gpa, u32, f.hi);
    try w.int(gpa, u64, f.args.start);
    try w.int(gpa, u32, @intCast(f.args.n));
    try w.int(gpa, u64, f.args.replay);
    try w.int(gpa, u32, f.next_rows);
}
fn getSlice(r: *Reader) !SliceFrame {
    const op = try r.int(u8);
    if (op > 1) return error.BadFrame;
    if (op == 1) {
        if (r.rest().len != 0) return error.BadFrame;
        return .{ .abort = true };
    }
    const f: SliceFrame = .{
        .lo = try r.int(u32),
        .hi = try r.int(u32),
        .args = .{ .start = @intCast(try r.int(u64)), .n = try r.int(u32), .replay = @intCast(try r.int(u64)) },
        .next_rows = try r.int(u32),
    };
    if (f.hi <= f.lo or f.args.n == 0 or r.rest().len != 0) return error.BadFrame;
    return f;
}

/// A frame being written.
pub const Wire = struct {
    buf: std.ArrayList(u8) = .empty,

    pub fn begin(w: *Wire, gpa: std.mem.Allocator, kind: Kind, step: u64) !void {
        w.buf.clearRetainingCapacity();
        try w.int(gpa, u8, @backingInt(kind));
        try w.int(gpa, u64, step);
    }

    pub fn int(w: *Wire, gpa: std.mem.Allocator, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try w.buf.appendSlice(gpa, &b);
    }

    pub fn ids(w: *Wire, gpa: std.mem.Allocator, v: []const i32) !void {
        try w.int(gpa, u32, @intCast(v.len));
        try w.buf.ensureUnusedCapacity(gpa, v.len * 4);
        for (v) |x| try w.int(gpa, i32, x);
    }

    pub fn deinit(w: *Wire, gpa: std.mem.Allocator) void {
        w.buf.deinit(gpa);
    }
};

/// Present only when startup word[6] bit24 agreed. Off is byte-for-byte legacy.
fn writeGreedyHeadMode(w: *Wire, gpa: std.mem.Allocator, enabled: bool, use_head: bool) !void {
    if (!enabled) {
        if (use_head) return error.BadGreedyHeadMode;
        return;
    }
    try w.int(gpa, u8, @intFromBool(use_head));
}

fn readGreedyHeadMode(r: *Reader, enabled: bool, unsupported: bool) !bool {
    if (!enabled) return false;
    const mode = try r.int(u8);
    if (mode > 1 or (mode == 1 and unsupported)) return error.BadGreedyHeadMode;
    return mode == 1;
}

/// Present only when startup word[6] bit25 agreed. Zero keeps full logits.
fn writeSampleHeadCount(w: *Wire, gpa: std.mem.Allocator, enabled: bool, count: usize) !void {
    if (count > 64 or (!enabled and count != 0)) return error.BadGreedyHeadMode;
    if (enabled) try w.int(gpa, u8, @intCast(count));
}

fn readSampleHeadCount(r: *Reader, enabled: bool, unsupported: bool) !usize {
    if (!enabled) return 0;
    const count = try r.int(u8);
    if (count > 64 or (unsupported and count != 0)) return error.BadGreedyHeadMode;
    return count;
}

/// A verify frame's window: slot, extent, first position, absorb, ids; TF_DS_KV_SHARD (`sharded`) adds its extent's
/// part (u8) and generation (u64), so a follower refuses a round for a reused extent. Off, the bytes are as ever.
pub fn writeWin(w: *Wire, gpa: std.mem.Allocator, win: Win, sharded: bool) !void {
    try w.int(gpa, u32, @intCast(win.slot));
    try w.int(gpa, u64, win.base);
    try w.int(gpa, u64, win.end);
    try w.int(gpa, u64, win.start);
    try w.int(gpa, u8, @intFromBool(win.absorb));
    try w.ids(gpa, win.ids);
    if (sharded) {
        if (win.part < 0 or win.part >= parts.max_parts) return error.BadPart;
        try w.int(gpa, u8, @intCast(win.part));
        try w.int(gpa, u64, win.gen);
    }
}

/// writeWin's window read back, its ids into `ids` (the window's slice points there).
pub fn readWin(r: *Reader, gpa: std.mem.Allocator, ids: *std.ArrayList(i32), sharded: bool) !Win {
    const slot: usize = try r.int(u32);
    const base: usize = @intCast(try r.int(u64));
    const end: usize = @intCast(try r.int(u64));
    const start: usize = @intCast(try r.int(u64));
    const absorb = try r.int(u8) != 0;
    try r.ids(gpa, ids);
    var win: Win = .{ .slot = slot, .base = base, .end = end, .start = start, .absorb = absorb, .ids = ids.items };
    if (sharded) {
        win.part = try r.int(u8);
        win.gen = try r.int(u64);
    }
    return win;
}

/// Exact idle-copy receipt parser, shared by the real TP3 transaction and mocks.
pub fn validateMoveReceipt(bytes: []const u8, phase: u8, rank: usize, generation: u64, copied: u64) !void {
    var r: Reader = .{ .b = bytes };
    if (try r.int(u8) != 12 or try r.int(u8) != phase or try r.int(u8) != rank or
        try r.int(u64) != generation or try r.int(u64) != copied or r.at != r.b.len) return error.TransferFailed;
}

/// A frame being read.
pub const Reader = struct {
    b: []const u8,
    at: usize = 0,

    pub fn int(r: *Reader, comptime T: type) !T {
        const n = @sizeOf(T);
        if (r.at + n > r.b.len) return error.ShortFrame;
        const v = std.mem.readInt(T, r.b[r.at..][0..n], .little);
        r.at += n;
        return v;
    }

    /// The frame's bytes after those read.
    pub fn rest(r: *Reader) []const u8 {
        const b = r.b[r.at..];
        r.at = r.b.len;
        return b;
    }

    /// A list of ids written by Wire.ids, into `out` (replacing it).
    pub fn ids(r: *Reader, gpa: std.mem.Allocator, out: *std.ArrayList(i32)) !void {
        const n = try r.int(u32);
        try out.resize(gpa, n);
        for (out.items) |*x| x.* = try r.int(i32);
    }
};

/// A window of a round as both ranks build it: its stream's slot and extent, its first position, whether its rows'
/// taps go into the drafter, its rows' ids; TF_DS_KV_SHARD: its extent's part (the rank holding its compressed rows;
/// -1: its base's, which round.Rows.ownerOf derives and checks) and its extent's generation (parts.zig: a follower
/// refuses a round naming a reused extent).
pub const Win = struct { slot: usize, base: usize, end: usize, start: usize, absorb: bool, ids: []const i32, part: i64 = -1, gen: u64 = 0 };

/// The rows of a round (round.Rows' slices point here).
pub const Built = struct {
    ids: [max_rows]i64 = undefined,
    pos: [max_rows]i64 = undefined,
    slots: [max_rows]i64 = undefined,
    bases: [max_rows]i64 = undefined,
    ends: [max_rows]i64 = undefined,
    parts: [max_rows]i64 = undefined, // TF_DS_KV_SHARD: each row's part, plain data for the attention (Win.part)
    wins: [max_streams]round.Window = undefined,
    absorb: [max_streams]bool = undefined,
    r: usize = 0,
    n: usize = 0,

    /// Each window's ids written at its positions in its slot's sequence (the rows past them dropped: the rejected
    /// rows of the last round), then the round's rows in window order.
    pub fn build(b: *Built, gpa: std.mem.Allocator, seqs: *[max_streams]std.ArrayList(i32), wins: []const Win) !round.Rows {
        if (wins.len == 0 or wins.len > max_streams) return error.BadRound;
        b.r = 0;
        b.n = wins.len;
        for (wins, 0..) |w, i| {
            if (w.slot >= max_streams or w.ids.len == 0) return error.BadRound;
            if (b.r + w.ids.len > max_rows) return error.WindowTooWide;
            const seq = &seqs[w.slot];
            if (seq.items.len < w.start) return error.SequenceGap;
            try seq.resize(gpa, w.start + w.ids.len);
            @memcpy(seq.items[w.start..], w.ids);
            for (w.ids, 0..) |id, j| {
                b.ids[b.r + j] = id;
                b.pos[b.r + j] = @intCast(w.start + j);
                b.slots[b.r + j] = @intCast(w.slot);
                b.bases[b.r + j] = @intCast(w.base);
                b.ends[b.r + j] = @intCast(w.end);
                b.parts[b.r + j] = w.part;
            }
            b.wins[i] = .{ .row = b.r, .n = w.ids.len, .seq = seq.items[0 .. w.start + w.ids.len] };
            b.absorb[i] = w.absorb;
            b.r += w.ids.len;
        }
        const R = b.r;
        return .{ .ids = b.ids[0..R], .pos = b.pos[0..R], .slots = b.slots[0..R], .bases = b.bases[0..R], .ends = b.ends[0..R], .windows = b.wins[0..b.n], .parts = b.parts[0..R] };
    }
};

/// The first chunk-aligned extent of `size` positions in [0, cap) that overlaps none of `used`.
/// Whether [lo, hi) meets any of `used`.
fn clashes(used: []const [2]usize, lo: usize, hi: usize) bool {
    for (used) |u| if (lo < u[1] and u[0] < hi) return true;
    return false;
}

/// Whether one of the prompt cache's marks has its chunk end at `at` (each kept at the chunk end at or before it).
fn markEnds(marks: []const u32, at: usize) bool {
    for (marks) |w| if (w / model.chunk_rows * model.chunk_rows == at) return true;
    return false;
}

// -- pictures in a resumed pass (recipe 0006; spill v2 restores take the same path as memory resumes) ------------------

/// A picture's span [at, at + tokens) of the runtime sequence `seq` (the prompt's ids, each the placeholder `image_id`
/// there) made dead: negative, so no Engram n-gram reaches into or across it.
pub fn markPicture(seq: []i32, at: usize, tokens: usize, image_id: u32) !void {
    if (tokens == 0 or at + tokens > seq.len) return error.BadImageSpan;
    for (seq[at..][0..tokens]) |*q| {
        if (q.* != @as(i64, image_id)) return error.BadImageSpan;
        q.* = -1;
    }
}

/// Whether a pass resuming at `resume_at` needs a picture's rows (the tower's run over it): a picture wholly before the
/// cut is in the resumed state's rows and needs none; one straddling or after it gets all of its rows, of which the
/// chunks embed only the positions at or past the cut (prompt.overlap).
pub fn pictureNeedsRows(at: usize, tokens: usize, resume_at: usize) bool {
    return at + tokens > resume_at;
}

/// A resumed sequence checked against its request: `ids` everywhere but the pictures' spans, which are negative
/// (spill_state.checkEngramSeq over the request's own pictures). The tests' oracle for what markPicture builds; a
/// restored resume needs no run of it (the pass's own comment says why).
pub fn checkPictureSeq(gpa: std.mem.Allocator, seq: []const i32, ids: []const u32, images: []const lanes.stream.Image) !void {
    const sp = try gpa.alloc([2]usize, images.len);
    defer gpa.free(sp);
    for (images, sp) |img, *x| x.* = .{ img.at, img.tokens };
    try model.Model.spill.checkEngramSeq(seq, ids, sp);
}

test "pictures in a resumed pass: spans dead in the sequence, rows only for those the cut does not hold, the Engram check" {
    const gpa = std.testing.allocator;
    const ph: u32 = 129264;
    // a prompt of 40 tokens: a picture at [4, 10), one straddling a cut at 16 ([14, 20)), one after it at [30, 34)
    var ids: [40]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + i);
    const imgs = [_]lanes.stream.Image{ .{ .at = 4, .tokens = 6, .bytes = "" }, .{ .at = 14, .tokens = 6, .bytes = "" }, .{ .at = 30, .tokens = 4, .bytes = "" } };
    for (imgs) |img| @memset(ids[img.at..][0..img.tokens], ph);
    var seq: [40]i32 = undefined;
    for (&seq, ids) |*q, id| q.* = @intCast(id);
    for (imgs) |img| try markPicture(&seq, img.at, img.tokens, ph);
    try checkPictureSeq(gpa, &seq, &ids, &imgs);
    // resuming at 16: the first picture is in the state, the straddling one and the later one run the tower
    try std.testing.expect(!pictureNeedsRows(4, 6, 16));
    try std.testing.expect(pictureNeedsRows(14, 6, 16));
    try std.testing.expect(pictureNeedsRows(30, 4, 16));
    try std.testing.expect(!pictureNeedsRows(14, 2, 16)); // (ending exactly at the cut: wholly before it)
    try std.testing.expect(pictureNeedsRows(4, 6, 0)); // a fresh pass: every picture
    // a sequence whose picture positions were not made dead, or a text id changed: refused
    var live = seq;
    live[15] = @intCast(ph);
    try std.testing.expectError(error.EngramSeqMismatch, checkPictureSeq(gpa, &live, &ids, &imgs));
    var other = seq;
    other[12] += 1;
    try std.testing.expectError(error.EngramSeqMismatch, checkPictureSeq(gpa, &other, &ids, &imgs));
    // a picture span that does not hold the placeholder, or runs past the prompt: refused before anything runs
    var bad = seq;
    try std.testing.expectError(error.BadImageSpan, markPicture(&bad, 10, 3, ph));
    try std.testing.expectError(error.BadImageSpan, markPicture(&bad, 38, 4, ph));
    try std.testing.expectError(error.BadImageSpan, markPicture(&bad, 4, 0, ph));
}

pub fn place(used: []const [2]usize, size: usize, cap: usize) ?usize {
    var base: usize = 0;
    while (base + size <= cap) {
        var clash: ?usize = null;
        for (used) |u| {
            if (base < u[1] and u[0] < base + size) clash = @max(clash orelse 0, u[1]);
        }
        const past = clash orelse return base;
        base = std.mem.alignForward(usize, past, model.chunk_rows);
    }
    return null;
}

/// Soft growth space, adapted from Bertholomus TensorFold PR26 (Apache-2.0); no extra KV allocation.
pub const growth_rows = 16 * model.chunk_rows;

/// Prefer growth room and full extents; under pressure fall back to the exact retained rows.
pub fn placeSoft(gpa: std.mem.Allocator, live: []const [2]usize, kept: []const [3]usize, size: usize, cap: usize) !?usize {
    var used: std.ArrayList([2]usize) = .empty;
    defer used.deinit(gpa);
    try used.ensureTotalCapacity(gpa, live.len + kept.len);
    used.appendSliceAssumeCapacity(live);
    for (kept) |k| used.appendAssumeCapacity(.{ k[0], @max(k[2], k[0] + k[1]) +| growth_rows });
    if (place(used.items, size, cap)) |base| return base;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| used.appendAssumeCapacity(.{ k[0], @max(k[2], k[0] + k[1]) });
    if (place(used.items, size, cap)) |base| return base;
    used.shrinkRetainingCapacity(live.len);
    for (kept) |k| used.appendAssumeCapacity(.{ k[0], k[0] + k[1] });
    return place(used.items, size, cap);
}

/// The token a row's logits draw at absolute position `position`: greedy without sampling (or at temperature 0),
/// else the served sampler with the stream's key.
pub fn draw(row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
    const sm = s orelse return @intCast(sampling.argmax(row));
    if (sm.temperature <= 0) return @intCast(sampling.argmax(row));
    const t = try sampling.sampleRow(row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p });
    return @intCast(t);
}

/// A row with every disallowed token at -inf, drawn by the unchanged sampler.
pub fn drawMasked(buf: []f32, row: []const f32, position: u64, sample: ?lanes.Sampling, mask: []const u32) !u32 {
    try maskRow(buf, row, mask);
    return draw(buf, position, sample);
}

fn maskRow(buf: []f32, row: []const f32, mask: []const u32) !void {
    if (buf.len != row.len) return error.BadMasks;
    var allowed = false;
    for (buf, row, 0..) |*x, value, id| {
        const on = lanes.grammar.allows(mask, id);
        x.* = if (on) value else -std.math.inf(f32);
        allowed = allowed or (on and value > -std.math.inf(f32));
    }
    if (!allowed) return error.NoAllowedToken;
}

pub fn maskWords(vocab: usize) usize { return (vocab + 31) / 32; }

fn masksFit(w: be.Window, words: usize) bool {
    return w.masks.len == 0 or w.masks.len == w.rows() * words;
}

/// Where rank 0's backend calls spend their time (ns, summed; --profile): a round's frame to rank 1, its forward
/// and absorb (synchronized apart), the logits' copy to the host and the draws; the drafter passes; the prefills.
pub const Profile = struct {
    send: u64 = 0,
    enqueue: u64 = 0,
    forward: u64 = 0,
    absorb: u64 = 0,
    logits: u64 = 0,
    sample: u64 = 0,
    pass: u64 = 0,
    prefill: u64 = 0,
    rounds: u64 = 0,
    rows: u64 = 0,
    passes: u64 = 0,
    pass_streams: u64 = 0,
    prefills: u64 = 0,
};

/// A round-cost table "30.0,35.2,..." (TF_DS_ROUND_MS's form): ms by rows from 1.
pub fn parseMs(a: std.mem.Allocator, text: []const u8) ![]const f64 {
    var out: std.ArrayList(f64) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |x| try out.append(a, try std.fmt.parseFloat(f64, std.mem.trim(u8, x, " ")));
    if (out.items.len == 0) return error.BadRoundMs;
    return out.items;
}

/// A stream's place in the pool and the drafts it holds; TF_DS_KV_SHARD: its planner stream, part and generation
/// (parts.zig's Live), and whether its first decode round ran (planDecode).
const Lane = struct {
    slot: usize,
    base: usize,
    end: usize,
    filled: usize = 0, // confirmed prefill positions; observation only
    id: u64 = 0,
    part: u8 = 0,
    gen: u64 = 0,
    decoding: bool = false,
    first: u64 = 0, // the prompt's draw (a handle)
    first_row: []f32 = &.{}, // this stream's full prompt row while a grammar may redraw it
    held: [draft.max_block]u32 = undefined,
    confs: [draft.max_block]f32 = undefined,
    nheld: usize = 0,
};

extern "c" fn exp(x: f64) f64;

/// A kept prompt state (the prompt cache's saved state, kept at a chunk end the prompt pass reached before its
/// replay): the slot's own part on the device (Model.snapCopy), and the extent its compressed rows stay in,
/// [base, base + at), held back from placement while it lives. Stale: a placement needed that room (its restore fails
/// and the prompt runs from 0).
pub const Snap = struct {
    id: u64,
    at: usize,
    base: usize,
    buf: cuda.DeviceBuffer,
    complete: bool = false,
    decoded: bool = false,
    stale: bool = false,
    wanted: u64 = 0, // when a request first waited for the live stream that holds these rows (placeKept; 0: none)
    end: usize = 0, // the extent end of the stream that kept it (placement holds [base, end) back while room allows)
    slot: u32 = 0, // the slot it was kept from (the spill tier's keep note: a follower reads the prompt's ids there)
    loading: bool = false, // a restore's destination: its rows held back, not usable until every rank installed it
    restores: u64 = 0,
    used: u64 = 0,
    generation: u64 = 0,
    restored: bool = false, // read back from the spill tier
    part: u8 = 0, // TF_DS_KV_SHARD: the part holding its rows (parts.zig's Kept; a live move changes it)
    last_use: u64 = 0, loss_reason: u8 = 0,
    held: u32 = 0, // requests that matched this state and wait for their pass: other placements leave its rows alone
    continuation_at: u64 = 0, continuation_state: u64 = 0,
    adaptive_density: u32 = 0, adaptive_at: u64 = 0,
    first_completed_at: u64 = 0,
};

/// Kept states this long or longer say so in the log when they go stale (their prompts then run again from 0).
const note_rows = 32 * model.chunk_rows;

fn noteStale(sn: *const Snap, why: []const u8) void {
    if (sn.at >= note_rows) std.log.info("dsv41: the kept state of {d} tokens at {d} goes stale: {s}", .{ sn.at, sn.base, why });
}

/// How long a request waits for the live stream that holds the kept state it resumes from to end (then it resumes
/// in place and every kept state stays) before other kept states give their room for a copy of it.
const held_wait_ns: u64 = 10 * std.time.ns_per_s;

/// First tokens a handle names (each prompt's draw, read back right after its prefill).
const ring = 1024;

pub const ResumeFacts = struct {
    restored: bool,
    pub fn capture(kept: ?*const Snap) ResumeFacts {
        return .{ .restored = if (kept) |snap| snap.restored else false };
    }
    pub fn finish(facts: ResumeFacts, model_: anytype, check: bool, slot: usize, len: usize) !void {
        if (check and facts.restored) try model_.spillCheckResumed(slot, len);
    }
};

pub const Lanes = struct {
    const Fill = struct {
        start: usize, len: usize, replay: usize, last_end: usize, head: bool,
        slot: usize, base: usize, source_facts: ResumeFacts,
        // Image rows belong to this fill, not the model's current slot. A later chunk must restore them after another
        // slot or a decode round has rebound the model; cached images never enter these lists.
        spans: std.ArrayList(@import("prompt.zig").Span) = .empty,
        rows: std.ArrayList(u8) = .empty,
        needs_resume: bool = false,
        progress: Progress = .{}, began: u64,
    };

    gpa: std.mem.Allocator,
    m: *Model,
    peer: ?link.Link, // rank 1 (null: one rank); every follower gets each frame (Model.followers)
    vision: ?*@import("vision.zig").Vision = null, // rank 0's tower (a vision checkpoint served with its kit's vision/)
    step: u64 = 0,
    rdma_idle: bool = false,
    rdma_park: bool = true, // TF_DS_RDMA_PARK=0 disables coordinated parking; independent of RDMA idle naps
    idle_retry_at: u64 = 0, // a busy follower postpones another all-rank vote instead of polling the links
    logit_dump_dir: ?[]const u8 = null, // closed-window diagnostics only; no production I/O by default
    logit_dump_serial: u64 = 0,
    short_continuation: bool = false, // experimental full-decoder resume; disabled unless explicitly qualified
    wire: Wire = .{},
    move_deadline: ?u64 = null,
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, Fill) = .empty,
    streams: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    wc: [window_rows]lanes.config.Cost = undefined, // the window costs (init)
    sc: [max_rows]lanes.config.Cost = undefined, // the shared forwards' costs (init)
    used: [max_streams]bool = @splat(false),
    seqs: [max_streams]std.ArrayList(i32) = @splat(.empty),
    built: Built = .{},
    prof: ?Profile = null,
    adaptive_learning: bool = false, // active online learning, opt-in
    fix_k: ?usize = null, // a measurement (tf-dsv41-lanes --fix-k): K drafts a stream, chances 1 to K and 0 past it
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    snaps: std.ArrayList(*Snap) = .empty, // kept prompt states, oldest first
    next_snap: u64 = 1,
    relocation_budget: move_budget.Budget = .{},
    relocation_enabled: bool = true,
    relocation_generation: u64 = 0,
    kv_clock: u64 = 0,
    pressure_planner: bool = false,
    short_reuse: bool = false,
    continuation_donors: bool = false,
    adaptive_prefix: bool = false,
    first_continuation: bool = false,
    relocate_max_bytes: usize = 0,
    sample_workspace: sampling.Workspace = .{},
    masked: []f32 = &.{}, // reused full-logit rows for grammar masks
    penalty_workers: ?*penalty_pool.Pool = null,
    penalty_disabled: bool = false,
    prefill_yield: bool = false, // TF_DS_PREFILL_YIELD=1; otherwise the whole prompt remains one host call

    spill: ?*SpillTier = null, // --spill-dir/--spill-gib: the spill tier (native.zig opens it; null: off, nothing changes)
    kv: ?*kv_lanes.Kv = null, // TF_DS_KV_SHARD: the parts' planner (startKv, at open; null: the switch off)
    kv_stats: lanes.kv_parts.Stats = .{}, // the last parts' stats kvPartsFn built (the engine loop reads them every
    kv_stats_at: u64 = 0, //   iteration: rebuilt at most each kv_stats_ns, Model.now; 0: never built)
    /// The pair's one rank_wire Leader (native.openPair: agreed with rank_identity's full fingerprint before anything
    /// spill- or shard-related crosses), shared by the spill tier and the KV parts; owned here (null: neither is on).
    rw: ?*rank_wire.Leader = null,

    fn drawToken(self: *Lanes, row: []const f32, position: u64, s: ?lanes.Sampling) !u32 {
        const sm = s orelse return @intCast(sampling.argmax(row));
        const t = try sampling.sampleRowWithWorkspace(&self.sample_workspace, self.gpa, row, position, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p });
        return @intCast(t);
    }

    pub fn init(gpa: std.mem.Allocator, m: *Model) Lanes {
        var ln: Lanes = .{ .gpa = gpa, .m = m, .peer = if (m.world > 1) m.peer() else null };
        // the round costs by rows: ROUND_MS[min(rows, len) - 1] (the served table, or TF_DS_ROUND_MS)
        const t = m.round_ms;
        for (&ln.wc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        for (&ln.sc, 0..) |*c, i| c.* = .{ .width = @intCast(i + 1), .ms = t[@min(i, t.len - 1)] };
        if (!@import("builtin").is_test) {
            ln.rdma_park = idle_protocol.enabled(getenv("TF_DS_RDMA_PARK"));
            if (std.c.getenv("TF_DS_PREFILL_YIELD")) |v| ln.prefill_yield = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_KV_PRESSURE_PLANNER")) |v| ln.pressure_planner = v[0] == '1';
        if (std.c.getenv("TF_KV_SHORT_REUSE")) |v| ln.short_reuse = std.mem.eql(u8, std.mem.span(v), "1");
        if (std.c.getenv("TF_KV_CONTINUATION_DONOR")) |v| ln.continuation_donors = std.mem.eql(u8, std.mem.span(v), "1");
        if (std.c.getenv("TF_CACHE_ADAPTIVE_PREFIX")) |v| ln.adaptive_prefix = std.mem.eql(u8, std.mem.span(v), "1");
        ln.first_continuation = cache_batch.Flags.fromEnv().first_continuation;
            if (std.c.getenv("TF_KV_RELOCATE_MAX_BYTES")) |v| {
                const n = std.fmt.parseInt(usize, std.mem.sliceTo(v, 0), 10) catch 0;
                ln.relocate_max_bytes = @min(n, pressure.hard_copy_bytes);
            }
        }
        return ln;
    }

    /// Tells rank 1 to stop, then frees the host state (the model is the caller's).
    pub fn deinit(self: *Lanes) void {
        self.wakeRdma();
        const cleanup = cuda.context_health.cleanupMark();
        self.stopKv();
        for (self.snaps.items) |sn| {
            sn.buf.free();
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
            self.gpa.destroy(sn);
        }
        self.snaps.deinit(self.gpa);
        if (self.penalty_workers) |p| p.destroy(self.gpa);
        self.sample_workspace.deinit(self.gpa);
        if (self.peer != null) self.send(.done) catch {};
        if (self.rw) |l| self.gpa.destroy(l); // (after the tier and the planner that borrowed it)
        self.rw = null;
        self.fills.deinit(self.gpa);
        var live = self.streams.valueIterator();
        while (live.next()) |lane| self.gpa.free(lane.first_row);
        self.streams.deinit(self.gpa);
        self.gpa.free(self.masked);
        for (&self.seqs) |*q| q.deinit(self.gpa);
        self.wire.deinit(self.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = if (self.prefill_yield) &vt_stepping else &vt_whole };
    }

    const vt_whole: be.Backend.VTable = .{
        .prefill = prefillFn,
        .first = firstFn, .first_masked = firstMaskedFn, .held = heldFn, .queue = queueFn, .read = readFn,
        .verify = verifyFn, .keep = keepFn, .draft = draftFn,
        .probabilities = probabilitiesFn, .release = releaseFn, .token_pool = tokenPoolFn,
        .begin_timing = beginTimingFn, .timing_clean = timingCleanFn,
    };
    const vt_stepping: be.Backend.VTable = blk: {
        var v = vt_whole;
        v.prefill_begin = prefillBeginFn;
        v.prefill_step = prefillStepFn;
        v.prefill_end = prefillEndFn;
        break :blk v;
    };

    test "prefill admission switch preserves whole-pass default and exposes stepping only when enabled" {
        var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null };
        const whole = ln.backend().vtable;
        try std.testing.expect(whole.prefill_begin == null and whole.prefill_step == null and whole.prefill_end == null);
        ln.prefill_yield = true;
        const stepped = ln.backend().vtable;
        try std.testing.expect(stepped.prefill_begin != null and stepped.prefill_step != null and stepped.prefill_end != null);
        try std.testing.expect(whole.prefill == stepped.prefill and whole.verify == stepped.verify);
    }

    /// The facts the round loop reads: windows of the pending row and up to the drafter's block, several streams'
    /// windows in one forward of at most 16 rows (four streams), the drafter's chances, the served lane's round ms.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.m.drafting();
        return .{
            .adaptive_learning = self.adaptive_learning,
            .draft_cal = self.m.draft_settings.weight,
            .first_draft_depth = self.m.draft_settings.first_depth,
            .exact_width = window_rows,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (self.m.dr) |dr| @intCast(dr.n) else 1,
            .window_costs = self.wc[0..],
            .mtp_step_ms = draft.draft_ms,
            .streams_exact = true,
            .hidden_rows = true,
            .batch_rows = @intCast(self.m.eng.round_rows),
            .max_streams = @intCast(self.m.streams),
            .shared_costs = self.sc[0..self.m.eng.round_rows],
            .draft_probabilities = drafting,
            .draft_streams = drafting,
        };
    }

    test "graph telemetry timing covers both caches and resets only at round boundary" {
        var m: model.Model = undefined;
        m.graphs = round.Graphs.init(std.testing.allocator, 128);
        m.pgraphs = draft.PassGraphs.init(std.testing.allocator, 128);
        m.graphs.?.telemetry.on = true;
        m.pgraphs.?.telemetry.on = true;
        var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null };
        const be_test = ln.backend();
        be_test.beginTiming();
        try std.testing.expect(be_test.timingClean());
        m.graphs.?.telemetry.miss();
        m.pgraphs.?.telemetry.hit();
        try std.testing.expect(!be_test.timingClean());
        be_test.beginTiming();
        try std.testing.expect(be_test.timingClean());
        m.pgraphs.?.telemetry.miss();
        m.graphs.?.telemetry.hit();
        try std.testing.expect(!be_test.timingClean());
        be_test.beginTiming();
        try std.testing.expect(be_test.timingClean());
        m.graphs = null;
        m.pgraphs = null;
        be_test.beginTiming();
        try std.testing.expect(be_test.timingClean());
    }

    fn beginTimingFn(ptr: *anyopaque) void {
        const m = of(ptr).m;
        if (m.graphs) |*g| g.telemetry.begin();
        if (m.pgraphs) |*g| g.telemetry.begin();
    }

    fn timingCleanFn(ptr: *anyopaque) bool {
        const m = of(ptr).m;
        return (if (m.graphs) |*g| g.telemetry.timingClean() else true) and
            (if (m.pgraphs) |*g| g.telemetry.timingClean() else true);
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The frame being built goes to rank 1 (after `begin` and the fields), or to ranks 1-3 on the four-node split.
    fn flush(self: *Lanes) !void {
        if (self.peer == null) return;
        for (self.m.followers()) |l| {
            const peer = l orelse return error.TransferFailed;
            if (self.move_deadline) |deadline| try peer.sendWithin(self.wire.buf.items, deadline) else try peer.send(self.wire.buf.items);
        }
    }

    fn begin(self: *Lanes, kind: Kind) !bool {
        self.wakeRdma(); // every command can precede CUDA, a graph vote, a restore or a relocation
        if (self.peer == null) return false;
        try self.wire.begin(self.gpa, kind, self.step);
        self.step += 1;
        return true;
    }

    fn send(self: *Lanes, kind: Kind) !void {
        if (try self.begin(kind)) try self.flush();
    }

    /// No proxy sleeps until every rank is quiescent and voted READY. A follower
    /// with pending spill copies votes BUSY, cancelling the entire attempt. Lost
    /// peers or uncertain frames end the TP world through its existing fatal path.
    pub fn idleRdma(self: *Lanes) void {
        if (!self.rdma_park or self.rdma_idle or self.streams.count() != 0 or
            self.fills.count() != 0 or self.move_deadline != null or !hasRdma(self.m)) return;
        if (self.spill) |tier| if (tier.idleBusy()) return;
        const now = link.Link.monotonicNs();
        if (now < self.idle_retry_at) return;
        self.m.joinAhead();
        const dev: Model.SpillDev = .{ .m = self.m };
        self.m.ctx.synchronize() catch |err| dev.failed(err);
        quiesceRdma(self.m, link.Link.monotonicNs() +| idle_protocol.budget_ns) catch |err| dev.failed(err);
        const deadline = link.Link.monotonicNs() +| idle_protocol.budget_ns;
        self.move_deadline = deadline;
        defer self.move_deadline = null;
        self.send(.rdma_idle) catch |err| dev.failed(err);
        if (!(idle_protocol.leader(self.m.followers(), deadline) catch |err| dev.failed(err))) {
            self.idle_retry_at = link.Link.monotonicNs() +| 250 * std.time.ns_per_ms;
            return;
        }
        self.idle_retry_at = 0;
        setRdmaIdle(self.m, true);
        self.rdma_idle = true;
    }

    pub fn wakeRdma(self: *Lanes) void {
        if (!self.rdma_idle) return;
        setRdmaIdle(self.m, false);
        self.rdma_idle = false;
    }

    fn spillWake(ctx: *anyopaque) void { of(ctx).wakeRdma(); }
    fn spillIdle(ctx: *anyopaque) void { of(ctx).idleRdma(); }

    /// Runtime agreement, outside the spill store's persistent fingerprint. Only the opt-in path adds this frame.
    pub fn agreeGraphVote(self: *Lanes) !void {
        if (!try self.begin(.graph_vote_agree)) return error.NoLink;
        try self.wire.int(self.gpa, u8, 1);
        try graph_vote.agreeSpill(self.m.followers(), self.wire.buf.items, self.m.gvote.?.timeout_ns);
    }

    /// Union of occupied intervals, without allocation or double-counting overlapping snapshots.
    fn retained(self: *const Lanes, active: []const [2]usize) usize {
        var cursor: usize = 0;
        var total: usize = 0;
        while (true) {
            var begin_: ?usize = null;
            for (active) |r| if (r[1] > r[0] and r[1] > cursor) {
                begin_ = @min(begin_ orelse r[0], @max(cursor, r[0]));
            };
            for (self.snaps.items) |s| if (!s.stale and s.at > 0 and s.base + s.at > cursor) {
                begin_ = @min(begin_ orelse s.base, @max(cursor, s.base));
            };
            const first = begin_ orelse break;
            var end = first;
            while (true) {
                const before = end;
                for (active) |r| if (r[0] <= end and r[1] > end) {
                    end = r[1];
                };
                for (self.snaps.items) |s| if (!s.stale and s.base <= end and s.base + s.at > end) {
                    end = s.base + s.at;
                };
                if (end == before) break;
            }
            if (end == first) break;
            total += end - first;
            cursor = end;
        }
        return total;
    }

    fn tokenPoolFn(ptr: *anyopaque) be.TokenPool {
        const self = of(ptr);
        var ranges: [max_streams][2]usize = undefined;
        var count: usize = 0;
        var active: usize = 0;
        var reserved: usize = 0;
        var it = self.streams.iterator();
        while (it.next()) |entry| {
            const lane = entry.value_ptr;
            const filled = @min(lane.end - lane.base, @max(lane.filled, entry.key_ptr.*.cache_len));
            ranges[count] = .{ lane.base, lane.base + filled };
            active += filled;
            reserved += lane.end - lane.base;
            count += 1;
        }
        const total = self.retained(ranges[0..count]);
        return .{ .capacity = self.m.pool_cap, .active = active, .cached = total -| active, .reserved = reserved };
    }

    /// GPU completion events are queried on the engine thread, never synchronized for telemetry.
    const Progress = struct {
        marks: std.ArrayList(struct { event: cuda.Event, at: u32 }) = .empty,
        consumed: usize = 0,
        fn deinit(p: *Progress, gpa: std.mem.Allocator) void {
            for (p.marks.items) |*mark| mark.event.deinit();
            p.marks.deinit(gpa);
        }
        fn record(p: *Progress, self: *Lanes, s: *lanes.Stream, at: usize) void {
            if (s.prefill_progress == null) return;
            var event = cuda.Event.init(self.m.stream.d, false) catch return;
            event.record(self.m.stream) catch {
                event.deinit();
                return;
            };
            p.marks.append(self.gpa, .{ .event = event, .at = @intCast(at) }) catch {
                event.deinit();
                return;
            };
            p.poll(self, s);
        }
        fn poll(p: *Progress, self: *Lanes, s: *lanes.Stream) void {
            var at: ?u32 = null;
            while (p.consumed < p.marks.items.len) {
                const mark = &p.marks.items[p.consumed];
                if (!(mark.event.done() catch false)) break;
                at = mark.at;
                p.consumed += 1;
            }
            if (at) |n| {
                self.streams.getPtr(s).?.filled = n;
                if (s.prefill_progress) |hook| hook.at(hook.ptr, s, n);
            }
        }
    };

    // -- TF_DS_KV_SHARD: kv_lanes.zig drives parts.zig's planner (docs/port/shard-lanes.md) ------------------------

    /// TF_DS_KV_SHARD: the parts' planner made and agreed with the followers (hello, shard_layout, shard_set, the
    /// first digest check), once, at open with the Lanes at their final address (the hooks point here): a follower
    /// started with other switches or another pool fails the start, never a request. Nothing with the switch off.
    pub fn startKv(self: *Lanes) !void {
        const m = self.m;
        if (m.kvs == null or self.kv != null) return;
        const settings = try kv_lanes.Settings.fromEnv(getenv);
        const k = try self.gpa.create(kv_lanes.Kv);
        errdefer self.gpa.destroy(k);
        k.* = try kv_lanes.Kv.init(self.gpa, kvConfig(m), settings, modelDev(m), .{ .ctx = self, .stale = kvStale, .moved = kvMoved, .stage = kvStage, .bump = kvBump });
        errdefer k.deinit();
        // the pair's one Leader (native.openPair agreed it with the full fingerprint: the spill tier, when on, posts
        // through the same one), so the followers heard one hello and one Tracker sees every frame
        if (self.peer != null) try k.lead(.{ .ctx = self, .rw = kvSendRw }, m.followers(), self.rw orelse return error.PairNotAgreed, 0, undefined);
        // with the spill tier each follower opens its store (every block file scanned, every manifest read) on its own
        // thread from the agreement, and native.openSpill waited for it (awaitFollowers) before this; start's asks
        // keep the hello's bound as a margin, not shard_ns' 10 s
        const l = self.rw;
        const shard_ns = if (l) |x| x.timeouts.shard_ns else 0;
        if (self.spill != null) if (l) |x| {
            x.timeouts.shard_ns = @max(shard_ns, x.timeouts.hello_ns);
        };
        defer if (l) |x| {
            x.timeouts.shard_ns = shard_ns;
        };
        k.start() catch |err| {
            if (self.spill != null) std.log.err("dsv41: the KV parts' start with the spill tier failed ({s}); a follower that logged \"dsv41 spill: rank N could not open its store\" names the cause", .{@errorName(err)});
            return err;
        };
        self.kv = k;
    }

    /// startKv undone (an open that fails after it).
    pub fn stopKv(self: *Lanes) void {
        const k = self.kv orelse return;
        k.deinit();
        self.gpa.destroy(k);
        self.kv = null;
    }

    fn kvSendRw(ctx: *anyopaque, sub: []const u8) anyerror!void {
        const self = of(ctx);
        if (try self.begin(.rank_wire)) {
            try self.wire.buf.appendSlice(self.gpa, sub);
            try self.flush();
        }
    }

    /// A kept state the planner gave up (a stale op): its Snap too; with the spill tier, its unpublished write stages
    /// or cancels here and on every follower (write_staged) before any row changes, as staleOver guards.
    fn kvStale(ctx: *anyopaque, kept: u64) void {
        const self = of(ctx);
        for (self.snaps.items) |sn| if (sn.id == kept) {
            if (!sn.stale) {
                noteStale(sn, "a placement needed its room (its part's oldest first)");
                if (self.spill) |t| t.guardSnap(sn.id, 0);
            }
            sn.loss_reason = 2;
            sn.stale = true;
        };
    }

    /// A stage op (TF_DS_KV_SHARD with the spill tier): kept state `kept`'s pending write stages its remaining rows
    /// now (or is cancelled) before the stale or move after it in the plan; followers stage their own at the same op.
    fn kvStage(ctx: *anyopaque, kept: u64) void {
        if (of(ctx).spill) |t| t.guardSnap(kept, 0);
    }

    /// A transfer's destination rows are about to change (copy, carry, move): the spill pins over them go stale.
    fn kvBump(ctx: *anyopaque, lo: usize, hi: usize) void {
        if (of(ctx).spill) |t| t.bump(lo, hi);
    }

    /// The planner's pending flags from the spill tier, before a plan (parts.Kept.pending: rank 0's own writes that
    /// still read a kept state's rows; a plan that stales or moves one stages it first).
    fn kvPending(self: *Lanes) void {
        const t = self.spill orelse return;
        const k = self.kv orelse return;
        // (one store lock a batch, not one a kept state: this runs before every plan, beside the store's writer)
        const batch = 64;
        var ids: [batch]u64 = undefined;
        var out: [batch]bool = undefined;
        const kept = k.planner.kept.items;
        var i: usize = 0;
        while (i < kept.len) : (i += batch) {
            const n = @min(batch, kept.len - i);
            for (kept[i..][0..n], ids[0..n]) |x, *id| id.* = x.id;
            t.writesPending(ids[0..n], out[0..n]);
            for (kept[i..][0..n], out[0..n]) |*x, o| x.pending = o;
        }
    }

    /// A live stream moved with its kept states: every Snap's and Lane's rows from the planner again (copyExtent,
    /// usable and the spill tier read them).
    fn kvMoved(ctx: *anyopaque, _: u64) void {
        of(ctx).kvSync();
    }

    fn patient(self: *const Lanes, sn: *Snap) bool {
        const now = self.m.now();
        if (sn.wanted == 0) sn.wanted = now;
        return now - sn.wanted < held_wait_ns;
    }

    fn kvSync(self: *Lanes) void {
        const p = &(self.kv orelse return).planner;
        for (self.snaps.items) |sn| if (p.keptAt(sn.id)) |k| {
            sn.base = k.base;
            sn.end = k.end;
            sn.part = k.part;
        };
        var it = self.streams.valueIterator();
        while (it.next()) |l| if (p.liveIndex(l.id)) |i| {
            const x = p.lives.items[i];
            l.base = x.base;
            l.end = x.base + x.size;
            l.part = x.part;
            l.gen = x.gen;
        };
    }

    const PlacedParts = struct { id: u64, part: u8, gen: u64, base: usize, kept: ?*Snap, moved: bool };

    /// prefillFn's placement with the switch on (kv_lanes.Kv.admit: plan, apply, send, act on the ops in order, the
    /// transfers finished with rank 0's ok): the extent in one part, and the kept state it resumes from when the plan
    /// resumed it (in place, or copied or carried there and finished ok).
    fn placeParts(self: *Lanes, k: *kv_lanes.Kv, s: *lanes.Stream, size: usize, slot: usize, len: usize) !PlacedParts {
        var src: ?*Snap = null;
        if (s.reuse.saved) |p| {
            const sn: *Snap = @ptrCast(@alignCast(p));
            if (self.usable(sn, len)) src = sn else s.reuse_failed = true;
        }
        // (lanes owns the clock: patient only when a live stream holds the state, as placeKept asks)
        const waits = if (src) |sn| k.planner.held(sn.id) and self.patient(sn) else false;
        self.kvPending();
        const a = try k.admit(@intFromPtr(s), std.hash.Wyhash.hash(0, s.id), @intCast(slot), size, if (src) |sn| sn.id else null, waits);
        if (src) |sn| {
            sn.wanted = 0;
            if (!a.resumed) {
                if (sn.at >= note_rows) std.log.info("dsv41: the kept state of {d} tokens at {d} was not resumed ({s}); the prompt runs from 0", .{ sn.at, sn.base, if (a.dropped) "no room in its part" else "its transfer failed" });
                s.reuse_failed = true;
            }
        }
        const kept = if (a.resumed) src else null;
        return .{ .id = a.id, .part = a.part, .gen = a.gen, .base = a.base, .kept = kept, .moved = if (kept) |sn| a.base != sn.base else false };
    }

    /// verifyFn's planner side: a stream's first round gives its twins back (planDecode), then the planner's clock and
    /// every window's rows with content once this round ran (its cache length plus the round's rows, which it
    /// writes at [cache_len, cache_len + rows): an upper bound, rejected drafts' rows included, so a live move before
    /// the next round copies every row this one wrote), and the periodic digest check.
    fn kvRound(self: *Lanes, k: *kv_lanes.Kv, windows: []const be.Window) !void {
        var vs: [max_streams]kv_lanes.Kv.Valid = undefined;
        for (windows, 0..) |w, i| {
            const l = self.streams.getPtr(w.stream) orelse return error.UnknownStream;
            if (!l.decoding) {
                try k.decodeStart(l.id);
                l.decoding = true;
            }
            vs[i] = .{ .id = l.id, .valid = @as(usize, @intCast(w.stream.cache_len)) + w.rows() };
        }
        try k.round(vs[0..windows.len]);
    }

    /// The parts' stats for /health and /metrics (native.Loaded.kv_parts: the host reads them on its engine thread,
    /// in noteLive, every engine-loop iteration). Rebuilt at most each kv_stats_ns: a rebuild scans the planner and
    /// sorts every part's spans (allocating), which base rounds never pay and a round should not wait for.
    pub fn kvPartsFn(ptr: *anyopaque, out: *lanes.kv_parts.Stats) void {
        const self = of(ptr);
        const k = self.kv orelse return;
        const now = self.m.now();
        if (self.kv_stats_at != 0 and now -| self.kv_stats_at < kv_stats_ns) {
            out.* = self.kv_stats;
            return;
        }
        self.kv_stats_at = now;
        k.stats(out);
        k.wantStats(); // (the followers' rows are asked for again at the next round, at most each stats_ms)
        if (self.m.kvs) |x| {
            const st = x.stats;
            out.prompt_fetches = st.prompt_fetches;
            out.prompt_rows = st.prompt_rows;
            out.prompt_picks = st.prompt_picks;
            out.prompt_bytes = st.prompt_bytes;
            out.prompt_syncs = st.prompt_syncs;
            out.fetch_rounds = st.fetch_rounds;
            out.owner_rounds = st.owner_rounds;
            out.copies = st.carries;
            out.copy_bytes = st.carry_bytes;
            shardRank(self.m, &out.rank[0]); // (rank 0's own row; the followers' ride their stats replies)
        }
        self.kv_stats = out.*;
    }

    /// How stale /health's parts' stats may be (kvPartsFn).
    const kv_stats_ns: u64 = 250 * std.time.ns_per_ms;

    // -- the vtable ---------------------------------------------------------------------------------------------

    /// The lowest free slot, the first extent that fits, the prompt in chunks of the served size (at multiples of it:
    /// the chunks the gates checked; the host's chunk plan is not followed), the head's draw of the first token.
    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        try prefillBeginFn(ptr, s);
        defer prefillEndFn(ptr, s);
        while (!try prefillStepFn(ptr, s)) {
            if (s.images.len == 0) if (s.between) |hook| if (hook.wanted(hook.ptr)) {
                // A decode round or tier restore may change the selected slot. Join read-ahead before it touches the
                // sequence and rebind on the next chunk; without an interleave the next chunk keeps its read-ahead.
                self.m.joinAhead();
                self.m.fill_seq = &.{};
                self.fills.getPtr(s).?.needs_resume = true;
                try hook.run(hook.ptr);
            };
        }
    }

    fn sendFillImages(self: *Lanes, spans: []const @import("prompt.zig").Span, rows: []const u8) !void {
        if (spans.len == 0) return;
        const gpa = self.gpa;
        var off: usize = 0;
        while (off < rows.len) {
            const n = @min(image_piece, rows.len - off);
            if (try self.begin(.fill_images)) {
                try self.wire.int(gpa, u32, @intCast(spans.len));
                for (spans) |sp| {
                    try self.wire.int(gpa, u64, sp.at);
                    try self.wire.int(gpa, u64, sp.len);
                }
                try self.wire.int(gpa, u64, off);
                try self.wire.buf.appendSlice(gpa, rows[off..][0..n]);
                try self.flush();
            }
            off += n;
        }
        try self.m.setImages(spans, rows);
    }

    fn prefillBeginFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        self.wakeRdma(); // placement may relocate idle KV before a stream is admitted
        const m = self.m;
        const gpa = self.gpa;
        if (self.fills.contains(s)) return error.PrefillAlreadyStarted;
        try self.fills.ensureUnusedCapacity(gpa, 1);
        const t0 = m.now();
        const ids = s.prompt();
        const len = ids.len;
        if (len == 0) return error.EmptyPrompt;
        if (m.prefill.af and len > m.prompt_cap) return error.PromptTooLong;
        const size = std.mem.alignForward(usize, len + s.max_new + m.eng.round_rows + 2, model.chunk_rows);
        if (size > m.request_cap) return error.PromptTooLong; // (the pool's, unless TF_DS_POOL or TF_DS_KV_SHARD part it)
        // TF_DS_KV_SHARD: parts.zig's planner places the extent inside one part (placeParts: part-local placeSoft and
        // eviction, borrowed twins, carries, live moves) and the lane carries its part into every round (Win.part)
        const kv = self.kv;
        if (kv == null and m.kvs != null) return error.KvShardNotStarted; // (startKv at open)
        if (self.streams.fetchRemove(s)) |old| {
            self.used[old.value.slot] = false;
            if (kv) |k| try k.release(old.value.id);
        }
        const slot = std.mem.indexOfScalar(bool, self.used[0..self.m.streams], false) orelse return error.NoFreeSlot;
        if (self.spill) |t| t.bumpSlot(slot);
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        // a kept state of this prompt's prefix: the pass resumes at its chunk end, in its extent, or in one of its own
        // with the kept rows copied there when a live stream holds that extent (a burst's second request on a prompt)
        s.cached = 0;
        const parted: ?PlacedParts = if (kv) |k| try self.placeParts(k, s, size, slot, len) else null;
        errdefer if (parted) |pp| kv.?.release(pp.id) catch |err| std.log.err("dsv41: a failed prompt pass kept its part's extent: {s}", .{@errorName(err)});
        var kept: ?*Snap = if (parted) |pp| pp.kept else null;
        var moved = if (parted) |pp| pp.moved else false;
        if (parted == null) if (s.reuse.saved) |p| {
            const sn: *Snap = @ptrCast(@alignCast(p));
            // The cache matched the full image identities, so the saved rows are this prompt's.
            if (self.usable(sn, len)) {
                kept = sn;
                moved = sn.base + size > m.pool_cap or clashes(taken[0..nt], sn.base, sn.base + size) or
                    self.overwritesKept(sn.base + sn.at, sn.base + size, sn);
            } else s.reuse_failed = true;
        };
        const base = if (parted) |pp| pp.base else if (kept != null and !moved) kept.?.base else try self.placeKept(taken[0..nt], size, kept);
        if (kept) |sn| {
            sn.wanted = 0;
            if (!moved) sn.end = @max(sn.end, base + size); // its extent now: the next resume in place finds it held
        }
        if (kept) |sn| if (moved and sn.stale) { // the placement needed the kept rows' room: the prompt runs from 0
            kept = null;
            s.reuse_failed = true;
        };
        // the rows this stream writes are no other kept state's
        if (parted == null) self.staleOver(if (kept != null and !moved) base + kept.?.at else base, base + size, kept);
        try self.streams.put(gpa, s, .{ .slot = slot, .base = base, .end = base + size, .id = if (parted) |pp| pp.id else 0, .part = if (parted) |pp| pp.part else 0, .gen = if (parted) |pp| pp.gen else 0 });
        self.used[slot] = true;
        errdefer {
            _ = self.streams.remove(s);
            self.used[slot] = false;
        }
        const seq = &self.seqs[slot];
        try seq.resize(gpa, len);
        for (seq.items, ids) |*q, id| q.* = @intCast(id);
        // images: each span's positions dead in the sequence (negative: no Engram n-gram reaches into or across one),
        // its rows ready (a gate's) or the tower's (rank 0), all of them sent to rank 1 after the fill's start
        var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
        errdefer spans.deinit(gpa);
        var rows: std.ArrayList(u8) = .empty;
        errdefer rows.deinit(gpa);
        if (s.images.len > 0) {
            const image_id = m.cfg.image_token orelse return error.NoVision;
            const row_bytes = m.cfg.hidden * 2;
            var at_row: usize = 0;
            for (s.images) |img| {
                try markPicture(seq.items, img.at, img.tokens, image_id);
                if (!pictureNeedsRows(img.at, img.tokens, if (kept) |sn| sn.at else 0)) continue;
                if (img.rows) {
                    if (img.bytes.len != img.tokens * row_bytes) return error.BadImageRows;
                    try rows.appendSlice(gpa, img.bytes);
                } else {
                    // a prepared picture (vision.zig): the tower's span rows, here on rank 0
                    const v = self.vision orelse return error.NoVisionTower;
                    if (try v.rows(img.bytes, gpa, &rows) != img.tokens) return error.BadImageSpan;
                }
                try spans.append(gpa, .{ .at = img.at, .len = img.tokens, .row = at_row });
                at_row += img.tokens;
            }
            // a state read back from disk resumes with this request's own sequence, built above from its ids and its
            // pictures' spans (the follower gets it at fill_begin). It is the point's by construction: the tier hit only
            // because the point's media name exactly these spans (lane_host.tierMedia binds Request.cache_media to
            // Request.images) and its keyed ids derive from them (format.mediaIds), so no check here could fail;
            // checkPictureSeq is the tests' oracle for that sequence.
        }
        if (try self.begin(.fill_begin)) {
            try self.wire.int(gpa, u32, @intCast(slot));
            try self.wire.int(gpa, u64, base);
            try self.wire.ids(gpa, seq.items);
            try self.flush();
        }
        try m.fillBegin(slot, base);
        // (its twins' view, bound as the follower binds it at fill_begin; it stays the fill on both ranks past the last
        // chunk, until the next fill or its release: the first decode round's untwin unbinds on both)
        if (parted) |pp| try kv.?.fillBegin(pp.id, base);
        // Only the uncached spans travel to followers. Keep their prepared rows with the slot so a yielded image
        // prefill can reselect its own embeddings without running the vision tower a second time.
        try self.sendFillImages(spans.items, rows.items);
        // A fully decoded prefix has valid preceding window keys. For a short
        // append, preserve that decoder history instead of discarding it at the
        // new prompt's CED boundary. Encoder-only snapshots retain bounded replay.
        // This path is opt-in and changes continuation numerics; qualify separately.
        const replay = if (kept) |sn| (if (self.shortResume(sn, len)) sn.at -| m.cfg.window else len -| m.cfg.window) else len -| m.cfg.window;
        var start: usize = 0;
        const source_facts = ResumeFacts.capture(kept);
        if (kept) |sn| {
            sn.restores +|= 1;
            sn.used = m.now();
            self.kv_clock +|= 1;
            sn.last_use = self.kv_clock;
            if (try self.begin(.snap_restore)) {
                try self.wire.int(gpa, u64, sn.id);
                try self.wire.int(gpa, u32, @intCast(slot));
                try self.wire.int(gpa, u64, if (parted != null) base else sn.base); // (TF_DS_KV_SHARD: the plan's transfer ran)
                try self.wire.int(gpa, u64, base);
                try self.wire.int(gpa, u64, sn.at);
                try self.flush();
            }
            if (base != sn.base and parted == null) {
                if (self.spill) |t| t.bump(base, base + sn.at);
                try m.copyExtent(sn.base, base, sn.at);
            }
            try m.snapCopy(slot, sn.buf.ptr, false);
            start = sn.at;
            s.cached = @intCast(sn.at);
        }
        // where this pass keeps states: the cache's marks and its last chunk end before the replay (the served lane's
        // kept prompt), each at the chunk end at or before it
        const last_end = std.mem.alignBackward(usize, @min(replay, len - 1), 2);
        self.streams.getPtr(s).?.filled = start;
        if (s.prefill_progress) |hook| hook.at(hook.ptr, s, @intCast(start));
        const held_spans = if (self.prefill_yield) spans else std.ArrayList(@import("prompt.zig").Span).empty;
        const held_rows = if (self.prefill_yield) rows else std.ArrayList(u8).empty;
        if (!self.prefill_yield) { // whole-pass requests never switch image slots; release preparation immediately
            spans.deinit(gpa);
            rows.deinit(gpa);
        }
        self.fills.putAssumeCapacity(s, .{ .start = start, .len = len, .replay = replay,
            .last_end = last_end, .head = start == len, .slot = slot, .base = base,
            .source_facts = source_facts, .spans = held_spans, .rows = held_rows, .began = t0 });
    }

    /// Reselect a yielded slot on all ranks without clearing its KV rows or drafter rings. Images retain only the
    /// uncached prepared rows, which must be installed again because fillBegin clears the model's image spans.
    fn resumeFill(self: *Lanes, s: *lanes.Stream, f: *Fill) !void {
        const m = self.m;
        if (f.needs_resume) {
            const l = self.streams.getPtr(s) orelse return error.UnknownStream;
            f.base = l.base; // a planner relocation between steps owns the current extent
            if (try self.begin(.fill_resume)) {
                try self.wire.int(self.gpa, u32, @intCast(f.slot));
                try self.wire.int(self.gpa, u64, f.base);
                try self.flush();
            }
            try m.fillBegin(f.slot, f.base);
            if (self.kv) |k| try k.fillBegin(l.id, f.base);
            try self.sendFillImages(f.spans.items, f.rows.items);
            f.needs_resume = false;
        }
        m.fill_seq = self.seqs[f.slot].items;
    }

    fn prefillEndFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        if (self.fills.fetchRemove(s)) |entry| {
            var f = entry.value;
            self.m.joinAhead();
            self.m.fill_seq = &.{};
            f.progress.deinit(self.gpa);
            f.spans.deinit(self.gpa);
            f.rows.deinit(self.gpa);
            if (self.prof) |*p| {
                p.prefill += self.m.now() - f.began;
                p.prefills += 1;
            }
        }
    }

    fn fillSliced(self: *Lanes, s: *lanes.Stream, seq: []const i32, args: model.SliceArgs) !bool {
        const m = self.m;
        var lo: usize = 0;
        while (true) {
            const hi = model.sliceHi(m.eng.c.layers, m.prefill.slice_layers, lo);
            if (try self.begin(.fill_slice)) {
                try putSlice(&self.wire, self.gpa, .{ .lo = @intCast(lo), .hi = @intCast(hi), .args = args, .next_rows = @intCast(m.next_engram_rows) });
                try self.flush();
            }
            const end = try m.fillSlice(seq[0 .. args.start + args.n], args, lo, hi);
            const b = s.between.?;
            // runSlice only enqueues: unsynchronized, the park's round (verifyRun) would wait out this slice's kernels
            // and the host would bill them to decode, cancelling the share's debt. Share off keeps the old overlap.
            if (end != .more) {
                if (b.timed and b.wanted(b.ptr)) try m.stream.synchronize();
                return end == .head;
            }
            lo = hi;
            if (b.wanted(b.ptr)) {
                m.joinAhead(); // shared Engram hashes and reader pool, not just row arrays
                if (b.timed) try m.stream.synchronize();
                try b.run(b.ptr);
            }
            if (s.isCancelled()) {
                if (try self.begin(.fill_slice)) {
                    try putSlice(&self.wire, self.gpa, .{ .abort = true });
                    try self.flush();
                }
                m.dropChunk();
                return error.Cancelled;
            }
        }
    }

    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        const f = self.fills.getPtr(s) orelse return error.PrefillNotStarted;
        if (s.isCancelled()) return error.Cancelled;
        try self.resumeFill(s, f);
        // The whole-pass path keeps Engram's next-chunk read-ahead alive. A yielded pass joins before returning
        // control to the scheduler, where another stream or a spill restore may reuse the model's sequence.
        defer if (self.prefill_yield) { m.joinAhead(); m.fill_seq = &.{}; };
        const seq = &self.seqs[f.slot];
        const len = f.len;
        const slot = f.slot;
        const replay = f.replay;
        var start = f.start;
        if (start < len) {
            const n = model.prefill128.nextRows(len, start, replay, m.compute_rows, s.reuse.marks);
            if (m.prefill.ae) m.next_engram_rows = model.prefill128.nextRows(len, start + n, replay, m.compute_rows, s.reuse.marks);
            const sliced = useLayerSlices(self.prefill_yield, m.prefill.slice_layers, self.kv != null, s.between != null);
            if (!sliced) if (try self.begin(.fill_chunk)) {
                try self.wire.int(gpa, u64, start);
                try self.wire.int(gpa, u32, @intCast(n));
                try self.wire.int(gpa, u64, replay);
                if (m.prefill.ae) try self.wire.int(gpa, u32, @intCast(m.next_engram_rows));
                try self.flush();
            };
            if (self.spill) |t| t.bump(f.base + start, f.base + start + n);
            const l = self.streams.getPtr(s).?;
            if (self.kv) |k| k.planner.setPartial(l.id, true);
            defer if (self.kv) |k| k.planner.setPartial(l.id, false);
            f.head = (if (sliced) self.fillSliced(s, seq.items, .{ .start = start, .n = n, .replay = replay }) else m.fillChunk(seq.items[0 .. start + n], start, n, replay)) catch |err| {
                // the followers may be waiting in this chunk's collectives: a rank-local Engram fault ends the pair
                if (round.endsPair(err) or m.rs.live) m.pairFault(err);
                m.dropChunk();
                return err;
            };
            start += n;
            f.start = start;
            f.progress.record(self, s, start);
            if (s.reuse.hook) |k| if (start < len and start <= replay and (start == f.last_end or markEnds(s.reuse.marks, start))) k.at(k.ptr, s, @intCast(start));
            if (start < len) {
                f.needs_resume = self.prefill_yield;
                return false;
            }
        }
        if (!f.head) return error.NoPromptLogits;
        // Full identical prompts reuse the complete state and logits. A later longer
        // prompt can use its encoder state only when bounded replay starts after it.
        if (s.images.len == 0) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(len));
        // a state read back from disk: the drafter's rings were rebuilt by the replay before the first draft
        try f.source_facts.finish(m, self.spill != null and m.drafting(), slot, len);
        var cand_count: usize = 0;
        if (m.cand != 0 and s.grammar == null and !s.prefill_only and !logits_sha and self.logit_dump_dir == null) {
            if (s.sampling) |sm| {
                if (sm.temperature > 0 and sm.top_k > 0 and sm.top_k <= sampling.max_candidates - sampling.margin)
                    cand_count = @min(m.vocab, sm.top_k + sampling.margin);
            }
        }
        const candidates = if (cand_count > 0) try m.promptCandidates(cand_count) else null;
        const use_candidates = if (candidates) |c| c.flags[0] == 0 else false;
        const logits: []const f32 = if (use_candidates) &.{} else try m.promptLogitsHost();
        f.progress.poll(self, s);
        self.streams.getPtr(s).?.filled = len;
        if (logits_sha) {
            var dg: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(logits), &dg, .{});
            std.debug.print("{{\"prompt_logits\": \"{s}\", \"sha256\": \"{s}\"}}\n", .{ s.id, std.fmt.bytesToHex(dg, .lower) });
        }
        if (self.logit_dump_dir) |dir| {
            self.logit_dump_serial += 1;
            var name: [4096]u8 = undefined;
            const path = try std.fmt.bufPrintSentinel(&name, "{s}/prompt-{d}.f32", .{ dir, self.logit_dump_serial }, 0);
            const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
            if (fd < 0) return error.LogitDiagnosticOpen;
            defer _ = std.c.close(fd);
            const bytes = std.mem.sliceAsBytes(logits);
            var written: usize = 0;
            while (written < bytes.len) {
                const n = std.c.write(fd, bytes.ptr + written, bytes.len - written);
                if (n <= 0) return error.LogitDiagnosticWrite;
                written += @intCast(n);
            }
            std.debug.print("{{\"cache_logit_dump\":{d},\"prompt_tokens\":{d},\"cached_tokens\":{d}}}\n", .{ self.logit_dump_serial, len, s.cached });
        }
        if (s.prefill_only) return true;
        const token: u32 = if (use_candidates) blk: {
            const c = candidates.?;
            const sm = s.sampling.?;
            break :blk @intCast(try sampling.sampleCandidates(c.vals, c.ids, len, .{ .seed = sm.seed, .temperature = sm.temperature, .top_k = sm.top_k, .top_p = sm.top_p, .min_p = sm.min_p }));
        } else try self.drawToken(logits, len, s.sampling);
        const lane = self.streams.getPtr(s).?;
        lane.first = self.take(token);
        if (s.grammar != null) {
            // TP3 can interleave 16 prefills; a single shared "last prompt row" loses another stream's draw.
            lane.first_row = try gpa.realloc(lane.first_row, logits.len);
            @memcpy(lane.first_row, logits);
        }
        return true;
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        return l.first;
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const self = of(ptr);
        const lane = self.streams.getPtr(s) orelse return error.UnknownStream;
        if (position != s.prompt_len) return error.PositionMismatch;
        if (lane.first_row.len == 0 or mask.len != maskWords(self.m.cfg.tokenVocab())) return error.NoPromptRow;
        self.masked = try self.gpa.realloc(self.masked, lane.first_row.len);
        try maskRow(self.masked, lane.first_row, mask);
        return self.take(try self.drawToken(self.masked, position, s.sampling));
    }

    fn heldFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
        const lane = of(ptr).streams.getPtr(s) orelse return error.UnknownStream;
        if (out.len > lane.nheld) return error.HeldMismatch;
        @memcpy(out, lane.held[0..out.len]);
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = .{ ptr, s, feed, position };
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Every window as one round at its stream's cache length: the pending token, the held drafts (or the host's), each
    /// row drawn at its keyed position.
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (windows.len > max_streams) return error.WindowTooWide;
        var ids: [max_streams][max_rows]i32 = undefined;
        var wins: [max_streams]Win = undefined;
        try self.roundWins(windows, &ids, &wins);
        const words = maskWords(m.cfg.tokenVocab()); // TP3 pads the physical head to 129408, grammar sees 129280
        const masked = for (windows) |w| { if (w.masks.len > 0) break true; } else false;
        const use_head = greedy_head.eligible(m.decode_greedy_head, m.two != null, windows);
        const sample_head_count = greedy_head.sampleCount(m.decode_sample_head and m.cand != 0, m.two != null, m.vocab, windows);
        var head_custody: greedy_head.Custody = .{};
        errdefer head_custody.failed(m) catch unreachable;
        const t0 = m.now();
        if (try self.begin(.verify)) {
            try self.wire.int(gpa, u32, @intCast(windows.len));
            try writeGreedyHeadMode(&self.wire, gpa, m.decode_greedy_head, use_head);
            try writeSampleHeadCount(&self.wire, gpa, m.decode_sample_head, sample_head_count);
            for (wins[0..windows.len]) |w| try writeWin(&self.wire, gpa, w, self.kv != null);
            head_custody.active = use_head or sample_head_count != 0; // BEFORE even a partial peer send
            try self.flush();
        }
        const t1 = m.now();
        if (self.spill) |t| for (wins[0..windows.len]) |w| t.bump(w.base + w.start, w.base + w.start + w.ids.len);
        const rows = try self.built.build(gpa, &self.seqs, wins[0..windows.len]);
        // Bounded two-row lookahead: all stream anchors before second rows.
        // Full-vocabulary unconstrained draws only; other modes use baseline.
        var requests: [penalty_pool.max_jobs]penalty_pool.Request = undefined;
        var count: usize = 0;
        for (0..2) |depth| for (windows) |w| {
            if (depth >= w.rows()) continue;
            if (w.stream.sampling) |sm| {
                if (sm.temperature <= 0 or !std.math.isFinite(sm.temperature) or
                    (sm.top_k != 0 and sm.top_k < m.vocab) or (sm.top_p > 0 and sm.top_p < 1)) continue;
                if (count == requests.len) break;
                requests[count] = .{ .seed = sm.seed, .position = w.positions[depth] };
                count += 1;
            }
        };
        if (count > 0 and self.penalty_workers == null and !self.penalty_disabled) {
            self.penalty_workers = penalty_pool.Pool.create(gpa, m.io, m.vocab) catch null;
            if (self.penalty_workers == null) self.penalty_disabled = true;
        }
        var active_pool: ?*penalty_pool.Pool = null;
        if (count > 0) if (self.penalty_workers) |p| {
            if (p.start(requests[0..count])) |_| {
                active_pool = p;
                self.sample_workspace.penalty_cache = p.cache();
            } else |_| {
                self.penalty_disabled = true; // preserve ordinary sampling if cache cannot start
            }
        };
        defer {
            self.sample_workspace.penalty_cache = null;
            if (active_pool) |p| p.finish(); // also on GPU/transport/sampling failure
        }
        // Registered after worker cleanup: pair-end before any unwind/drain.
        head_custody.active = use_head or sample_head_count != 0;
        errdefer head_custody.failed(m) catch unreachable;
        if (sample_head_count != 0)
            try greedy_head.verifySample(m, rows, self.built.absorb[0..windows.len], sample_head_count)
        else
            try greedy_head.verify(m, rows, self.built.absorb[0..windows.len], use_head);
        head_custody.active = false; // output and absorb are stream-fenced on every rank
        const t2 = m.now();
        // The windows without a grammar take the path they would take alone: every one greedy, the device's argmax (R
        // tokens back); else bounded candidates, or the logits for the host's draws. A grammar window reads only its own
        // rows' logits, so one structured reply does not move the round's other streams off the device paths.
        var greedy = true;
        var plain = false;
        for (windows) |w| {
            if (w.masks.len > 0) continue;
            plain = true;
            if (w.stream.sampling) |sm| if (sm.temperature > 0) {
                greedy = false;
            };
        }
        const tokens: []const u32 = if (greedy and plain) try m.roundArgmax(rows.ids.len) else &.{};
        var cand_count: usize = 0;
        if (!greedy and m.cand != 0) {
            cand_count = 1;
            for (windows) |w| {
                if (w.masks.len > 0) continue;
                const sm = w.stream.sampling orelse continue;
                if (std.math.isNan(sm.temperature)) {
                    cand_count = 0;
                    break;
                }
                if (sm.temperature > 0) {
                    if (sm.top_k == 0 or sm.top_k > sampling.max_candidates - sampling.margin) {
                        cand_count = 0;
                        break;
                    }
                    cand_count = @max(cand_count, @min(m.vocab, sm.top_k + sampling.margin));
                }
            }
        }
        const candidates = if (cand_count > 0) try m.roundCandidates(rows.ids.len, cand_count) else null;
        var whole = candidates == null;
        // any row's fallback, a grammar row's too: copyCandidates then copies no row's candidates (model.zig)
        if (candidates) |c| for (c.flags) |flag| {
            if (flag != 0) whole = true;
        };
        const logits: []const f32 = if (!greedy and whole) try m.roundLogits(rows.ids.len) else if (masked) blk: {
            for (windows, 0..) |w, k| if (w.masks.len > 0) try m.roundLogitRows(self.built.wins[k].row, w.rows());
            break :blk m.logits[0 .. rows.ids.len * m.vocab];
        } else &.{};
        const t3 = m.now();
        defer if (self.prof) |*p| {
            p.send += t1 - t0;
            p.enqueue += m.t_enqueue;
            p.forward += m.t_forward;
            p.absorb += m.t_absorb;
            p.logits += t3 - t2;
            p.sample += m.now() - t3;
            p.rounds += 1;
            p.rows += rows.ids.len;
        };
        const V = m.vocab;
        for (windows, out, 0..) |w, *o, k| {
            const row0 = self.built.wins[k].row;
            const l = self.streams.getPtr(w.stream).?;
            @memcpy(o.drafts[0..w.held], l.held[0..w.held]);
            @memcpy(o.drafts[w.held..][0..w.tokens.len], w.tokens);
            if (greedy and w.masks.len == 0) {
                @memcpy(o.sampled[0..w.rows()], tokens[row0..][0..w.rows()]);
            } else {
                const sm: ?sampling.Sampling = if (w.stream.sampling) |s| .{ .seed = s.seed, .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p } else null;
                if (w.masks.len > 0) {
                    self.masked = try gpa.realloc(self.masked, w.rows() * V);
                    for (0..w.rows()) |r| {
                        try maskRow(self.masked[r * V ..][0..V], logits[(row0 + r) * V ..][0..V], w.masks[r * words ..][0..words]);
                    }
                    _ = try sampling.sampleWindow(&self.sample_workspace, gpa, self.masked[0 .. w.rows() * V], V, w.positions, o.drafts[0 .. w.rows() - 1], sm, w.prefix_only, o.sampled[0..w.rows()]);
                } else if (whole) {
                    _ = try sampling.sampleWindow(&self.sample_workspace, gpa, logits[row0 * V ..][0 .. w.rows() * V], V, w.positions, o.drafts[0 .. w.rows() - 1], sm, w.prefix_only, o.sampled[0..w.rows()]);
                } else {
                    const c = candidates.?;
                    _ = try sampling.sampleCandidateWindow(c.vals[row0 * cand_count ..][0 .. w.rows() * cand_count], c.ids[row0 * cand_count ..][0 .. w.rows() * cand_count], cand_count, V, w.positions, o.drafts[0 .. w.rows() - 1], sm, w.prefix_only, o.sampled[0..w.rows()]);
                }
            }
            l.nheld = 0;
        }
    }

    /// verifyFn's windows as both ranks build them (each stream's Lane: slot, extent, part, generation; its pending
    /// row, held drafts and host drafts), then the planner's round (kvRound: its frames go out before the verify's).
    fn roundWins(self: *Lanes, windows: []const be.Window, ids: *[max_streams][max_rows]i32, wins: *[max_streams]Win) !void {
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = self.streams.getPtr(w.stream) orelse return error.UnknownStream;
            const rows = w.rows();
            if (rows > window_rows) return error.WindowTooWide;
            const start: usize = @intCast(w.stream.cache_len);
            for (w.positions, 0..) |p, r| if (p != start + 1 + r) return error.PositionMismatch;
            if (w.held > l.nheld) return error.HeldMismatch;
            if (!masksFit(w, maskWords(self.m.cfg.tokenVocab()))) return error.BadMasks;
            ids[k][0] = @intCast(w.pending);
            for (l.held[0..w.held], 0..) |t, j| ids[k][1 + j] = @intCast(t);
            for (w.tokens, 0..) |t, j| ids[k][1 + w.held + j] = @intCast(t);
            wins[k] = .{ .slot = l.slot, .base = l.base, .end = l.end, .start = start, .absorb = self.m.drafting() and w.stream.drafts, .ids = ids[k][0..rows], .part = if (self.kv != null) l.part else -1, .gen = l.gen };
        }
        if (self.kv) |kvp| try self.kvRound(kvp, windows);
    }

    /// The caches are positional: the next window starts at the stream's new length and writes over the rows past it.
    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        _ = ptr;
        for (windows, paths) |_, path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        }
    }

    /// One drafter pass for every stream asked: its pending token at its new length, `depth` drafts held.
    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        const m = self.m;
        const gpa = self.gpa;
        if (m.dr == null) return error.NoDraftHead;
        const block = m.dr.?.n;
        var tokens: [max_streams]i64 = undefined;
        var q0: [max_streams]i64 = undefined;
        var slots: [max_streams]i64 = undefined;
        var depths: [max_streams]usize = undefined;
        var seeds: [max_streams]u64 = @splat(0);
        var temps: [max_streams]f64 = @splat(0);
        var top_ks: [max_streams]u32 = @splat(0);
        var sampled = false;
        var asked: [max_streams]*Lane = undefined;
        var n: usize = 0;
        var steps: usize = 0;
        const mode = keyed_draft.filterMode();
        for (requests) |r| {
            if (r.lanes != null or r.early) return error.TreesNotBuilt;
            const l = self.streams.getPtr(r.stream) orelse return error.UnknownStream;
            l.nheld = 0;
            if (r.depth == 0) continue;
            if (n == max_streams) return error.TooManyStreams;
            if (r.position != r.stream.cache_len + 1) return error.PositionMismatch;
            const token: u32 = if (r.rows == null) self.value(r.first orelse return error.NoFirstToken) else (if (r.follow.len > 0) r.follow[r.follow.len - 1] else return error.NoFollowToken);
            tokens[n] = token;
            q0[n] = @intCast(r.position - 1);
            slots[n] = @intCast(l.slot);
            if (r.stream.sampling) |sm| if (r.stream.grammar == null and keyed_draft.eligible(sm, m.vocab, mode)) {
                seeds[n] = sm.seed;
                temps[n] = sm.temperature;
                top_ks[n] = keyed_draft.draftTopK(sm, m.vocab, mode);
                sampled = true;
            };
            depths[n] = @min(r.depth, block);
            asked[n] = l;
            steps = @max(steps, depths[n]);
            n += 1;
        }
        if (n == 0) return;
        const keyed = sampled and m.dr.?.nz != 0;
        const keys: ?@import("draft.zig").Keyed = if (keyed) .{ .seeds = seeds[0..n], .temps = temps[0..n], .top_ks = top_ks[0..n] } else null;
        if (try self.begin(if (keyed) .pass_keyed else .pass)) {
            try writePass(&self.wire, gpa, tokens[0..n], q0[0..n], slots[0..n], steps, keys);
            try self.flush(); // the same frame goes to BOTH TP3 followers
        }
        const t0 = m.now();
        try draftPass(m, tokens[0..n], q0[0..n], slots[0..n], steps, keys);
        if (self.prof) |*p| {
            p.pass += m.now() - t0;
            p.passes += 1;
            p.pass_streams += n;
        }
        const d = &m.dr.?;
        for (asked[0..n], depths[0..n], 0..) |l, k, i| {
            for (0..k) |j| {
                l.held[j] = @intCast(d.drafts[i][j]);
                l.confs[j] = d.confs[i][j];
            }
            l.nheld = k;
        }
    }

    /// The chance that each held draft lands with every earlier one, as the round loop's allocation reads it: the
    /// running product of the confidences' sigmoids (multi.py's conf rule: a sigmoid is the draft's survival once
    /// the earlier drafts survived, and the prefix survivals' sum is the expected kept drafts; draft.chooseK).
    fn probabilitiesFn(ptr: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
        const self = of(ptr);
        const l = self.streams.getPtr(s) orelse return false;
        if (out.len > l.nheld) return false;
        if (self.fix_k) |k| {
            for (out, 0..) |*o, j| o.* = if (j < k) 1.0 else 0.0;
            return true;
        }
        survivals(l.confs[0..out.len], out);
        return true;
    }

    /// Whether a kept state still restores for a prompt of `len` tokens: live, not stale, a chunk end before the
    /// prompt's replay (the rows before it ran the encoder layers alone, as a fresh pass of this prompt runs them).
    fn usable(self: *const Lanes, sn: *const Snap, len: usize) bool {
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, @constCast(sn)) == null) return false;
        if (self.kv) |k| if (k.planner.keptIndex(sn.id) == null) return false;
        if (sn.stale or sn.loading or sn.at == 0) return false;
        if (sn.complete and sn.at == len) return true;
        if (self.shortResume(sn, len)) return true;
        return sn.at < len and sn.at % 2 == 0 and sn.at <= len -| self.m.cfg.window;
    }

    fn shortResume(self: *const Lanes, sn: *const Snap, len: usize) bool {
        return self.short_continuation and (sn.complete or sn.decoded) and
            sn.at > 0 and sn.at % 2 == 0 and sn.at < len and len - sn.at <= self.m.cfg.window;
    }

    pub fn snapLiveFn(ptr: *anyopaque, saved: *anyopaque) bool {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return std.mem.indexOfScalar(*Snap, self.snaps.items, sn) != null and !sn.stale;
    }

    /// Core cache ABI: 0 unknown, 1 physical overlap, 2 placement pressure.
    pub fn snapDeadReasonFn(_: *anyopaque, saved: *anyopaque) u8 {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return sn.loss_reason;
    }

    pub fn snapContinuationHintFn(ptr: *anyopaque, saved: *anyopaque, core_state: u64) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (!self.continuation_donors or sn.stale or core_state == 0) return;
        // Existing save/restore event clock: no timer, fake hit or GPU operation.
        sn.continuation_at = self.kv_clock;
        sn.continuation_state = core_state;
    }

    pub fn snapFirstHintFn(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (self.first_continuation and !sn.stale and !sn.loading and sn.first_completed_at == 0)
            sn.first_completed_at = self.kv_clock;
    }

    pub fn snapAdaptiveHintFn(ptr: *anyopaque, saved: *anyopaque, density: u32) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (!self.adaptive_prefix or sn.stale) return;
        sn.adaptive_density = density;
        sn.adaptive_at = self.kv_clock;
    }

    /// Pure numeric diagnostic contract, captured before the snapshot is freed.
    /// [exact replay length or 0, minimum longer compatible length or maxInt].
    pub fn ghostRange(at: usize, complete: bool, decoded: bool, window: usize, short: bool) [2]u32 {
        return .{ if (complete) @intCast(at) else 0,
            if (at > 0 and at % 2 == 0) @intCast(at + (if (short and (complete or decoded)) @as(usize,1) else window)) else std.math.maxInt(u32) };
    }
    pub fn snapGhostRangeFn(ptr: *anyopaque, saved: *anyopaque) [2]u32 {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return ghostRange(sn.at, sn.complete, sn.decoded, self.m.cfg.window, self.short_continuation);
    }

    pub fn snapCompatibleFn(ptr: *anyopaque, saved: *anyopaque, len: usize) bool {
        return of(ptr).usable(@ptrCast(@alignCast(saved)), len);
    }

    /// Pressure-only X1-X4 placement. Non-copy candidates use the same complete
    /// victim validation and spill guards as legacy placement. Relocation uses
    /// the existing two-phase all-rank idle transaction, never a local-only move.
    fn planPressure(self: *Lanes, live: []const [2]usize, size: usize, protected: ?*Snap) !?usize {
        if (self.kv != null) return null; // sharded parts have their own planner and ownership
        const t0 = self.m.now();
        var states: std.ArrayList(pressure.State) = .empty;
        defer states.deinit(self.gpa);
        var loading = false;
        for (self.snaps.items) |sn| if (!sn.stale) {
            loading = loading or sn.loading;
            try states.append(self.gpa, .{
                .key = sn.id, .base = sn.base,
                .len = if (sn.loading) @max(sn.at, sn.end -| sn.base) else sn.at,
                .restores = sn.restores, .used = sn.last_use, .protected = sn == protected or sn.held > 0,
                .loading = sn.loading, .continuation_at = sn.continuation_at,
                .adaptive_density = sn.adaptive_density, .adaptive_at = sn.adaptive_at,
                .first_completed_at = sn.first_completed_at,
            });
        };
        const idle = !loading and self.relocation_enabled and self.streams.count() == 0 and live.len == 0;
        var options: pressure.Options = .{
            .short_reuse = self.short_reuse, .short_clock = self.kv_clock,
            .continuation_donors = self.continuation_donors, .continuation_clock = self.kv_clock,
            .adaptive_prefix = self.adaptive_prefix, .short_return_rescue = self.adaptive_prefix,
            .first_continuation = self.first_continuation,
            .deadline_ns = if (self.first_continuation) t0 +| 2 * std.time.ns_per_ms else std.math.maxInt(u64),
        };
        if (idle and self.relocate_max_bytes > 0 and self.m.pool_cap >= 2 * model.chunk_rows) {
            const unit = try self.m.extentBytes(0, model.chunk_rows, model.chunk_rows);
            const credit = @min(self.relocate_max_bytes, self.relocation_budget.available(t0));
            options.max_move_rows = @min(
                move_budget.maxMoveRows(&self.relocation_budget, t0, unit, model.chunk_rows),
                @as(usize, @intCast(credit / unit)) * model.chunk_rows);
        }
        var p = (try pressure.plan(self.gpa, states.items, live, size, self.m.pool_cap, options)) orelse return null;
        if (p.move) |mv| {
            const bytes = self.m.extentBytes(mv.from, mv.to, mv.len) catch 0;
            if (bytes == 0 or bytes > self.relocate_max_bytes or bytes > self.relocation_budget.available(t0)) {
                options.max_move_rows = 0;
                p = (try pressure.plan(self.gpa, states.items, live, size, self.m.pool_cap, options)) orelse return null;
            }
        }
        if (self.first_continuation) std.log.info("kv_first_continuation: event_ns={d} base={d} gate={s} evals={d} reject_proven={d} saved_rows={d} before_us={d} after_us={d}", .{
            link.Link.monotonicNs(), p.base, @tagName(p.first.gate), p.first.evaluated, p.first.proven, p.first.saved, p.first.before_us, p.first.after_us,
        });
        var copied: u64 = 0;
        if (p.move) |mv| {
            copied = try self.m.extentBytes(mv.from, mv.to, mv.len);
            self.idleMove(mv) catch |err| (Model.SpillDev{ .m = self.m }).failed(err);
        } else {
            var keys: [pressure.max_states]u64 = undefined;
            const victims = try pressure.collectVictims(states.items, size, p, &keys);
            for (victims) |key| {
                const victim = for (self.snaps.items) |sn| {
                    if (sn.id == key and !sn.stale and !sn.loading) break sn;
                } else return error.StaleExtent;
                if (self.spill) |t| {
                    t.guardSnap(victim.id, 0);
                    if (t.broken) |err| return err;
                }
                victim.loss_reason = 2;
                victim.stale = true;
            }
        }
        std.log.info("kv_pressure: base={d} size={d} lost={d} copied_bytes_per_rank={d} evals={d} truncated={} baseline_base={d} baseline_lost={d} donor_groups={d} donor_direct={d} donor_hints={d} reject_unproven={d} reject_protected={d} reject_credit={d} reject_overlap={d} reject_opening={d} reject_target={d} reject_fit={d} continuation_move={} adaptive_applied={} adaptive_before_us={d} adaptive_after_us={d} adaptive_extra_rows={d} adaptive_evals={d} adaptive_geometry_skip={} credit_bytes={d} charged_bytes={d} moves={d}",
            .{ p.base, size, p.lost, copied, p.evaluations, p.truncated, p.baseline_base,
                p.baseline_lost, p.donors.groups, p.donors.direct, p.donors.hinted,
                p.donors.unproven, p.donors.protected, p.donors.credit, p.donors.overlap,
                p.donors.opening, p.donors.target, p.donors.fit, p.continuation_move,
                p.adaptive_applied, p.adaptive_before_us, p.adaptive_after_us,
                p.adaptive_extra_rows, p.adaptive_evaluations, p.adaptive_geometry_skip,
                self.relocation_budget.credit, self.relocation_budget.charged, self.relocation_budget.moves });
        std.log.info("kv_refine: base={d} gate={s} evaluated={d} invalid={d} extra={d} long_guard={d} no_short_gain={d} score={d} winners={d} row_limit={d} rescue_relaxed={} rescue_saved_rows={d} final_guard={}",
            .{ p.base, @tagName(p.refine.gate), p.refine.evaluated, p.refine.invalid,
                p.refine.extra, p.refine.long_guard, p.refine.no_short_gain, p.refine.score,
                p.refine.winners, p.refine.row_limit, p.refine.rescue_relaxed,
                p.refine.rescue_saved, p.refine.final_guard });
        return p.base;
    }

    /// The first extent that fits around the live streams and the kept states; the oldest kept states go stale
    /// until it does.
    fn placeKept(self: *Lanes, live: []const [2]usize, size: usize, protected: ?*Snap) !usize {
        const gpa = self.gpa;
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(gpa);
        while (true) {
            kept.clearRetainingCapacity();
            for (self.snaps.items) |sn| if (!sn.stale) try kept.append(gpa, .{ sn.base, if (sn.loading) @max(sn.at, sn.end -| sn.base) else sn.at, sn.end });
            if (try placeSoft(gpa, live, kept.items, size, self.m.pool_cap)) |b| return b;
            // Plan the complete victim set before invalidating any snapshot. Live,
            // borrowed and loading reservations remain hard exclusions.
            var states: std.ArrayList(pressure.State) = .empty;
            defer states.deinit(gpa);
            for (self.snaps.items) |sn| if (!sn.stale) {
                try states.append(gpa, .{
                    .key = sn.id, .base = sn.base,
                    .len = if (sn.loading) @max(sn.at, sn.end -| sn.base) else sn.at,
                    .restores = sn.restores, .used = sn.used,
                    .protected = sn == protected or sn.held > 0,
                    .loading = sn.loading,
                });
            };
            var options: pressure.Options = .{};
            // Idle relocations run only with no live stream, no loading snapshot,
            // and no sharded planner. Existing sharded ownership remains untouched.
            const loading = for (self.snaps.items) |sn| { if (sn.loading) break true; } else false;
            const idle = !loading and self.relocation_enabled and self.streams.count() == 0 and self.kv == null and live.len == 0 and self.m.pool_cap >= 2 * model.chunk_rows;
            if (idle) {
                for (states.items) |*state| {
                    _ = self.m.extentBytes(state.base, state.base, state.len) catch {
                        state.movable = false;
                        continue;
                    };
                }
                const unit = try self.m.extentBytes(0, model.chunk_rows, model.chunk_rows);
                options.max_move_rows = move_budget.maxMoveRows(&self.relocation_budget, self.m.now(), unit, model.chunk_rows);
            }
            if (self.pressure_planner) if (try self.planPressure(live, size, protected)) |base| return base;
            var choice = try pressure.plan(gpa, states.items, live, size, self.m.pool_cap, options);
            if (choice == null) {
                // Only true pressure may sacrifice the requested source. The
                // caller already handles its invalidation by restarting cold.
                for (states.items) |*state| state.protected = false;
                choice = try pressure.plan(gpa, states.items, live, size, self.m.pool_cap, .{});
            }
            const plan = choice orelse return error.ContextFull;
            if (plan.move) |mv| {
                self.idleMove(mv) catch |err| (Model.SpillDev{ .m = self.m }).failed(err);
                return plan.base;
            }
            var keys: [pressure.max_states]u64 = undefined;
            const victims = try pressure.collectVictims(states.items, size, plan, &keys);
            // All metadata is owned by this lane thread; no CUDA or callback
            // occurs between projection and complete-set validation.
            for (victims) |key| {
                const victim = for (self.snaps.items) |sn| {
                    if (sn.id == key and !sn.stale and !sn.loading) break sn;
                } else return error.StaleExtent;
                if (self.spill) |t| t.guardSnap(victim.id, 0);
                victim.loss_reason = 2;
                victim.stale = true;
            }
            return plan.base;
        }
    }

    fn waitIdleCopy(m: *Model, deadline: u64) !void {
        try idle_relocation.waitCopy(&m.stream, deadline);
    }

    /// Two-phase, all-rank validated idle copy. Uncertain transfers fence the
    /// world; metadata never advances on a missing or malformed receipt.
    fn idleMove(self: *Lanes, mv: pressure.Move) !void {
        self.wakeRdma();
        if (self.kv != null or self.streams.count() != 0) return error.StaleExtent;
        for (self.snaps.items) |sn| if (sn.loading) return error.StaleExtent;
        const bytes = try self.m.extentBytes(mv.from, mv.to, mv.len);
        self.relocation_generation += 1;
        const Host = struct {
            lanes: *Lanes,
            pub fn enter(h: *@This(), deadline: u64) !void {
                if (h.lanes.m.ahead != null) return error.TransferFailed;
                h.lanes.move_deadline = deadline;
                if (h.lanes.spill) |t| t.side.operation_deadline = deadline;
            }
            pub fn leave(h: *@This()) void {
                h.lanes.move_deadline = null;
                if (h.lanes.spill) |t| t.side.operation_deadline = null;
            }
            pub fn guard(h: *@This(), move: idle_relocation.Move) !void {
                const l = h.lanes;
                for (l.snaps.items) |sn| if (!sn.stale and sn.base == move.from) {
                    if (l.spill) |t| {
                        t.guardSnap(sn.id, 0);
                        if (t.broken) |err| return err;
                    }
                };
                if (l.spill) |t| t.bump(@intCast(move.to), @intCast(move.to + move.len));
                if (!l.relocation_budget.consume(l.m.now(), move.bytes)) return error.TransferFailed;
            }
            pub fn copy(h: *@This(), move: idle_relocation.Move, deadline: u64) !void {
                try h.lanes.m.copyExtent(@intCast(move.from), @intCast(move.to), @intCast(move.len));
                try waitIdleCopy(h.lanes.m, deadline);
            }
            pub fn publishMove(h: *@This(), move: idle_relocation.Move) void {
                idle_relocation.publish(h.lanes.snaps.items, move);
            }
        };
        var host: Host = .{ .lanes = self };
        try idle_relocation.leader(&host, .{ .generation = self.relocation_generation, .from = mv.from, .to = mv.to, .len = mv.len, .bytes = bytes }, &self.m.fan.links, self.peer, self.m.world, self.m.rank, &self.step, idle_relocation.budget_ns);
    }

    /// Would the writable tail destroy another retained prefix? In-place growth
    /// must obey the same reservations as placement of a fresh extent.
    fn overwritesKept(self: *const Lanes, lo: usize, hi: usize, except: ?*Snap) bool {
        for (self.snaps.items) |sn| {
            if (!sn.stale and sn != except and sn.base < hi and lo < sn.base + sn.at) return true;
        }
        return false;
    }

    /// Kept states with rows in [lo, hi) go stale (a stream writes there), `except` aside.
    fn staleOver(self: *Lanes, lo: usize, hi: usize, except: ?*Snap) void {
        for (self.snaps.items) |sn| if (sn != except and sn.base < hi and lo < sn.base + sn.at) {
            if (!sn.stale) {
                noteStale(sn, "a stream writes over its rows");
                if (self.spill) |t| t.guardSnap(sn.id, @intCast(@max(lo, sn.base) - sn.base));
            }
            sn.loss_reason = 1;
            sn.stale = true;
        };
    }

    /// The prompt cache's Snapshots.usable (with the spill tier): a stale, loading or ineligible state is passed over
    /// before lengths are compared, so it never shadows a usable state or a durable point.
    pub fn usableFn(ptr: *anyopaque, saved: *anyopaque, len: u32) bool {
        return of(ptr).usable(@ptrCast(@alignCast(saved)), len);
    }

    pub fn snapShortEligibleFn(ptr: *anyopaque, saved: *anyopaque, len: u32) bool {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        return self.usable(sn, len) and self.shortResume(sn, len);
    }

    /// The prompt cache's Snapshots functions (native.zig hands them to the CUDA host).
    pub fn snapBytesFn(ptr: *anyopaque, at: u32) u64 {
        _ = at;
        return of(ptr).m.snapBytes();
    }

    /// The live state at `at` (the pass stands at that chunk end): the slot's own part copied, the extent's rows kept.
    pub fn snapSaveFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const self = of(ptr);
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const l = self.streams.getPtr(s) orelse return error.UnknownStream;
        const complete = at == s.prompt_len and s.emitted().len == 0;
        const continuation = s.finished and (s.reason == .stop or s.reason == .length) and at <= s.cache_len;
        if (at == 0 or (!complete and (at % 2 != 0 or (!continuation and at > s.prompt_len -| self.m.cfg.window)))) return error.NotAChunkEnd;
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .complete = complete, .decoded = complete or continuation, .id = self.next_snap, .at = at, .base = l.base, .end = l.end, .part = l.part, .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .slot = @intCast(l.slot) };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        self.kv_clock +|= 1;
        sn.last_use = self.kv_clock;
        if (try self.begin(.snap_save)) {
            try self.wire.int(self.gpa, u64, sn.id);
            try self.wire.int(self.gpa, u32, @intCast(l.slot));
            try self.flush();
        }
        try self.m.snapCopy(l.slot, sn.buf.ptr, true);
        // TF_DS_KV_SHARD: the planner keeps it too (its rows held in its part); a refused keep leaves the state unusable
        // (usable asks the planner), never one whose rows a placement may take
        if (self.kv) |k| k.keep(sn.id, l.id, at) catch |err| std.log.err("dsv41: the parts' planner did not keep a prompt state: {s}", .{@errorName(err)});
        // (the spill tier's keep follows: spillInfo gives it the planner's part salt, saltOf, for its block lineage;
        // its write's pending flag reaches the planner before each plan, kvPending)
        return sn;
    }

    fn checkpointPlan(self: *Lanes, saved: ?*anyopaque, len: usize) @import("core").checkpoint_plan.Native {
        const sn: ?*Snap = if (saved) |v| @ptrCast(@alignCast(v)) else null;
        const replay = if (sn) |v| (if (self.shortResume(v, len)) v.at -| self.m.cfg.window else len -| self.m.cfg.window) else len -| self.m.cfg.window;
        return .{ .len = @intCast(len), .from = if (sn) |v| @intCast(v.at) else 0,
            .replay = @intCast(replay), .chunk = model.chunk_rows };
    }

    pub fn snapMapMarkFn(ptr: *anyopaque, saved: ?*anyopaque, len: usize, wanted: u32) u32 {
        return of(ptr).checkpointPlan(saved, len).effective(wanted);
    }

    pub fn snapAutomaticMarksFn(ptr: *anyopaque, saved: ?*anyopaque, len: usize, out: *[2]u32) usize {
        return of(ptr).checkpointPlan(saved, len).automatic(out);
    }

    pub fn snapRestoreFn(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores; // the prompt pass restores, before its first chunk
    }

    /// Fatal on any uncertainty: the core must not uncharge partially freed state.
    pub fn snapDropManyFn(ptr: *anyopaque, saved: []const *anyopaque) void {
        const self = of(ptr);
        self.dropMany(saved) catch |err| (Model.SpillDev{ .m = self.m }).failed(err);
    }

    const DropBatch = struct {
        lanes_: *Lanes,
        saved: []const *anyopaque,
        step: u64,
        deadline: u64,
        sent: u64 = 0,
        freed: u64 = 0,
        confirmed: u64 = 0,
        pub fn followerCount(b: *DropBatch) usize {
            return if (b.lanes_.peer != null) b.lanes_.m.followers().len else 0;
        }
        pub fn send(b: *DropBatch, rank: usize) !void {
            const peer = b.lanes_.m.followers()[rank] orelse return error.TransferFailed;
            try peer.sendWithin(b.lanes_.wire.buf.items, b.deadline);
        }
        pub fn freeLocal(b: *DropBatch) !void {
            b.sent = link.Link.monotonicNs();
            for (b.saved) |p| {
                const sn: *Snap = @ptrCast(@alignCast(p));
                sn.buf.free(); // The existing synchronous cuMemFree custody contract.
            }
            b.freed = link.Link.monotonicNs();
        }
        pub fn receive(b: *DropBatch, rank: usize) !void {
            const peer = b.lanes_.m.followers()[rank] orelse return error.TransferFailed;
            var buf: [32]u8 = undefined;
            try cache_batch.checkReceipt(try peer.recvBy(&buf, b.deadline), b.step, b.saved.len);
        }
        pub fn commit(b: *DropBatch) void {
            b.confirmed = link.Link.monotonicNs();
            for (b.saved) |p| {
                const sn: *Snap = @ptrCast(@alignCast(p));
                _ = b.lanes_.snaps.orderedRemove(std.mem.indexOfScalar(*Snap, b.lanes_.snaps.items, sn).?);
                b.lanes_.gpa.destroy(sn);
            }
        }
    };

    fn dropMany(self: *Lanes, saved: []const *anyopaque) !void {
        if (saved.len < 2 or saved.len > cache_batch.max_count) return error.BadDropBatch;
        var ids: [cache_batch.max_count]u64 = undefined;
        // Validate every pointer and dead state before custody, dispatch or free.
        for (saved, 0..) |p, i| {
            const sn: *Snap = @ptrCast(@alignCast(p));
            if (std.mem.indexOfScalar(*Snap, self.snaps.items, sn) == null) return error.UnknownSnapshot;
            if (!sn.stale or sn.loading) return error.BadDropBatch;
            ids[i] = sn.id;
        }
        try cache_batch.validate(ids[0..saved.len]);
        // These are existing spill/shard custody prerequisites, not asynchronous
        // reclamation. They may exchange their own guarded metadata frames.
        for (ids[0..saved.len]) |id| {
            if (self.spill) |t| {
                t.dropSnap(id);
                if (t.broken) |err| return err;
            }
            if (self.kv) |k| try k.drop(id);
        }
        const start = link.Link.monotonicNs();
        const step = self.step;
        const deadline = start +| 5 * std.time.ns_per_s;
        if (try self.begin(.snap_drop_many)) {
            try self.wire.int(self.gpa, u32, @intCast(saved.len));
            for (ids[0..saved.len]) |id| try self.wire.int(self.gpa, u64, id);
        }
        var transaction: DropBatch = .{ .lanes_ = self, .saved = saved, .step = step, .deadline = deadline };
        try cache_batch.release(&transaction);
        std.log.info("checkpoint_drop_batch: count={d} followers={d} send_ns={d} local_free_ns={d} receipt_wait_ns={d} confirmed=true", .{
            saved.len, transaction.followerCount(), transaction.sent -| start, transaction.freed -| transaction.sent, transaction.confirmed -| transaction.freed,
        });
        self.idleRdma();
    }

    /// A request matched this state and waits for its pass, or stopped waiting.
    pub fn snapHoldFn(_: *anyopaque, saved: *anyopaque, delta: i32) void {
        const sn: *Snap = @ptrCast(@alignCast(saved));
        sn.held = if (delta > 0) sn.held + 1 else sn.held -| 1;
    }

    pub fn snapDropFn(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        const i = std.mem.indexOfScalar(*Snap, self.snaps.items, sn) orelse return;
        _ = self.snaps.orderedRemove(i);
        if (self.spill) |t| t.dropSnap(sn.id); // (rule b: a pending small-state capture of its buffer done first)
        if (self.begin(.snap_drop) catch false) {
            self.wire.int(self.gpa, u64, sn.id) catch {};
            self.flush() catch |err| std.log.err("dsv41: rank 1 kept a dropped prompt state: {s}", .{@errorName(err)});
        }
        // TF_DS_KV_SHARD: the planner's record goes too (deferred while a copy or carry still reads its rows)
        if (self.kv) |k| k.drop(sn.id) catch |err| std.log.err("dsv41: the parts' planner kept a dropped prompt state: {s}", .{@errorName(err)});
        sn.buf.free();
        self.gpa.destroy(sn);
        // A metadata-only drop wakes every proxy through begin(), but without a
        // spill pump or another request there is no later parking opportunity.
        // Retry after both the tier and the planner have released this state.
        self.idleRdma();
    }

    // -- the spill tier's family and wire (spill_tier.Family, spill_tier.Wire) ---------------------------------------

    pub fn spillFamily(self: *Lanes) spill_tier.Family {
        return .{ .ptr = self, .reserve = spillReserve, .install = spillInstall, .discard = spillDiscard, .info = spillInfo };
    }

    pub fn spillWire(self: *Lanes) spill_tier.Wire {
        return .{ .ctx = self, .frame = spillFrame, .links = self.m.followers(), .wake = spillWake, .idle = spillIdle };
    }

    /// A restore's destination: an extent placed like a stream's (kept states give their room oldest first, guarded),
    /// a fresh Snap holding it back (loading: not usable until every rank installed it) and its buffer. `room`: the
    /// waiting prompt's length; when an extent of it and a growth margin fits without taking any kept state's room,
    /// the Snap's extent (`end`) holds that much, so the resume runs in place (no copyExtent of the restored rows).
    fn spillReserve(ptr: *anyopaque, at: usize, room: usize, part: ?u8) anyerror!spill_tier.Dest {
        const self = of(ptr);
        if (self.kv) |k| return self.spillReserveParts(k, at, room, part orelse return error.SpillNoPart);
        var taken: [max_streams][2]usize = undefined;
        var nt: usize = 0;
        var it = self.streams.valueIterator();
        while (it.next()) |l| {
            taken[nt] = .{ l.base, l.end };
            nt += 1;
        }
        const want = @min(std.mem.alignForward(usize, @max(room, at) + growth_rows, model.chunk_rows), self.m.pool_cap);
        const roomy = if (want > at) try self.placeFree(taken[0..nt], want) else null;
        const base = roomy orelse try self.placeKept(taken[0..nt], at, null);
        self.staleOver(base, base + at, null);
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = self.next_snap, .at = at, .base = base, .end = base + (if (roomy != null) want else at), .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .loading = true };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        return .{ .saved = sn, .id = sn.id, .base = base, .buf = sn.buf.ptr };
    }

    /// TF_DS_KV_SHARD: the destination is a planner hold in the point's recorded part (its owner holds the codes
    /// there: Kv.restore over planRestore, on every rank alike), the waiting prompt's room with a growth margin when
    /// that stales nobody, else the point's rows; the hold becomes a kept state at install (restoredOp) or goes at
    /// discard (unholdOp).
    fn spillReserveParts(self: *Lanes, k: *kv_lanes.Kv, at: usize, room: usize, part: u8) !spill_tier.Dest {
        self.kvPending();
        const want = std.mem.alignForward(usize, @max(room, at) + growth_rows, model.chunk_rows);
        const id = self.next_snap;
        const r = try k.restore(id, want, at, part);
        errdefer k.unhold(id) catch |err| std.log.warn("dsv41: a refused restore kept its part's hold: {s}", .{@errorName(err)});
        // (its generation: rank_wire Reserve.generation, each follower checks its own planner's hold against it)
        const gen = for (k.planner.holds.items) |x| {
            if (x.why == .restore and x.id == id) break x.gen;
        } else return error.UnknownHold;
        const sn = try self.gpa.create(Snap);
        errdefer self.gpa.destroy(sn);
        sn.* = .{ .id = id, .at = at, .base = r.base, .end = r.base + r.size, .part = part, .buf = try cuda.DeviceBuffer.alloc(self.m.ctx.d, self.m.snapBytes()), .loading = true };
        errdefer sn.buf.free();
        try self.snaps.append(self.gpa, sn);
        self.next_snap += 1;
        return .{ .saved = sn, .id = sn.id, .base = sn.base, .buf = sn.buf.ptr, .gen = gen, .size = r.size };
    }

    /// An extent of `size` around the live streams and the kept states' rows (placeSoft's rules: their held extents
    /// first), staling nobody: null when none.
    fn placeFree(self: *Lanes, live: []const [2]usize, size: usize) !?usize {
        var kept: std.ArrayList([3]usize) = .empty;
        defer kept.deinit(self.gpa);
        for (self.snaps.items) |sn| if (!sn.stale) try kept.append(self.gpa, .{ sn.base, sn.at, sn.end });
        return placeSoft(self.gpa, live, kept.items, size, self.m.pool_cap);
    }

    fn spillInstall(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        sn.loading = false;
        sn.restored = true;
        // TF_DS_KV_SHARD: the hold a kept state of the planner too (every rank's at this op); refused, its hold goes
        // and the Snap is stale (no room of its own in placement, never usable; the prompt cache drops it later, and
        // Kv.drop ignores an id the planner never kept)
        if (self.kv) |k| k.restored(sn.id, sn.id, sn.at) catch |err| {
            std.log.err("dsv41: the parts' planner did not keep a restored prompt state: {s}", .{@errorName(err)});
            k.unhold(sn.id) catch {};
            sn.loss_reason = 2;
            sn.stale = true;
        };
    }

    /// A restore that did not commit: its Snap goes (the followers drop theirs with the abort, no snap_drop).
    fn spillDiscard(ptr: *anyopaque, saved: *anyopaque) void {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        if (self.kv) |k| k.unhold(sn.id) catch |err| std.log.warn("dsv41: a discarded restore kept its part's hold: {s}", .{@errorName(err)});
        if (std.mem.indexOfScalar(*Snap, self.snaps.items, sn)) |i| _ = self.snaps.orderedRemove(i);
        sn.buf.free();
        self.gpa.destroy(sn);
    }

    fn spillInfo(ptr: *anyopaque, saved: *anyopaque) spill_tier.SnapInfo {
        const self = of(ptr);
        const sn: *Snap = @ptrCast(@alignCast(saved));
        // TF_DS_KV_SHARD: the part its rows lie in now, as the planner holds it (its block lineage's salt)
        const part: ?u8 = if (self.kv) |k| k.planner.saltOf(sn.id) else null;
        return .{ .id = sn.id, .base = sn.base, .at = sn.at, .buf = sn.buf.ptr, .slot = sn.slot, .part = part };
    }

    /// A spill frame (Kind rank_wire) to every follower, in the ordered stream.
    pub fn spillFrame(ctx: *anyopaque, kind: u8, sub: []const u8) anyerror!void {
        const self = of(ctx);
        const k: Kind = switch (kind) {
            10 => .rank_wire,
            else => return error.BadFrame,
        };
        if (try self.begin(k)) {
            try self.wire.buf.appendSlice(self.gpa, sub);
            try self.flush();
        }
    }

    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        prefillEndFn(ptr, s);
        if (!self.streams.contains(s)) {
            if (self.kv) |k| k.forgetKey(@intFromPtr(s));
            self.idleRdma();
            return;
        }
        // Only accepted target rows. Speculative rows beyond cache_len are excluded
        // from the key and compressed extent; the ring's extra rows protect the tail.
        if (s.images.len == 0 and s.finished and (s.reason == .stop or s.reason == .length)) {
            const at = std.mem.alignBackward(usize, @intCast(s.cache_len), 2);
            if (at > s.prompt_len and at <= s.context.items.len) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(at));
        }
        const kv = self.streams.fetchRemove(s).?;
        self.gpa.free(kv.value.first_row);
        self.used[kv.value.slot] = false;
        if (self.kv) |k| k.release(kv.value.id) catch |err| std.log.err("dsv41: the parts' planner kept a released stream: {s}", .{@errorName(err)});
        self.idleRdma();
    }
};

fn hasRdma(m: *const Model) bool { return m.ring != null or m.rdma_rings != null; }

fn quiesceRdma(m: *Model, deadline: u64) !void {
    if (m.ring) |r| try r.quiesce(deadline);
    if (m.rdma_rings) |rs| {
        for ([_]*@import("rdma.zig").Ring{ rs.all4, rs.col, rs.pair }) |r| try r.quiesce(deadline);
    }
}

fn setRdmaIdle(m: *Model, idle: bool) void {
    if (m.ring) |r| { if (idle) r.park() else r.wake(); }
    if (m.rdma_rings) |rs| {
        for ([_]*@import("rdma.zig").Ring{ rs.all4, rs.col, rs.pair }) |r| {
            if (idle) r.park() else r.wake();
        }
    }
}

/// Prefix survivals: out[j] = the product of sigmoid(conf[i]) for i <= j, in float64 (libm's exp, as chooseK).
pub fn survivals(conf: []const f32, out: []f64) void {
    var surv: f64 = 1.0;
    for (out, conf[0..out.len]) |*o, cf| {
        surv *= 1.0 / (1.0 + exp(-@as(f64, cf)));
        o.* = surv;
    }
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The parts' planner for this model: its agreed pool in one part a rank, and the owner-only bytes a carried token
/// costs (carryBytes over the model's kv sources).
pub fn kvConfig(m: *const Model) parts.Config {
    var buf: [16]shard.Source = undefined;
    return .{ .cap = m.pool_cap, .parts = @intCast(m.world), .carry_bytes = carryBytes(m.kvSources(&buf)) };
}

/// The bytes a carried token costs, rounded up: every source's compressed codes and scales at its ratio (the owner's
/// rows, shard.planeBytes over a whole number of every ratio's blocks, so no ratio's share truncates; the index
/// keys copy on every rank and are not sent). (3/2 + 1) x 288 = 720 at this checkpoint.
pub fn carryBytes(sources: []const shard.Source) usize {
    var owner: [16]shard.Source = undefined; // (Model.kvSources' bound)
    std.debug.assert(sources.len <= owner.len);
    var tokens: usize = 1;
    for (sources, owner[0..sources.len]) |s, *o| {
        o.* = .{ .ratio = s.ratio, .comp_row = s.comp_row, .idx_row = 0 };
        tokens = tokens / std.math.gcd(tokens, s.ratio) * s.ratio;
    }
    const bytes = shard.planeBytes(owner[0..sources.len], tokens, 1, false);
    return (bytes + tokens - 1) / tokens;
}

/// The device side of the planner's ops on this rank (Model.copyExtent, Model.bindTwin).
/// A rank's attention-side counters (Model.kvs' stats) as its kv_shard row has them.
fn shardRank(m: *Model, out: *lanes.kv_parts.Rank) void {
    const x = m.kvs orelse return;
    const st = x.stats;
    out.prompt_fetches = st.prompt_fetches;
    out.prompt_rows = st.prompt_rows;
    out.prompt_picks = st.prompt_picks;
    out.prompt_bytes = st.prompt_bytes;
    out.prompt_syncs = st.prompt_syncs;
    out.fetch_rounds = st.fetch_rounds;
    out.owner_rounds = st.owner_rounds;
    out.copies = st.carries;
    out.copy_bytes = st.carry_bytes;
}

fn readShardRank(ctx: *anyopaque, out: *lanes.kv_parts.Rank) void {
    shardRank(@ptrCast(@alignCast(ctx)), out);
}

fn modelDev(m: *Model) kv_lanes.Dev {
    return .{ .ctx = m, .check = devCheck, .copy = devCopy, .twin = devTwin, .wait = devWait };
}

fn devCheck(ctx: *anyopaque, from: usize, to: usize, rows: usize) anyerror!void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    try m.checkExtent(from, to, rows);
}

/// The transfers enqueued so far done within `timeout_ns`: an event after them polled (a stream synchronize would
/// wait forever for an NCCL half whose peer never enqueued its own).
fn devWait(ctx: *anyopaque, timeout_ns: u64) anyerror!void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    var ev = try cuda.Event.init(m.ctx.d, false);
    defer ev.deinit();
    try ev.record(m.stream);
    const deadline = m.now() +| timeout_ns;
    while (!try ev.done()) {
        if (m.now() > deadline) return error.TransferTimeout;
        std.atomic.spinLoopHint();
    }
}

fn devCopy(ctx: *anyopaque, from: usize, to: usize, rows: usize) anyerror!void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    try m.copyExtent(from, to, rows);
}

fn devTwin(ctx: *anyopaque, base: usize, on: bool, local: ?usize) anyerror!void {
    const m: *Model = @ptrCast(@alignCast(ctx));
    try m.bindTwin(base, on, local);
}

/// Rank 0 has already sent .pass: like prompt fill, a rank-local pair-ending
/// fault must terminate before returning to request cleanup. Generic only for CPU tests;
/// Model.pairFault is noreturn in production.
fn draftPass(m: anytype, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, keyed: ?@import("draft.zig").Keyed) !void {
    m.pass(tokens, q0, slots, steps, keyed) catch |err| {
        if (round.endsPair(err)) return m.pairFault(err);
        return err;
    };
}

// Socketpair peers model links, not GPU collectives.
test "rank zero drafter qualification faults take pairFault before request cleanup" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const Fake = struct {
        fan: @import("prompt2d.zig").Fan = .{},
        injected: ?anyerror = null,
        fault: ?anyerror = null,
        passes: usize = 0,
        faults: usize = 0,

        pub fn pass(m: *@This(), tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, keyed: ?@import("draft.zig").Keyed) !void {
            try std.testing.expect(keyed == null);
            m.passes += 1;
            try std.testing.expectEqual(@as(usize, 13), tokens.len);
            try std.testing.expectEqual(tokens.len, q0.len);
            try std.testing.expectEqual(tokens.len, slots.len);
            try std.testing.expectEqual(@as(usize, 5), steps);
            if (m.injected) |err| return err;
        }

        pub fn pairFault(m: *@This(), err: anyerror) !void {
            m.faults += 1;
            m.fault = err;
            m.fan.close();
            // Substitute a distinguished test result for production's noreturn exit 75.
            return error.TestPairEnded;
        }
    };
    const errors = [_]?anyerror{
        error.RaggedMmaNumericalMismatch,
        model.prefill128.qualificationFault(error.OutOfMemory),
        error.OutOfMemory, // only qualification's mapped allocation error is pair-ending
        null,
    };
    for (errors) |injected| {
        var m: Fake = .{ .injected = injected };
        defer m.fan.close();
        var peers: [2]?link.Link = .{ null, null };
        defer for (&peers) |*peer| if (peer.*) |*p| p.close();
        for (&peers, 0..) |*peer, i| {
            var pair: [2]std.c.fd_t = undefined;
            try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &pair) == 0);
            m.fan.links[i] = .{ .fd = pair[0] };
            peer.* = .{ .fd = pair[1] };
            try std.testing.expect(!peer.*.?.peerGone());
        }
        // Thirteen five-row blocks represent the >=64-row eager drafter shape.
        // The failure is injected at the Model.pass boundary, not in a CUDA kernel.
        const tokens: [13]i64 = @splat(0);
        const result = draftPass(&m, &tokens, &tokens, &tokens, 5, null);
        const fatal = if (injected) |err| err == error.RaggedMmaNumericalMismatch or err == error.PrefillQualificationFailed else false;
        if (fatal) {
            try std.testing.expectError(error.TestPairEnded, result);
            try std.testing.expectEqual(injected, m.fault);
            try std.testing.expectEqual(@as(usize, 1), m.faults);
        } else {
            if (injected) |err| try std.testing.expectError(err, result) else try result;
            try std.testing.expectEqual(@as(usize, 0), m.faults);
            try std.testing.expect(m.fault == null);
        }
        try std.testing.expectEqual(@as(usize, 1), m.passes);
        for (peers) |peer| {
            const gone = peer.?.peerGone();
            try std.testing.expectEqual(fatal, gone);
            const grace = std.time.ns_per_s;
            // Existing follower-watch decision: busy follower, grace elapsed, lead link closed.
            try std.testing.expectEqual(fatal, FollowerWatch.shouldExit(1, 1 + grace, gone, grace));
        }
    }
}

/// A primitive this rank refused: rank 0 runs the same one on the same inputs and refuses it at the same step (its
/// request ends with the error), so the follower reports it and takes the next frame instead of ending.
fn refused(err: anyerror) void {
    std.debug.print("{{\"follow_refused\": \"{s}\"}}\n", .{@errorName(err)});
}

/// A round this rank could not finish: refused as above, unless the error means the ranks cannot go on together
/// (round.pair_ending: the ring's order or rows broken, a sized send or the peer's collect pending, the ring or the
/// communicator dead). Then the follower ends with it (rank 0 ends the pair for the same error, native.pair_fatal, or
/// sees this rank's link go down) instead of taking the next frame.
fn roundRefused(err: anyerror) !void {
    refused(err);
    if (round.endsPair(err)) return err;
}

/// A follower's verify times with Model.prof (a profile on ranks 1-3: rounds, issue and wall ns summed).
pub var follow_prof: struct { rounds: u64 = 0, enqueue: u64 = 0, forward: u64 = 0 } = .{};

/// Rank 1 (each of ranks 1-3 on the four-node split): the primitives rank 0 sends, run in order until it says done.
pub fn follow(gpa: std.mem.Allocator, m: *Model) !void {
    return followWith(gpa, m, null);
}

/// A follower's kept states as its spill side sees them (spill_tier.FollowerFamily).
const FollowKept = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    kept: *std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer),
    loading: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty, // restores' destinations, not yet installed
    seqs: *[max_streams]std.ArrayList(i32),

    fn family(f: *FollowKept) spill_tier.FollowerFamily {
        return .{ .ptr = f, .reserve = reserve, .install = install, .discard = discard, .buf = buf, .ids = ids };
    }

    fn of(p: *anyopaque) *FollowKept {
        return @ptrCast(@alignCast(p));
    }

    fn reserve(p: *anyopaque, id: u64) anyerror!u64 {
        const f = of(p);
        if (f.kept.contains(id) or f.loading.contains(id)) return error.SnapshotExists;
        try f.loading.ensureUnusedCapacity(f.gpa, 1);
        const b = try cuda.DeviceBuffer.alloc(f.m.ctx.d, f.m.snapBytes());
        f.loading.putAssumeCapacity(id, b);
        return b.ptr;
    }

    fn install(p: *anyopaque, id: u64) anyerror!void {
        const f = of(p);
        const kv = f.loading.fetchRemove(id) orelse return error.UnknownSnapshot;
        f.kept.put(f.gpa, id, kv.value) catch |e| {
            var b = kv.value;
            b.free();
            return e;
        };
    }

    fn discard(p: *anyopaque, id: u64) void {
        const f = of(p);
        var kv = f.loading.fetchRemove(id) orelse return;
        kv.value.free();
    }

    fn buf(p: *anyopaque, id: u64) ?u64 {
        return if (of(p).kept.get(id)) |b| b.ptr else null;
    }

    fn ids(p: *anyopaque, slot: u32) []const i32 {
        const f = of(p);
        return if (slot < max_streams) f.seqs[slot].items else &.{};
    }

    fn deinit(f: *FollowKept) void {
        var it = f.loading.valueIterator();
        while (it.next()) |b| b.free();
        f.loading.deinit(f.gpa);
    }
};

/// A follower's one rank_wire Follower (spill-shard round 3): rank 0's Leader (Lanes.rw) sends the spill tier's and
/// the KV parts' commands in one ordered stream, so one Tracker admits every frame here too; the spill methods go to
/// the spill side, shard* to the planner's Follower, each failing closed (error.Unsupported) when that side is off on
/// this rank (the fingerprint's features should have refused the start already). Heap only (rank_wire.Follower is
/// about 40 KiB).
pub const PairFollower = PairFollowerOf(SpillFollower);

/// PairFollower over a spill follower type `SF` (spill_tier.Follower of the Model's device; the tests' byte array).
pub fn PairFollowerOf(comptime SF: type) type {
    return struct {
        const Self = @This();
        wf: rank_wire.Follower,
        spill: ?*SF,
        kv: ?*kv_lanes.Follower,

        /// `fx`: this rank's full fingerprint (native.rankExchange; the spill side's own when it has one).
        pub fn create(gpa: std.mem.Allocator, rank: u16, fx: fingerprint.Exchange, spill: ?*SF, kv: ?*kv_lanes.Follower) !*Self {
            const p = try gpa.create(Self);
            p.* = .{ .wf = rank_wire.Follower.init(rank, fx), .spill = spill, .kv = kv };
            return p;
        }

        /// A lanes Kind.rank_wire frame: decoded, admitted in order, dispatched; an ask answered on `l`.
        pub fn onFrame(p: *Self, l: link.Link, payload: []const u8) !void {
            try rank_wire.serve(&p.wf, l, payload, p);
            if (p.kv) |k| if (k.diverged) return error.ShardDiverged;
        }

        fn sp(p: *Self) !*SF {
            return p.spill orelse error.Unsupported;
        }

        fn kf(p: *Self) !*kv_lanes.Follower {
            return p.kv orelse error.Unsupported;
        }

        const H = rank_wire.Head;

        pub fn agreed(p: *Self, a: fingerprint.Agreement) !void {
            // (its store opens under its own compat on its own thread: every block file scanned, every manifest read;
            // rank 0's awaitFollowers waits for it)
            if (p.spill) |s| s.agreed(a) catch |err| {
                std.log.err("dsv41 spill: rank {d} could not open its store at the ranks' agreement: {s}", .{ p.wf.rank, @errorName(err) });
                return err;
            };
        }
        pub fn has(p: *Self, h: H, m: rank_wire.Has) !rank_wire.HasReply {
            return (try p.sp()).has(h, m);
        }
        /// owner_sharded (a nonzero generation): the destination must be this rank's planner hold for the kept id, at
        /// that base, size and generation (planRestore's, applied from rank 0's plan just before): a restore never lands
        /// outside the part and the rows the planner reserved for it (a mismatch is a diverged pair: this rank ends).
        pub fn reserve(p: *Self, h: H, m: rank_wire.Reserve) !void {
            const s = try p.sp();
            if (m.generation != 0) {
                const k = try p.kf();
                const pl = &(k.planner orelse return error.NotAgreed);
                const hold = for (pl.holds.items) |x| {
                    if (x.why == .restore and x.id == m.kept) break x;
                } else return error.UnknownHold;
                if (hold.gen != m.generation or hold.base != m.base or hold.part != m.part or hold.size != m.size or hold.size < m.at) return error.StaleExtent;
            }
            return s.reserve(h, m);
        }
        pub fn status(p: *Self, h: H, m: rank_wire.RestoreStatus) !rank_wire.RestoreStatusReply {
            return (try p.sp()).status(h, m);
        }
        pub fn keep(p: *Self, h: H, m: rank_wire.Keep) !void {
            return (try p.sp()).keep(h, m);
        }
        pub fn share(p: *Self, h: H, m: rank_wire.Share) !void {
            return (try p.sp()).share(h, m);
        }
        pub fn touch(p: *Self, h: H, m: rank_wire.Touch) !void {
            return (try p.sp()).touch(h, m);
        }
        pub fn forget(p: *Self, h: H, m: rank_wire.Forget) !void {
            return (try p.sp()).forget(h, m);
        }
        pub fn start(p: *Self, h: H, m: rank_wire.Start) !void {
            return (try p.sp()).start(h, m);
        }
        pub fn progress(p: *Self, h: H, m: rank_wire.Progress) !void {
            return (try p.sp()).progress(h, m);
        }
        pub fn validate(p: *Self, h: H, m: rank_wire.Validate) !rank_wire.ValidateReply {
            return (try p.sp()).validate(h, m);
        }
        pub fn commit(p: *Self, h: H, m: rank_wire.Commit) !void {
            return (try p.sp()).commit(h, m);
        }
        pub fn abort(p: *Self, h: H, m: rank_wire.Abort) !void {
            return (try p.sp()).abort(h, m);
        }
        pub fn writeStaged(p: *Self, h: H, m: rank_wire.WriteStaged) !void {
            return (try p.sp()).writeStaged(h, m);
        }
        pub fn writeCancelled(p: *Self, h: H, m: rank_wire.WriteCancelled) !void {
            return (try p.sp()).writeCancelled(h, m);
        }
        /// Each set to its side: `tier` the spill side's counters, `shard` the planner side's (a side this rank does not
        /// run answers its set empty, as each side alone does for the other's).
        pub fn stats(p: *Self, h: H, m: rank_wire.Stats) !rank_wire.StatsReply {
            return switch (m.set) {
                .tier => if (p.spill) |x| x.stats(h, m) else .{ .set = .tier, .n = 0, .values = @splat(0) },
                .shard => if (p.kv) |x| x.stats(h, m) else .{ .set = .shard, .n = 0, .values = @splat(0) },
                .open => if (p.spill) |x| x.stats(h, m) else .{ .set = .open, .n = 0, .values = @splat(0) },
            };
        }
        pub fn shardLayout(p: *Self, h: H, m: rank_wire.ShardLayout) !rank_wire.ShardLayoutReply {
            return (try p.kf()).shardLayout(h, m);
        }
        pub fn shardSet(p: *Self, h: H, m: rank_wire.ShardSet) !void {
            return (try p.kf()).shardSet(h, m);
        }
        pub fn shardOps(p: *Self, h: H, m: rank_wire.ShardOps) !void {
            return (try p.kf()).shardOps(h, m);
        }
        pub fn shardCheck(p: *Self, h: H, m: rank_wire.ShardCheck) !rank_wire.ShardCheckReply {
            return (try p.kf()).shardCheck(h, m);
        }
        pub fn shardXfer(p: *Self, h: H, m: rank_wire.ShardXfer) !rank_wire.ShardXferReply {
            return (try p.kf()).shardXfer(h, m);
        }
        pub fn shardFinish(p: *Self, h: H, m: rank_wire.ShardFinish) !void {
            return (try p.kf()).shardFinish(h, m);
        }
    };
}

/// The follower's spill side as its planner's ops reach it (kv_lanes.SpillHooks over a spill follower `sd`).
pub fn followerSpillHooks(sd: anytype) kv_lanes.SpillHooks {
    const SF = @typeInfo(@TypeOf(sd)).pointer.child;
    return .{ .ctx = sd, .guard = struct {
        fn f(ctx: *anyopaque, kept: u64, from: u32) void {
            const x: *SF = @ptrCast(@alignCast(ctx));
            x.guard(kept, from);
        }
    }.f, .bump = struct {
        fn f(ctx: *anyopaque, lo: usize, hi: usize) void {
            const x: *SF = @ptrCast(@alignCast(ctx));
            x.bump(lo, hi);
        }
    }.f };
}

/// follow, with this rank's spill side when the tier is on (`spill`: made before the model's pool, its store opened at
/// the startup agreement rank 0 starts; the caller destroys it after this returns).
pub fn followWith(gpa: std.mem.Allocator, m: *Model, spill: ?*SpillFollower) !void {
    return followPair(gpa, m, spill, null);
}

/// followWith, with `fx` for the pair's one rank_wire Follower's hello instead of its default: the spill side's full
/// fingerprint when the tier is on, else (TF_DS_KV_SHARD alone) round 2's shard-only hello (kv_lanes.Follower.exchange).
/// A follower's own bounded exit (rank 0 has core/peer_watch): when rank 0's link is gone while this rank is inside a
/// frame (busy since `busy` ns, 0 when waiting for the next one) for longer than the grace, the engine thread may be
/// blocked in a GPU collective rank 0 will never join, so the process ends with 75 at once: no locks, no CUDA
/// teardown. Outside a frame the follow loop sees the closed link itself and ends as it always has.
pub const FollowerWatch = struct {
    peer: link.Link,
    busy: std.atomic.Value(u64) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    grace_ns: u64 = std.time.ns_per_s,

    extern "c" fn _exit(status: c_int) noreturn;

    pub fn shouldExit(busy_since: u64, now: u64, gone: bool, grace_ns: u64) bool {
        return gone and busy_since != 0 and now -| busy_since >= grace_ns;
    }

    pub fn fatalVoteExit(with_spill: bool, vote_enabled: bool, agreement_seen: bool) bool {
        return with_spill and (vote_enabled or agreement_seen);
    }

    fn run(w: *FollowerWatch) void {
        while (!w.stop.load(.acquire)) {
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
            _ = std.c.nanosleep(&ts, null);
            const since = w.busy.load(.acquire);
            if (since == 0) continue;
            if (shouldExit(since, link.Link.monotonicNs(), w.peer.peerGone(), w.grace_ns)) {
                std.log.err("dsv41 follower: rank 0's link closed while this rank was inside a frame; exiting 75", .{});
                _exit(75);
            }
        }
    }
};

test "follower watch exits only when rank 0 is gone during a frame, after the grace" {
    const g = std.time.ns_per_s;
    try std.testing.expect(!FollowerWatch.shouldExit(0, 10 * g, true, g)); // waiting for a frame: the loop handles it
    try std.testing.expect(!FollowerWatch.shouldExit(5 * g, 5 * g + g / 2, true, g)); // inside the grace
    try std.testing.expect(!FollowerWatch.shouldExit(5 * g, 9 * g, false, g)); // a long frame with rank 0 alive
    try std.testing.expect(FollowerWatch.shouldExit(5 * g, 6 * g, true, g));
}

test "spill vote failures bypass unbounded follower cleanup, other modes keep it" {
    try std.testing.expect(FollowerWatch.fatalVoteExit(true, true, false));
    try std.testing.expect(FollowerWatch.fatalVoteExit(true, false, true)); // rank 0 opted in, this rank did not
    try std.testing.expect(!FollowerWatch.fatalVoteExit(true, false, false)); // spill with the vote off, including KV shard
    try std.testing.expect(!FollowerWatch.fatalVoteExit(false, true, false)); // the spill-off vote's existing teardown
}

pub fn followPair(gpa: std.mem.Allocator, m: *Model, spill: ?*SpillFollower, fx: ?fingerprint.Exchange) !void {
    const peer = m.peer();
    const buf = try gpa.alloc(u8, (m.pool_cap + 1) * 4 + (1 << 16));
    defer gpa.free(buf);
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    var win_ids: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&win_ids) |*q| q.deinit(gpa);
    var built: Built = .{};
    var step: u64 = 0;
    var rdma_idle = false;
    var slot: usize = 0;
    var img_rows: std.ArrayList(u8) = .empty; // a prompt's image rows as their pieces come
    defer img_rows.deinit(gpa);
    var kept: std.AutoHashMapUnmanaged(u64, cuda.DeviceBuffer) = .empty; // rank 0's kept states, this rank's part
    defer {
        var ki = kept.valueIterator();
        while (ki.next()) |b| b.free();
        kept.deinit(gpa);
    }
    var fk: FollowKept = .{ .gpa = gpa, .m = m, .kept = &kept, .seqs = &seqs };
    defer fk.deinit();
    if (spill) |sd| sd.attach(fk.family());
    // on every exit, while fk and kept are alive: open restores ended (their buffers discarded through fk), the
    // family let go, then every copy still on the device waited for (no copy may read a buffer the defers above free)
    defer if (spill) |sd| {
        sd.detach();
        sd.quiesce();
    };
    var move_credit: move_budget.Budget = .{};
    var move_state: idle_relocation.State = .{};
    var move_deadline: u64 = 0;
    // This errdefer is registered after cleanup defers: unknown transfer state
    // terminates custody before detach, quiesce, or retained-buffer recycling.
    var relocation_custody = false;
    errdefer if (relocation_custody) (Model.SpillDev{ .m = m }).failed(error.TransferFailed);

    var fill_base: usize = 0;
    // TF_DS_KV_SHARD: this rank's copy of the parts' planner, fed rank 0's ops; with the spill tier its stage, stale
    // and move ops guard this rank's own writes and its transfers bump their destinations (kv_lanes.SpillHooks)
    const kvf: ?*kv_lanes.Follower = if (m.kvs != null) try kv_lanes.Follower.init(gpa, m.rank, kvConfig(m), try kv_lanes.Settings.fromEnv(getenv), modelDev(m)) else null;
    defer if (kvf) |k| k.deinit();
    if (kvf) |k| k.counters = .{ .ctx = m, .read = readShardRank }; // (its own row on rank 0's /health)
    if (kvf) |k| if (spill) |sd| {
        k.spill = followerSpillHooks(sd);
    };
    // one rank_wire Follower for both (rank 0's one Leader): the caller's fingerprint, else the spill side's, else
    // the planner's shard-only one (native.openPair's hello for TF_DS_KV_SHARD alone)
    const pair_fx: ?fingerprint.Exchange = fx orelse if (spill) |sd| sd.wf.fx else if (kvf) |k| k.exchange() else null;
    const pf: ?*PairFollower = if (pair_fx) |x| try PairFollower.create(gpa, @intCast(m.rank), x, spill, kvf) else null;
    defer if (pf) |p| gpa.destroy(p);
    const vote_spill = pf != null and m.keepGraphVote(spill != null);
    if (pf != null and !vote_spill) m.disableGraphVote();
    var vote_agreed = false;
    var watch: FollowerWatch = .{ .peer = peer };
    const watcher: ?std.Thread = std.Thread.spawn(.{}, FollowerWatch.run, .{&watch}) catch |err| blk: {
        std.log.warn("dsv41 follower: no watch thread ({s}); a blocked rank relies on its supervisor", .{@errorName(err)});
        break :blk null;
    };
    defer if (watcher) |t| {
        watch.stop.store(true, .release);
        t.join();
    };
    var vote_agreement_seen = false;
    // Fatal spill-vote errors cannot run the normal defers: a store scan or CUDA event can stall quiesce past the
    // vote deadline. Exit without freeing buffers still owned by DMA, even if the peer watcher could not start.
    errdefer if (FollowerWatch.fatalVoteExit(spill != null, vote_spill, vote_agreement_seen)) {
        std.log.err("dsv41 follower: spill graph-vote session failed; exiting 75 without teardown", .{});
        FollowerWatch._exit(75);
    };
    while (true) {
        watch.busy.store(0, .release); // waiting for rank 0's next frame
        // with spill writes or restores pending, they progress while rank 0 sends nothing (2 ms polls)
        if (!m.rs.live and move_state.pending == null) if (spill) |sd| while (sd.busy() and !try peer.readable(2 * std.time.ns_per_ms)) sd.service(std.time.ns_per_ms);
        // the next primitive, polled for 20 ms before the read blocks (rank 0 sends it a fraction of a millisecond
        // after this rank's GPU goes idle in a decode; a sleeping thread would add its core's wake-up to every frame)
        var admission_deadline: u64 = 0;
        // A partial command is irrecoverable framing uncertainty, even PREPARE.
        const incoming = if (move_state.pending != null) peer.recvBy(buf, move_deadline) else peer.recvCommand(buf, &admission_deadline, 5 * std.time.ns_per_s);
        var r: Reader = .{ .b = incoming catch |err| (Model.SpillDev{ .m = m }).failed(err) };
        const kind = try decodeOuterKind(try r.int(u8));
        if (rdma_idle) {
            setRdmaIdle(m, false); // before any local spill, capture, restore, relocation or RDMA command
            rdma_idle = false;
        }
        vote_agreement_seen = kind == .graph_vote_agree;
        if (try r.int(u64) != step) return error.OutOfStep;
        step += 1;
        // An enabled follower must see the opt-in immediately after rank_wire's agreed frame. Otherwise it
        // might wait inside a graph for a rank 0 that never votes; refuse before the first collective instead.
        if (vote_spill and pf.?.wf.t.agreed and !vote_agreed and kind != .graph_vote_agree) return error.GraphVoteMismatch;
        watch.busy.store(link.Link.monotonicNs(), .release); // inside this frame until the loop comes round
        if (kind == .idle_move) relocation_custody = true;
        if (move_state.pending != null and kind != .idle_move and kind != .rank_wire) return error.TransferFailed;
        if (kind == .idle_move) {
            if (m.ahead != null) return error.TransferFailed;
            if (move_state.pending == null) move_deadline = admission_deadline;
            if (spill) |sd| if (sd.side) |side| { side.operation_deadline = move_deadline; };
        }
        if (m.rs.live and !openChunkAllows(kind)) return error.OpenPrefillChunk;
        if (!m.rs.live and move_state.pending == null and kind != .idle_move and kind != .fill_slice) if (spill) |sd| sd.service(std.time.ns_per_ms); // relocation does not service unrelated work
        // a chunk's read-ahead of the next is done before anything but that chunk (a prompt's ids may change)
        if (kind != .fill_chunk and kind != .fill_slice and kind != .snap_save and kind != .snap_restore and kind != .fill_images and kind != .rank_wire) {
            if (kind != .idle_move) m.joinAhead();
            if (!m.rs.live) m.fill_seq = &.{};
        }
        switch (kind) {
            .rdma_idle => {
                const dev: Model.SpillDev = .{ .m = m };
                // m.joinAhead ran before dispatch; synchronize all CUDA streams,
                // including graph capture/replay, before voting for parking.
                m.ctx.synchronize() catch |err| dev.failed(err);
                const can_park = idle_protocol.enabled(getenv("TF_DS_RDMA_PARK")) and
                    move_state.pending == null and (spill == null or !spill.?.idleBusy());
                if (can_park) quiesceRdma(m, admission_deadline) catch |err| dev.failed(err);
                const agreed = idle_protocol.follower(peer, can_park, admission_deadline) catch |err| dev.failed(err);
                if (agreed) {
                    setRdmaIdle(m, true);
                    rdma_idle = true;
                }
            },
            .graph_vote_agree => {
                if (pf == null or !pf.?.wf.t.agreed or vote_agreed) return error.GraphVoteMismatch;
                const value = r.rest();
                if (value.len != 1 or value[0] != 1) return error.GraphVoteMismatch;
                const reply = [_]u8{@intFromBool(vote_spill)};
                try peer.sendWithin(&reply, admission_deadline);
                if (!vote_spill) return error.GraphVoteMismatch;
                vote_agreed = true;
            },
            .idle_move => {
                const Host = struct {
                    m: *Model,
                    spill: ?*SpillFollower,
                    credit: *move_budget.Budget,
                    sharded: bool,
                    pub fn rank(h: *@This()) u8 { return @intCast(h.m.rank); }
                    pub fn validate(h: *@This(), mv: idle_relocation.Move) !void {
                        if (h.sharded or h.m.ahead != null or mv.bytes != try h.m.extentBytes(@intCast(mv.from), @intCast(mv.to), @intCast(mv.len)) or mv.bytes > h.credit.available(h.m.now())) return error.TransferFailed;
                    }
                    pub fn copy(h: *@This(), mv: idle_relocation.Move, deadline: u64) !void {
                        if (!h.credit.consume(h.m.now(), mv.bytes)) return error.TransferFailed;
                        if (h.spill) |sd| sd.bump(@intCast(mv.to), @intCast(mv.to + mv.len));
                        try h.m.copyExtent(@intCast(mv.from), @intCast(mv.to), @intCast(mv.len));
                        try Lanes.waitIdleCopy(h.m, deadline);
                    }
                };
                var host: Host = .{ .m = m, .spill = spill, .credit = &move_credit, .sharded = kvf != null };
                var relocation_step = step - 1;
                const receipt = try idle_relocation.dispatch(&move_state, r.b, admission_deadline -| idle_relocation.budget_ns, &relocation_step, &host);
                move_deadline = move_state.deadline;
                if (spill) |sd| if (sd.side) |side| { side.operation_deadline = move_deadline; };
                try peer.sendWithin(&receipt, move_deadline);
                if (move_state.pending == null) {
                    relocation_custody = false;
                    if (spill) |sd| if (sd.side) |side| { side.operation_deadline = null; };
                }
            },
            .fill_begin => {
                slot = try r.int(u32);
                const base = try r.int(u64);
                if (slot >= max_streams) return error.BadSlot;
                img_rows.clearRetainingCapacity();
                try r.ids(gpa, &seqs[slot]);
                fill_base = @intCast(base);
                if (spill) |sd| sd.bumpSlot(slot);
                if (m.fillBegin(slot, @intCast(base))) {
                    if (kvf) |k| try k.fillBegin(slot, @intCast(base)); // (its twins' view, as rank 0 binds it)
                } else |err| refused(err);
                m.fill_seq = seqs[slot].items;
            },
            .fill_resume => {
                slot = try r.int(u32);
                fill_base = @intCast(try r.int(u64));
                if (slot >= max_streams or seqs[slot].items.len == 0) return error.BadSlot;
                try m.fillBegin(slot, fill_base);
                if (kvf) |k| try k.fillBegin(slot, fill_base);
                m.fill_seq = seqs[slot].items;
            },
            .fill_slice => {
                if (m.prefill.slice_layers == 0 or kvf != null) return error.BadFrame;
                const sf = try getSlice(&r);
                if (sf.abort) {
                    if (!m.rs.live) return error.BadSlice;
                    m.dropChunk();
                    continue;
                }
                if (sf.args.start > seqs[slot].items.len or sf.args.n > seqs[slot].items.len - sf.args.start) return error.BadChunk;
                m.fill_seq = seqs[slot].items;
                m.next_engram_rows = sf.next_rows;
                if (sf.lo == 0) if (spill) |sd| sd.bump(fill_base + sf.args.start, fill_base + sf.args.start + sf.args.n);
                if (m.fillSlice(m.fill_seq[0 .. sf.args.start + sf.args.n], sf.args, sf.lo, sf.hi)) |_| {} else |err| try roundRefused(err);
            },
            .fill_chunk => {
                const start: usize = @intCast(try r.int(u64));
                const n: usize = try r.int(u32);
                const replay: usize = @intCast(try r.int(u64));
                if (m.prefill.ae) m.next_engram_rows = try r.int(u32);
                if (start + n > seqs[slot].items.len) return error.BadChunk;
                m.fill_seq = seqs[slot].items; // (a round between two chunks cleared it: the next chunk's read-ahead again)
                if (spill) |sd| sd.bump(fill_base + start, fill_base + start + n);
                // an error that parts the ranks (an Engram read this rank could not make) ends the follower; rank 0 ends the pair
                if (m.fillChunk(seqs[slot].items[0 .. start + n], start, n, replay)) |_| {} else |err| try roundRefused(err);
            },
            .verify => {
                // New-format parse failures can strand a leader already in a
                // collective. This inner errdefer precedes every outer teardown.
                var head_custody: greedy_head.Custody = .{ .active = m.decode_greedy_head or m.decode_sample_head };
                errdefer head_custody.failed(m) catch unreachable;
                const nw = try r.int(u32);
                if (nw == 0 or nw > max_streams) return error.BadFrame;
                const use_head = try readGreedyHeadMode(&r, m.decode_greedy_head, m.two != null);
                const sample_head_count = try readSampleHeadCount(&r, m.decode_sample_head, m.two != null or use_head);
                var wins: [max_streams]Win = undefined;
                for (0..nw) |k| {
                    wins[k] = try readWin(&r, gpa, &win_ids[k], kvf != null);
                    // TF_DS_KV_SHARD: a round naming a reused extent (its part or generation not the planner's) fails closed
                    if (kvf) |kf| try kf.checkWin(wins[k].slot, wins[k].base, wins[k].end, @intCast(wins[k].part), wins[k].gen);
                }
                if ((m.decode_greedy_head or m.decode_sample_head) and r.at != r.b.len) return error.BadFrame;
                const rows = try built.build(gpa, &seqs, wins[0..nw]);
                if (spill) |sd| for (wins[0..nw]) |w| sd.bump(w.base + w.start, w.base + w.start + w.ids.len);
                head_custody.active = use_head or sample_head_count != 0; // baseline execution keeps its old error policy
                const verified = if (sample_head_count != 0)
                    greedy_head.verifySample(m, rows, built.absorb[0..nw], sample_head_count)
                else
                    greedy_head.verify(m, rows, built.absorb[0..nw], use_head);
                verified catch |err| {
                    if (use_head or sample_head_count != 0) return err; // inner custody pair-ends before outer defers
                    try roundRefused(err);
                };
                head_custody.active = false;
                if (m.prof) {
                    follow_prof.rounds += 1;
                    follow_prof.enqueue += m.t_enqueue;
                    follow_prof.forward += m.t_forward;
                }
            },
            .pass, .pass_keyed => {
                const keyed = kind == .pass_keyed;
                const frame = try readPass(&r, keyed, m.dr != null and m.dr.?.nz != 0);
                m.pass(frame.tokens[0..frame.n], frame.q0[0..frame.n], frame.slots[0..frame.n], frame.steps,
                    if (keyed) .{ .seeds = frame.seeds[0..frame.n], .temps = frame.temps[0..frame.n], .top_ks = frame.top_ks[0..frame.n] } else null) catch |err| try roundRefused(err);
            },
            .fill_images => {
                const count = try r.int(u32);
                var spans: std.ArrayList(@import("prompt.zig").Span) = .empty;
                defer spans.deinit(gpa);
                var row: usize = 0;
                for (0..count) |_| {
                    const at: usize = @intCast(try r.int(u64));
                    const n: usize = @intCast(try r.int(u64));
                    try spans.append(gpa, .{ .at = at, .len = n, .row = row });
                    row += n;
                }
                const off: usize = @intCast(try r.int(u64));
                const piece = r.rest();
                const total = row * m.cfg.hidden * 2;
                if (off != img_rows.items.len or off + piece.len > total or piece.len == 0) return error.BadFrame;
                try img_rows.appendSlice(gpa, piece);
                if (img_rows.items.len == total) {
                    m.setImages(spans.items, img_rows.items) catch |err| refused(err);
                    img_rows.clearRetainingCapacity();
                }
            },
            .snap_save => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                var b = try cuda.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
                m.snapCopy(s, b.ptr, true) catch |err| {
                    b.free();
                    refused(err);
                    continue;
                };
                try kept.put(gpa, id, b);
            },
            .snap_restore => {
                const id = try r.int(u64);
                const s: usize = try r.int(u32);
                if (s >= max_streams) return error.BadSlot;
                const from: usize = @intCast(try r.int(u64));
                const to: usize = @intCast(try r.int(u64));
                const at: usize = @intCast(try r.int(u64));
                const b = kept.get(id) orelse return error.UnknownSnapshot; // rank 0 restores only states both ranks keep
                if (from != to) {
                    if (spill) |sd| sd.bump(to, to + at);
                    m.copyExtent(from, to, at) catch |err| refused(err);
                }
                m.snapCopy(s, b.ptr, false) catch |err| refused(err);
            },
            .snap_drop => {
                const id = try r.int(u64);
                if (spill) |sd| sd.dropSnap(id); // its write guarded, its small-state capture done, before the free
                if (kept.fetchRemove(id)) |kv| {
                    var b = kv.value;
                    b.free();
                }
            },
            .snap_drop_many => {
                // Exact frame size/count/uniqueness and all IDs checked before any free.
                const batch = try cache_batch.decode(r.rest());
                for (batch.slice()) |id| if (!kept.contains(id)) return error.UnknownSnapshot;
                for (batch.slice()) |id| {
                    if (spill) |sd| sd.dropSnap(id);
                    var b = kept.fetchRemove(id).?.value;
                    b.free();
                }
                const ack = cache_batch.receipt(step - 1, batch.count);
                try peer.sendWithin(&ack, admission_deadline);
            },
            // one rank_wire Follower a rank, for the spill tier's and the KV parts' frames alike
            .rank_wire => try (pf orelse return error.SpillNotStarted).onFrame(peer, r.rest()), // (rank 0 runs the spill tier)
            .done => {
                if (spill) |sd| {
                    const left = sd.flush(sd.flush_ns);
                    if (left > 0) std.log.info("dsv41 spill: {d} kept states were still unpublished at the flush deadline", .{left});
                }
                return;
            },
        }
    }
}

test "draft chances are the prefix survivals chooseK sums" {
    var out: [3]f64 = undefined;
    survivals(&.{ 0.0, 0.0, 2.0 }, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), out[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.25 / (1.0 + @exp(@as(f64, -2.0))), out[2], 1e-12);
}

test "extents take the first chunk-aligned gap" {
    const used = [_][2]usize{ .{ 0, 4096 }, .{ 6144, 8192 } };
    try std.testing.expectEqual(@as(?usize, 4096), place(&used, 2048, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 8192), place(&used, 4096, 1 << 20));
    try std.testing.expectEqual(@as(?usize, 0), place(&.{}, 2048, 2048));
    try std.testing.expectEqual(@as(?usize, null), place(&used, 4096, 10240));
}

test "a frame's fields read back as written" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try w.begin(gpa, .verify, 7);
    try w.int(gpa, u64, 1 << 40);
    try w.ids(gpa, &.{ 1, -2, 129279 });
    var r: Reader = .{ .b = w.buf.items };
    try std.testing.expectEqual(@as(u8, @backingInt(Kind.verify)), try r.int(u8));
    try std.testing.expectEqual(@as(u64, 7), try r.int(u64));
    try std.testing.expectEqual(@as(u64, 1 << 40), try r.int(u64));
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    try r.ids(gpa, &got);
    try std.testing.expectEqualSlices(i32, &.{ 1, -2, 129279 }, got.items);
    try std.testing.expectError(error.ShortFrame, r.int(u8));
}

test "a round's rows: each window at its positions, its ids written over the rejected rows" {
    const gpa = std.testing.allocator;
    var seqs: [max_streams]std.ArrayList(i32) = @splat(.empty);
    defer for (&seqs) |*q| q.deinit(gpa);
    try seqs[0].appendSlice(gpa, &.{ 10, 11, 12, 13, 99, 98 }); // positions 4.. hold a rejected window
    try seqs[2].appendSlice(gpa, &.{ 20, 21 });
    var b: Built = .{};
    const rows = try b.build(gpa, &seqs, &.{
        .{ .slot = 0, .base = 0, .end = 4096, .start = 4, .absorb = true, .ids = &.{ 14, 15 } },
        .{ .slot = 2, .base = 4096, .end = 8192, .start = 2, .absorb = false, .ids = &.{22} },
    });
    try std.testing.expectEqualSlices(i64, &.{ 14, 15, 22 }, rows.ids);
    try std.testing.expectEqualSlices(i64, &.{ 4, 5, 2 }, rows.pos);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0, 2 }, rows.slots.?);
    try std.testing.expectEqualSlices(i64, &.{ 4096, 4096, 8192 }, rows.ends.?);
    try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 14, 15 }, rows.windows.?[0].seq);
    try std.testing.expectEqualSlices(i32, &.{ 20, 21, 22 }, rows.windows.?[1].seq);
    try std.testing.expectEqual(@as(usize, 2), rows.windows.?[1].row);
    try std.testing.expectError(error.SequenceGap, b.build(gpa, &seqs, &.{.{ .slot = 1, .base = 0, .end = 2048, .start = 3, .absorb = false, .ids = &.{1} }}));
}

test "greedy draws take the first largest logit" {
    const row = [_]f32{ 0.5, 2.0, -1.0, 2.0 };
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, null));
    try std.testing.expectEqual(@as(u32, 1), try draw(&row, 9, .{ .seed = 3, .temperature = 0 }));
}

test "prefix growth relocates instead of overwriting a retained neighbour" {
    var a: Snap = .{ .id = 1, .at = 6144, .base = 0, .buf = undefined };
    var b: Snap = .{ .id = 2, .at = 6144, .base = 6144, .buf = undefined };
    var m: Model = undefined;
    m.rs = .{};
    m.pool_cap = 3300000;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.kv = null;
    l.spill = null;
    l.streams = .empty;
    l.relocation_enabled = false;
    l.m = &m;
    l.snaps = .empty;
    defer l.snaps.deinit(l.gpa);
    try l.snaps.appendSlice(l.gpa, &.{ &a, &b });
    try std.testing.expect(l.overwritesKept(6144, 10240, &a));
    try std.testing.expectEqual(@as(usize, 12288 + growth_rows), try l.placeKept(&.{}, 10240, &a));
    try std.testing.expect(!a.stale and !b.stale);
    try std.testing.expect(!l.overwritesKept(12288, 22528, &a));
}

test "placement keeps restore source until genuine pool pressure requires cold fallback" {
    var a: Snap = .{ .id = 1, .at = 6144, .base = 0, .buf = undefined };
    var b: Snap = .{ .id = 2, .at = 6144, .base = 6144, .buf = undefined };
    var m: Model = undefined;
    m.rs = .{};
    m.pool_cap = 16384;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.kv = null;
    l.spill = null;
    l.streams = .empty;
    l.relocation_enabled = false;
    l.m = &m;
    l.snaps = .empty;
    defer l.snaps.deinit(l.gpa);
    try l.snaps.appendSlice(l.gpa, &.{ &a, &b });
    try std.testing.expectEqual(@as(usize, 6144), try l.placeKept(&.{}, 10240, &a));
    try std.testing.expect(!a.stale and b.stale);
    m.pool_cap = 10240;
    try std.testing.expectEqual(@as(usize, 0), try l.placeKept(&.{}, 10240, &a));
    try std.testing.expect(a.stale);
}

test "a state a waiting request matched survives other placements until only it can give room" {
    var a: Snap = .{ .id = 1, .at = 6144, .base = 0, .buf = undefined, .held = 1 };
    var b: Snap = .{ .id = 2, .at = 6144, .base = 6144, .buf = undefined };
    var m: Model = undefined;
    m.rs = .{};
    m.pool_cap = 16384;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.kv = null;
    l.spill = null;
    l.streams = .empty;
    l.relocation_enabled = false;
    l.m = &m;
    l.snaps = .empty;
    defer l.snaps.deinit(l.gpa);
    try l.snaps.appendSlice(l.gpa, &.{ &a, &b });
    try std.testing.expectEqual(@as(usize, 6144), try l.placeKept(&.{}, 10240, null));
    try std.testing.expect(!a.stale and b.stale);
    m.pool_cap = 10240;
    try std.testing.expectEqual(@as(usize, 0), try l.placeKept(&.{}, 10240, null));
    try std.testing.expect(a.stale);
}

test "full prompt snapshots are exact-length only inside bounded replay by default" {
    var m: Model = undefined;
    m.rs = .{};
    m.cfg.window = 128;
    var sn: Snap = .{ .id = 1, .at = 8191, .base = 0, .buf = undefined, .complete = true };
    var l: Lanes = undefined;
    l.kv = null;
    l.spill = null;
    l.streams = .empty;
    l.relocation_enabled = false;
    l.m = &m;
    l.gpa = std.testing.allocator;
    l.snaps = .empty;
    defer l.snaps.deinit(l.gpa);
    try l.snaps.append(l.gpa, &sn);
    l.short_continuation = false;
    try std.testing.expect(l.usable(&sn, 8191));
    try std.testing.expect(!l.usable(&sn, 8192));
    try std.testing.expect(!l.usable(&sn, 8400)); // pending ratio-2 row: not a prompt chunk start
    sn.at = 8190;
    try std.testing.expect(l.usable(&sn, 8318));
    try std.testing.expect(!l.usable(&sn, 8317));
    sn.stale = true;
    try std.testing.expect(!l.usable(&sn, 8190));
}

test "opt-in short continuation requires a live aligned fully decoded prefix" {
    var m: Model = undefined;
    m.rs = .{};
    m.cfg.window = 128;
    var sn: Snap = .{ .id = 1, .at = 9126, .base = 0, .buf = undefined, .decoded = true };
    var l: Lanes = undefined;
    l.kv = null;
    l.spill = null;
    l.streams = .empty;
    l.relocation_enabled = false;
    l.m = &m;
    l.gpa = std.testing.allocator;
    l.snaps = .empty;
    l.short_continuation = false;
    defer l.snaps.deinit(l.gpa);
    try l.snaps.append(l.gpa, &sn);
    try std.testing.expect(!l.usable(&sn, 9145)); // actual workflow miss remains the default
    l.short_continuation = true;
    try std.testing.expect(l.usable(&sn, 9145));
    try std.testing.expect(Lanes.snapShortEligibleFn(&l, &sn, 9145));
    try std.testing.expect(l.shortResume(&sn, 9254));
    try std.testing.expect(!l.shortResume(&sn, 9255));
    try std.testing.expect(l.usable(&sn, 9255)); // ordinary encoder reuse, no fast continuation
    try std.testing.expect(!l.usable(&sn, 9126)); // accepted state does not carry its exact head logits
    try std.testing.expect(!l.usable(&sn, 9125));
    sn.decoded = false;
    try std.testing.expect(!l.usable(&sn, 9145));
    try std.testing.expect(!Lanes.snapShortEligibleFn(&l, &sn, 9145)); // encoder-only remains protected
    sn.complete = true;
    try std.testing.expect(l.usable(&sn, 9126));
    try std.testing.expect(l.usable(&sn, 9145));
    sn.at = 9127;
    try std.testing.expect(!l.usable(&sn, 9145)); // ratio-2 partial group cannot start a fill chunk
    try std.testing.expect(l.usable(&sn, 9127)); // exact head-only restore is still allowed
    sn.at = 9126;
    sn.stale = true;
    try std.testing.expect(!l.usable(&sn, 9145));
}

test "telemetry unions snapshot overlaps and excludes stale and live overlap" {
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var a: Snap = .{ .id = 1, .base = 100, .at = 200, .buf = undefined };
    var b: Snap = .{ .id = 2, .base = 200, .at = 200, .buf = undefined };
    var stale: Snap = .{ .id = 3, .base = 1000, .at = 200, .buf = undefined, .stale = true };
    try ln.snaps.append(std.testing.allocator, &a);
    try ln.snaps.append(std.testing.allocator, &b);
    try ln.snaps.append(std.testing.allocator, &stale);
    const active = [_][2]usize{ .{ 0, 150 }, .{ 500, 550 } };
    try std.testing.expectEqual(@as(usize, 450), ln.retained(&active));
    try std.testing.expectEqual(@as(usize, 300), ln.retained(&.{}));
    b.stale = true;
    try std.testing.expectEqual(@as(usize, 350), ln.retained(&active));
}

test "a short shared sibling cannot overwrite the main conversation's longer prefix" {
    var common: Snap = .{ .id = 1, .at = 8192, .base = 0, .buf = undefined };
    var main: Snap = .{ .id = 2, .at = 98304, .base = 0, .end = 131072, .buf = undefined };
    var m: Model = undefined;
    m.rs = .{};
    m.pool_cap = 524288;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.kv = null;
    l.spill = null;
    l.streams = .empty;
    l.relocation_enabled = false;
    l.m = &m;
    l.snaps = .empty;
    defer l.snaps.deinit(l.gpa);
    try l.snaps.appendSlice(l.gpa, &.{ &common, &main });
    // A1 retained main; B1 resumes common. This invariant already holds in CDEF.
    try std.testing.expect(l.overwritesKept(common.at, 16384, &common));
    const b = try l.placeKept(&.{}, 16384, &common);
    l.staleOver(b, b + 16384, &common);
    try std.testing.expect(!main.stale and !common.stale);
    try std.testing.expect(b >= main.end);
    // A2 can still reuse main in place after B1 finishes, without changing its prefix.
    try std.testing.expect(!l.overwritesKept(main.at, main.end, &main));
}

test "soft reservations yield to packing before any kept prefix is evicted" {
    const c = model.chunk_rows;
    const a = std.testing.allocator;
    const kept = [_][3]usize{.{ 0, 4 * c, 8 * c }};
    try std.testing.expectEqual(@as(?usize, 8 * c + growth_rows), try placeSoft(a, &.{}, &kept, 4 * c, 128 * c));
    try std.testing.expectEqual(@as(?usize, 8 * c), try placeSoft(a, &.{}, &kept, 4 * c, 12 * c));
    try std.testing.expectEqual(@as(?usize, 4 * c), try placeSoft(a, &.{}, &kept, 4 * c, 8 * c));
    try std.testing.expectEqual(@as(?usize, null), try placeSoft(a, &.{.{ 4 * c, 8 * c }}, &kept, 4 * c, 8 * c));
}

test "pool pressure cannot evict a retained prefix whose rows remain owned by a live stream" {
    var snap: Snap = .{ .id = 1, .at = 8192, .base = 0, .end = 16384, .buf = undefined };
    var m: Model = undefined;
    m.rs = .{};
    m.pool_cap = 16384;
    var l: Lanes = undefined;
    l.gpa = std.testing.allocator;
    l.kv = null;
    l.spill = null;
    l.streams = .empty;
    l.relocation_enabled = false;
    l.m = &m;
    l.snaps = .empty;
    defer l.snaps.deinit(l.gpa);
    try l.snaps.append(l.gpa, &snap);
    try std.testing.expectError(error.ContextFull, l.placeKept(&.{.{ 0, 16384 }}, 2048, null));
    try std.testing.expect(!snap.stale);
}

test "K2a empty active interval cannot stop union before later retained snapshots" {
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var a: Snap = .{ .id = 1, .base = 10, .at = 10, .buf = undefined };
    var b: Snap = .{ .id = 2, .base = 100, .at = 20, .buf = undefined };
    try ln.snaps.appendSlice(ln.gpa, &.{ &a, &b });
    const active = [_][2]usize{ .{ 0, 0 }, .{ 30, 30 }, .{ 200, 200 }, .{ 15, 25 } };
    try std.testing.expectEqual(@as(usize, 35), ln.retained(&active));
    try std.testing.expectEqual(@as(usize, 30), ln.retained(&.{.{ 30, 30 }}));
    a.stale = true;
    b.stale = true;
    try std.testing.expectEqual(@as(usize, 0), ln.retained(&.{ .{ 0, 0 }, .{ 10, 10 } }));
}

test "K2a retained union agrees with independent bitmap oracle in 10000 interval cases" {
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null };
    defer ln.snaps.deinit(std.testing.allocator);
    var snaps: [8]Snap = undefined;
    for (&snaps, 0..) |*sn, i| {
        sn.* = .{ .id = @intCast(i), .base = 0, .at = 0, .buf = undefined };
        try ln.snaps.append(ln.gpa, sn);
    }
    var prng = std.Random.DefaultPrng.init(0x4b32564b);
    const rng = prng.random();
    for (0..10000) |_| {
        var occupied: [256]bool = undefined;
        @memset(&occupied, false);
        var active: [8][2]usize = undefined;
        for (&active) |*r| {
            r[0] = rng.uintLessThan(usize, 256);
            r[1] = r[0] + rng.uintLessThan(usize, 257 - r[0]);
            for (r[0]..r[1]) |j| occupied[j] = true;
        }
        for (&snaps) |*sn| {
            sn.base = rng.uintLessThan(usize, 256);
            sn.at = rng.uintLessThan(usize, 257 - sn.base);
            sn.stale = rng.boolean();
            if (!sn.stale) for (sn.base..sn.base + sn.at) |j| {
                occupied[j] = true;
            };
        }
        var expected: usize = 0;
        for (occupied) |yes| if (yes) {
            expected += 1;
        };
        try std.testing.expectEqual(expected, ln.retained(&active));
    }
}

test "a restore's destination is held back from placement, never a victim and never usable until installed" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.rs = .{};
    m.pool_cap = 16 * c;
    m.cfg.window = 128;
    const gpa = std.testing.allocator;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    // a restore reading into [0, 4c), an ordinary kept state at [4c, 8c)
    var loading: Snap = .{ .id = 1, .at = 4 * c, .base = 0, .end = 4 * c, .buf = undefined, .loading = true };
    var kept: Snap = .{ .id = 2, .at = 4 * c, .base = 4 * c, .end = 8 * c, .buf = undefined };
    try ln.snaps.append(gpa, &loading);
    try ln.snaps.append(gpa, &kept);
    try std.testing.expect(!ln.usable(&loading, 8 * c)); // not until every rank installed it
    try std.testing.expect(ln.usable(&kept, 9 * c));
    // a placement that needs room past the free 8 chunks takes the ordinary kept state's, not the destination's
    const base = try ln.placeKept(&.{}, 12 * c, null);
    try std.testing.expect(kept.stale and !loading.stale);
    try std.testing.expectEqual(4 * c, base);
    // with nothing else to give, the destination still holds: the pool is full for this request
    kept.stale = false;
    kept.base = 4 * c;
    try std.testing.expectError(error.ContextFull, ln.placeKept(&.{.{ 4 * c, 16 * c }}, 4 * c, null));
    loading.loading = false;
    loading.restored = true;
    try std.testing.expect(ln.usable(&loading, 8 * c));
    // with the tier off nothing spill-related runs: the frame kinds before it keep their values
    try std.testing.expectEqual(@as(u8, 9), @backingInt(Kind.done));
    try std.testing.expectEqual(@as(u8, 10), @backingInt(Kind.rank_wire));
    try std.testing.expectEqual(Kind.rdma_idle, std.enums.fromInt(Kind, 11).?); // not the old spill notes
    try std.testing.expect(std.enums.fromInt(Kind, 14) == null); // the nested vote is never an outer command
    try std.testing.expectEqual(@as(u8, 15), @backingInt(Kind.graph_vote_agree));
    try std.testing.expect(ln.spill == null);
}

test "pool pressure uses X3 learned placement without bypassing snapshot custody" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.rs = .{};
    m.io = std.testing.io;
    m.pool_cap = 82 * c;
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = &m, .peer = null,
        .relocation_enabled = false, .pressure_planner = true, .adaptive_prefix = true, .kv_clock = 100 };
    defer ln.snaps.deinit(std.testing.allocator);
    var hot: Snap = .{ .id = 1, .at = 40 * c, .base = 0, .end = 40 * c,
        .buf = undefined, .adaptive_density = 400, .adaptive_at = 100, .last_use = 100 };
    var cold: Snap = .{ .id = 2, .at = 42 * c, .base = 40 * c, .end = 82 * c,
        .buf = undefined, .adaptive_density = 20, .adaptive_at = 100, .last_use = 90 };
    try ln.snaps.append(std.testing.allocator, &hot);
    try ln.snaps.append(std.testing.allocator, &cold);
    const base = try ln.placeKept(&.{}, 40 * c, null);
    try std.testing.expect(base >= 40 * c and !hot.stale and cold.stale);
    try std.testing.expectEqual(@as(u8, 2), cold.loss_reason);
}

test "a restore's destination holds the waiting prompt's room past its rows while that fits clear of the others" {
    const c = model.chunk_rows;
    var m: Model = undefined;
    m.rs = .{};
    m.pool_cap = 64 * c;
    m.cfg.window = 128;
    const gpa = std.testing.allocator;
    var ln: Lanes = .{ .gpa = gpa, .m = &m, .peer = null };
    defer ln.snaps.deinit(gpa);
    var kept: Snap = .{ .id = 1, .at = 4 * c, .base = 0, .end = 5 * c, .buf = undefined };
    try ln.snaps.append(gpa, &kept);
    // clear of the kept state's held extent and its growth: placed past it, nothing staled
    try std.testing.expectEqual(@as(?usize, 5 * c + growth_rows), try ln.placeFree(&.{}, 8 * c));
    try std.testing.expect(!kept.stale);
    // no room for the hint anywhere: null (spillReserve then places the rows alone, as placeKept does)
    try std.testing.expectEqual(@as(?usize, null), try ln.placeFree(&.{.{ 4 * c, 64 * c }}, 8 * c));
    try std.testing.expect(!kept.stale);
}

test "a verify window's frame bytes: as ever with the switch off, its part and generation added with it on" {
    const gpa = std.testing.allocator;
    const win: Win = .{ .slot = 2, .base = 4096, .end = 8192, .start = 7, .absorb = true, .ids = &.{ 5, 6 }, .part = 1, .gen = 9 };
    var w: Wire = .{};
    defer w.deinit(gpa);
    try writeWin(&w, gpa, win, false);
    // the layout before TF_DS_KV_SHARD: verifyFn's fields in their order
    var old: Wire = .{};
    defer old.deinit(gpa);
    try old.int(gpa, u32, 2);
    try old.int(gpa, u64, 4096);
    try old.int(gpa, u64, 8192);
    try old.int(gpa, u64, 7);
    try old.int(gpa, u8, 1);
    try old.ids(gpa, &.{ 5, 6 });
    try std.testing.expectEqualSlices(u8, old.buf.items, w.buf.items);
    var ids: std.ArrayList(i32) = .empty;
    defer ids.deinit(gpa);
    var r: Reader = .{ .b = w.buf.items };
    const off = try readWin(&r, gpa, &ids, false);
    try std.testing.expectEqual(@as(i64, -1), off.part);
    try std.testing.expectEqual(@as(u64, 0), off.gen);
    try std.testing.expectEqual(w.buf.items.len, r.at);
    w.buf.clearRetainingCapacity();
    try writeWin(&w, gpa, win, true);
    try std.testing.expectEqual(old.buf.items.len + 9, w.buf.items.len);
    r = .{ .b = w.buf.items };
    const on = try readWin(&r, gpa, &ids, true);
    try std.testing.expectEqual(@as(i64, 1), on.part);
    try std.testing.expectEqual(@as(u64, 9), on.gen);
    try std.testing.expectEqualSlices(i32, &.{ 5, 6 }, on.ids);
    var bad = win;
    bad.part = -1;
    try std.testing.expectError(error.BadPart, writeWin(&w, gpa, bad, true));
}

/// The device side for the lanes tests: transfers and twin binds counted.
const TestDev = struct {
    copies: usize = 0,
    binds: usize = 0,

    fn dev(d: *TestDev) kv_lanes.Dev {
        return .{ .ctx = d, .check = checkFn, .copy = copyFn, .twin = twinFn, .wait = waitFn };
    }

    fn checkFn(_: *anyopaque, _: usize, _: usize, _: usize) anyerror!void {}

    fn waitFn(_: *anyopaque, _: u64) anyerror!void {}

    fn copyFn(ctx: *anyopaque, _: usize, _: usize, _: usize) anyerror!void {
        const d: *TestDev = @ptrCast(@alignCast(ctx));
        d.copies += 1;
    }

    fn twinFn(ctx: *anyopaque, _: usize, _: bool, _: ?usize) anyerror!void {
        const d: *TestDev = @ptrCast(@alignCast(ctx));
        d.binds += 1;
    }
};

/// A stream as placeParts and releaseFn read it.
fn testStream(id: []const u8, saved: ?*Snap) lanes.Stream {
    var s: lanes.Stream = undefined;
    s.id = id;
    s.reuse = .{ .saved = if (saved) |sn| @ptrCast(sn) else null };
    s.reuse_failed = false;
    s.images = &.{};
    s.finished = false;
    return s;
}

fn testLanes(m: *Model) Lanes {
    // CPU fixture has no RDMA rings or read-ahead thread. Uninitialized
    // optionals can otherwise send releaseFn's idle parking down a GPU path.
    m.ring = null;
    m.rdma_rings = null;
    m.ahead = null;
    m.pool_cap = 64 * model.chunk_rows;
    m.cfg.window = 0;
    m.io = std.testing.io;
    return .{ .gpa = std.testing.allocator, .m = m, .peer = null };
}

fn testPut(ln: *Lanes, s: *const lanes.Stream, slot: usize, size: usize, p: anytype) !void {
    try ln.streams.put(ln.gpa, s, .{ .slot = slot, .base = p.base, .end = p.base + size, .id = p.id, .part = p.part, .gen = p.gen });
    ln.used[slot] = true;
}

test "TF_DS_KV_SHARD lanes: part-local placement, a kept state resumed in place, the planner's stales reach the Snaps" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    var m: Model = undefined;
    m.rs = .{};
    var ln = testLanes(&m);
    defer ln.snaps.deinit(gpa);
    defer ln.streams.deinit(gpa);
    var dev: TestDev = .{};
    var k = try kv_lanes.Kv.init(gpa, .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 }, .{ .twins = false }, dev.dev(), .{ .ctx = &ln, .stale = Lanes.kvStale, .moved = Lanes.kvMoved });
    defer k.deinit();
    ln.kv = &k;
    // a conversation's first turn, kept at 8 blocks, then released
    var sa = testStream("a", null);
    const pa = try ln.placeParts(&k, &sa, 10 * c, 0, 10 * c);
    try std.testing.expect(pa.kept == null and pa.base + 10 * c <= 32 * c);
    try testPut(&ln, &sa, 0, 10 * c, pa);
    var s1: Snap = .{ .id = 1, .at = 8 * c, .base = pa.base, .end = pa.base + 10 * c, .buf = undefined, .part = pa.part };
    try ln.snaps.append(gpa, &s1);
    try k.keep(1, pa.id, 8 * c);
    Lanes.releaseFn(&ln, &sa);
    try std.testing.expectEqual(@as(usize, 0), k.planner.lives.items.len);
    // its next turn resumes it in place (the extent grows into its slack: the planner's kept end follows)
    var sb = testStream("b", &s1);
    const pb = try ln.placeParts(&k, &sb, 12 * c, 1, 9 * c);
    try std.testing.expect(pb.kept == &s1 and !pb.moved and !sb.reuse_failed);
    try std.testing.expectEqual(s1.base, pb.base);
    try std.testing.expectEqual(pa.base + 12 * c, k.planner.keptAt(1).?.end);
    try testPut(&ln, &sb, 1, 12 * c, pb);
    try std.testing.expectEqual(@as(usize, 0), dev.copies); // (in place: no transfer)
    // a long stream takes the other part; the conversation ends; a request that needs part 0 whole gives up the state
    var sc = testStream("c", null);
    const pc = try ln.placeParts(&k, &sc, 30 * c, 2, 30 * c);
    try std.testing.expect(pc.part != pb.part);
    try testPut(&ln, &sc, 2, 30 * c, pc);
    Lanes.releaseFn(&ln, &sb);
    var sd = testStream("d", null);
    const pd = try ln.placeParts(&k, &sd, 30 * c, 1, 30 * c);
    try std.testing.expectEqual(pb.part, pd.part);
    try std.testing.expect(s1.stale); // (kvStale: the planner's stale op)
    try std.testing.expect(!ln.usable(&s1, 9 * c));
    // a waiting request is counted once and forgotten when the host gives it up
    var se = testStream("e", null);
    try std.testing.expectError(error.ContextFull, ln.placeParts(&k, &se, 20 * c, 0, 20 * c));
    try std.testing.expectEqual(@as(usize, 1), k.waiting.items.len);
    Lanes.releaseFn(&ln, &se);
    try std.testing.expectEqual(@as(usize, 0), k.waiting.items.len);
    try testPut(&ln, &sd, 1, 30 * c, pd);
    Lanes.releaseFn(&ln, &sc);
    Lanes.releaseFn(&ln, &sd);
    try std.testing.expectEqual(@as(usize, 0), k.planner.lives.items.len);
}

test "TF_DS_KV_SHARD lanes: a live move rewrites the moved Lane and the Snaps inside it" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    var m: Model = undefined;
    m.rs = .{};
    var ln = testLanes(&m);
    defer ln.snaps.deinit(gpa);
    defer ln.streams.deinit(gpa);
    var dev: TestDev = .{};
    var k = try kv_lanes.Kv.init(gpa, .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 }, .{ .twins = false }, dev.dev(), .{ .ctx = &ln, .stale = Lanes.kvStale, .moved = Lanes.kvMoved });
    defer k.deinit();
    ln.kv = &k;
    var sx = testStream("x", null);
    const px = try ln.placeParts(&k, &sx, 14 * c, 0, 14 * c);
    try testPut(&ln, &sx, 0, 14 * c, px);
    var sz = testStream("z", null);
    const pz = try ln.placeParts(&k, &sz, 10 * c, 1, 10 * c);
    try testPut(&ln, &sz, 1, 10 * c, pz);
    try std.testing.expect(px.part != pz.part);
    var s2: Snap = .{ .id = 2, .at = 2 * c, .base = px.base, .end = px.base + 14 * c, .buf = undefined, .part = px.part };
    try ln.snaps.append(gpa, &s2);
    try k.keep(2, px.id, 2 * c);
    try k.decodeStart(px.id);
    try k.decodeStart(pz.id);
    try k.round(&.{ .{ .id = px.id, .valid = 2 * c }, .{ .id = pz.id, .valid = 3 * c } });
    // a 24-block request fits neither part: x (two blocks used) moves, with its kept state
    var sw = testStream("w", null);
    const pw = try ln.placeParts(&k, &sw, 24 * c, 2, 24 * c);
    const lx = ln.streams.get(&sx).?;
    try std.testing.expectEqual(pz.part, lx.part);
    try std.testing.expect(lx.base != px.base and lx.gen != px.gen);
    try std.testing.expectEqual(lx.base + 14 * c, lx.end);
    try std.testing.expectEqual(lx.base, s2.base);
    try std.testing.expectEqual(lx.part, s2.part);
    try std.testing.expectEqual(px.part, pw.part);
    try std.testing.expectEqual(@as(usize, 1), dev.copies);
    try testPut(&ln, &sw, 2, 24 * c, pw);
    // the next round names the moved extent's generation (a follower checks it)
    try std.testing.expectEqual(k.planner.lives.items[k.planner.liveIndex(lx.id).?].gen, lx.gen);
}

/// A window of the pending row and `n - 1` host drafts at the stream's cache length (verifyFn's input).
fn testWindow(s: *lanes.Stream, positions: []u64, tokens: []const u32) be.Window {
    for (positions, 0..) |*p, r| p.* = s.cache_len + 1 + r;
    return .{ .stream = s, .pending = 1, .held = 0, .tokens = tokens, .parents = null, .positions = positions[0 .. 1 + tokens.len] };
}

test "TF_DS_KV_SHARD lanes: a round's windows reach the follower with part and generation; a move right after a round crossing a block copies its rows; the old generation is refused" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    var m: Model = undefined;
    m.rs = .{};
    m.dr = null;
    var ln = testLanes(&m);
    defer ln.snaps.deinit(gpa);
    defer ln.streams.deinit(gpa);
    // rank 0's Kv with an in-process follower (kv_lanes.Pair), its hooks the lanes'
    const pr = try kv_lanes.Pair.initWith(gpa, .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 }, .{ .twins = false }, .{ .ctx = &ln, .stale = Lanes.kvStale, .moved = Lanes.kvMoved });
    defer pr.deinit();
    ln.kv = &pr.kv;
    var sx = testStream("x", null);
    const px = try ln.placeParts(&pr.kv, &sx, 14 * c, 0, 14 * c);
    try testPut(&ln, &sx, 0, 14 * c, px);
    var sz = testStream("z", null);
    const pz = try ln.placeParts(&pr.kv, &sz, 10 * c, 1, 10 * c);
    try testPut(&ln, &sz, 1, 10 * c, pz);
    // x's round starts 3 rows before its second block ends and writes 6 rows into its third
    sx.cache_len = 2 * c - 3;
    sx.drafts = false;
    sz.cache_len = 3 * c;
    sz.drafts = false;
    var pos: [2][window_rows]u64 = undefined;
    const drafts = [_]u32{ 2, 3, 4, 5, 6 };
    const windows = [_]be.Window{ testWindow(&sx, &pos[0], &drafts), testWindow(&sz, &pos[1], drafts[0..1]) };
    var ids: [max_streams][max_rows]i32 = undefined;
    var wins: [max_streams]Win = undefined;
    try ln.roundWins(&windows, &ids, &wins);
    try std.testing.expectEqual(@as(usize, 2 * c + 3), pr.kv.planner.lives.items[pr.kv.planner.liveIndex(px.id).?].valid);
    try std.testing.expectEqual(@as(usize, 3 * c + 2), pr.kv.planner.lives.items[pr.kv.planner.liveIndex(pz.id).?].valid);
    try pr.same(); // (the first rounds' untwin plans reached the follower)
    // the verify frame's windows as the follower reads and checks them
    var w: Wire = .{};
    defer w.deinit(gpa);
    for (wins[0..2]) |x| try writeWin(&w, gpa, x, true);
    var r: Reader = .{ .b = w.buf.items };
    var got: std.ArrayList(i32) = .empty;
    defer got.deinit(gpa);
    var old_x: Win = undefined;
    for (0..2) |k| {
        const x = try readWin(&r, gpa, &got, true);
        try pr.fol.checkWin(x.slot, x.base, x.end, @intCast(x.part), x.gen);
        if (k == 0) old_x = x;
    }
    // a 24-block request fits neither part: x moves with every row the round wrote (whole blocks: 3, not 2)
    var sw = testStream("w", null);
    const pw = try ln.placeParts(&pr.kv, &sw, 24 * c, 2, 24 * c);
    try testPut(&ln, &sw, 2, 24 * c, pw);
    try std.testing.expectEqual(@as(usize, 1), pr.dev0.copies.items.len);
    try std.testing.expectEqual(@as(usize, 3 * c), pr.dev0.copies.items[0][2]);
    try std.testing.expectEqualSlices([3]usize, pr.dev0.copies.items, pr.dev1.copies.items);
    try pr.same();
    // the follower refuses a round naming x's old extent, and takes the next one at the new generation
    try std.testing.expectError(error.StaleExtent, pr.fol.checkWin(old_x.slot, old_x.base, old_x.end, @intCast(old_x.part), old_x.gen));
    sx.cache_len += 6;
    const next = [_]be.Window{testWindow(&sx, &pos[0], drafts[0..1])};
    try ln.roundWins(&next, &ids, &wins);
    try std.testing.expect(wins[0].gen != old_x.gen and wins[0].base != old_x.base);
    try pr.fol.checkWin(wins[0].slot, wins[0].base, wins[0].end, @intCast(wins[0].part), wins[0].gen);
    for ([_]*lanes.Stream{ &sx, &sz, &sw }) |s| Lanes.releaseFn(&ln, s);
    try pr.same();
    try std.testing.expectEqual(@as(usize, 0), pr.fol.planner.?.lives.items.len);
}

test "TF_DS_KV_SHARD lanes: a prompt pass failing after its placement (prefillFn's errdefer) frees the extent on both ranks" {
    const c = model.chunk_rows;
    const gpa = std.testing.allocator;
    var m: Model = undefined;
    m.rs = .{};
    var ln = testLanes(&m);
    defer ln.snaps.deinit(gpa);
    defer ln.streams.deinit(gpa);
    const pr = try kv_lanes.Pair.initWith(gpa, .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 }, .{}, .{ .ctx = &ln, .stale = Lanes.kvStale, .moved = Lanes.kvMoved });
    defer pr.deinit();
    ln.kv = &pr.kv;
    var sa = testStream("a", null);
    const pa = try ln.placeParts(&pr.kv, &sa, 30 * c, 0, 30 * c);
    try pr.same();
    try std.testing.expect(pr.fol.planner.?.liveIndex(pa.id) != null);
    // (prefillFn's errdefer after placeParts: a fill chunk failed, the request cancelled)
    try pr.kv.release(pa.id);
    try pr.same();
    try std.testing.expectEqual(@as(usize, 0), pr.fol.planner.?.lives.items.len);
    // slot 0 and the whole part take the next request
    var sb = testStream("b", null);
    const pb = try ln.placeParts(&pr.kv, &sb, 30 * c, 0, 30 * c);
    try testPut(&ln, &sb, 0, 30 * c, pb);
    Lanes.releaseFn(&ln, &sb);
    try pr.same();
}

test "TF_DS_KV_SHARD lanes: a carried token's bytes are the owners' rows over whole blocks of every ratio, rounded up" {
    // this checkpoint's four kv sources (ratios 2, 2, 2, 1): (3/2 + 1) x 288 = 720, the index keys not sent
    const row = shard.comp_row_bytes;
    const idx = shard.idx_row_bytes;
    try std.testing.expectEqual(@as(usize, 720), carryBytes(&.{ .{ .ratio = 2, .comp_row = row, .idx_row = idx }, .{ .ratio = 2, .comp_row = row, .idx_row = idx }, .{ .ratio = 2, .comp_row = row, .idx_row = idx }, .{ .ratio = 1, .comp_row = row, .idx_row = idx } }));
    // ratios that do not divide the row: 72 + 2.25 a token rounds up to 75 (per-layer integer division said 74)
    try std.testing.expectEqual(@as(usize, 75), carryBytes(&.{ .{ .ratio = 4, .comp_row = row, .idx_row = idx }, .{ .ratio = 128, .comp_row = row, .idx_row = 0 } }));
    // the spill tier's owner_sharded layout of the checkpoint stores the same bytes a token for its part's points
    // (native.openPair refuses a start where the two differ)
    const config = @import("config.zig");
    var why: config.Why = .{};
    const cfg = try config.parse(std.testing.allocator, config.test_config, &why);
    const lay = try spill_tier_state.StateLayout.fromConfig(&cfg, spill_tier_state.Geometry.of(&cfg, model.chunk_rows, 16, 64), .owner_sharded);
    try std.testing.expectEqual(@as(usize, 720), lay.owner_token_bytes);
}

/// A byte-array device for a follower's spill side in the composite's tests (no copy is made there: only the store
/// opens at the agreement and the handlers answer).
const MiniDev = struct {
    pub const Event = u32;
    pub fn blockPlane(_: MiniDev, _: spill_tier_state.Plane) u64 {
        return 4096;
    }
    pub fn slotPlane(_: MiniDev, _: spill_tier_state.Plane, _: usize) u64 {
        return 4096;
    }
    pub fn capacity(_: MiniDev) usize {
        return 64 * model.chunk_rows;
    }
    pub fn slots(_: MiniDev) usize {
        return 4;
    }
    pub fn d2h(_: MiniDev, _: []u8, _: u64) !void {}
    pub fn h2d(_: MiniDev, _: u64, _: []const u8) !void {}
    pub fn fill(_: MiniDev, _: u64, _: u8, _: usize) !void {}
    pub fn record(_: MiniDev, _: u32) !void {}
    pub fn done(_: MiniDev, _: u32) bool {
        return true;
    }
    pub fn wait(_: MiniDev, _: u32) void {}
    // spill_tier calls a device's failed() on an unrecoverable copy; no copy is made here, so it never runs
    pub fn failed(_: MiniDev, err: anyerror) noreturn {
        std.debug.panic("MiniDev failed: {s}", .{@errorName(err)});
    }
};
const spill_tier_state = @import("spill_state.zig");
const MiniSpill = spill_tier.Follower(MiniDev);

fn miniFx(rank: u8) fingerprint.Exchange {
    var x: fingerprint.Exchange = .{ .common = @splat(3), .rank = @splat(20 + rank) };
    @memset(&x.fields, .{ .id = 0, .scope = .common, .digest = @splat(0) });
    return x;
}

/// The follower side of the composite test: its spill side (an owner-sharded store on TF_SPILL_TEST_DIR, opened at the
/// agreement), its planner's Follower and the one rank_wire Follower over both, on its own thread.
const CompositePeer = struct {
    port: u16,
    cfg: parts.Config,
    root: [:0]const u8,
    err: ?anyerror = null,
    spill_opened: bool = false,
    has_answered: bool = false,
    digest: [32]u8 = undefined,

    fn run(p: *CompositePeer) void {
        p.loop() catch |e| {
            p.err = e;
        };
    }

    fn loop(p: *CompositePeer) !void {
        const gpa = std.testing.allocator;
        var l = try link.Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p.port, 10);
        defer l.close();
        const shapes = [_]spill_tier_state.LayerShape{ .{ .ratio = 2, .kv = true, .idx = true, .raw = true }, .{ .ratio = 1, .kv = true, .idx = true } };
        const raw = try spill_tier_state.StateLayout.init(&shapes, .{ .block_tokens = model.chunk_rows, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .owner_sharded);
        const layout = try raw.withParts(p.cfg.cap / 2);
        const settings: spill_tier.Settings = .{ .dir = p.root, .gib = 1, .min_free_gib = 0, .min_tokens = model.chunk_rows, .staging_bytes = 64 << 20, .readers = 1 };
        const plan = try spill_tier.stagingPlan(settings, &layout);
        const staging = try gpa.alignedAlloc(u8, .fromByteUnits(4096), plan.bytes());
        defer gpa.free(staging);
        const events = try gpa.alloc(u32, spill_tier.eventCount(plan));
        defer gpa.free(events);
        for (events, 0..) |*e, i| e.* = @intCast(i);
        const sf = try MiniSpill.create(gpa, std.testing.io, 1, miniFx(1), null, .{ .dev = .{}, .layout = layout, .gens = try spill_tier_state.Generations.init(gpa, p.cfg.cap, model.chunk_rows), .settings = settings, .max_tokens = @intCast(p.cfg.cap / 2), .staging = staging, .events = events });
        defer sf.destroy();
        var dev: kv_lanes.FakeDev = .{ .gpa = gpa };
        defer dev.deinit();
        const kf = try kv_lanes.Follower.init(gpa, 1, p.cfg, .{ .twins = false }, dev.dev());
        defer kf.deinit();
        kf.spill = followerSpillHooks(sf);
        const pf = try PairFollowerOf(MiniSpill).create(gpa, 1, miniFx(1), sf, kf);
        defer gpa.destroy(pf);
        const buf = try gpa.alloc(u8, rank_wire.max_frame);
        defer gpa.free(buf);
        while (true) {
            const bytes = try l.recv(buf);
            if (bytes.len == 0) break;
            try pf.onFrame(l, bytes);
            p.spill_opened = sf.side != null;
        }
        p.digest = kf.planner.?.digest();
    }
};

test "spill + shard: one rank_wire Follower serves both sides over loopback (agreement opens the spill store, shard ops reach the planner, spill asks the store)" {
    const gpa = std.testing.allocator;
    const c = model.chunk_rows;
    const core = @import("core");
    var path: [256]u8 = undefined;
    const root = try core.spill.fsio.testRoot(&path, "lanes-composite");
    const fd = try core.spill.fsio.freshDir(gpa, root);
    _ = std.c.close(fd);
    defer core.spill.fsio.dropDir(gpa, root);
    var b: [2]u8 = undefined;
    std.testing.io.random(&b);
    const port = 30000 + std.mem.readInt(u16, &b, .little) % 20000;
    const cfg: parts.Config = .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 };
    var peer: CompositePeer = .{ .port = port, .cfg = cfg, .root = root };
    const th = try std.Thread.spawn(.{}, CompositePeer.run, .{&peer});
    var joined = false;
    var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
    defer {
        if (!joined) {
            l.send(&.{}) catch {};
            th.join();
        }
        l.close();
    }
    const Out = struct {
        l: link.Link,
        fn rw(ctx: *anyopaque, sub: []const u8) anyerror!void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            try o.l.send(sub);
        }
        fn sink(o: *@This()) rank_wire.Sink {
            return .{ .ctx = o, .send = rw };
        }
    };
    var out: Out = .{ .l = l };
    const links = [_]?link.Link{l};
    // rank 0: the pair's one Leader, agreed once, then the planner over it (Lanes.startKv's order)
    const lead = try gpa.create(rank_wire.Leader);
    defer gpa.destroy(lead);
    lead.* = rank_wire.Leader.init(2, 91, miniFx(0));
    _ = try lead.agree(out.sink(), &links);
    var dev0: kv_lanes.FakeDev = .{ .gpa = gpa };
    defer dev0.deinit();
    const Rec = struct {
        fn stale(_: *anyopaque, _: u64) void {}
        fn moved(_: *anyopaque, _: u64) void {}
    };
    var k = try kv_lanes.Kv.init(gpa, cfg, .{ .twins = false }, dev0.dev(), .{ .ctx = &dev0, .stale = Rec.stale, .moved = Rec.moved });
    defer k.deinit();
    try k.lead(.{ .ctx = &out, .rw = Out.rw }, &links, lead, 0, undefined);
    try k.start(); // shard_layout (answered by the planner side), shard_set, shard_check over the same Tracker
    // a spill ask between plans: the follower's spill side answers from its own store (opened at the agreement)
    var replies: [1]rank_wire.Frame = undefined;
    _ = try lead.ask(out.sink(), &links, 0, .{ .has = .{ .point = @splat(7), .compat = lead.compat(), .at = 2 * c, .blocks = 2, .layout = .owner_sharded, .part = 1 } }, &replies);
    try std.testing.expectEqual(rank_wire.Absence.missing, replies[0].msg.has_reply.answer);
    _ = try lead.post(out.sink(), replies[0].head.txn, .{ .abort = .{ .why = .presence, .rank = 1, .forget = false } });
    // a plan with a stage and a stale op (rank 0's pending write), interleaved with a spill post: one stream
    const a = try k.admit(1, 1, 0, 10 * c, null, false);
    try k.keep(500, a.id, 8 * c);
    try k.release(a.id);
    _ = try lead.post(out.sink(), 0, .{ .write_staged = .{ .point = @splat(9), .kept = 500, .upto = 0 } });
    const f1 = try k.admit(2, 2, 1, 32 * c, null, false);
    k.planner.setPending(500, true);
    const f2 = try k.admit(3, 3, 2, 30 * c, null, false);
    try std.testing.expect(k.planner.keptAt(500).?.stale);
    _ = try lead.ask(out.sink(), &links, 0, .{ .stats = .{ .set = .tier } }, &replies);
    try std.testing.expectEqual(rank_wire.StatsSet.tier, replies[0].msg.stats_reply.set);
    for ([_]u64{ f1.id, f2.id }) |id| try k.release(id);
    try k.check();
    try l.send(&.{});
    th.join();
    joined = true;
    try std.testing.expectEqual(@as(?anyerror, null), peer.err);
    try std.testing.expect(peer.spill_opened);
    try std.testing.expectEqualSlices(u8, &k.planner.digest(), &peer.digest);
}

test "spill + shard: the composite fails closed for a side this rank does not run" {
    const gpa = std.testing.allocator;
    var dev: kv_lanes.FakeDev = .{ .gpa = gpa };
    defer dev.deinit();
    const kf = try kv_lanes.Follower.init(gpa, 1, .{ .cap = 64 * model.chunk_rows, .parts = 2, .carry_bytes = 720 }, .{}, dev.dev());
    defer kf.deinit();
    const only_kv = try PairFollowerOf(MiniSpill).create(gpa, 1, miniFx(1), null, kf);
    defer gpa.destroy(only_kv);
    const h: rank_wire.Head = .{ .op = .has, .from = 0, .epoch = 1, .txn = 1, .seq = 2 };
    try std.testing.expectError(error.Unsupported, only_kv.has(h, .{ .point = @splat(1), .compat = @splat(0), .at = 2048, .blocks = 1, .layout = .owner_sharded, .part = 0 }));
    try std.testing.expectError(error.Unsupported, only_kv.writeStaged(h, .{ .point = @splat(1), .kept = 1, .upto = 0 }));
    try only_kv.agreed(.{ .common = @splat(0), .world = 2 }); // (no spill side: nothing to open)
    const none = try PairFollowerOf(MiniSpill).create(gpa, 1, miniFx(1), null, null);
    defer gpa.destroy(none);
    try std.testing.expectError(error.Unsupported, none.shardSet(h, .{ .version = parts.layout_version, .parts = 2, .part_rows = 32 * model.chunk_rows, .owner_epoch = 1 }));
}

test "spill + shard: an owner-sharded reserve lands only on the follower planner's own hold (base, part, generation)" {
    const gpa = std.testing.allocator;
    const c = model.chunk_rows;
    const cfg: parts.Config = .{ .cap = 64 * c, .parts = 2, .carry_bytes = 720 };
    var dev: kv_lanes.FakeDev = .{ .gpa = gpa };
    defer dev.deinit();
    const kf = try kv_lanes.Follower.init(gpa, 1, cfg, .{ .twins = false }, dev.dev());
    defer kf.deinit();
    const h: rank_wire.Head = .{ .op = .shard_set, .from = 0, .epoch = 1, .txn = 1, .seq = 2 };
    try kf.shardSet(h, .{ .version = parts.layout_version, .parts = 2, .part_rows = kf.cfg.partRows(), .owner_epoch = 1 });
    // rank 0's restore plan as the follower applies it: a hold of 8 blocks in part 1 for kept id 77
    const gen = kf.planner.?.next_gen;
    const hold: kv_lanes.WireOp = .{ .hold = .{ .id = 77, .part = 1, .base = 40 * c, .size = 8 * c, .gen = gen } };
    try kf.shardOps(h, .{ .owner_epoch = 2, .index = 0, .last = true, .ops = &.{hold} });
    // a spill side not yet agreed (no store): what passes the hold check reaches it and is refused there
    const shapes = [_]spill_tier_state.LayerShape{.{ .ratio = 1, .kv = true, .idx = true }};
    const raw = try spill_tier_state.StateLayout.init(&shapes, .{ .block_tokens = c, .window = 8, .ring_extra = 4, .raw_rows = 4, .head_dim = 64, .index_head_dim = 64 }, .owner_sharded);
    const layout = try raw.withParts(32 * c);
    const sf = try MiniSpill.create(gpa, std.testing.io, 1, miniFx(1), null, .{ .dev = .{}, .layout = layout, .gens = try spill_tier_state.Generations.init(gpa, cfg.cap, c), .settings = .{ .dir = "/nonexistent", .gib = 1 }, .max_tokens = @intCast(32 * c), .staging = &.{}, .events = &.{} });
    defer sf.destroy();
    const pf = try PairFollowerOf(MiniSpill).create(gpa, 1, miniFx(1), sf, kf);
    defer gpa.destroy(pf);
    const want: rank_wire.Reserve = .{ .point = @splat(5), .kept = 77, .slot = 0, .base = 40 * c, .size = 8 * c, .at = 6 * c, .blocks = 6, .generation = gen, .part = 1 };
    try std.testing.expectError(error.SpillNotAgreed, pf.reserve(h, want)); // (the hold matched)
    var other = want;
    other.generation = gen + 1;
    try std.testing.expectError(error.StaleExtent, pf.reserve(h, other));
    other = want;
    other.base = 48 * c;
    try std.testing.expectError(error.StaleExtent, pf.reserve(h, other));
    other = want;
    other.size = 9 * c; // (an extent rank 0 never held: the hold's size is the Reserve's)
    try std.testing.expectError(error.StaleExtent, pf.reserve(h, other));
    other = want;
    other.kept = 78;
    try std.testing.expectError(error.UnknownHold, pf.reserve(h, other));
    // a replicated reserve (generation 0) asks no planner
    other = want;
    other.generation = 0;
    try std.testing.expectError(error.SpillNotAgreed, pf.reserve(h, other));
}

test "spill resume facts survive a callback that poisons and frees the source" {
    const gpa = std.testing.allocator;
    const source = try gpa.create(Snap);
    source.* = .{ .id = 1, .at = 2048, .base = 0, .buf = undefined, .restored = true };
    const facts = ResumeFacts.capture(source);
    const Callback = struct {
        fn keep(ptr: *Snap) void {
            @memset(std.mem.asBytes(ptr), 0xa5);
            std.testing.allocator.destroy(ptr);
        }
    };
    Callback.keep(source);
    const Fake = struct {
        checks: u32 = 0,
        pub fn spillCheckResumed(self: *@This(), _: usize, _: usize) !void { self.checks += 1; }
    };
    var target: Fake = .{};
    try facts.finish(&target, true, 0, 4096);
    try std.testing.expectEqual(@as(u32, 1), target.checks);
    try ResumeFacts.capture(null).finish(&target, true, 0, 4096);
    try std.testing.expectEqual(@as(u32, 1), target.checks);
}

test "idle relocation receipt validates phase rank generation exact bytes and trailing data" {
    var wire: Wire = .{};
    defer wire.deinit(std.testing.allocator);
    try wire.int(std.testing.allocator, u8, 12);
    try wire.int(std.testing.allocator, u8, 1);
    try wire.int(std.testing.allocator, u8, 2);
    try wire.int(std.testing.allocator, u64, 91);
    try wire.int(std.testing.allocator, u64, 4096);
    try validateMoveReceipt(wire.buf.items, 1, 2, 91, 4096);
    try std.testing.expectError(error.TransferFailed, validateMoveReceipt(wire.buf.items, 0, 2, 91, 4096));
    try std.testing.expectError(error.TransferFailed, validateMoveReceipt(wire.buf.items, 1, 1, 91, 4096));
    try std.testing.expectError(error.TransferFailed, validateMoveReceipt(wire.buf.items, 1, 2, 90, 4096));
    try std.testing.expectError(error.TransferFailed, validateMoveReceipt(wire.buf.items, 1, 2, 91, 4095));
    try wire.int(std.testing.allocator, u8, 0);
    try std.testing.expectError(error.TransferFailed, validateMoveReceipt(wire.buf.items, 1, 2, 91, 4096));
}

test "X2 native hint is opt-in metadata, never a restore, and ignores stale saves" {
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null, .kv_clock = 15 };
    var sn: Snap = .{ .id = 1, .at = 47518, .base = 0, .buf = undefined };
    Lanes.snapContinuationHintFn(&ln, &sn, 3742);
    try std.testing.expectEqual(@as(u64, 0), sn.continuation_at);
    ln.continuation_donors = true;
    Lanes.snapContinuationHintFn(&ln, &sn, 0);
    try std.testing.expectEqual(@as(u64, 0), sn.continuation_at);
    Lanes.snapContinuationHintFn(&ln, &sn, 3742);
    try std.testing.expectEqual(@as(u64, 15), sn.continuation_at);
    try std.testing.expectEqual(@as(u64, 3742), sn.continuation_state);
    try std.testing.expectEqual(@as(u64, 0), sn.restores);
    try std.testing.expectEqual(@as(u64, 0), sn.last_use);
    try std.testing.expectEqual(@as(u64, 15), ln.kv_clock);
    ln.kv_clock = 20;
    sn.stale = true;
    Lanes.snapContinuationHintFn(&ln, &sn, 4000);
    try std.testing.expectEqual(@as(u64, 15), sn.continuation_at);
    try std.testing.expectEqual(@as(u64, 3742), sn.continuation_state);
}

test "X3 native retention hint is opt-in CPU metadata, not a restore or device access" {
    var ln: Lanes = .{ .gpa=std.testing.allocator, .m=undefined, .peer=null, .kv_clock=15 };
    var sn: Snap = .{ .id=1, .at=87484, .base=0, .buf=undefined };
    Lanes.snapAdaptiveHintFn(&ln,&sn,400);
    try std.testing.expectEqual(@as(u32,0),sn.adaptive_density);
    ln.adaptive_prefix=true;
    Lanes.snapAdaptiveHintFn(&ln,&sn,400);
    try std.testing.expectEqual(@as(u32,400),sn.adaptive_density);
    try std.testing.expectEqual(@as(u64,15),sn.adaptive_at);
    try std.testing.expectEqual(@as(u64,0),sn.restores);
    try std.testing.expectEqual(@as(u64,0),sn.last_use);
    try std.testing.expectEqual(@as(u64,15),ln.kv_clock);
    sn.stale=true;
    ln.kv_clock=20;
    Lanes.snapAdaptiveHintFn(&ln,&sn,900);
    try std.testing.expectEqual(@as(u32,400),sn.adaptive_density);
    try std.testing.expectEqual(@as(u64,15),sn.adaptive_at);
}


test "X4 lost-state compatibility matches native parity window and exact replay" {
    try std.testing.expectEqual([2]u32{22189,std.math.maxInt(u32)},Lanes.ghostRange(22189,true,true,128,true));
    try std.testing.expectEqual([2]u32{0,22188},Lanes.ghostRange(22060,false,false,128,true));
    try std.testing.expectEqual([2]u32{22252,22253},Lanes.ghostRange(22252,true,true,128,true));
    try std.testing.expectEqual([2]u32{0,22380},Lanes.ghostRange(22252,false,true,128,false));
    try std.testing.expectEqual([2]u32{0,22253},Lanes.ghostRange(22252,false,true,128,true));
}

test "structured output: masked TP3 rows exclude padding and preserve keyed draws" {
    const V = 129408;
    const logical = 129280;
    const gpa = std.testing.allocator;
    const row = try gpa.alloc(f32, V);
    defer gpa.free(row);
    const masked_row = try gpa.alloc(f32, V);
    defer gpa.free(masked_row);
    const expected = try gpa.alloc(f32, V);
    defer gpa.free(expected);
    for (row, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 37) % 1009)) / 127.0;
    const bits = try gpa.alloc(u32, maskWords(logical));
    defer gpa.free(bits);
    @memset(bits, 0);
    for ([_]usize{ 5, 219, logical - 1 }) |id| bits[id / 32] |= @as(u32, 1) << @intCast(id % 32);
    for (expected, row, 0..) |*x, v, id| x.* = if (lanes.grammar.allows(bits, id)) v else -std.math.inf(f32);
    for ([_]?lanes.Sampling{ null, .{ .seed = 87, .temperature = 0.8, .top_k = 30, .top_p = 0.9 }, .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 0.95 } }) |s| {
        const got = try drawMasked(masked_row, row, 733, s, bits);
        try std.testing.expect(lanes.grammar.allows(bits, got));
        try std.testing.expect(got < logical);
        try std.testing.expectEqual(try draw(expected, 733, s), got);
    }
    const blank = try gpa.alloc(u32, maskWords(logical));
    defer gpa.free(blank);
    @memset(blank, 0);
    try std.testing.expectError(error.NoAllowedToken, drawMasked(masked_row, row, 733, null, blank));
    var stream = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &.{1}, .max_new = 8 });
    defer stream.deinit(gpa);
    const w: be.Window = .{ .stream = &stream, .pending = 1, .held = 0, .tokens = &.{2}, .parents = null, .positions = &.{2, 3} };
    try std.testing.expect(masksFit(w, maskWords(logical)));
    var with = w;
    with.masks = bits;
    try std.testing.expect(!masksFit(with, maskWords(logical)));
}

test "cache batch outer opcode does not collide with RDMA idle or graph vote" {
    try std.testing.expectEqual(Kind.snap_drop_many, try decodeOuterKind(cache_batch.kind));
    try std.testing.expectEqual(Kind.rdma_idle, try decodeOuterKind(11));
    try std.testing.expectEqual(Kind.graph_vote_agree, try decodeOuterKind(15));
}
test "cache batch first hint is metadata only and ignores disabled stale loading or repeated state" {
    var ln: Lanes = .{ .gpa = std.testing.allocator, .m = undefined, .peer = null, .kv_clock = 100 };
    var sn: Snap = .{ .id = 1, .at = 8192, .base = 0, .buf = undefined };
    Lanes.snapFirstHintFn(&ln, &sn);
    try std.testing.expectEqual(@as(u64, 0), sn.first_completed_at);
    ln.first_continuation = true;
    sn.stale = true;
    Lanes.snapFirstHintFn(&ln, &sn);
    sn.stale = false;
    sn.loading = true;
    Lanes.snapFirstHintFn(&ln, &sn);
    try std.testing.expectEqual(@as(u64, 0), sn.first_completed_at);
    sn.loading = false;
    Lanes.snapFirstHintFn(&ln, &sn);
    ln.kv_clock = 101;
    Lanes.snapFirstHintFn(&ln, &sn);
    try std.testing.expectEqual(@as(u64, 100), sn.first_completed_at);
    try std.testing.expectEqual(@as(u64, 0), sn.restores);
}

const CacheBatchLoopback = struct {
    heads: [2]link.Link,
    followers: [2]link.Link,
    frame: []const u8,
    count: usize,
    held: usize = 16,
    released: [3]usize = .{ 0, 0, 0 },
    received: usize = 0,
    deadline: u64,
    pub fn followerCount(_: *CacheBatchLoopback) usize { return 2; }
    pub fn send(t: *CacheBatchLoopback, rank: usize) !void {
        try std.testing.expectEqual(@as(usize, 16), t.held);
        try t.heads[rank].sendWithin(t.frame, t.deadline);
    }
    pub fn freeLocal(t: *CacheBatchLoopback) !void {
        t.released[0] = t.count;
        for (t.followers, 0..) |peer, rank| {
            var buf: [256]u8 = undefined;
            var r: Reader = .{ .b = try peer.recvBy(&buf, t.deadline) };
            try std.testing.expectEqual(Kind.snap_drop_many, try decodeOuterKind(try r.int(u8)));
            const step = try r.int(u64);
            try std.testing.expectEqual(@as(u64, 42), step);
            const batch = try cache_batch.decode(r.rest());
            try std.testing.expectEqual(t.count, batch.count);
            for (batch.slice(), 0..) |id, i| try std.testing.expectEqual(@as(u64, i + 1), id);
            t.released[rank + 1] = batch.count;
            const ack = cache_batch.receipt(step, batch.count);
            try peer.sendWithin(&ack, t.deadline);
        }
    }
    pub fn receive(t: *CacheBatchLoopback, rank: usize) !void {
        try std.testing.expectEqual(@as(usize, 16), t.held);
        var buf: [32]u8 = undefined;
        try cache_batch.checkReceipt(try t.heads[rank].recvBy(&buf, t.deadline), 42, t.count);
        t.received += 1;
    }
    pub fn commit(t: *CacheBatchLoopback) void {
        std.debug.assert(t.received == 2);
        t.held -= t.count;
    }
};
test "cache batch TP3 drop frame and receipts traverse two independent follower links" {
    var sockets: [2][2]std.c.fd_t = undefined;
    try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &sockets[0]) == 0);
    defer for (sockets[0]) |fd| { _ = std.c.close(fd); };
    try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &sockets[1]) == 0);
    defer for (sockets[1]) |fd| { _ = std.c.close(fd); };
    for ([_]usize{ 2, 16 }) |count| {
        var wire: Wire = .{};
        defer wire.deinit(std.testing.allocator);
        try wire.begin(std.testing.allocator, .snap_drop_many, 42);
        try wire.int(std.testing.allocator, u32, @intCast(count));
        for (0..count) |i| try wire.int(std.testing.allocator, u64, i + 1);
        var t: CacheBatchLoopback = .{
            .heads = .{ .{ .fd = sockets[0][0] }, .{ .fd = sockets[1][0] } },
            .followers = .{ .{ .fd = sockets[0][1] }, .{ .fd = sockets[1][1] } },
            .frame = wire.buf.items, .count = count,
            .deadline = link.Link.monotonicNs() + 5 * std.time.ns_per_s,
        };
        try cache_batch.release(&t);
        try std.testing.expectEqualSlices(usize, &.{ count, count, count }, &t.released);
        try std.testing.expectEqual(16 - count, t.held);
    }
}

// pass_keyed uses 17 (16 is reserved for the concurrent cache-batch port). Outer 14 stays reserved for the nested vote protocol.
fn writePass(w: *Wire, gpa: std.mem.Allocator, tokens: []const i64, q0: []const i64, slots: []const i64, steps: usize, keyed: ?@import("draft.zig").Keyed) !void {
    try w.int(gpa, u32, @intCast(tokens.len));
    try w.int(gpa, u32, @intCast(steps));
    for (tokens, q0, slots, 0..) |token, pos, slot, i| {
        try w.int(gpa, i64, token);
        try w.int(gpa, i64, pos);
        try w.int(gpa, i64, slot);
        if (keyed) |k| {
            try w.int(gpa, u64, k.seeds[i]);
            try w.int(gpa, u64, @bitCast(k.temps[i]));
            try w.int(gpa, u32, k.top_ks[i]);
        }
    }
}

const PassFrame = struct {
    n: usize,
    steps: usize,
    tokens: [max_streams]i64 = undefined,
    q0: [max_streams]i64 = undefined,
    slots: [max_streams]i64 = undefined,
    seeds: [max_streams]u64 = undefined,
    temps: [max_streams]f64 = undefined,
    top_ks: [max_streams]u32 = undefined,
};

fn readPass(r: *Reader, keyed: bool, enabled: bool) !PassFrame {
    if (keyed and !enabled) return error.KeyedDraftOff;
    var f: PassFrame = .{ .n = try r.int(u32), .steps = try r.int(u32) };
    if (f.n == 0 or f.n > max_streams or f.steps == 0 or f.steps > @import("draft.zig").max_block) return error.BadFrame;
    for (0..f.n) |i| {
        f.tokens[i] = try r.int(i64);
        f.q0[i] = try r.int(i64);
        f.slots[i] = try r.int(i64);
        if (keyed) {
            f.seeds[i] = try r.int(u64);
            f.temps[i] = @bitCast(try r.int(u64));
            f.top_ks[i] = try r.int(u32);
            if (f.top_ks[i] > keyed_draft.max_top_k) return error.BadFrame;
            if (!std.math.isFinite(f.temps[i]) or f.temps[i] < 0) return error.BadFrame;
            _ = try keyed_draft.firstPosition(f.q0[i]);
        }
    }
    return f;
}

test "keyed draft frame round trip through both TP3 follower links and off bytes" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const a = std.testing.allocator;
    var m: Model = undefined;
    m.world = 3;
    m.rank = 0;
    // flush uses Model.followers (the TP3 fan), not a single TP2 peer.
    m.fan = .{};
    defer m.fan.close();
    var peers: [2]link.Link = undefined;
    for (&peers, 0..) |*peer, i| {
        var pair: [2]std.c.fd_t = undefined;
        try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &pair) == 0);
        m.fan.links[i] = .{ .fd = pair[0] };
        peer.* = .{ .fd = pair[1] };
    }
    defer for (&peers) |*peer| peer.close();
    var ln: Lanes = .{ .gpa = a, .m = &m, .peer = m.fan.links[0] };
    defer ln.wire.deinit(a);
    const tokens = [_]i64{ 11, 29 };
    const q0 = [_]i64{ 100, 2000 };
    const slots = [_]i64{ 0, 7 };
    const seeds = [_]u64{ 0xffffffffffffffff, 4242 };
    const temps = [_]f64{ 0.7, 0 };
    const top_ks = [_]u32{ 20, 1023 };
    for ([_]bool{ false, true }) |keyed| {
        try ln.wire.begin(a, if (keyed) .pass_keyed else .pass, 31);
        try writePass(&ln.wire, a, &tokens, &q0, &slots, 5, if (keyed) .{ .seeds = &seeds, .temps = &temps, .top_ks = &top_ks } else null);
        if (!keyed) {
            var legacy: Wire = .{};
            defer legacy.deinit(a);
            try legacy.begin(a, .pass, 31);
            try legacy.int(a, u32, 2);
            try legacy.int(a, u32, 5);
            for (tokens, q0, slots) |t, p, slot| {
                try legacy.int(a, i64, t);
                try legacy.int(a, i64, p);
                try legacy.int(a, i64, slot);
            }
            try std.testing.expectEqualSlices(u8, legacy.buf.items, ln.wire.buf.items);
        }
        try ln.flush();
        for (&peers) |*peer| {
            var buf: [512]u8 = undefined;
            const bytes = try peer.recv(&buf);
            var r: Reader = .{ .b = bytes };
            const kind = try decodeOuterKind(try r.int(u8));
            try std.testing.expectEqual(if (keyed) Kind.pass_keyed else Kind.pass, kind);
            try std.testing.expectEqual(@as(u64, 31), try r.int(u64));
            if (keyed) {
                var disabled_reader = r;
                try std.testing.expectError(error.KeyedDraftOff, readPass(&disabled_reader, true, false));
                try std.testing.expectEqual(r.at, disabled_reader.at);
            }
            const f = try readPass(&r, keyed, true);
            try std.testing.expectEqual(@as(usize, 2), f.n);
            try std.testing.expectEqual(@as(usize, 5), f.steps);
            try std.testing.expectEqualSlices(i64, &tokens, f.tokens[0..f.n]);
            try std.testing.expectEqualSlices(i64, &q0, f.q0[0..f.n]);
            try std.testing.expectEqualSlices(i64, &slots, f.slots[0..f.n]);
            if (keyed) {
                try std.testing.expectEqualSlices(u64, &seeds, f.seeds[0..f.n]);
                try std.testing.expectEqualSlices(f64, &temps, f.temps[0..f.n]);
                try std.testing.expectEqualSlices(u32, &top_ks, f.top_ks[0..f.n]);
            }
            try std.testing.expectEqual(r.b.len, r.at);
        }
    }
    // both follower links reject a top_k the kernel cannot mirror.
    const too_large = [_]u32{ 20, 1024 };
    try ln.wire.begin(a, .pass_keyed, 32);
    try writePass(&ln.wire, a, &tokens, &q0, &slots, 5, .{ .seeds = &seeds, .temps = &temps, .top_ks = &too_large });
    try ln.flush();
    for (&peers) |*peer| {
        var buf: [512]u8 = undefined;
        var r: Reader = .{ .b = try peer.recv(&buf) };
        try std.testing.expectEqual(Kind.pass_keyed, try decodeOuterKind(try r.int(u8)));
        try std.testing.expectEqual(@as(u64, 32), try r.int(u64));
        try std.testing.expectError(error.BadFrame, readPass(&r, true, true));
    }
}

test "decode greedy head actual conditional verify wire preserves off bytes and round trips modes" {
    const a = std.testing.allocator;
    const win: Win = .{ .slot = 0, .base = 0, .end = 2048, .start = 0, .absorb = false, .ids = &.{ 7, 8 } };
    var old: Wire = .{};
    defer old.deinit(a);
    try old.int(a, u32, 1);
    try writeWin(&old, a, win, false);
    for ([_]bool{ false, true }) |on| for ([_]bool{ false, true }) |mode| {
        var wire: Wire = .{};
        defer wire.deinit(a);
        try wire.int(a, u32, 1);
        if (!on and mode) {
            try std.testing.expectError(error.BadGreedyHeadMode, writeGreedyHeadMode(&wire, a, on, mode));
            continue;
        }
        try writeGreedyHeadMode(&wire, a, on, mode);
        try writeWin(&wire, a, win, false);
        if (!on) try std.testing.expectEqualSlices(u8, old.buf.items, wire.buf.items);
        try std.testing.expectEqual(old.buf.items.len + @intFromBool(on), wire.buf.items.len);
        var reader: Reader = .{ .b = wire.buf.items };
        try std.testing.expectEqual(@as(u32, 1), try reader.int(u32));
        try std.testing.expectEqual(mode, try readGreedyHeadMode(&reader, on, false));
        var ids: std.ArrayList(i32) = .empty;
        defer ids.deinit(a);
        const got = try readWin(&reader, a, &ids, false);
        try std.testing.expectEqualSlices(i32, win.ids, got.ids);
        try std.testing.expectEqual(wire.buf.items.len, reader.at);
    };
}

test "decode greedy head actual follower mode and truncated payload refuse before collective" {
    const a = std.testing.allocator;
    var reader: Reader = .{ .b = &.{} };
    try std.testing.expectError(error.ShortFrame, readGreedyHeadMode(&reader, true, false));
    try std.testing.expect(!try readGreedyHeadMode(&reader, false, false));
    try std.testing.expectEqual(@as(usize, 0), reader.at);
    for ([_]u8{ 2, 127, 255 }) |mode| {
        reader = .{ .b = &.{mode} };
        try std.testing.expectError(error.BadGreedyHeadMode, readGreedyHeadMode(&reader, true, false));
    }
    reader = .{ .b = &.{1} };
    try std.testing.expectError(error.BadGreedyHeadMode, readGreedyHeadMode(&reader, true, true));
    reader = .{ .b = &.{0} };
    try std.testing.expect(!try readGreedyHeadMode(&reader, true, true));
    var wire: Wire = .{};
    defer wire.deinit(a);
    try writeGreedyHeadMode(&wire, a, true, true);
    try writeWin(&wire, a, .{ .slot = 0, .base = 0, .end = 2048, .start = 0, .absorb = false, .ids = &.{7} }, false);
    for (1..wire.buf.items.len) |len| {
        reader = .{ .b = wire.buf.items[0..len] };
        try std.testing.expect(try readGreedyHeadMode(&reader, true, false));
        var ids: std.ArrayList(i32) = .empty;
        defer ids.deinit(a);
        try std.testing.expectError(error.ShortFrame, readWin(&reader, a, &ids, false));
    }
}

test "decode greedy head actual custody and verify seams end links before cleanup without drain" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const Fake = struct {
        fan: @import("prompt2d.zig").Fan = .{},
        failed: bool = false,
        cleanup: bool = false,
        inject: bool = true,
        calls: usize = 0,
        pub fn verifyHead(m: *@This(), _: void, _: []const bool, _: bool) !void {
            m.calls += 1;
            if (m.inject) return error.Injected;
        }
        pub fn pairFault(m: *@This(), _: anyerror) !void {
            try std.testing.expect(!m.cleanup);
            m.failed = true;
            m.fan.close();
            return error.TestPairEnded; // production exits 75, never returns
        }
    };
    for (0..5) |stage| {
        var m: Fake = .{};
        defer m.fan.close();
        var pair: [2]std.c.fd_t = undefined;
        try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &pair) == 0);
        m.fan.links[0] = .{ .fd = pair[0] };
        var peer: link.Link = .{ .fd = pair[1] };
        defer peer.close();
        // Shared production Custody covers partial send, parser and build errors;
        // shared verify covers kernel/gather/merge/final-fence errors on either rank.
        if (stage < 3) {
            const custody: greedy_head.Custody = .{ .active = true };
            try std.testing.expectError(error.TestPairEnded, custody.failed(&m));
        } else if (stage == 3) {
            try std.testing.expectError(error.TestPairEnded, greedy_head.verify(&m, {}, &.{}, true));
        } else {
            try std.testing.expectError(error.Injected, greedy_head.verify(&m, {}, &.{}, false));
        }
        const fatal = stage != 4;
        try std.testing.expectEqual(fatal, m.failed);
        try std.testing.expectEqual(fatal, peer.peerGone());
        try std.testing.expectEqual(fatal, FollowerWatch.shouldExit(1, 1 + std.time.ns_per_s, peer.peerGone(), std.time.ns_per_s));
        m.cleanup = true; // only after the pair-fault boundary (no driver/drain exists)
    }
    var m: Fake = .{ .inject = false };
    try greedy_head.verify(&m, {}, &.{}, true);
    try std.testing.expect(!m.failed);
    try (greedy_head.Custody{}).failed(&m);
    try std.testing.expect(!m.failed);
}

test "decode greedy head sampled wire is absent when off and validates every count" {
    const a = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(a);
    try writeSampleHeadCount(&w, a, false, 0);
    try std.testing.expectEqual(@as(usize, 0), w.buf.items.len);
    try std.testing.expectError(error.BadGreedyHeadMode, writeSampleHeadCount(&w, a, false, 1));
    for (0..65) |count| {
        w.buf.clearRetainingCapacity();
        try writeSampleHeadCount(&w, a, true, count);
        var r: Reader = .{ .b = w.buf.items };
        try std.testing.expectEqual(count, try readSampleHeadCount(&r, true, false));
        try std.testing.expectEqual(@as(usize, 1), r.at);
    }
    var bad: Reader = .{ .b = &.{65} };
    try std.testing.expectError(error.BadGreedyHeadMode, readSampleHeadCount(&bad, true, false));
    var conflict: Reader = .{ .b = &.{28} };
    try std.testing.expectError(error.BadGreedyHeadMode, readSampleHeadCount(&conflict, true, true));
    var short: Reader = .{ .b = &.{} };
    try std.testing.expectError(error.ShortFrame, readSampleHeadCount(&short, true, false));
    try std.testing.expectEqual(@as(usize, 0), try readSampleHeadCount(&short, false, false));
}

test "prefill slice frame round trip is identical for both TP3 followers and abort" {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    const frames = [_]SliceFrame{
        .{ .lo = 0, .hi = 5, .args = .{ .start = 8192, .n = 8192, .replay = 32768 }, .next_rows = 8064 },
        .{ .lo = 5, .hi = 10, .args = .{ .start = 8192, .n = 8192, .replay = 32768 }, .next_rows = 8064 },
        .{ .abort = true },
    };
    for (frames, 0..) |frame, step| {
        try w.begin(gpa, .fill_slice, step);
        try putSlice(&w, gpa, frame);
        for (0..2) |_| {
            var reader: Reader = .{ .b = w.buf.items };
            try std.testing.expectEqual(Kind.fill_slice, try decodeOuterKind(try reader.int(u8)));
            try std.testing.expectEqual(@as(u64, step), try reader.int(u64));
            try std.testing.expectEqualDeep(frame, try getSlice(&reader));
        }
        try w.int(gpa, u8, 1);
        var bad: Reader = .{ .b = w.buf.items[9..] };
        try std.testing.expectError(error.BadFrame, getSlice(&bad));
    }
    try std.testing.expectEqual(@as(u8, 18), @backingInt(Kind.fill_slice));
    try std.testing.expectEqual(@as(u8, 13), @backingInt(Kind.fill_resume));
    var bad_op: Reader = .{ .b = &.{2} };
    try std.testing.expectError(error.BadFrame, getSlice(&bad_op));
}

fn useLayerSlices(yielded: bool, per: usize, sharded: bool, callback: bool) bool {
    return yielded and per > 0 and !sharded and callback;
}
test "prefill slice off and unsupported scheduler paths keep whole chunk protocol" {
    try std.testing.expect(useLayerSlices(true, 5, false, true));
    try std.testing.expect(!useLayerSlices(true, 0, false, true));
    try std.testing.expect(!useLayerSlices(false, 5, false, true));
    try std.testing.expect(!useLayerSlices(true, 5, true, true));
    try std.testing.expect(!useLayerSlices(true, 5, false, false));
}
