//! Default-off prompt scratch aliasing, adapted from bertholomus/TensorFold
//! a57deb9d470b27e1e3fab2a9494915a0049415aa (MIT), prompt.zig Regions.
//! Only addresses change. Persistent state, grouping and initialized constants
//! stay private. All users enqueue on Engine.s; no side-stream buffer is shared.
const std = @import("std");

pub const Member = enum {
    q,
    o,
    u,
    pa,
    ga,
    xf,
    pm,
    gm,
    iq,
    iq4,
    keys,
    score,
    pool,
    pidx,
    cand,
    ek,
    ekg,
    kv,
    xg,
    xu,
    xd,
    z,
    xh,
    w,
    taps,
};

pub fn region(m: Member) usize {
    return switch (m) {
        .ekg, .z, .w, .cand, .gm, .ga, .score, .xf => 0,
        .ek, .keys, .pool, .taps => 1,
        .xg, .xd, .kv, .o, .pa, .pm, .pidx, .iq => 2,
        .xu, .q, .u => 3,
        .xh, .iq4 => 4,
    };
}

/// Kernel boundaries, not host completion times. In particular sparseAttn
/// reads Q while writing O, and gate/up reads both XG and XU while writing Z.
/// The MoE gather survives endLayer until the NEXT pending_post. Engram and
/// taps post it before running; replayCut posts it through attnMixes first.
pub const Stage = enum {
    pending_post,
    engram_project,
    engram_gather,
    engram_sum,
    engram_gate,
    mix,
    query_project,
    query_norm,
    query_rope,
    kv_update,
    index_query,
    index_pack,
    index_weight,
    keyed_select,
    scored_select,
    sparse_attn,
    output_rope,
    output_a,
    output_b,
    attention_gather,
    ffn_mix,
    route,
    expert_rot,
    expert_gu,
    expert_epilogue,
    expert_down,
    expert_combine,
    moe_gather,
    head,
};

fn mask(comptime stages: []const Stage) u64 {
    var bits: u64 = 0;
    for (stages) |s| bits |= @as(u64, 1) << @backingInt(s);
    return bits;
}

const all_stages = (1 << @typeInfo(Stage).@"enum".field_names.len) - 1;
const gemms = mask(&.{ .engram_project, .query_project, .kv_update, .index_query, .output_a, .output_b, .head });

pub const Schedule = struct {
    candidate_source: usize,
    last_engram: ?usize = null,
    last_index: ?usize = null,
    first_tap: ?usize = null,

    /// The pool persists BETWEEN layers; taps persist through drafter.absorb.
    /// Refuse a reordered model rather than silently reusing their bytes.
    pub fn validate(s: Schedule) !void {
        if (s.last_engram) |last| if (last >= s.candidate_source) return error.PromptAliasLayers;
        if (s.first_tap) |first| {
            if (s.last_engram) |last| if (first <= last) return error.PromptAliasLayers;
            if (s.last_index) |last| if (first <= last) return error.PromptAliasLayers;
        }
    }
};

const Life = struct {
    stages: u64,
    first: usize = 0,
    last: usize = std.math.maxInt(usize),

    fn overlaps(a: Life, b: Life) bool {
        return a.stages & b.stages != 0 and @max(a.first, b.first) <= @min(a.last, b.last);
    }
};

/// Conservative live intervals, including inputs and outputs of the SAME
/// kernel, grouped decode-sized linears, owned gathers and the CED prepass.
fn life(m: Member, s: Schedule) Life {
    return switch (m) {
        .ekg => .{ .stages = mask(&.{ .engram_gather, .engram_sum }) },
        .z => .{ .stages = mask(&.{ .expert_gu, .expert_epilogue, .expert_down, .expert_combine }) },
        .w, .xh => .{ .stages = gemms },
        .cand, .keys => .{ .stages = mask(&.{.keyed_select}), .last = s.candidate_source -| 1 },
        .gm => .{ .stages = mask(&.{ .moe_gather, .pending_post }) },
        .ga => .{ .stages = mask(&.{ .attention_gather, .ffn_mix }) },
        .score => .{ .stages = mask(&.{.scored_select}) },
        .xf => .{ .stages = mask(&.{ .index_weight, .route }) },
        .ek => .{ .stages = if (s.last_engram != null) mask(&.{ .engram_project, .engram_gather }) else 0, .last = s.last_engram orelse 0 },
        .pool => .{ .stages = all_stages, .first = s.candidate_source, .last = s.last_index orelse 0 },
        .taps => .{ .stages = if (s.first_tap != null) all_stages else 0, .first = s.first_tap orelse 0 },
        .xg, .xu => .{ .stages = mask(&.{ .expert_rot, .expert_gu }) },
        .xd => .{ .stages = mask(&.{ .expert_epilogue, .expert_down }) },
        .kv => .{ .stages = mask(&.{ .engram_sum, .engram_gate }) },
        .o => .{ .stages = mask(&.{ .sparse_attn, .output_rope, .output_a }) },
        .pa => .{ .stages = mask(&.{ .output_b, .attention_gather }) },
        .pm => .{ .stages = mask(&.{ .expert_combine, .moe_gather }) },
        .pidx => .{ .stages = mask(&.{.scored_select}) },
        .iq => .{ .stages = mask(&.{ .index_query, .index_pack }) },
        .q => .{ .stages = mask(&.{ .query_project, .query_norm, .query_rope, .kv_update, .index_query, .index_pack, .index_weight, .keyed_select, .scored_select, .sparse_attn }) },
        .u => .{ .stages = mask(&.{ .output_a, .output_b }) },
        .iq4 => .{ .stages = mask(&.{ .index_pack, .index_weight, .keyed_select, .scored_select }) },
    };
}

/// Off: call the real Arena.take immediately, preserving its byte count, call
/// order, alignment and block policy exactly. On: reserve maxima once, then
/// bind fields in the caller's final Chunk (never pointers to copied locals).
pub const Regions = struct {
    on: bool,
    schedule: Schedule,
    size: [5]usize = @splat(0),
    bytes: usize = 0,
    total: usize = 0,
    entries: [@typeInfo(Member).@"enum".field_names.len]Entry = undefined,
    n: usize = 0,

    const Entry = struct { member: Member, ptr: *u64, bytes: usize };

    pub fn take(r: *Regions, a: anytype, m: Member, ptr: *u64, bytes: usize) !void {
        if (!r.on) {
            ptr.* = try a.take(bytes);
            return;
        }
        if (r.n == r.entries.len) return error.TooManySharedBuffers;
        r.entries[r.n] = .{ .member = m, .ptr = ptr, .bytes = bytes };
        r.n += 1;
        r.size[region(m)] = @max(r.size[region(m)], bytes);
        r.bytes += bytes;
        ptr.* = 0;
    }

    /// Always checked when enabled, even in ReleaseFast. A mistake in a new
    /// grouping or layer layout fails setup, before any prompt can be served.
    pub fn validate(r: *const Regions) !void {
        try r.schedule.validate();
        for (r.entries[0..r.n], 0..) |a, i| {
            for (r.entries[i + 1 .. r.n]) |b| {
                if (region(a.member) == region(b.member) and a.bytes != 0 and b.bytes != 0 and life(a.member, r.schedule).overlaps(life(b.member, r.schedule))) return error.PromptAliasOverlap;
            }
        }
    }

    pub fn place(r: *Regions, a: anytype) !void {
        if (!r.on) return;
        try r.validate();
        var bases: [5]u64 = @splat(0);
        for (r.size, &bases) |bytes, *base| {
            if (bytes == 0) continue;
            base.* = try a.take(bytes);
            r.total += bytes;
        }
        for (r.entries[0..r.n]) |entry| entry.ptr.* = bases[region(entry.member)];
    }
};

const FakeArena = struct {
    at: usize = 0x100000,
    used: usize = 0,
    calls: usize = 0,
    fn take(a: *FakeArena, bytes: usize) !u64 {
        const ptr = std.mem.alignForward(usize, a.at, 256);
        a.at = ptr + bytes;
        a.used += bytes;
        a.calls += 1;
        return ptr;
    }
};
const test_schedule: Schedule = .{ .candidate_source = 20, .last_engram = 14, .last_index = 36, .first_tap = 37 };

test "prefill128 prompt alias off preserves every allocation and private gap" {
    var actual: FakeArena = .{};
    var expected: FakeArena = .{};
    var r: Regions = .{ .on = false, .schedule = test_schedule };
    for (0..@typeInfo(Member).@"enum".field_names.len) |i| {
        var ptr: u64 = 0;
        const bytes = (i + 1) * 257;
        try r.take(&actual, @fromBackingInt(@intCast(i)), &ptr, bytes);
        try std.testing.expectEqual(try expected.take(bytes), ptr);
        try std.testing.expectEqual(try expected.take(19), try actual.take(19));
    }
    try r.place(&actual);
    try std.testing.expectEqualDeep(expected, actual);
    try std.testing.expectEqual(@as(usize, 0), r.n);
}

test "prefill128 prompt alias every live pair has disjoint storage at all stages" {
    const fields = @typeInfo(Member).@"enum".field_names;
    for ([_]usize{ 1, 16, 63, 64, 128, 384, 2048, 4096, 6144, 8192 }) |rows| {
        var r: Regions = .{ .on = true, .schedule = test_schedule };
        var ptrs: [fields.len]u64 = @splat(0);
        var a: FakeArena = .{};
        for (0..fields.len) |i| try r.take(&a, @fromBackingInt(@intCast(i)), &ptrs[i], rows * (i + 1) * 256);
        try r.place(&a);
        try std.testing.expectEqual(@as(usize, 5), a.calls);
        try std.testing.expect(r.total < r.bytes);
        for (0..40) |layer| {
            for (0..@typeInfo(Stage).@"enum".field_names.len) |stage| {
                for (0..fields.len) |i| {
                    const la = life(@fromBackingInt(@intCast(i)), test_schedule);
                    if (la.stages & (@as(u64, 1) << @as(u6, @intCast(stage))) != 0 and layer >= la.first and layer <= la.last) {
                        for (0..fields.len) |j| {
                            if (j > i) {
                                const lb = life(@fromBackingInt(@intCast(j)), test_schedule);
                                if (lb.stages & (@as(u64, 1) << @as(u6, @intCast(stage))) != 0 and layer >= lb.first and layer <= lb.last) {
                                    const ba = rows * (i + 1) * 256;
                                    const bb = rows * (j + 1) * 256;
                                    try std.testing.expect(ptrs[i] + ba <= ptrs[j] or ptrs[j] + bb <= ptrs[i]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

test "prefill128 prompt alias rejects lifetime and ordering faults" {
    try std.testing.expect(life(.q, test_schedule).overlaps(life(.o, test_schedule)));
    try std.testing.expect(life(.xg, test_schedule).overlaps(life(.xu, test_schedule)));
    try std.testing.expect(life(.gm, test_schedule).stages & mask(&.{.pending_post}) != 0);
    var r: Regions = .{ .on = true, .schedule = test_schedule };
    var a: u64 = 0;
    var b: u64 = 0;
    var arena: FakeArena = .{};
    try r.take(&arena, .gm, &a, 256);
    try r.take(&arena, .gm, &b, 256);
    try std.testing.expectError(error.PromptAliasOverlap, r.validate());
    try std.testing.expectError(error.PromptAliasLayers, (Schedule{ .candidate_source = 20, .last_engram = 20 }).validate());
    try std.testing.expectError(error.PromptAliasLayers, (Schedule{ .candidate_source = 20, .last_index = 37, .first_tap = 37 }).validate());
}

test "prefill128 prompt alias byte budget matches TP3 native scratch sizes" {
    const Scratch = @import("exl3_experts.zig").Scratch;
    for ([_]usize{ 2048, 4096, 8192 }, [_]usize{ 1613, 2806, 5192 }) |rows, saved_mib| {
        const sc = Scratch.sizes(rows, 7, 5120, 768, 385);
        const sizes = [_]usize{
            rows * 24 * 512 * 2, rows * 24 * 512 * 2,  rows * 3 * 1024 * 2,
            rows * 5120 * 4,     rows * 3 * 5120 * 4,  rows * 5120 * 4,
            rows * 5120 * 4,     rows * 3 * 5120 * 4,  rows * 32 * 128 * 2,
            rows * 32 * 128 * 2, (1 << 24) * 8,        (1 << 24) * 4,
            rows * 131328,       rows * 2048 * 8,      512 * 512 * 64 * 8,
            rows * 25600 * 4,    rows * 3 * 25600 * 4, rows * 25600 * 2,
            sc[0],               sc[1],                sc[2],
            sc[3],               rows * 5120 * 2,      2048 * 25600 * 2,
            3 * rows * 5120 * 2,
        };
        var ptrs: [sizes.len]u64 = @splat(0);
        var r: Regions = .{ .on = true, .schedule = test_schedule };
        var arena: FakeArena = .{};
        for (sizes, 0..) |bytes, i| try r.take(&arena, @fromBackingInt(@intCast(i)), &ptrs[i], bytes);
        try r.place(&arena);
        try std.testing.expectEqual(saved_mib << 20, r.bytes - r.total);
        try std.testing.expectEqual(rows * (307200 + 131328 + 71680 + 71680 + 10240), r.total);
        // Every logical view retains its full extent. In particular these
        // do not rely on an owned third, a 128-row tail, or a PIDX row cap.
        for ([_]Member{ .ga, .gm, .ekg, .pool, .pidx }) |m| {
            const i = @backingInt(m);
            try std.testing.expect(r.size[region(m)] >= sizes[i]);
            try std.testing.expectEqual(sizes[i], r.entries[i].bytes);
            // No replay-dependent rebinding: all views of a region start at
            // its one setup allocation, including later full-cap decoders.
            for (r.entries[0..r.n]) |other| {
                if (region(other.member) == region(m)) try std.testing.expectEqual(ptrs[i], other.ptr.*);
            }
        }
    }
}
