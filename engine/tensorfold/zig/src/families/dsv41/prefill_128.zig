//! LakyFx 1.2.8 prefill policy adapted to our owned-row and spill engine.
//! Pure planning is shared by the scheduler and Engram read-ahead; no GPU work.
const std = @import("std");

/// Keep physical KV, spill blocks, snapshot keys and growth independent of compute.
pub const cache_quantum: usize = 2048;
pub const legacy_rows: usize = 2048;
pub const ac_rows: usize = 4096;
/// Largest compute window TF_DS_PREFILL_COMPUTE_ROWS may pick (Owen's TP3 6K/8K budgets).
pub const max_rows: usize = 8192;

pub const Flags = struct {
    ae: bool = false,
    w: bool = false,
    k: bool = false,
    ac: bool = false,
    af: bool = false,
    e: bool = false,
    f: bool = false,
    d: bool = false,
    /// TF_DS_PREFILL_W_RANGE16: qualify W's (2,16) templates; requires W.
    w_range16: bool = false,
    /// TF_DS_PREFILL_SHARED_SPLIT: routed W plus original shared; requires W.
    shared_split: bool = false,
    /// TF_DS_PREFILL_COMPUTE_ROWS: an AC compute window other than 4096; 0 keeps AC's 4096.
    rows: usize = 0,
    /// TF_DS_PROMPT_ALIAS: five non-overlapping prompt scratch lifetimes.
    prompt_alias: bool = false,

    pub fn fromEnv() Flags {
        return .{ .ae = enabled("TF_DS_PREFILL_AE"), .w = enabled("TF_DS_PREFILL_W"), .k = enabled("TF_DS_PREFILL_K"), .ac = enabled("TF_DS_PREFILL_AC"), .af = enabled("TF_DS_PREFILL_AF"), .e = enabled("TF_DS_PREFILL_E"), .f = enabled("TF_DS_PREFILL_F"), .d = enabled("TF_DS_PREFILL_D"), .rows = envRows("TF_DS_PREFILL_COMPUTE_ROWS"), .w_range16 = enabled("TF_DS_PREFILL_W_RANGE16"), .prompt_alias = enabled("TF_DS_PROMPT_ALIAS"), .shared_split = enabled("TF_DS_PREFILL_SHARED_SPLIT") };
    }

    /// A window needs AC, a whole number of 2K cache quanta and at most max_rows.
    pub fn validate(f: Flags) !void {
        if (f.rows == 0) return;
        if (!f.ac or f.rows % cache_quantum != 0 or f.rows < ac_rows or f.rows > max_rows) return error.BadPrefillComputeRows;
    }

    /// Check only after rank agreement, so a bad dependency ends every rank.
    pub fn validateRagged(f: Flags) !void {
        if (f.w_range16 and !f.w) return error.WRange16RequiresW;
        if (f.shared_split and !f.w) return error.SharedSplitRequiresW;
    }

    pub fn computeRows(f: Flags) usize {
        return if (!f.ac) legacy_rows else if (f.rows != 0) f.rows else ac_rows;
    }

    /// Bits 8-11 carry the window; 15 agrees aliasing, 16 W range16, 17 shared split.
    /// Bits 12-14 belong to other experiments. Off keeps every existing key.
    pub fn bits(f: Flags) u64 {
        return @as(u64, @intFromBool(f.ae)) | (@as(u64, @intFromBool(f.w)) << 1) | (@as(u64, @intFromBool(f.k)) << 2) | (@as(u64, @intFromBool(f.ac)) << 3) | (@as(u64, @intFromBool(f.af)) << 4) | (@as(u64, @intFromBool(f.e)) << 5) | (@as(u64, @intFromBool(f.f)) << 6) | (@as(u64, @intFromBool(f.d)) << 7) | (@as(u64, @intCast(f.rows / cache_quantum & 0xf)) << 8) | (@as(u64, @intFromBool(f.w_range16)) << 16) | (@as(u64, @intFromBool(f.prompt_alias)) << 15) | (@as(u64, @intFromBool(f.shared_split)) << 17);
    }
};

/// An unparsable value reads as an impossible window, which validate rejects.
fn envRows(comptime name: [:0]const u8) usize {
    const v = std.c.getenv(name) orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1;
}

pub fn enabled(comptime name: [:0]const u8) bool {
    return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
}

/// Exactly the next chunk, including the replay cut and our snapshot boundaries.
/// Marks retain their 2K cache grid even when the compute window grows to 4K.
pub fn nextRows(total: usize, start: usize, replay: usize, compute_rows: usize, marks: []const u32) usize {
    if (start >= total) return 0;
    const last_end = std.mem.alignBackward(usize, @min(replay, total - 1), 2);
    var end = if (start < last_end) last_end else total;
    // A raw mark snapshots at its 2K floor (lanes.markEnds), so cut there too.
    for (marks) |mark| {
        const cut = mark / cache_quantum * cache_quantum;
        if (cut > start and cut < end) end = cut;
    }
    return @min(compute_rows, end - start);
}

// LakyFx's AE replay-tail tests, extended for both compute sizes and our marks.
test "prefill128 AE predicts replay tails and snapshot cuts" {
    try std.testing.expectEqual(@as(usize, 4096), nextRows(32768, 24576, 32640, 4096, &.{}));
    try std.testing.expectEqual(@as(usize, 3968), nextRows(32768, 28672, 32640, 4096, &.{}));
    try std.testing.expectEqual(@as(usize, 128), nextRows(32768, 32640, 32640, 4096, &.{}));
    try std.testing.expectEqual(@as(usize, 2048), nextRows(32768, 4096, 32640, 4096, &.{ 6144, 8192 }));
    // An unaligned raw mark still stops a 4K window at its 2K snapshot cut.
    try std.testing.expectEqual(@as(usize, 2048), nextRows(16384, 4096, 16256, 4096, &.{7000}));
    try std.testing.expectEqual(@as(usize, 0), nextRows(0, 0, 0, 2048, &.{}));
    try std.testing.expectEqual(@as(usize, 0), nextRows(17, 17, 0, 2048, &.{}));
    for ([_]usize{ 2048, 4096, 6144, 8192 }) |compute| {
        for ([_]usize{ 1, 17, 1024, 2048, 4096, 8192, 32768, 65536, 131072, 301001, 401408 }) |total| {
            for ([_]usize{ 0, 1, 64, 128, 4095 }) |tail| {
                const replay = total - @min(total, tail);
                const last_end = std.mem.alignBackward(usize, @min(replay, total - 1), 2);
                const marks = [_]u32{ 2048, 6144, 7000, 16384, 32768 };
                var at: usize = 0;
                while (at < total) {
                    var end = if (at < last_end) @min(total, last_end) else total;
                    for (marks) |mark| {
                        const cut = mark / cache_quantum * cache_quantum;
                        if (cut > at and cut < end) end = cut;
                    }
                    const n = @min(compute, end - at);
                    try std.testing.expectEqual(n, nextRows(total, at, replay, compute, &marks));
                    try std.testing.expect(n > 0);
                    at += n;
                }
            }
        }
    }
}

test "prefill128 flags default off with independent rank agreement bits" {
    try std.testing.expectEqual(@as(u64, 0), (Flags{}).bits());
    try std.testing.expectEqual(@as(u64, 1), (Flags{ .ae = true }).bits());
}

test "prefill128 shared split defaults off and agrees bit 17 after W dependency" {
    try std.testing.expect(!(Flags{}).shared_split);
    try std.testing.expectError(error.SharedSplitRequiresW, (Flags{ .shared_split = true }).validateRagged());
    try (Flags{ .w = true, .shared_split = true }).validateRagged();
    try (Flags{ .w = true, .shared_split = true, .w_range16 = true }).validateRagged();
    try std.testing.expectEqual(@as(u64, 1 << 17), (Flags{ .shared_split = true }).bits());
    for (0..256) |mask| for ([_]usize{ 0, 6144, 8192 }) |rows| {
        var f: Flags = .{ .ae = mask & 1 != 0, .w = mask & 2 != 0, .k = mask & 4 != 0, .ac = mask & 8 != 0, .af = mask & 16 != 0, .e = mask & 32 != 0, .f = mask & 64 != 0, .d = mask & 128 != 0, .rows = rows, .prompt_alias = true, .w_range16 = true };
        const old = f.bits();
        const compute = f.computeRows();
        f.shared_split = true;
        try std.testing.expectEqual(old | (1 << 17), f.bits());
        try std.testing.expectEqual(compute, f.computeRows());
        try std.testing.expectEqual(@as(u64, 0), f.bits() & (7 << 12));
    };
}

test "prefill128 W range16 requires W and leaves all legacy bits unchanged" {
    try std.testing.expect(!(Flags{}).w_range16);
    try (Flags{}).validateRagged();
    try (Flags{ .w = true, .w_range16 = true }).validateRagged();
    try std.testing.expectError(error.WRange16RequiresW, (Flags{ .w_range16 = true }).validateRagged());
    const bit: u64 = 1 << 16;
    try std.testing.expectEqual(bit, (Flags{ .w_range16 = true }).bits());
    for (0..256) |mask| {
        for ([_]usize{ 0, 4096, 6144, 8192 }) |rows| {
            var f: Flags = .{ .ae = mask & 1 != 0, .w = mask & 2 != 0, .k = mask & 4 != 0, .ac = mask & 8 != 0, .af = mask & 16 != 0, .e = mask & 32 != 0, .f = mask & 64 != 0, .d = mask & 128 != 0, .rows = rows };
            const legacy: u64 = mask | (rows / cache_quantum << 8);
            try std.testing.expectEqual(legacy, f.bits());
            const compute = f.computeRows();
            f.w_range16 = true;
            try std.testing.expectEqual(legacy | bit, f.bits());
            try std.testing.expectEqual(compute, f.computeRows());
        }
    }
}

/// AF reserves the request's verification tail but never reduces physical KV.
/// Our shard.plan already bounds request_cap by the agreed window, unlike
/// Laky's 1.2.7 pool-sized scratch. Keep that existing optimization intact.
pub fn workspaceTokens(request_cap: usize, context: usize, verify_rows: usize) !usize {
    if (request_cap == 0 or context > request_cap) return error.BadPromptContext;
    if (context == 0) return request_cap;
    const margin = @addWithOverflow(verify_rows, 2);
    if (margin[1] != 0) return error.BadPromptContext;
    const tail = @addWithOverflow(context, margin[0]);
    if (tail[1] != 0) return error.BadPromptContext;
    const padded = @addWithOverflow(tail[0], cache_quantum - 1);
    if (padded[1] != 0) return error.BadPromptContext;
    return @min(request_cap, padded[0] & ~@as(usize, cache_quantum - 1));
}

test "prefill128 AC changes compute without touching KV spill grid or growth" {
    try std.testing.expectEqual(@as(usize, 2048), (Flags{}).computeRows());
    try std.testing.expectEqual(@as(usize, 4096), (Flags{ .ac = true }).computeRows());
    try std.testing.expectEqual(@as(usize, 2048), cache_quantum);
    try std.testing.expectEqual(@as(usize, 32768), 16 * cache_quantum);
    // A retained 2K snapshot must still end a 4K compute chunk at its exact mark.
    try std.testing.expectEqual(@as(usize, 2048), nextRows(16384, 0, 16256, 4096, &.{2048}));
    const stride = std.mem.alignForward(usize, (ac_rows + 2) / 3, 16);
    const expected = [_]usize{ 1376, 1376, 1344 };
    for (expected, 0..) |n, rank| try std.testing.expectEqual(n, @min(stride, ac_rows - rank * stride));
}

test "prefill128 compute rows picks 6K and 8K windows only with AC" {
    try std.testing.expectEqual(@as(usize, 6144), (Flags{ .ac = true, .rows = 6144 }).computeRows());
    try std.testing.expectEqual(@as(usize, 8192), (Flags{ .ac = true, .rows = 8192 }).computeRows());
    try (Flags{ .ac = true, .rows = 6144 }).validate();
    try (Flags{}).validate();
    try std.testing.expectError(error.BadPrefillComputeRows, (Flags{ .rows = 6144 }).validate());
    try std.testing.expectError(error.BadPrefillComputeRows, (Flags{ .ac = true, .rows = 7000 }).validate());
    try std.testing.expectError(error.BadPrefillComputeRows, (Flags{ .ac = true, .rows = 10240 }).validate());
    try std.testing.expectError(error.BadPrefillComputeRows, (Flags{ .ac = true, .rows = 2048 }).validate());
    try std.testing.expectError(error.BadPrefillComputeRows, (Flags{ .ac = true, .rows = 1 }).validate());
    // Default AC keeps its old agreement bits; explicit windows differ from it and each other.
    try std.testing.expectEqual(@as(u64, 8), (Flags{ .ac = true }).bits());
    try std.testing.expect((Flags{ .ac = true, .rows = 6144 }).bits() != (Flags{ .ac = true, .rows = 8192 }).bits());
    // 2K snapshot marks still cut a wide window; owned thirds stay 16-row aligned.
    try std.testing.expectEqual(@as(usize, 6144), nextRows(65536, 0, 65408, 8192, &.{6144}));
    for ([_][3]usize{ .{ 2048, 2048, 2048 }, .{ 2736, 2736, 2720 } }, [_]usize{ 6144, 8192 }) |expected, rows| {
        const stride = std.mem.alignForward(usize, (rows + 2) / 3, 16);
        for (expected, 0..) |n, rank| try std.testing.expectEqual(n, @min(stride, rows - rank * stride));
    }
}

test "prefill128 AF covers full requests and rejects overflow without pool truncation" {
    try std.testing.expectEqual(@as(usize, 403456), try workspaceTokens(7000000, 401408, 48));
    try std.testing.expectEqual(@as(usize, 7000000), try workspaceTokens(7000000, 0, 48));
    // Our actual configured 1M context is not Laky's 401408-token context.
    try std.testing.expectEqual(@as(usize, 1050624), try workspaceTokens(1050624, 1048576, 48));
    try std.testing.expectEqual(@as(usize, 401408), try workspaceTokens(401408, 401408, 48));
    for ([_]usize{ 1, 1024, 32768, 301001, 401408, 1048576 }) |context| {
        const cap = try workspaceTokens(7000000, context, 48);
        try std.testing.expect(cap >= context + 50 and cap <= 7000000 and cap % cache_quantum == 0);
    }
    try std.testing.expectError(error.BadPromptContext, workspaceTokens(0, 0, 48));
    try std.testing.expectError(error.BadPromptContext, workspaceTokens(401408, 401409, 48));
    try std.testing.expectError(error.BadPromptContext, workspaceTokens(std.math.maxInt(usize), std.math.maxInt(usize), 48));
    try std.testing.expectError(error.BadPromptContext, workspaceTokens(7000000, 1, std.math.maxInt(usize)));
}

/// LakyFx E: an incomplete candidate pool must never survive the CED cut.
/// Retain its >=768-row qualification even though our row ownership starts at384.
pub fn indexEligible(enabled_flag: bool, owned: bool, rows: usize, compressed: usize, source: usize, layers: usize) bool {
    return enabled_flag and owned and rows >= 768 and compressed >= 512 and source >= layers / 2;
}

pub fn attentionNeedsFullX(has_index: bool, has_compressor: bool) bool {
    return has_index or has_compressor;
}

test "prefill128 E preserves complete candidate masks across CED" {
    try std.testing.expect(indexEligible(true, true, 4096, 512, 32, 64));
    try std.testing.expect(!indexEligible(false, true, 4096, 512, 32, 64));
    try std.testing.expect(!indexEligible(true, false, 4096, 512, 32, 64));
    try std.testing.expect(!indexEligible(true, true, 767, 512, 32, 64));
    try std.testing.expect(!indexEligible(true, true, 4096, 511, 32, 64));
    try std.testing.expect(!indexEligible(true, true, 4096, 512, 31, 64));
}

test "prefill128 F skips only attention layers without full X consumers" {
    try std.testing.expect(!attentionNeedsFullX(false, false));
    try std.testing.expect(attentionNeedsFullX(true, false));
    try std.testing.expect(attentionNeedsFullX(false, true));
    try std.testing.expect(attentionNeedsFullX(true, true));
}

test "prefill128 every new flag has an independent cross-rank bit" {
    const singles = [_]Flags{ .{ .ae = true }, .{ .w = true }, .{ .k = true }, .{ .ac = true }, .{ .af = true }, .{ .e = true }, .{ .f = true }, .{ .d = true } };
    var combined: u64 = 0;
    for (singles, 0..) |flag, bit| {
        const expected = @as(u64, 1) << @as(u6, @intCast(bit));
        try std.testing.expectEqual(expected, flag.bits());
        combined |= flag.bits();
    }
    try std.testing.expectEqual(@as(u64, 255), combined);
}

/// A rank-local failure during optional qualification cannot leave peers in
/// the following collective, including when host reference allocation fails.
pub fn qualificationFault(err: anyerror) anyerror {
    return switch (err) {
        error.MaskedAttentionNumericalMismatch, error.RaggedMmaNumericalMismatch => err,
        else => error.PrefillQualificationFailed,
    };
}

test "prefill128 qualification allocation and CUDA failures end the pair" {
    try std.testing.expectEqual(@as(anyerror, error.PrefillQualificationFailed), qualificationFault(error.OutOfMemory));
    try std.testing.expectEqual(@as(anyerror, error.PrefillQualificationFailed), qualificationFault(error.CudaFailure));
    try std.testing.expectEqual(@as(anyerror, error.RaggedMmaNumericalMismatch), qualificationFault(error.RaggedMmaNumericalMismatch));
    try std.testing.expectEqual(@as(anyerror, error.MaskedAttentionNumericalMismatch), qualificationFault(error.MaskedAttentionNumericalMismatch));
}

test "prefill128 prompt alias is default off with independent agreement bit" {
    try std.testing.expect(!(Flags{}).prompt_alias);
    try std.testing.expectEqual(@as(u64, 1 << 15), (Flags{ .prompt_alias = true }).bits());
    const old = Flags{ .ae = true, .w = true, .k = true, .ac = true, .af = true, .e = true, .f = true, .d = true, .rows = max_rows };
    var aliased = old;
    aliased.prompt_alias = true;
    try std.testing.expectEqual(@as(u64, 1279), old.bits());
    try std.testing.expectEqual(old.bits() | (1 << 15), aliased.bits());
    try std.testing.expectEqual(@as(u64, 0), aliased.bits() & (7 << 12));
    try std.testing.expectEqual(old.computeRows(), aliased.computeRows());
    try aliased.validate();
    // Existing combinations and explicit compute windows retain their bytes.
    for (0..256) |bits| {
        for ([_]usize{ 0, ac_rows, 6144, max_rows }) |rows| {
            var f: Flags = .{ .ae = bits & 1 != 0, .w = bits & 2 != 0, .k = bits & 4 != 0, .ac = bits & 8 != 0, .af = bits & 16 != 0, .e = bits & 32 != 0, .f = bits & 64 != 0, .d = bits & 128 != 0, .rows = rows };
            const before = f.bits();
            try std.testing.expectEqual(@as(u64, bits | (rows / cache_quantum << 8)), before);
            f.prompt_alias = true;
            try std.testing.expectEqual(before | (1 << 15), f.bits());
            try std.testing.expectEqual(@as(u64, 0), f.bits() & (7 << 12));
        }
    }
}
