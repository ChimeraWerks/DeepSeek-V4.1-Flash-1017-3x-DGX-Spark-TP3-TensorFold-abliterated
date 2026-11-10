//! Default-off removal of vocabulary rows with no Markov consumer.
const std = @import("std");
pub const agreement_bit: u64 = @as(u64, 1) << 54;

pub fn enabled() bool {
    const value = std.c.getenv("TF_DS_COMPACT_DRAFT_HEAD") orelse return false;
    return parse(std.mem.span(value));
}

pub fn parse(value: ?[]const u8) bool {
    return if (value) |v| std.mem.eql(u8, v, "1") else false;
}

pub fn compact(on: bool, block: usize, steps: usize, probe: bool) bool {
    return on and !probe and steps < block;
}

pub fn settingBit(on: bool) u64 {
    return if (on) agreement_bit else 0;
}

pub fn checkAgreement(rank: u64, leader: u64) !void {
    if ((rank ^ leader) & agreement_bit != 0) return error.CompactDraftHeadMismatch;
}

test "draft head opt in and diagnostic probes keep the old layout" {
    try std.testing.expect(!parse(null));
    for ([_][]const u8{ "", "0", "true", "01", "2" }) |v| try std.testing.expect(!parse(v));
    try std.testing.expect(parse("1"));
    for (1..9) |block| for (1..block + 1) |steps| {
        try std.testing.expect(!compact(false, block, steps, false));
        try std.testing.expect(!compact(true, block, steps, true));
        try std.testing.expectEqual(steps < block, compact(true, block, steps, false));
    };
}

test "draft head agreement checks both TP3 followers without changing off words" {
    const old: u64 = 0x800f_0000_030f_ffff;
    try std.testing.expectEqual(@as(u64, 0), old & agreement_bit);
    try std.testing.expectEqual(old, old | settingBit(false));
    for ([_]bool{ false, true }) |head| for ([_]bool{ false, true }) |one| for ([_]bool{ false, true }) |two| {
        const leader = old | settingBit(head);
        for ([_]bool{ one, two }) |f| {
            if (f == head) try checkAgreement(old | settingBit(f), leader) else
                try std.testing.expectError(error.CompactDraftHeadMismatch, checkAgreement(old | settingBit(f), leader));
        }
    };
}

test "draft head poisoned unprojected rows have no logical consumers" {
    // Bit patterns, including NaNs, survive the byte copy. Never touch the tail.
    const poison: u32 = 0x7f81dead;
    for (1..17) |streams| for (1..9) |block| for (1..block + 1) |steps| {
        if (streams * block > 80) continue;
        var source: [80]u32 = @splat(poison);
        var compacted: [80]u32 = @splat(poison);
        for (0..streams) |s| for (0..steps) |j| {
            source[s * block + j] = @intCast(1 + s * block + j);
            compacted[s * steps + j] = source[s * block + j];
        };
        for (0..steps) |j| for (0..streams) |s| {
            const consumed = compacted[s * steps + j];
            try std.testing.expect(consumed != poison);
            try std.testing.expectEqual(source[s * block + j], consumed);
        };
        for (compacted[streams * steps ..]) |v| try std.testing.expectEqual(poison, v);
    };
}

test "draft head consumer inventory requires review before any new local read" {
    const source = @embedFile("draft.zig");
    const allowed = [_][]const u8{
        "try x.e.ops.keyedNoise(x.e.s, x.dr.local + 4 * step * lw, x.block * lw, lw, mk.col0(), x.dr.nz, step, x.n);",
        "try x.e.ops.keyedFilter(x.e.s, dr.local + 4 * step * lw, x.block * lw, lw, mk.col0(), dr.nz, step, x.n,",
        "dr.local = try a.take(R * e.w.head.n * 4);",
        "try prompt.grouped(e, ch, head_rows, &.{e.w.head}, &.{head_in}, &.{d}, &.{dr.local}, &.{e.w.head.n}, &.{.f32});",
        "try e.d.check(e.d.api.cuMemsetD32Async(dr.local + (r * width + valid) * 4,",
        "try Probe.check(probe, .local, S, dr.local);",
        "try dr.mk.stepsWith(t, dr.local, dr.out, ts, N, block, steps, dr.msc, .{ .ctx = &g, .run = MarkovGather.run }, pre_tap, mid);",
    };
    var found: usize = 0;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "dr.local") == null) continue;
        try std.testing.expect(found < allowed.len);
        try std.testing.expectEqualStrings(allowed[found], std.mem.trim(u8, line, " \r"));
        found += 1;
    }
    try std.testing.expectEqual(allowed.len, found);
    // Both keyed hooks and Markov must share the compact per-stream stride.
    for ([_][]const u8{
        "var nt: NoiseTap = .{ .e = e, .dr = dr, .n = N, .block = block };",
        "var ft: FilterTap = .{ .e = e, .dr = dr, .n = N, .ts = ts, .block = block };",
        "const compact = draft_head.compact(dr.compact_head, n, steps, probe != null);",
        "const block = if (compact) steps else n;",
        "if (valid < width) for (0..head_rows) |r| {",
    }) |contract| try std.testing.expect(std.mem.indexOf(u8, source, contract) != null);
}
