//! Wall-time debt for resumable prompt chunks, matching TensorFold PR1188/1287.
const std = @import("std");
pub const Share = struct {
    p: f64 = 1,
    debt_ns: i64 = 0,
    park_max_ns: i64 = 250 * std.time.ns_per_ms,
    pub const short_rows = 512;
    pub fn fromEnv() Share {
        var s = if (std.c.getenv("TENSORFOLD_PREFILL_SHARE")) |v| fromText(std.mem.span(v)) else Share{};
        if (std.c.getenv("TF_DS_PARK_MAX_MS")) |v| s.park_max_ns = parkMaxFromText(std.mem.span(v));
        return s;
    }
    pub fn fromText(text: []const u8) Share {
        const p = std.fmt.parseFloat(f64, text) catch return .{};
        return .{ .p = if (p >= 0.01 and p <= 1) p else 1 };
    }
    pub fn parkMaxFromText(text: []const u8) i64 {
        const ms = std.fmt.parseInt(u64, text, 10) catch return 250 * std.time.ns_per_ms;
        const bounded: u64 = @min(ms, std.math.maxInt(i64) / std.time.ns_per_ms);
        return @intCast(bounded * std.time.ns_per_ms);
    }
    pub fn stepped(s: *Share, ns: i64, decoding: bool) void {
        if (s.p >= 1 or !decoding) return;
        const owed = @as(f64, @floatFromInt(@max(0, ns))) * (1 - s.p) / s.p;
        s.debt_ns +|= @intFromFloat(@min(owed, @as(f64, @floatFromInt(std.math.maxInt(i64) / 2))));
    }
    /// A long chunk's nested decode parks are excluded from fill debt and pay it.
    pub fn sliced(s: *Share, total_ns: i64, decode_ns: i64, decoding: bool) void {
        const park = @min(@max(0, total_ns), @max(0, decode_ns));
        s.stepped(@max(0, total_ns) - park, decoding);
        s.rounded(park);
    }
    pub fn rounded(s: *Share, ns: i64) void { s.debt_ns = @max(0, s.debt_ns - @max(0, ns)); }
};
test "stepped prefill share saturation and invalid configuration" {
    var s: Share = .{ .p = 0.5 };
    s.stepped(1000, true);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s.stepped(1000, false);
    s.rounded(400);
    try std.testing.expectEqual(@as(i64, 600), s.debt_ns);
    s.rounded(800);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    for ([_][]const u8{ "nan", "inf", "0", "0.009", "2", "x" }) |v| try std.testing.expectEqual(@as(f64, 1), Share.fromText(v).p);
    var huge: Share = .{ .p = 0.01 };
    huge.stepped(std.math.maxInt(i64), true);
    huge.stepped(std.math.maxInt(i64), true);
    try std.testing.expectEqual(std.math.maxInt(i64), huge.debt_ns);
}

// UNRUN: deterministic shared occupancy arithmetic, not a native GPU timing qualification.
test "one debt account includes fill restore and pump occupancy" {
    var s: Share = .{ .p = 0.5 };
    s.stepped(10, true); // long fill
    s.stepped(40, true); // restore begin/poll
    s.stepped(50, true); // pump
    s.rounded(10);
    try std.testing.expectEqual(@as(i64, 90), s.debt_ns);
    s.rounded(90);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    s.stepped(100, false); // paused decoders do not earn future debt
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
}

test "prefill slice decode occupancy pays debt instead of counting as fill" {
    var s: Share = .{ .p = 0.5 };
    s.sliced(1000, 200, true);
    try std.testing.expectEqual(@as(i64, 600), s.debt_ns);
    s = .{ .p = 0.5 };
    s.sliced(1000, 0, true);
    try std.testing.expectEqual(@as(i64, 1000), s.debt_ns);
    s = .{ .p = 0.5 };
    s.sliced(1000, 2000, true);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    s = .{};
    s.sliced(1000, 200, true);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
}

test "prefill slice park cap configuration" {
    try std.testing.expectEqual(@as(i64, 250_000_000), (Share{}).park_max_ns);
    try std.testing.expectEqual(@as(i64, 0), Share.parkMaxFromText("0"));
    try std.testing.expectEqual(@as(i64, 123_000_000), Share.parkMaxFromText("123"));
    for ([_][]const u8{ "", "-1", "1.5", "x", "18446744073709551616" }) |v|
        try std.testing.expectEqual(@as(i64, 250_000_000), Share.parkMaxFromText(v));
    try std.testing.expectEqual(@as(i64, 9_223_372_036_854_000_000), Share.parkMaxFromText("18446744073709551615"));
}

test "prefill slice local park budgets do not double charge the chunk" {
    var s: Share = .{ .p = 0.5 };
    var park: Share = .{ .p = s.p };
    park.stepped(100, true);
    park.rounded(40);
    try std.testing.expectEqual(@as(i64, 60), park.debt_ns);
    park.rounded(80);
    try std.testing.expectEqual(@as(i64, 0), park.debt_ns);
    try std.testing.expectEqual(@as(i64, 0), s.debt_ns);
    s.sliced(420, 120, true); // 300 fill, 120 paid, including the unsliced tail
    try std.testing.expectEqual(@as(i64, 180), s.debt_ns);
}
