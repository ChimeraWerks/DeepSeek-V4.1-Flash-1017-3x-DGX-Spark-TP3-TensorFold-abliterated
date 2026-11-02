//! Wall-time debt for resumable prompt chunks, matching TensorFold PR1188/1287.
const std = @import("std");
pub const Share = struct {
    p: f64 = 1,
    debt_ns: i64 = 0,
    pub const short_rows = 512;
    pub fn fromEnv() Share {
        return fromText(std.mem.span(std.c.getenv("TENSORFOLD_PREFILL_SHARE") orelse return .{}));
    }
    pub fn fromText(text: []const u8) Share {
        const p = std.fmt.parseFloat(f64, text) catch return .{};
        return .{ .p = if (p >= 0.01 and p <= 1) p else 1 };
    }
    pub fn stepped(s: *Share, ns: i64, decoding: bool) void {
        if (s.p >= 1 or !decoding) return;
        const owed = @as(f64, @floatFromInt(@max(0, ns))) * (1 - s.p) / s.p;
        s.debt_ns +|= @intFromFloat(@min(owed, @as(f64, @floatFromInt(std.math.maxInt(i64) / 2))));
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
