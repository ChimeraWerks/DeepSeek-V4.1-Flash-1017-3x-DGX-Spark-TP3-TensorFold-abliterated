//! Opt-in draft allocation settings, compared by every TP rank before serving.
const std = @import("std");
pub const Settings = struct {
    weight: f64 = 0,
    first_depth: bool = false,

    pub fn parse(weight: ?[]const u8, first: ?[]const u8) !Settings {
        const w = if (weight) |s| try parseWeight(s) else 0;
        const f = if (first) |s| blk: {
            if (std.mem.eql(u8, s, "1")) break :blk true;
            if (std.mem.eql(u8, s, "0")) break :blk false;
            return error.BadFirstDraftDepth;
        } else false;
        return .{ .weight = if (w == 0) 0 else w, .first_depth = f };
    }
    pub fn words(s: Settings) [2]u64 {
        return .{ @bitCast(s.weight), @intFromBool(s.first_depth) };
    }
};
pub fn parseWeight(text: []const u8) !f64 {
    const v = std.fmt.parseFloat(f64, std.mem.trim(u8, text, " \t\r\n")) catch return error.BadDraftCal;
    if (!std.math.isFinite(v) or v < 0 or v > 1) return error.BadDraftCal;
    return v;
}
pub const invalid = [2]u64{ std.math.maxInt(u64), std.math.maxInt(u64) };
pub fn check(a: [2]u64, b: [2]u64) !void {
    if (std.mem.eql(u64, &a, &invalid) or !std.mem.eql(u64, &a, &b)) return error.DraftSettingMismatch;
}
test "draft settings default off and canonical numeric agreement" {
    const a = try Settings.parse(null, null);
    try std.testing.expectEqualSlices(u64, &.{0, 0}, &a.words());
    const b = try Settings.parse("0.05", "1");
    const c = try Settings.parse(" 0.050 ", "1");
    try check(b.words(), c.words());
    try check((try Settings.parse("-0", "0")).words(), a.words());
    try std.testing.expectError(error.DraftSettingMismatch, check(a.words(), b.words()));
    try std.testing.expectError(error.DraftSettingMismatch, check(a.words(), invalid));
    try std.testing.expectError(error.DraftSettingMismatch, check(invalid, invalid));
}
test "draft settings rejects malformed flags and weights" {
    for ([_][]const u8{ "-1", "1.1", "nan", "inf", "", "abc" }) |s|
        try std.testing.expectError(error.BadDraftCal, Settings.parse(s, null));
    for ([_][]const u8{ "", "yes", "2", " 1" }) |s|
        try std.testing.expectError(error.BadFirstDraftDepth, Settings.parse(null, s));
}
test "draft settings TP3 compares both followers independently" {
    const off = (try Settings.parse(null, null)).words();
    const on = (try Settings.parse("0.05", "1")).words();
    for ([_][3][2]u64{ .{off, off, on}, .{off, on, off} }) |ranks| {
        var mismatches: usize = 0;
        for (ranks[1..]) |rank| check(ranks[0], rank) catch { mismatches += 1; };
        try std.testing.expectEqual(@as(usize, 1), mismatches);
    }
}
