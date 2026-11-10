//! TP3 opt-in penalty protocol. Legacy request frames remain unchanged.
const std = @import("std");
pub const core = @import("lanes").penalties;
pub const Settings = core.Settings;
pub const Counts = core.Counts;
pub const Params = core.Params;
/// Startup agreement word 6 bit 57 (54 compact draft head, 55 round timing, 56 prefill checkpoints, 63 keyed drafts).
pub const agreement_bit: u64 = @as(u64, 1) << 57;
pub const verify_kind: u8 = 19;
pub const prompt_kind: u8 = 20;
pub fn enabled() bool {
    if (@import("builtin").is_test) return false;
    return if (std.c.getenv("TF_DS_PENALTIES")) |s| std.mem.eql(u8, std.mem.span(s), "1") else false;
}
pub fn agreement(on: bool) u64 {
    return if (on) agreement_bit else 0;
}
pub fn agree(a: u64, b: u64) !void {
    if ((a ^ b) & agreement_bit != 0) return error.PenaltySettingMismatch;
}
pub fn get(s: ?@import("lanes").Sampling) Settings {
    return if (s) |v| v.penalties else .{};
}
pub const Request = struct { settings: Settings = .{}, prompt_len: usize = 0 };
pub fn write(w: anytype, a: std.mem.Allocator, p: Request) !void {
    try p.settings.validate();
    try w.int(a, u64, @bitCast(p.settings.presence));
    try w.int(a, u64, @bitCast(p.settings.frequency));
    try w.int(a, u64, @bitCast(p.settings.repetition));
    try w.int(a, u32, p.settings.window);
    try w.int(a, u8, @intFromBool(p.settings.include_prompt));
    try w.int(a, u64, p.prompt_len);
}
pub fn read(r: anytype) !Request {
    var p: Request = .{};
    p.settings.presence = @bitCast(try r.int(u64));
    p.settings.frequency = @bitCast(try r.int(u64));
    p.settings.repetition = @bitCast(try r.int(u64));
    p.settings.window = try r.int(u32);
    const include = try r.int(u8);
    if (include > 1) return error.BadPenalties;
    p.settings.include_prompt = include != 0;
    p.prompt_len = @intCast(try r.int(u64));
    try p.settings.validate();
    return p;
}
test "penalties startup agreement is opt in and explicit" {
    try std.testing.expectEqual(@as(u64, 0), agreement(false));
    try std.testing.expectEqual(@as(u64, 1) << 57, agreement(true));
    const taken = @import("draft_head.zig").agreement_bit | @import("round_timing.zig").agreement_bit | @import("keyed_draft.zig").agreement_bit | (@as(u64, 1) << 56) | (@as(u64, 1) << 32);
    try std.testing.expectEqual(@as(u64, 0), taken & agreement_bit);
    try agree(12, 12);
    try std.testing.expectError(error.PenaltySettingMismatch, agree(0, agreement(true)));
}

pub fn admit(s: Settings, on: bool, world: usize, two: bool, reduced: bool, images: bool) !void {
    try s.validate();
    if (!s.active()) return;
    if (!on or world != 3 or two) return error.PenaltiesUnavailable;
    if (reduced) return error.PenaltiesReducedHead;
    if (images) return error.PenaltiesImages;
}
test "penalties unsupported native paths refuse while neutral remains available" {
    const active: Settings = .{ .frequency = 1 };
    try admit(.{}, false, 2, true, true, true);
    try admit(active, true, 3, false, false, false);
    try std.testing.expectError(error.PenaltiesUnavailable, admit(active, false, 3, false, false, false));
    try std.testing.expectError(error.PenaltiesUnavailable, admit(active, true, 2, false, false, false));
    try std.testing.expectError(error.PenaltiesUnavailable, admit(active, true, 4, true, false, false));
    try std.testing.expectError(error.PenaltiesReducedHead, admit(active, true, 3, false, true, false));
    try std.testing.expectError(error.PenaltiesImages, admit(active, true, 3, false, false, true));
}
