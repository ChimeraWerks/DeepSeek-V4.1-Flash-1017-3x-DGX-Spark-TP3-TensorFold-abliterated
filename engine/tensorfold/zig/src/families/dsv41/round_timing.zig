//! Opt-in steady-state CPU round timing, independent of diagnostic logging.
const std = @import("std");

pub const agreement_bit: u64 = @as(u64, 1) << 55; // 54 is the compact draft head
pub fn enabledText(text: ?[]const u8) bool {
    return if (text) |v| std.mem.eql(u8, v, "1") else false;
}
pub fn enabled() bool {
    return enabledText(if (std.c.getenv("TF_DS_ROUND_TIMING")) |v| std.mem.span(v) else null);
}
pub fn agreement(on: bool) u64 {
    return if (on) agreement_bit else 0;
}
pub fn check(leader: u64, follower: u64) !void {
    if ((leader ^ follower) & agreement_bit != 0) return error.RoundTimingMismatch;
}

/// Before the core starts its clock: finish the preceding prompt's readers and GPU queue.
/// The round's existing verify/pass readbacks provide its ending fence.
pub fn begin(on: bool, model: anytype) !void {
    if (!on) return;
    model.joinAhead();
    try model.stream.synchronize();
}

test "round timing is opt in with an unchanged off agreement word" {
    try std.testing.expect(!enabledText(null));
    for ([_][]const u8{ "", "0", "true", "01", " 1", "1 " }) |v| try std.testing.expect(!enabledText(v));
    try std.testing.expect(enabledText("1"));
    const old: u64 = 0x803f_ffff_ffff_ffff;
    try std.testing.expectEqual(@as(u64, 0), old & agreement_bit);
    try std.testing.expectEqual(old, old | agreement(false));
    try check(old, old);
    try check(old | agreement(true), old | agreement(true));
    try std.testing.expectError(error.RoundTimingMismatch, check(old, old | agreement(true)));
    try std.testing.expectError(error.RoundTimingMismatch, check(old | agreement(true), old));
}

test "round timing fences prior queued work before the caller's clock" {
    const Fake = struct {
        const Stream = struct {
            clock: *u64,
            queued: *u64,
            joins: *u32,
            calls: u32 = 0,
            fail: bool = false,
            pub fn synchronize(s: *@This()) !void {
                try std.testing.expectEqual(@as(u32, 1), s.joins.*);
                s.calls += 1;
                if (s.fail) return error.DeviceFault;
                s.clock.* += s.queued.*;
                s.queued.* = 0;
            }
        };
        joins: *u32,
        stream: Stream,
        pub fn joinAhead(m: *@This()) void { m.joins.* += 1; }
    };
    var clock: u64 = 0;
    var queued: u64 = 100;
    var joins: u32 = 0;
    var m: Fake = .{ .joins = &joins, .stream = .{ .clock = &clock, .queued = &queued, .joins = &joins } };
    try begin(false, &m);
    try std.testing.expectEqual(@as(u32, 0), joins);
    try std.testing.expectEqual(@as(u32, 0), m.stream.calls);
    try std.testing.expectEqual(@as(u64, 100), queued);
    try begin(true, &m);
    const start = clock;
    clock += 20; // synchronized round completion
    try std.testing.expectEqual(@as(u64, 20), clock - start);
    try std.testing.expectEqual(@as(u64, 0), queued);
    joins = 0;
    m.stream.fail = true;
    try std.testing.expectError(error.DeviceFault, begin(true, &m));
}
