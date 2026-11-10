//! Park replay never creates a cache entry or enters a graph vote.
const std = @import("std");

pub const bit: u64 = @as(u64, 1) << 60;
pub fn agreement(on: bool) u64 { return if (on) bit else 0; }
pub fn check(a: u64, b: u64) !void {
    if ((a ^ b) & bit != 0) return error.ParkGraphsMismatch;
}
pub const Mode = enum { eager, normal, replay };
pub fn mode(on: bool, live: bool, coherent: bool) Mode {
    if (!live) return .normal;
    return if (on and coherent) .replay else .eager;
}
pub const Action = enum { eager, replay, capture };
pub fn action(replay_only: bool, disabled: bool, present: bool, graph: bool) Action {
    if (disabled) return .eager;
    if (present) return if (graph) .replay else .eager;
    return if (replay_only) .eager else .capture;
}

test "prefill slice park graphs agreement and capture fence" {
    try std.testing.expectEqual(@as(u64, 0), agreement(false));
    try std.testing.expectEqual(@as(u64, 1) << 60, agreement(true));
    try std.testing.expectError(error.ParkGraphsMismatch, check(0, bit));
    try std.testing.expectError(error.ParkGraphsMismatch, check(bit, 0));
    try check(bit | 7, bit | 9);
    for ([_]bool{ false, true }) |on| for ([_]bool{ false, true }) |coherent| {
        try std.testing.expectEqual(Mode.normal, mode(on, false, coherent));
        try std.testing.expectEqual(if (on and coherent) Mode.replay else Mode.eager, mode(on, true, coherent));
    };
    for ([_]bool{ false, true }) |disabled| for ([_]bool{ false, true }) |present| for ([_]bool{ false, true }) |graph| {
        try std.testing.expect(action(true, disabled, present, graph) != .capture);
    };
    try std.testing.expectEqual(Action.eager, action(true, false, false, false));
    try std.testing.expectEqual(Action.replay, action(true, false, true, true));
    try std.testing.expectEqual(Action.capture, action(false, false, false, false));
}
