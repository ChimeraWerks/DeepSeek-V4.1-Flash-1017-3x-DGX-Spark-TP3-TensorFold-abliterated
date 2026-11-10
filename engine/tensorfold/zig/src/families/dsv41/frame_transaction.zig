//! A stream step belongs to a complete frame, never to its construction.
const std = @import("std");

pub const State = enum { never_sent, partly_sent, committed };
pub const Transaction = struct {
    state: State = .never_sent,
    next_peer: usize = 0,
    peers: usize = 0,

    pub fn reset(t: *Transaction) !void {
        if (t.state == .partly_sent) return error.TransferFailed;
        t.* = .{};
    }

    pub fn preflight(t: *const Transaction, frame: []const u8, step: u64) !void {
        if (t.state != .never_sent) return error.TransferFailed;
        if (frame.len < 9 or frame.len > std.math.maxInt(u32)) return error.BadFrame;
        if (std.mem.readInt(u64, frame[1..9], .little) != step) return error.OutOfStep;
        if (step == std.math.maxInt(u64)) return error.TransferFailed;
    }

    /// Enter before calling the first transport write, including its length prefix.
    /// A failed write does not prove zero bytes reached the peer.
    pub fn enterPeer(t: *Transaction, step: *u64, index: usize, count: usize) !void {
        if (count == 0 or index >= count or index != t.next_peer) return error.TransferFailed;
        if (index == 0) {
            if (t.state != .never_sent or step.* == std.math.maxInt(u64)) return error.TransferFailed;
            t.state = .partly_sent;
            t.peers = count;
            step.* += 1;
        } else if (t.state != .partly_sent or t.peers != count) return error.TransferFailed;
    }

    pub fn sentPeer(t: *Transaction, index: usize, count: usize) void {
        std.debug.assert(t.state == .partly_sent and index == t.next_peer and count == t.peers);
        t.next_peer += 1;
        if (t.next_peer == count) t.state = .committed;
    }
};

test "frame transaction failure before and after each TP3 peer is never replayable" {
    for (0..3) |fail_after| {
        var t: Transaction = .{};
        var step: u64 = 41;
        const bytes = [_]u8{ 9, 41, 0, 0, 0, 0, 0, 0, 0 };
        try t.preflight(&bytes, step);
        var received = [_]?u64{ null, null };
        for (0..2) |peer| {
            try t.enterPeer(&step, peer, 2);
            if (fail_after == peer) break;
            received[peer] = 41;
            t.sentPeer(peer, 2);
        }
        try std.testing.expectEqual(@as(u64, 42), step);
        if (fail_after < 2) {
            try std.testing.expectEqual(State.partly_sent, t.state);
            try std.testing.expectError(error.TransferFailed, t.reset());
            try std.testing.expectError(error.TransferFailed, t.preflight(&bytes, 41));
            for (received[0..fail_after]) |seen| try std.testing.expectEqual(@as(?u64, 41), seen);
        } else {
            try std.testing.expectEqual(State.committed, t.state);
            try std.testing.expectEqualSlices(?u64, &.{ 41, 41 }, &received);
            try t.reset();
        }
    }
}

test "frame transaction preflight never consumes a step" {
    var t: Transaction = .{};
    const bytes = [_]u8{ 9, 41, 0, 0, 0, 0, 0, 0, 0 };
    try std.testing.expectError(error.BadFrame, t.preflight(bytes[0..8], 41));
    try std.testing.expectError(error.OutOfStep, t.preflight(&bytes, 42));
    try std.testing.expectEqual(State.never_sent, t.state);
    try t.reset();
    try t.preflight(&bytes, 41);
}
