//! READY/COMMIT handshake over the ordered command links. A busy rank votes
//! against parking; a missing or malformed peer is fatal to the entire world.
const std = @import("std");
const link = @import("link.zig");

pub const budget_ns = 5 * std.time.ns_per_s;

/// Upstream parks by default; exactly "0" is the explicit off switch.
pub fn enabled(value: ?[]const u8) bool {
    return if (value) |v| !std.mem.eql(u8, v, "0") else true;
}

test "parking is on by default and has an explicit off switch" {
    try std.testing.expect(enabled(null));
    try std.testing.expect(enabled("1"));
    try std.testing.expect(!enabled("0"));
}

const ready = "rdma-idle-ready-v1";
const busy = "rdma-idle-busy-v1";
const commit = "rdma-idle-commit-v1";
const cancel = "rdma-idle-cancel-v1";

/// The outer idle command has already been sent to every follower. All votes
/// arrive before any rank receives COMMIT. One BUSY cancels the whole attempt.
pub fn leader(peers: []const ?link.Link, deadline: u64) !bool {
    var all_ready = true;
    var buf: [64]u8 = undefined;
    for (peers) |maybe_peer| {
        const peer = maybe_peer orelse return error.PeerDown;
        const vote = try peer.recvBy(&buf, deadline);
        if (std.mem.eql(u8, vote, busy)) {
            all_ready = false;
        } else if (!std.mem.eql(u8, vote, ready)) return error.BadIdleVote;
    }
    for (peers) |maybe_peer| {
        const peer = maybe_peer orelse return error.PeerDown;
        try peer.sendWithin(if (all_ready) commit else cancel, deadline);
    }
    return all_ready;
}

/// A follower synchronizes CUDA and checks local background work before
/// voting. It parks only on COMMIT, never while another rank voted BUSY.
pub fn follower(peer: link.Link, can_park: bool, deadline: u64) !bool {
    try peer.sendWithin(if (can_park) ready else busy, deadline);
    var buf: [64]u8 = undefined;
    const decision = try peer.recvBy(&buf, deadline);
    if (std.mem.eql(u8, decision, cancel)) return false;
    if (!can_park or !std.mem.eql(u8, decision, commit)) return error.BadIdleCommit;
    return true;
}

test "TP3 READY/COMMIT parks only when both followers agree" {
    const c = std.c;
    for ([_]bool{ true, false }) |second_ready| {
        var pairs: [2][2]c.fd_t = undefined;
        for (&pairs) |*pair| try std.testing.expect(c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, pair) == 0);
        defer for (&pairs) |*pair| { _ = c.close(pair[0]); _ = c.close(pair[1]); };
        const Peer = struct {
            fn run(p: link.Link, yes: bool, result: *?bool) void {
                var command: [16]u8 = undefined;
                _ = p.recvBy(&command, link.Link.monotonicNs() + budget_ns) catch return;
                result.* = follower(p, yes, link.Link.monotonicNs() + budget_ns) catch null;
            }
        };
        var results: [2]?bool = .{ null, null };
        const a = try std.Thread.spawn(.{}, Peer.run, .{ link.Link{ .fd = pairs[0][1] }, true, &results[0] });
        const b = try std.Thread.spawn(.{}, Peer.run, .{ link.Link{ .fd = pairs[1][1] }, second_ready, &results[1] });
        const peers = [_]?link.Link{ .{ .fd = pairs[0][0] }, .{ .fd = pairs[1][0] } };
        for (peers) |p| try p.?.sendWithin("idle", link.Link.monotonicNs() + budget_ns);
        try std.testing.expectEqual(second_ready, try leader(&peers, link.Link.monotonicNs() + budget_ns));
        a.join();
        b.join();
        try std.testing.expectEqual(@as(?bool, second_ready), results[0]);
        try std.testing.expectEqual(@as(?bool, second_ready), results[1]);
    }
}

test "missing follower and malformed vote refuse the idle transition" {
    const c = std.c;
    var pair: [2]c.fd_t = undefined;
    try std.testing.expect(c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &pair) == 0);
    defer _ = c.close(pair[0]);
    const head: link.Link = .{ .fd = pair[0] };
    var peers = [_]?link.Link{head};
    const follower_peer: link.Link = .{ .fd = pair[1] };
    try follower_peer.send("invalid");
    try std.testing.expectError(error.BadIdleVote, leader(&peers, link.Link.monotonicNs() + budget_ns));
    _ = c.close(pair[1]);
    try std.testing.expectError(error.PeerDown, leader(&peers, link.Link.monotonicNs() + budget_ns));
}
