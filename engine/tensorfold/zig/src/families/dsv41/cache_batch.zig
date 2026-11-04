//! Laky 4.0.2 bounded dead-checkpoint release adapted to our TP3 protocol.
//! Opcode 11 is RDMA idle here. No CUDA work or storage allocation in this module.
const std = @import("std");
pub const kind: u8 = 16;
pub const max_count = 16;
pub const Flags = struct {
    batch_dead: bool = false,
    first_continuation: bool = false,
    pub fn fromEnv() Flags {
        return .{ .batch_dead = enabled("TF_CACHE_BATCH_DEAD"), .first_continuation = enabled("TF_CACHE_FIRST_CONTINUATION") };
    }
    pub fn bits(f: Flags) u64 {
        return @as(u64, @intFromBool(f.batch_dead)) | (@as(u64, @intFromBool(f.first_continuation)) << 1);
    }
    /// Existing startup all-gather word, disjoint from current prefill bits 0-19.
    pub fn agreement(f: Flags, legacy: u64) u64 {
        std.debug.assert(legacy & (@as(u64, 3) << 48) == 0);
        return legacy | (f.bits() << 48);
    }
};
fn enabled(comptime name: [:0]const u8) bool {
    return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
}
pub const Batch = struct {
    ids: [max_count]u64 = undefined,
    count: usize,
    pub fn slice(b: *const Batch) []const u64 { return b.ids[0..b.count]; }
};
pub fn validate(ids: []const u64) !void {
    if (ids.len < 2 or ids.len > max_count) return error.BadDropBatch;
    for (ids, 0..) |id, i| {
        if (id == 0 or std.mem.indexOfScalar(u64, ids[0..i], id) != null) return error.BadDropBatch;
    }
}
/// Parse the complete payload before any follower drops storage.
pub fn decode(payload: []const u8) !Batch {
    if (payload.len < 4) return error.BadDropBatch;
    const n = std.mem.readInt(u32, payload[0..4], .little);
    if (n < 2 or n > max_count or payload.len != 4 + @as(usize, n) * 8) return error.BadDropBatch;
    var b: Batch = .{ .count = n };
    for (b.ids[0..n], 0..) |*id, i| id.* = std.mem.readInt(u64, payload[4 + i * 8 ..][0..8], .little);
    try validate(b.slice());
    return b;
}
pub fn receipt(step: u64, count: usize) [13]u8 {
    var bytes: [13]u8 = undefined;
    bytes[0] = kind;
    std.mem.writeInt(u64, bytes[1..9], step, .little);
    std.mem.writeInt(u32, bytes[9..13], @intCast(count), .little);
    return bytes;
}
pub fn checkReceipt(bytes: []const u8, step: u64, count: usize) !void {
    const expected = receipt(step, count);
    if (!std.mem.eql(u8, bytes, &expected)) return error.BadDropReceipt;
}
/// Shared by the real leader and TP3 fault tests. Accounting/metadata survives
/// every send, free and receipt, including a slow or failed second follower.
pub fn release(runtime: anytype) !void {
    for (0..runtime.followerCount()) |rank| try runtime.send(rank);
    try runtime.freeLocal();
    for (0..runtime.followerCount()) |rank| try runtime.receive(rank);
    runtime.commit();
}

test "cache batch flags default off and startup agreement detects either TP3 follower" {
    const off: Flags = .{};
    try std.testing.expectEqual(@as(u64, 0), off.bits());
    // Keyed drafts reserve bit 63 in the same word during port integration.
    const legacy: u64 = 0x100003 | (@as(u64, 1) << 63);
    try std.testing.expectEqual(legacy, off.agreement(legacy));
    for ([_]Flags{ .{ .batch_dead = true }, .{ .first_continuation = true }, .{ .batch_dead = true, .first_continuation = true } }) |flags| {
        for (1..3) |rank| {
            var words = [_]u64{ legacy, legacy, legacy };
            words[rank] = flags.agreement(legacy);
            try std.testing.expect(words[rank] != words[0]);
        }
    }
}
test "cache batch drop frame rejects malformed duplicate unknown and unbounded payloads" {
    var payload: [4 + 16 * 8]u8 = undefined;
    for (2..17) |n| {
        std.mem.writeInt(u32, payload[0..4], @intCast(n), .little);
        for (0..n) |i| std.mem.writeInt(u64, payload[4 + i * 8 ..][0..8], i + 1, .little);
        const b = try decode(payload[0 .. 4 + n * 8]);
        try std.testing.expectEqual(n, b.count);
        try std.testing.expectEqual(@as(u64, n), b.ids[n - 1]);
        try std.testing.expectError(error.BadDropBatch, decode(payload[0 .. 3 + n * 8]));
    }
    for ([_]u32{ 0, 1, 17, std.math.maxInt(u32) }) |n| {
        std.mem.writeInt(u32, payload[0..4], n, .little);
        try std.testing.expectError(error.BadDropBatch, decode(&payload));
    }
    try std.testing.expectError(error.BadDropBatch, validate(&.{ 1, 1 }));
    try std.testing.expectError(error.BadDropBatch, validate(&.{ 1, 0 }));
}
const Fake = struct {
    count: usize = 2,
    sends: usize = 0, frees: usize = 0, receipts: usize = 0,
    held: usize = 16, committed: bool = false,
    fail_send: ?usize = null, fail_receipt: ?usize = null, bad_receipt: ?usize = null,
    pub fn followerCount(_: *Fake) usize { return 2; }
    pub fn send(f: *Fake, rank: usize) !void {
        try std.testing.expectEqual(@as(usize, 16), f.held);
        try std.testing.expectEqual(@as(usize, 0), f.frees);
        if (f.fail_send == rank) return error.PeerLost;
        try std.testing.expectEqual(rank, f.sends);
        f.sends += 1;
    }
    pub fn freeLocal(f: *Fake) !void {
        try std.testing.expectEqual(@as(usize, 2), f.sends);
        try std.testing.expectEqual(@as(usize, 16), f.held);
        f.frees += f.count;
    }
    pub fn receive(f: *Fake, rank: usize) !void {
        try std.testing.expectEqual(@as(usize, 16), f.held);
        try std.testing.expect(!f.committed);
        try std.testing.expectEqual(rank, f.receipts);
        if (f.fail_receipt == rank) return error.PeerLost;
        const ack = receipt(42, if (f.bad_receipt == rank) f.count + 1 else f.count);
        try checkReceipt(&ack, 42, f.count);
        f.receipts += 1;
    }
    pub fn commit(f: *Fake) void {
        std.debug.assert(f.receipts == 2 and f.sends == 2);
        f.held -= f.count;
        f.committed = true;
    }
};
test "cache batch TP3 one frame each and both receipts before accounting release" {
    for (2..17) |n| {
        var f: Fake = .{ .count = n };
        try release(&f);
        try std.testing.expectEqual(@as(usize, 2), f.sends);
        try std.testing.expectEqual(n, f.frees);
        try std.testing.expectEqual(@as(usize, 2), f.receipts);
        try std.testing.expectEqual(16 - n, f.held);
        try std.testing.expect(f.committed);
    }
}
test "cache batch TP3 either follower send or receipt failure prevents commit" {
    for (0..2) |rank| {
        var send_fail: Fake = .{ .fail_send = rank };
        try std.testing.expectError(error.PeerLost, release(&send_fail));
        try std.testing.expectEqual(@as(usize, 0), send_fail.frees);
        var lost: Fake = .{ .fail_receipt = rank };
        try std.testing.expectError(error.PeerLost, release(&lost));
        try std.testing.expectEqual(@as(usize, 16), lost.held);
        try std.testing.expect(!lost.committed);
        var corrupt: Fake = .{ .bad_receipt = rank };
        try std.testing.expectError(error.BadDropReceipt, release(&corrupt));
        try std.testing.expectEqual(@as(usize, 16), corrupt.held);
        try std.testing.expect(!corrupt.committed);
    }
    const ack = receipt(42, 2);
    try std.testing.expectError(error.BadDropReceipt, checkReceipt(&ack, 43, 2));
    try std.testing.expectError(error.BadDropReceipt, checkReceipt(ack[0..12], 42, 2));
}
