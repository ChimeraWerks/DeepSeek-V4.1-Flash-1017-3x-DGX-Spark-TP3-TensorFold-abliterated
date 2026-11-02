//! The RDMA ring's sequence bookkeeping (rdma.zig's proxy thread), apart from the verbs so it can be tested: a
//! slot's parts over the devices, a peer's parts back to whole sequences, and when a slot's writes have all been sent.
const std = @import("std");

pub const max_ranks = 8;
pub const max_devices = 4;
pub const max_slots = 16;

/// Per-peer local device masks, comma-separated decimal masks in global-rank
/// order. Own rank must be0, each peer nonzero and limited to selected devices.
/// Null retains the original all-devices-to-every-peer switch topology.
pub fn peerMasks(text: ?[]const u8, rank: u32, world: u32, nd: u32) error{BadPeerMasks}!?[max_ranks]u8 {
    if (world < 2 or world > max_ranks or rank >= world or nd == 0 or nd > max_devices) return error.BadPeerMasks;
    const value = text orelse return null;
    var out: [max_ranks]u8 = @splat(0);
    var it = std.mem.splitScalar(u8, value, ',');
    const allowed: u8 = (@as(u8, 1) << @intCast(nd)) - 1;
    for (0..world) |p| {
        const field = it.next() orelse return error.BadPeerMasks;
        if (field.len == 0) return error.BadPeerMasks;
        for (field) |c| if (c < '0' or c > '9') return error.BadPeerMasks;
        const mask = std.fmt.parseInt(u8, field, 10) catch return error.BadPeerMasks;
        if (mask & ~allowed != 0 or (p == rank and mask != 0) or (p != rank and mask == 0)) return error.BadPeerMasks;
        out[p] = mask;
    }
    if (it.next() != null) return error.BadPeerMasks;
    return out;
}

pub fn deviceMask(masks: ?[max_ranks]u8, peer: usize, nd: u32) u8 {
    return if (masks) |m| m[peer] else (@as(u8, 1) << @intCast(nd)) - 1;
}

/// Logical wire-rail index of a local NIC selected by mask; both ends pack
/// their independently chosen local NICs in ascending order.
pub fn railIndex(mask: u8, device: usize) ?u32 {
    const bit: u8 = @as(u8, 1) << @intCast(device);
    if (mask & bit == 0) return null;
    return @popCount(mask & (bit - 1));
}

/// Device `d`'s part of an `n`-byte slot over `nd` devices: whole 16-byte units each, the last device takes the rest.
pub fn part(n: u64, nd: u32, d: u32) struct { off: u64, len: u64 } {
    const unit = n / 16 / nd * 16;
    const off = unit * d;
    return .{ .off = off, .len = if (d == nd - 1) n - off else unit };
}

/// The immediate a write carries: its sequence's low 32 bits, in network order as the Python build's ring sends it.
pub fn immediate(q: u64) u32 {
    return std.mem.nativeToBig(u32, @truncate(q));
}

/// A receive completion's immediate back to a sequence. A device delivers a peer's parts in order, so it extends `last`,
/// the sequence that device delivered from that peer before.
pub fn extend(last: u64, imm: u32) u64 {
    var q = (last & ~@as(u64, 0xffff_ffff)) | std.mem.bigToNative(u32, imm);
    if (q <= last) q += 1 << 32;
    return q;
}

/// What each peer has delivered: the last sequence each device brought and the last one published in the flags.
pub const Arrivals = struct {
    nd: u32,
    masks: ?[max_ranks]u8 = null,
    got: [max_ranks][max_devices]u64 = @splat(@splat(0)),
    published: [max_ranks]u64 = @splat(0),

    /// A receive completion from `peer` on device `d`: the sequences `from`..`to` (inclusive; none when from > to) whose
    /// every part is now in, to publish in order (a peer may already be a gather ahead on one device).
    pub fn arrive(a: *Arrivals, peer: usize, d: usize, imm: u32) struct { from: u64, to: u64 } {
        a.got[peer][d] = extend(a.got[peer][d], imm);
        const mask = deviceMask(a.masks, peer, a.nd);
        std.debug.assert(railIndex(mask, d) != null);
        var in: u64 = std.math.maxInt(u64);
        for (a.got[peer][0..a.nd], 0..) |g, dev| if (railIndex(mask, dev) != null) { in = @min(in, g); };
        const from = a.published[peer] + 1;
        if (in > a.published[peer]) a.published[peer] = in;
        return .{ .from = from, .to = in };
    }
};

/// Write completions a gather's slot still owes: one a peer and device.
pub const Sends = struct {
    need: u32,
    slots: u32,
    seq: [max_slots]u64 = @splat(0),
    done: [max_slots]u32 = @splat(0),

    /// One completed write of sequence `q`: q when it was the last its slot owed (the slot may be staged again), else null.
    pub fn complete(s: *Sends, q: u64) ?u64 {
        const slot: usize = @intCast(q % s.slots);
        if (s.seq[slot] != q) {
            s.seq[slot] = q;
            s.done[slot] = 0;
        }
        s.done[slot] += 1;
        return if (s.done[slot] == s.need) q else null;
    }
};

test "a slot's parts: whole 16-byte units a device, the rest on the last" {
    const a = part(120_000, 2, 0);
    const b = part(120_000, 2, 1);
    try std.testing.expectEqual(@as(u64, 0), a.off);
    try std.testing.expectEqual(@as(u64, 60_000), a.len);
    try std.testing.expectEqual(@as(u64, 60_000), b.off);
    try std.testing.expectEqual(@as(u64, 60_000), b.len);
    const c = part(48, 2, 1);
    try std.testing.expectEqual(@as(u64, 16), c.off);
    try std.testing.expectEqual(@as(u64, 32), c.len);
    try std.testing.expectEqual(@as(u64, 0), part(0, 2, 0).len);
    try std.testing.expectEqual(@as(u64, 0), part(0, 2, 1).len);
    try std.testing.expectEqual(@as(u64, 4096), part(4096, 1, 0).len);
}

test "immediates extend a device's last sequence, across the 32-bit wrap" {
    try std.testing.expectEqual(@as(u64, 1), extend(0, immediate(1)));
    try std.testing.expectEqual(@as(u64, 7), extend(6, immediate(7)));
    try std.testing.expectEqual(@as(u64, 1 << 32), extend(0xffff_ffff, immediate(1 << 32)));
    try std.testing.expectEqual(@as(u64, (1 << 32) + 3), extend((1 << 32) + 2, immediate((1 << 32) + 3)));
    // the Python build's htonl: the wire bytes are big-endian
    try std.testing.expectEqual(@as(u32, 0x0102_0304), std.mem.bigToNative(u32, immediate(0x0102_0304)));
}

test "a sequence is published once every device has delivered its part" {
    var a: Arrivals = .{ .nd = 2 };
    var r = a.arrive(1, 0, immediate(1));
    try std.testing.expect(r.from > r.to);
    r = a.arrive(1, 0, immediate(2));
    try std.testing.expect(r.from > r.to);
    r = a.arrive(1, 1, immediate(1));
    try std.testing.expectEqual(@as(u64, 1), r.from);
    try std.testing.expectEqual(@as(u64, 1), r.to);
    r = a.arrive(1, 1, immediate(2));
    try std.testing.expectEqual(@as(u64, 2), r.from);
    try std.testing.expectEqual(@as(u64, 2), r.to);
    // another peer is independent
    r = a.arrive(3, 1, immediate(1));
    try std.testing.expect(r.from > r.to);
    // one device: every completion publishes its sequence
    var b: Arrivals = .{ .nd = 1 };
    r = b.arrive(0, 0, immediate(1));
    try std.testing.expectEqual(@as(u64, 1), r.to);
}

test "a slot is sent after its last write: peers x devices completions" {
    var s: Sends = .{ .need = 6, .slots = 4 };
    for (0..5) |_| try std.testing.expectEqual(@as(?u64, null), s.complete(1));
    try std.testing.expectEqual(@as(?u64, 1), s.complete(1));
    // sequence 5 reuses slot 1 and starts its own count
    for (0..5) |_| try std.testing.expectEqual(@as(?u64, null), s.complete(5));
    try std.testing.expectEqual(@as(?u64, null), s.complete(2));
    try std.testing.expectEqual(@as(?u64, 5), s.complete(5));
}

test "peer rail masks validate topology and retain legacy defaults" {
    try std.testing.expectEqual(@as(?[max_ranks]u8, null), try peerMasks(null, 0, 3, 4));
    const masks = (try peerMasks("0,5,10", 0, 3, 4)).?;
    try std.testing.expectEqualSlices(u8, &.{ 0, 5, 10 }, masks[0..3]);
    try std.testing.expectEqual(@as(u8, 15), deviceMask(null, 1, 4));
    try std.testing.expectEqual(@as(?u32, 0), railIndex(5, 0));
    try std.testing.expectEqual(@as(?u32, null), railIndex(5, 1));
    try std.testing.expectEqual(@as(?u32, 1), railIndex(5, 2));
    for ([_][]const u8{ "", "0,5", "0,5,10,0", "0,0,10", "1,5,10", "0,16,10", "0,-1,10", "0, 5,10", "0,5,256" }) |bad|
        try std.testing.expectError(error.BadPeerMasks, peerMasks(bad, 0, 3, 4));
    try std.testing.expectError(error.BadPeerMasks, peerMasks("0,5,10", 0, 3, 2));
    try std.testing.expectError(error.BadPeerMasks, peerMasks(null, 3, 3, 4));
}

test "direct triangle arrivals wait only for each peer's selected local NICs" {
    const masks = try peerMasks("0,5,10", 0, 3, 4);
    var a: Arrivals = .{ .nd = 4, .masks = masks };
    var r = a.arrive(1, 2, immediate(1));
    try std.testing.expect(r.from > r.to);
    r = a.arrive(2, 3, immediate(1));
    try std.testing.expect(r.from > r.to);
    r = a.arrive(1, 0, immediate(1));
    try std.testing.expectEqual(@as(u64, 1), r.to);
    try std.testing.expectEqual(@as(u64, 0), a.published[2]);
    r = a.arrive(2, 1, immediate(1));
    try std.testing.expectEqual(@as(u64, 1), r.to);
    r = a.arrive(1, 0, immediate(2));
    try std.testing.expect(r.from > r.to);
    r = a.arrive(1, 2, immediate(2));
    try std.testing.expectEqual(@as(u64, 2), r.to);
    try std.testing.expectEqual(@as(u64, 1), a.published[2]);
}

test "sparse device masks still partition a peer's entire payload without holes" {
    for (1..16) |bits| {
        const mask: u8 = @intCast(bits);
        for ([_]u64{ 16, 48, 1024, 120000 }) |n| {
            var cursor: u64 = 0;
            for (0..4) |device| {
                const wire = railIndex(mask, device) orelse continue;
                const p = part(n, @popCount(mask), wire);
                try std.testing.expectEqual(cursor, p.off);
                cursor += p.len;
            }
            try std.testing.expectEqual(n, cursor);
        }
    }
}
