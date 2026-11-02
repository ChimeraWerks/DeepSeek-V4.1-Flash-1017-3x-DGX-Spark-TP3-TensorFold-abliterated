//! The production idle relocation phase machine. Device effects are supplied by
//! the native host, not emulated by this module. Wire bytes are fixed little endian.
const std = @import("std");
const Link = @import("link.zig").Link;
pub const Move = struct { generation: u64, from: u64, to: u64, len: u64, bytes: u64 };
pub const budget_ns = 5 * std.time.ns_per_s;
pub const State = struct {
    pending: ?Move = null,
    last: u64 = 0,
    deadline: u64 = 0,

    pub fn phase(self: *State, payload: []const u8, admission: u64, host: anytype) ![43]u8 {
        if (payload.len != 49) return error.TransferFailed;
        const phase_id = payload[0];
        var words: [6]u64 = undefined;
        for (&words, 0..) |*word, i| word.* = std.mem.readInt(u64, payload[1 + i * 8 ..][0..8], .little);
        const mv: Move = .{ .generation = words[0], .from = words[1], .to = words[2], .len = words[3], .bytes = words[4] };
        if (phase_id > 1 or words[5] == 0 or words[5] > budget_ns) return error.TransferFailed;
        if (phase_id == 0) {
            if (self.pending != null or mv.generation <= self.last) return error.TransferFailed;
            self.deadline = @min(admission +| budget_ns, admission +| words[5]);
        }
        if (Link.monotonicNs() >= self.deadline) return error.TransferTimeout;
        try host.validate(mv);
        if (phase_id == 0) {
            self.pending = mv;
        } else {
            const pending = self.pending orelse return error.TransferFailed;
            if (!same(pending, mv)) return error.TransferFailed;
            try host.copy(mv, self.deadline);
            self.last = mv.generation;
            self.pending = null;
        }
        return receipt(phase_id, host.rank(), mv);
    }
};
pub fn membership(world: usize, rank: usize, peers: []const ?Link, primary: ?Link) !void {
    if (rank != 0 or world < 1 or world > peers.len + 1) return error.TransferFailed;
    if ((world > 1) != (primary != null)) return error.TransferFailed;
    for (peers, 0..) |peer, i| {
        if ((i < world - 1) != (peer != null)) return error.TransferFailed;
        if (peer) |a| {
            if (a.fd < 0) return error.TransferFailed;
            if (i == 0 and primary.?.fd != a.fd) return error.TransferFailed;
            for (peers[0..i]) |prior| if (prior) |b| if (a.fd == b.fd) return error.TransferFailed;
        }
    }
}

/// Complete leader transaction, shared by Lanes and host integration tests.
/// Host owns device/source interfaces; transport, ordering and publication live here.
pub fn leader(host: anytype, mv: Move, peers: []const ?Link, primary: ?Link, world: usize, rank: usize, step: *u64, duration: u64) !void {
    try membership(world, rank, peers, primary);
    const deadline = Link.monotonicNs() +| duration;
    if (Link.monotonicNs() >= deadline) return error.TransferTimeout;
    try host.enter(deadline);
    defer host.leave();
    if (Link.monotonicNs() >= deadline) return error.TransferTimeout;
    for ([_]u8{ 0, 1 }) |phase_id| {
        if (Link.monotonicNs() >= deadline) return error.TransferTimeout;
        if (phase_id == 1) try host.guard(mv);
        if (Link.monotonicNs() >= deadline) return error.TransferTimeout;
        var frame: [58]u8 = undefined;
        frame[0] = 12;
        std.mem.writeInt(u64, frame[1..9], step.*, .little);
        step.* += 1;
        const payload = request(phase_id, mv, deadline -| Link.monotonicNs());
        @memcpy(frame[9..], &payload);
        for (peers[0 .. world - 1]) |p| try p.?.sendWithin(&frame, deadline);
        if (phase_id == 1) {
            if (Link.monotonicNs() >= deadline) return error.TransferTimeout;
            try host.copy(mv, deadline);
        }
        for (peers[0 .. world - 1], 1..) |p, id| {
            var buf: [64]u8 = undefined;
            const got = p.?.recvBy(&buf, deadline) catch return error.TransferTimeout;
            try validateReceipt(got, phase_id, @intCast(id), mv);
        }
    }
    if (Link.monotonicNs() >= deadline) return error.TransferTimeout;
    host.publishMove(mv);
}

/// Stream query loop shared by both native roles; never synchronizes on failure.
pub fn waitCopy(stream: anytype, deadline: u64) !void {
    while (true) {
        if (Link.monotonicNs() >= deadline) return error.TransferTimeout;
        const complete = try stream.done();
        if (Link.monotonicNs() >= deadline) return error.TransferTimeout;
        if (complete) return;
        std.Thread.yield() catch {};
    }
}

/// Outer relocation command parser, including command kind and stream step.
pub fn dispatch(state: *State, frame: []const u8, admission: u64, step: *u64, host: anytype) ![43]u8 {
    if (frame.len != 58 or frame[0] != 12) return error.BadFrame;
    if (std.mem.readInt(u64, frame[1..9], .little) != step.*) return error.OutOfStep;
    step.* += 1;
    return state.phase(frame[9..], admission, host);
}

/// Publication is one host-owned pass after every validated rank/phase receipt.
pub fn publish(snaps: anytype, mv: Move) void {
    for (snaps) |sn| if (!sn.stale and sn.base == mv.from) {
        sn.base = @intCast(mv.to);
        sn.end = @as(@TypeOf(sn.end), @intCast(mv.to)) + sn.at;
        sn.generation = mv.generation;
    };
}

pub fn same(a: Move, b: Move) bool {
    return a.generation == b.generation and a.from == b.from and a.to == b.to and a.len == b.len and a.bytes == b.bytes;
}
pub fn receipt(phase: u8, rank: u8, mv: Move) [43]u8 {
    var out: [43]u8 = undefined;
    out[0] = 12; out[1] = phase; out[2] = rank;
    for ([_]u64{ mv.generation, mv.bytes, mv.from, mv.to, mv.len }, 0..) |v, i| std.mem.writeInt(u64, out[3 + i * 8 ..][0..8], v, .little);
    return out;
}
pub fn validateReceipt(bytes: []const u8, phase: u8, rank: u8, mv: Move) !void {
    const expected = receipt(phase, rank, mv);
    if (!std.mem.eql(u8, bytes, &expected)) return error.TransferFailed;
}
pub fn request(phase: u8, mv: Move, remaining: u64) [49]u8 {
    var out: [49]u8 = undefined;
    out[0] = phase;
    for ([_]u64{ mv.generation, mv.from, mv.to, mv.len, mv.bytes, remaining }, 0..) |v, i| std.mem.writeInt(u64, out[1 + i * 8 ..][0..8], v, .little);
    return out;
}

const Fake = struct {
    id: u8, copies: usize = 0, failure: enum { none, enqueue, query } = .none,
    pub fn rank(h: *@This()) u8 { return h.id; }
    pub fn validate(_: *@This(), mv: Move) !void { if (mv.bytes != mv.len * 4) return error.TransferFailed; }
    pub fn copy(h: *@This(), _: Move, _: u64) !void {
        h.copies += 1;
        if (h.failure == .enqueue) return error.EnqueueFailed;
        if (h.failure == .query) return error.QueryFailed;
    }
};
test "idle relocation actual phase TP3 socketpair parser receipt binding and faults" {
    const c = std.c;
    const mv: Move = .{ .generation = 7, .from = 0, .to = 32, .len = 8, .bytes = 32 };
    for (1..3) |rank| {
        var pair: [2]c.fd_t = undefined;
        try std.testing.expect(c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &pair) == 0);
        defer _ = c.close(pair[0]); defer _ = c.close(pair[1]);
        const head: Link = .{ .fd = pair[0] }; const follower: Link = .{ .fd = pair[1] };
        var state: State = .{}; var host: Fake = .{ .id = @intCast(rank) };
        const start = Link.monotonicNs(); const deadline = start + budget_ns;
        var buf: [80]u8 = undefined;
        for (0..2) |phase| {
            const req = request(@intCast(phase), mv, budget_ns);
            try head.sendWithin(&req, deadline);
            const reply = try state.phase(try follower.recvBy(&buf, deadline), start, &host);
            try follower.sendWithin(&reply, deadline);
            const got = try head.recvBy(&buf, deadline);
            try validateReceipt(got, @intCast(phase), @intCast(rank), mv);
            // Every byte belongs to the production 43-byte binding.
            for (0..43) |i| {
                var wrong = reply; wrong[i] ^= 1;
                try std.testing.expectError(error.TransferFailed, validateReceipt(&wrong, @intCast(phase), @intCast(rank), mv));
            }
            try std.testing.expectError(error.TransferFailed, validateReceipt(got[0..42], @intCast(phase), @intCast(rank), mv));
        }
        try std.testing.expectEqual(@as(usize, 1), host.copies);
        const replay = request(0, mv, budget_ns);
        try std.testing.expectError(error.TransferFailed, state.phase(&replay, start, &host));
        for ([_]@TypeOf(host.failure){ .enqueue, .query }) |fault| {
            var failed: State = .{}; host.failure = fault;
            const prepare = request(0, mv, budget_ns); const copy = request(1, mv, budget_ns);
            _ = try failed.phase(&prepare, start, &host);
            if (fault == .enqueue) try std.testing.expectError(error.EnqueueFailed, failed.phase(&copy, start, &host)) else try std.testing.expectError(error.QueryFailed, failed.phase(&copy, start, &host));
            try std.testing.expect(failed.pending != null and failed.last == 0);
        }
    }
}

test "idle relocation membership complete before source side effects" {
    const a: Link = .{ .fd = 4 }; const b: Link = .{ .fd = 5 };
    try membership(3, 0, &.{ a, b, null }, a);
    try std.testing.expectError(error.TransferFailed, membership(3, 0, &.{ a, null, null }, a));
    try std.testing.expectError(error.TransferFailed, membership(3, 0, &.{ a, a, null }, a));
    try std.testing.expectError(error.TransferFailed, membership(3, 0, &.{ a, b, a }, a));
    try std.testing.expectError(error.TransferFailed, membership(3, 1, &.{ a, b, null }, a));
    try std.testing.expectError(error.TransferFailed, membership(3, 0, &.{ a, b, null }, b));
}

test "idle relocation publication preserves stale unrelated aliases" {
    const Snap = struct { stale: bool = false, base: usize = 0, end: usize = 8, at: usize = 8, generation: u64 = 0 };
    var a: Snap = .{}; var b: Snap = .{}; var stale: Snap = .{ .stale = true }; var other: Snap = .{ .base = 80 };
    publish(&[_]*Snap{ &a, &b, &stale, &other }, .{ .generation = 7, .from = 0, .to = 32, .len = 8, .bytes = 32 });
    try std.testing.expect(a.base == 32 and b.base == 32 and a.end == 40 and a.generation == 7);
    try std.testing.expect(stale.base == 0 and other.base == 80);
}

test "idle relocation partial PREPARE body and missing each TP3 receipt expire" {
    const c = std.c;
    for (1..3) |_| {
        var pair: [2]c.fd_t = undefined;
        try std.testing.expect(c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &pair) == 0);
        defer _ = c.close(pair[0]); defer _ = c.close(pair[1]);
        const head: Link = .{ .fd = pair[0] }; const follower: Link = .{ .fd = pair[1] };
        var buf: [80]u8 = undefined;
        try std.testing.expectError(error.Timeout, head.recvBy(&buf, Link.monotonicNs() + 2 * std.time.ns_per_ms));
        // Full header, partial PREPARE body: the same command admission used
        // by followPair cannot fall back to blocking recvSpin.
        const partial = [_]u8{49, 0, 0, 0, 0, 7};
        try std.testing.expectEqual(@as(isize, partial.len), c.send(head.fd, &partial, partial.len, c.MSG.NOSIGNAL));
        var admission: u64 = 0;
        try std.testing.expectError(error.Timeout, follower.recvCommand(&buf, &admission, 2 * std.time.ns_per_ms));
        try std.testing.expect(admission != 0);
    }
}

test "idle relocation receive admission never renews PREPARE budget on COPY" {
    var state: State = .{}; var host: Fake = .{ .id = 1 };
    const mv: Move = .{ .generation = 1, .from = 0, .to = 32, .len = 8, .bytes = 32 };
    const request0 = request(0, mv, budget_ns);
    const expired = Link.monotonicNs() -| budget_ns;
    try std.testing.expectError(error.TransferTimeout, state.phase(&request0, expired, &host));
    try std.testing.expect(state.pending == null and host.copies == 0);
    const now = Link.monotonicNs();
    _ = try state.phase(&request0, now, &host);
    state.deadline = now;
    const request1 = request(1, mv, budget_ns);
    try std.testing.expectError(error.TransferTimeout, state.phase(&request1, Link.monotonicNs(), &host));
    try std.testing.expect(host.copies == 0 and state.pending != null);
}

const IntegrationHost = struct {
    id: u8 = 0,
    copies: usize = 0,
    guards: usize = 0,
    published: bool = false,
    entered: bool = false,
    fault: enum { none, enqueue, query, timeout } = .none,
    pub fn rank(h: *@This()) u8 { return h.id; }
    pub fn validate(_: *@This(), mv: Move) !void { if (mv.bytes != mv.len * 4) return error.TransferFailed; }
    pub fn enter(h: *@This(), _: u64) !void { h.entered = true; }
    pub fn leave(h: *@This()) void { h.entered = false; }
    pub fn guard(h: *@This(), _: Move) !void { h.guards += 1; }
    pub fn publishMove(h: *@This(), _: Move) void { h.published = true; }
    pub fn done(h: *@This()) !bool {
        if (h.fault == .query) return error.QueryFailed;
        return h.fault != .timeout;
    }
    pub fn copy(h: *@This(), _: Move, deadline: u64) !void {
        h.copies += 1;
        if (h.fault == .enqueue) return error.EnqueueFailed;
        try waitCopy(h, deadline);
    }
};
const ConcurrentFollower = struct {
    peer: Link,
    host: IntegrationHost,
    phase_fault: u8 = 0,
    // -1 clean, -2 missing, -3 partial framing, 0..42 corrupt exact field.
    corrupt: i8 = -1,
    replay: bool = false,
    failure: ?anyerror = null,
    terminal: bool = false,
    fn run(f: *@This()) void { f.execute() catch |err| { if (f.terminal) std.process.exit(75); f.failure = err; }; }
    fn execute(f: *@This()) !void {
        var state: State = .{};
        var step: u64 = 0;
        var buf: [80]u8 = undefined;
        var previous: [43]u8 = undefined;
        for (0..2) |phase_id| {
            var admission_end: u64 = 0;
            const frame = try f.peer.recvCommand(&buf, &admission_end, 30 * std.time.ns_per_ms);
            var reply = try dispatch(&state, frame, admission_end -| 30 * std.time.ns_per_ms, &step, &f.host);
            if (phase_id == f.phase_fault) {
                if (f.corrupt == -2) return;
                if (f.corrupt == -3) {
                    const partial = [_]u8{43, 0, 0, 0, 12};
                    _ = std.c.send(f.peer.fd, &partial, partial.len, std.c.MSG.NOSIGNAL);
                    return;
                }
                if (f.corrupt >= 0) reply[@intCast(f.corrupt)] ^= 1;
                if (f.replay) reply = previous;
            }
            try f.peer.sendWithin(&reply, state.deadline);
            previous = reply;
        }
    }
};

fn wholeTransaction(bad_rank: usize, phase_id: u8, corrupt: i8, replay: bool, fault: @FieldType(IntegrationHost, "fault")) !void {
    var pairs: [2][2]std.c.fd_t = undefined;
    for (&pairs) |*pair| try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, pair) == 0);
    defer for (pairs) |pair| { _ = std.c.close(pair[0]); _ = std.c.close(pair[1]); };
    var peers: [2]?Link = .{ .{ .fd = pairs[0][0] }, .{ .fd = pairs[1][0] } };
    var followers: [2]ConcurrentFollower = undefined;
    var threads: [2]std.Thread = undefined;
    for (&followers, &threads, 0..) |*f, *thread, i| {
        f.* = .{ .peer = .{ .fd = pairs[i][1] }, .host = .{ .id = @intCast(i + 1) }, .phase_fault = phase_id,
            .corrupt = if (bad_rank == i + 1) corrupt else -1, .replay = bad_rank == i + 1 and replay };
        thread.* = try std.Thread.spawn(.{}, ConcurrentFollower.run, .{f});
    }
    var host: IntegrationHost = .{ .fault = fault };
    var step: u64 = 0;
    const result = leader(&host, .{ .generation = 7, .from = 0, .to = 32, .len = 8, .bytes = 32 }, &peers, peers[0], 3, 0, &step, 20 * std.time.ns_per_ms);
    // Wake peers waiting for the next command on all failure exits, then join both.
    for (pairs) |pair| _ = std.c.shutdown(pair[0], std.c.SHUT.RDWR);
    for (threads) |thread| thread.join();
    if (bad_rank == 0 and fault == .none) {
        try result;
        try std.testing.expect(host.published and host.guards == 1 and host.copies == 1);
        for (followers) |f| { try std.testing.expect(f.failure == null and f.host.copies == 1); }
    } else {
        try std.testing.expect(if (result) |_| false else |_| true);
        try std.testing.expect(!host.published);
    }
    try std.testing.expect(!host.entered);
}

test "idle relocation complete leader concurrent TP3 receipt matrix" {
    try wholeTransaction(0, 0, -1, false, .none);
    for (1..3) |rank_id| for (0..2) |phase_id| {
        for ([_]i8{ -2, -3, 0, 1, 2, 3, 11, 19, 27, 35 }) |corrupt|
            try wholeTransaction(rank_id, @intCast(phase_id), corrupt, false, .none);
        if (phase_id == 1) try wholeTransaction(rank_id, 1, -1, true, .none);
    };
}

test "idle relocation complete leader actual copy polling failures suppress publication" {
    for ([_]@FieldType(IntegrationHost, "fault"){ .enqueue, .query, .timeout }) |fault|
        try wholeTransaction(0, 0, -1, false, fault);
}

const Integration = struct {
    const Fault = enum { none, missing, partial, wrong, replay, enqueue, query, timeout };
    const Stream = struct {
        fault: Fault = .none,
        queries: usize = 0,
        pub fn done(s: *@This()) !bool {
            s.queries += 1;
            if (s.fault == .query) return error.QueryFailed;
            return s.fault != .timeout;
        }
    };
    id: u8 = 0,
    peer: Link = .{ .fd = -1 },
    fault: Fault = .none,
    bad_phase: u8 = 0,
    bad_byte: usize = 0,
    stream: Stream = .{},
    copies: usize = 0,
    guarded: bool = false,
    published: bool = false,
    entered: bool = false,
    failure: ?anyerror = null,
    state: State = .{},
    pub fn enter(h: *@This(), _: u64) !void { h.entered = true; }
    pub fn leave(h: *@This()) void { h.entered = false; }
    pub fn guard(h: *@This(), _: Move) !void { h.guarded = true; }
    pub fn publishMove(h: *@This(), _: Move) void { h.published = true; }
    pub fn rank(h: *@This()) u8 { return h.id; }
    pub fn validate(_: *@This(), mv: Move) !void { if (mv.bytes != mv.len * 4) return error.TransferFailed; }
    pub fn copy(h: *@This(), _: Move, deadline: u64) !void {
        h.copies += 1;
        if (h.fault == .enqueue) return error.EnqueueFailed;
        h.stream.fault = h.fault;
        try waitCopy(&h.stream, deadline);
    }
    fn run(h: *@This()) void {
        h.work() catch |err| { h.failure = err; };
    }
    fn work(h: *@This()) !void {
        var step: u64 = 0;
        var previous: [43]u8 = undefined;
        for (0..2) |phase_id| {
            var frame: [80]u8 = undefined;
            var admission_deadline: u64 = 0;
            const incoming = if (h.state.pending != null) try h.peer.recvBy(&frame, h.state.deadline) else try h.peer.recvCommand(&frame, &admission_deadline, budget_ns);
            var reply = try dispatch(&h.state, incoming, admission_deadline -| budget_ns, &step, h);
            const deadline = h.state.deadline;
            if (phase_id == h.bad_phase) switch (h.fault) {
                .missing => return,
                .partial => {
                    const raw = [_]u8{ 43, 0, 0, 0, 12 };
                    if (std.c.send(h.peer.fd, &raw, raw.len, std.c.MSG.NOSIGNAL) != raw.len) return error.SendFailed;
                    return;
                },
                .wrong => reply[h.bad_byte] ^= 1,
                .replay => { if (phase_id == 0) reply[3] -%= 1 else reply = previous; },
                else => {},
            };
            try h.peer.sendWithin(&reply, deadline);
            previous = reply;
        }
    }
};

test "idle relocation complete concurrent TP3 leader receipt fault matrix" {
    const mv: Move = .{ .generation = 7, .from = 0, .to = 32, .len = 8, .bytes = 32 };
    // Both followers always run concurrently, including every rank/phase fault.
    for (0..3) |fault_rank| {
        for (0..2) |phase_id| {
            for ([_]Integration.Fault{ .none, .missing, .partial, .wrong, .replay }) |fault| {
                if (fault_rank == 0 and fault != .none) continue;
                if (fault_rank != 0 and fault == .none) continue;
                const fields = [_]usize{ 0, 1, 2, 3, 11, 19, 27, 35, 42 };
                for (fields) |byte| {
                    if (fault != .wrong and byte != 0) continue;
                    var pairs: [2][2]std.c.fd_t = undefined;
                    for (&pairs) |*pair| try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, pair) == 0);
                    defer for (pairs) |pair| { _ = std.c.close(pair[0]); _ = std.c.close(pair[1]); };
                    var peers: [2]?Link = undefined;
                    var followers: [2]Integration = undefined;
                    var threads: [2]std.Thread = undefined;
                    for (0..2) |i| {
                        peers[i] = .{ .fd = pairs[i][0] };
                        followers[i] = .{ .id = @intCast(i + 1), .peer = .{ .fd = pairs[i][1] }, .fault = if (i + 1 == fault_rank) fault else .none, .bad_phase = @intCast(phase_id), .bad_byte = byte };
                        threads[i] = try std.Thread.spawn(.{}, Integration.run, .{&followers[i]});
                    }
                    var host: Integration = .{};
                    var step: u64 = 0;
                    const result = leader(&host, mv, &peers, peers[0], 3, 0, &step, 100 * std.time.ns_per_ms);
                    // Close the direction to wake a follower waiting for COPY after a refused PREPARE.
                    for (peers) |peer| _ = std.c.shutdown(peer.?.fd, std.c.SHUT.RDWR);
                    for (threads) |thread| thread.join();
                    if (fault_rank == 0) {
                        try result;
                        try std.testing.expect(host.published and host.guarded and host.copies == 1);
                        for (followers) |f| try std.testing.expect(f.failure == null and f.state.last == mv.generation and f.copies == 1);
                    } else {
                        try std.testing.expect(if (result) |_| false else |_| true);
                        try std.testing.expect(!host.published);
                        if (phase_id == 0) try std.testing.expect(!host.guarded and host.copies == 0);
                    }
                    try std.testing.expect(!host.entered);
                }
            }
        }
    }
}

test "idle relocation production query loop enqueue query timeout no publication" {
    for ([_]Integration.Fault{ .enqueue, .query, .timeout }) |fault| {
        var host: Integration = .{ .fault = fault };
        var step: u64 = 0;
        const result = leader(&host, .{ .generation = 1, .from = 0, .to = 32, .len = 8, .bytes = 32 }, &.{}, null, 1, 0, &step, std.time.ns_per_ms);
        if (fault == .enqueue) try std.testing.expectError(error.EnqueueFailed, result);
        if (fault == .query) try std.testing.expectError(error.QueryFailed, result);
        if (fault == .timeout) try std.testing.expectError(error.TransferTimeout, result);
        try std.testing.expect(!host.published and !host.entered and host.guarded);
        if (fault != .enqueue) try std.testing.expect(host.stream.queries != 0);
    }
}


test "idle relocation concurrent follower copy query and timeout retain terminal disposition" {
    for (1..3) |bad_rank| for ([_]@FieldType(IntegrationHost, "fault"){ .enqueue, .query, .timeout }) |fault| {
        const child = std.c.fork();
        if (child == 0) {
            var pairs: [2][2]std.c.fd_t = undefined;
            for (&pairs) |*pair| if (std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, pair) != 0) std.process.exit(2);
            var peers: [2]?Link = .{ .{ .fd = pairs[0][0] }, .{ .fd = pairs[1][0] } };
            var followers: [2]ConcurrentFollower = undefined;
            var threads: [2]std.Thread = undefined;
            for (&followers, &threads, 0..) |*f, *thread, i| {
                f.* = .{ .peer = .{ .fd = pairs[i][1] }, .host = .{ .id = @intCast(i + 1), .fault = if (i + 1 == bad_rank) fault else .none }, .terminal = true };
                thread.* = std.Thread.spawn(.{}, ConcurrentFollower.run, .{f}) catch std.process.exit(3);
            }
            var host: IntegrationHost = .{};
            var step: u64 = 0;
            leader(&host, .{ .generation = 7, .from = 0, .to = 32, .len = 8, .bytes = 32 }, &peers, peers[0], 3, 0, &step, 100 * std.time.ns_per_ms) catch {
                // Follower timeout can race the leader deadline; join the fatal worker.
                for (threads) |thread| thread.join();
                std.process.exit(4);
            };
            std.process.exit(5);
        }
        if (child < 0) return error.ForkFailed;
        var status: c_int = 0;
        try std.testing.expectEqual(child, std.c.waitpid(child, &status, 0));
        try std.testing.expectEqual(@as(c_int, 75 << 8), status);
    };
}


test "idle relocation expired ready query is never admitted" {
    const Ready = struct {
        queries: usize = 0,
        pub fn done(self: *@This()) !bool {
            self.queries += 1;
            return true;
        }
    };
    var ready: Ready = .{};
    try std.testing.expectError(error.TransferTimeout, waitCopy(&ready, 0));
    try std.testing.expectEqual(@as(usize, 0), ready.queries);
}

test "idle relocation expired entry never reaches side effects" {
    var host: Integration = .{};
    var step: u64 = 0;
    try std.testing.expectError(error.TransferTimeout, leader(&host,
        .{ .generation = 1, .from = 0, .to = 32, .len = 8, .bytes = 32 },
        &.{}, null, 1, 0, &step, 0));
    try std.testing.expect(!host.entered and !host.guarded and !host.published);
    try std.testing.expectEqual(@as(u64, 0), step);
}
