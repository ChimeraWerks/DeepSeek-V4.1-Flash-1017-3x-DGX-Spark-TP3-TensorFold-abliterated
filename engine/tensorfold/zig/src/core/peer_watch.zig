//! Bounded rank-0 failure exit, inspired by Robert Wojciechowski's Mia 0020.
//! This is a TP-world-neutral watchdog, independent of spill and KV sharding.
const std = @import("std");
extern "c" fn _exit(status: c_int) noreturn;

pub const Reason = enum { peer_lost, stalled };
pub fn fatalError(err: anyerror) bool {
    return switch (err) {
        error.CudaContextLost, error.PeerDown, error.RingFailed, error.NcclFailed, error.ShardDiverged, error.StaleExtent, error.TransferFailed, error.TransferTimeout, error.KvShardNotStarted, error.ShardRingOrder, error.BadRingRows, error.RingOrder, error.GraphVoteFatal, error.GraphStorageChanged, error.GraphStateMismatch => true,
        else => false,
    };
}

pub const Probe = struct {
    ctx: *anyopaque,
    failed: *const fn (*anyopaque) bool,
    exit: *const fn (*anyopaque, u8) void = exitProcess,
};

fn exitProcess(_: *anyopaque, code: u8) void {
    // Do not wait for CUDA teardown or take locks held by a hung engine thread.
    _exit(code);
}

pub const Policy = struct {
    stall_ms: u64 = 120000,
    grace_ms: u64 = 1000,
    poll_ms: u32 = 100,

    pub fn fromEnv() !Policy {
        var p: Policy = .{};
        if (std.c.getenv("TF_DS_PEER_STALL_SECONDS")) |s| {
            const n = try std.fmt.parseInt(u32, std.mem.span(s), 10);
            if (n == 0 or n > 3600) return error.BadPeerStall;
            p.stall_ms = @as(u64, n) * 1000;
        }
        if (std.c.getenv("TF_DS_PEER_EXIT_GRACE_MS")) |s| {
            const n = try std.fmt.parseInt(u32, std.mem.span(s), 10);
            if (n > 15000) return error.BadPeerGrace;
            p.grace_ms = n;
        }
        return p;
    }
};

pub const State = struct {
    reason: ?Reason = null,
    since: u64 = 0,

    pub fn tick(s: *State, p: Policy, now: u64, busy: bool, last: u64, peer_lost: bool) bool {
        if (s.reason == null) {
            if (peer_lost) s.reason = .peer_lost else if (busy and now -| last >= p.stall_ms) {
                s.reason = .stalled;
            }
            if (s.reason != null) s.since = now;
        }
        return s.reason != null and now -| s.since >= p.grace_ms;
    }
};

pub const Monitor = struct {
    io: std.Io,
    probe: Probe,
    policy: Policy,
    pending: std.atomic.Value(u32) = .init(0),
    last_ms: std.atomic.Value(u64) = .init(0),
    tripped: std.atomic.Value(bool) = .init(false),
    forced: std.atomic.Value(bool) = .init(false),
    stopping: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    fn now(m: *const Monitor) u64 {
        return @intCast(@divTrunc(std.Io.Clock.awake.now(m.io).toNanoseconds(), std.time.ns_per_ms));
    }

    pub fn start(m: *Monitor) !void {
        m.progress();
        m.thread = try std.Thread.spawn(.{}, run, .{m});
    }

    pub fn stop(m: *Monitor) void {
        m.stopping.store(true, .release);
        if (m.thread) |thread| thread.join();
        m.thread = null;
    }

    pub fn progress(m: *Monitor) void {
        m.last_ms.store(m.now(), .release);
    }

    pub fn signal(m: *Monitor, err: anyerror) void {
        if (fatalError(err)) m.forced.store(true, .release);
    }

    pub fn submitted(m: *Monitor) void {
        // Reset the clock before the first pending request becomes visible.
        if (m.pending.load(.acquire) == 0) m.progress();
        _ = m.pending.fetchAdd(1, .acq_rel);
    }

    pub fn finished(m: *Monitor) void {
        _ = m.pending.fetchSub(1, .acq_rel);
        m.progress();
    }

    fn run(m: *Monitor) void {
        var state: State = .{};
        while (!m.stopping.load(.acquire)) {
            const exit_now = state.tick(m.policy, m.now(), m.pending.load(.acquire) > 0, m.last_ms.load(.acquire), m.forced.load(.acquire) or m.probe.failed(m.probe.ctx));
            if (state.reason != null and !m.tripped.swap(true, .acq_rel)) {
                if (!@import("builtin").is_test) std.log.err("rank group broken ({s}); rank 0 exits 75 after {d} ms so its supervisor can replace every rank", .{ @tagName(state.reason.?), m.policy.grace_ms });
            }
            if (exit_now) {
                m.probe.exit(m.probe.ctx, 75);
                return;
            }
            std.Io.sleep(m.io, .fromMilliseconds(m.policy.poll_ms), .awake) catch {};
        }
    }
};

test "peer-watch ignores idle time, observes progress and latches failures" {
    const p: Policy = .{ .stall_ms = 100, .grace_ms = 10 };
    var s: State = .{};
    try std.testing.expect(!s.tick(p, 10000, false, 0, false));
    try std.testing.expect(!s.tick(p, 10000, true, 9999, false));
    try std.testing.expect(!s.tick(p, 10100, true, 9999, false));
    try std.testing.expectEqual(Reason.stalled, s.reason.?);
    // Late progress cannot repair an invalid collective world after a trip.
    try std.testing.expect(s.tick(p, 10110, false, 10110, false));
    var lost: State = .{};
    try std.testing.expect(!lost.tick(p, 2, false, 2, true));
    try std.testing.expectEqual(Reason.peer_lost, lost.reason.?);
    try std.testing.expect(lost.tick(p, 12, false, 12, false));
}

test "peer-watch exits once with 75 and joins cleanly on a CPU probe" {
    const Fake = struct {
        exits: std.atomic.Value(u32) = .init(0),
        code: std.atomic.Value(u8) = .init(0),
        fn failed(_: *anyopaque) bool {
            return true;
        }
        fn exit(ctx: *anyopaque, code: u8) void {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            f.code.store(code, .release);
            _ = f.exits.fetchAdd(1, .acq_rel);
        }
    };
    var f: Fake = .{};
    var m: Monitor = .{ .io = std.testing.io, .probe = .{ .ctx = &f, .failed = Fake.failed, .exit = Fake.exit }, .policy = .{ .stall_ms = 100, .grace_ms = 1, .poll_ms = 1 } };
    try m.start();
    defer m.stop();
    for (0..500) |_| {
        if (f.exits.load(.acquire) > 0) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    try std.testing.expectEqual(@as(u8, 75), f.code.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), f.exits.load(.acquire));
}

test "peer-watch classifies fatal collectives without treating normal refusals as fatal" {
    for ([_]anyerror{ error.CudaContextLost, error.PeerDown, error.RingFailed, error.NcclFailed, error.ShardDiverged, error.StaleExtent, error.TransferFailed, error.TransferTimeout, error.KvShardNotStarted, error.ShardRingOrder, error.BadRingRows, error.RingOrder, error.GraphVoteFatal, error.GraphStorageChanged, error.GraphStateMismatch }) |err|
        try std.testing.expect(fatalError(err));
    for ([_]anyerror{ error.Cancelled, error.ContextFull, error.OutOfMemory, error.BadRequest }) |err|
        try std.testing.expect(!fatalError(err));
}
