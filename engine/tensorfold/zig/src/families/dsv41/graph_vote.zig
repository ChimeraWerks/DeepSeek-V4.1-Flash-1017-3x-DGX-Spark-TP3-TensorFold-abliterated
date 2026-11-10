//! Host-only nested graph-admission protocol. Never decoded as an outer command.
const std = @import("std");
const link = @import("link.zig");

pub const size = 48;
pub const discriminator: u8 = 14;
pub const version: u8 = 3; // 3: keyed pass frames carry each stream's top_k
pub const Message = enum(u8) { start = 1, vote = 2, decision = 3 };
pub const Stage = enum(u8) { reserve = 1, ready = 2 };
pub const Family = enum(u8) { round = 1, draft = 2 };
pub const Status = enum(u8) { none = 0, yes = 1, no = 2, disable = 3, fatal = 4 };
pub const Key = struct { a: u64, b: u64, c: u64 };
pub const Frame = struct {
    message: Message,
    stage: Stage,
    sender: u16,
    world: u16,
    step: u64,
    ordinal: u32,
    family: Family,
    status: Status,
    key: Key,
};

fn valid(f: Frame) bool {
    if (f.world == 0 or f.sender >= f.world) return false;
    // Draft keys: c is the keyed-draft bit (draft.zig PassGraphs.Key.voteKey), so exactly 0 or 1.
    if (f.family == .draft and (f.ordinal != 0 or f.key.c > 1)) return false;
    if (f.family == .round and (f.key.c & 0xff) > 2) return false;
    return switch (f.message) {
        .start => f.sender == 0 and f.status == .none,
        .vote => f.sender != 0 and f.status != .none and (f.stage == .reserve or f.status != .disable),
        .decision => f.sender == 0 and f.status != .none and (f.stage == .reserve or (f.status != .no and f.status != .disable)),
    };
}

pub fn encode(f: Frame) error{BadFrame}![size]u8 {
    if (!valid(f)) return error.BadFrame;
    var b: [size]u8 = @splat(0);
    b[0] = discriminator; b[1] = version; b[2] = @intFromEnum(f.message); b[3] = @intFromEnum(f.stage);
    std.mem.writeInt(u16, b[4..6], f.sender, .little);
    std.mem.writeInt(u16, b[6..8], f.world, .little);
    std.mem.writeInt(u64, b[8..16], f.step, .little);
    std.mem.writeInt(u32, b[16..20], f.ordinal, .little);
    b[20] = @intFromEnum(f.family); b[21] = @intFromEnum(f.status);
    std.mem.writeInt(u64, b[24..32], f.key.a, .little);
    std.mem.writeInt(u64, b[32..40], f.key.b, .little);
    std.mem.writeInt(u64, b[40..48], f.key.c, .little);
    return b;
}

pub fn decode(b: []const u8) error{BadFrame}!Frame {
    if (b.len != size or b[0] != discriminator or b[1] != version or b[22] != 0 or b[23] != 0) return error.BadFrame;
    const f: Frame = .{
        .message = std.enums.fromInt(Message, b[2]) orelse return error.BadFrame,
        .stage = std.enums.fromInt(Stage, b[3]) orelse return error.BadFrame,
        .sender = std.mem.readInt(u16, b[4..6], .little),
        .world = std.mem.readInt(u16, b[6..8], .little),
        .step = std.mem.readInt(u64, b[8..16], .little),
        .ordinal = std.mem.readInt(u32, b[16..20], .little),
        .family = std.enums.fromInt(Family, b[20]) orelse return error.BadFrame,
        .status = std.enums.fromInt(Status, b[21]) orelse return error.BadFrame,
        .key = .{ .a = std.mem.readInt(u64, b[24..32], .little), .b = std.mem.readInt(u64, b[32..40], .little), .c = std.mem.readInt(u64, b[40..48], .little) },
    };
    if (!valid(f)) return error.BadFrame;
    return f;
}

/// Fatal always wins; allocation failure disables the whole cache.
pub fn decide(stage: Stage, votes: []const Status) Status {
    var result: Status = .yes;
    for (votes) |v| switch (v) {
        .fatal, .none => return .fatal,
        .disable => if (stage == .ready) { return .fatal; } else { result = .disable; },
        .no => if (stage == .ready) { return .fatal; } else if (result != .disable) { result = .no; },
        .yes => {},
    };
    return result;
}

/// Only the spill-only pair may opt in. KV shard keeps its existing vote-off path.
pub fn enabled(graphs: bool, world: u32, spill: bool, shard: bool, spill_opt_in: bool) bool {
    return graphs and (world == 1 or (!shard and (!spill or spill_opt_in)));
}

pub fn spillOptIn(text: ?[]const u8) bool {
    return if (text) |v| std.mem.eql(u8, v, "1") else false;
}

/// The opt-in's outer command was built by lanes; finish sending it before collecting replies.
/// One deadline bounds both directions. Any partial frame or disagreement ends the pair.
pub fn agreeSpill(links: []const ?link.Link, frame: []const u8, timeout_ns: u64) !void {
    const deadline = link.Link.monotonicNs() +| timeout_ns;
    for (links) |l| try (l orelse return error.NoLink).sendWithin(frame, deadline);
    for (links) |l| {
        var reply: [1]u8 = undefined;
        const bytes = try (l orelse return error.NoLink).recvBy(&reply, deadline);
        if (bytes.len != 1 or bytes[0] != 1) return error.GraphVoteMismatch;
    }
}

pub const Phase = enum { reserve, ready, launch_first };
pub const Fault = struct {
    rank: u16,
    phase: Phase,
    remaining: u32,
    pub fn fire(f: *Fault, rank: u16, phase: Phase) bool {
        if (f.rank != rank or f.phase != phase or f.remaining == 0) return false;
        f.remaining -= 1;
        std.log.warn("dsv41 graph fault rank {d} phase {s}", .{ rank, @tagName(phase) });
        return true;
    }
};
pub fn parseFault(s: []const u8) ?Fault {
    var it = std.mem.splitScalar(u8, s, ':');
    const rank = std.fmt.parseInt(u16, it.next() orelse return null, 10) catch return null;
    const p = it.next() orelse return null;
    const phase: Phase = if (std.mem.eql(u8, p, "reserve")) .reserve else if (std.mem.eql(u8, p, "ready")) .ready else if (std.mem.eql(u8, p, "launch-first")) .launch_first else return null;
    const count = if (it.next()) |v| std.fmt.parseInt(u32, v, 10) catch return null else 1;
    if (count == 0 or it.next() != null) return null;
    return .{ .rank = rank, .phase = phase, .remaining = count };
}

/// Per-peer operations use one absolute local deadline. A failed operation is never retried.
pub const Transport = struct {
    ctx: *anyopaque,
    send: *const fn (*anyopaque, u16, []const u8, u64) anyerror!void,
    recv: *const fn (*anyopaque, u16, []u8, u64) anyerror![]u8,
};

pub const Control = struct {
    rank: u16,
    world: u16,
    /// Every rank meets the same misses in the same order, so a shared count binds each exchange; a rank that
    /// disagrees sees a mismatched frame and fails closed.
    step: u64 = 0,
    timeout_ns: u64,
    transport: ?Transport,
    clock: *const fn () u64,
    healthy_ctx: *anyopaque,
    healthy: *const fn (*anyopaque) bool,
    fault: ?Fault = null,

    pub fn isHealthy(c: *const Control) bool { return c.healthy(c.healthy_ctx); }
    pub fn inject(c: *Control, phase: Phase) bool {
        if (c.fault) |*f| return f.fire(c.rank, phase);
        return false;
    }
    pub fn stage(c: *Control, which: Stage, family: Family, ordinal: u32, key: Key, vote: Status) error{GraphVoteFatal}!Status {
        const frame: Frame = .{ .message = .start, .stage = which, .sender = 0, .world = c.world, .step = c.step, .ordinal = ordinal, .family = family, .status = .none, .key = key };
        c.step +%= 1;
        return exchange(c.transport, c.rank, frame, vote, c.clock() +| c.timeout_ns);
    }
};

/// TF_DS_GRAPH_VOTE_MS: each stage's bound in milliseconds (default 10 s; zero or unparsable keeps the default).
pub fn timeoutNs(text: ?[]const u8) u64 {
    const default_ms: u64 = 10_000;
    const ms = if (text) |t| std.fmt.parseUnsigned(u64, std.mem.trim(u8, t, " \t"), 10) catch default_ms else default_ms;
    return (if (ms == 0) default_ms else @min(ms, 3_600_000)) * std.time.ns_per_ms;
}

fn matching(got: Frame, expected: Frame) bool {
    return got.message == expected.message and got.stage == expected.stage and got.sender == expected.sender and
        got.world == expected.world and got.step == expected.step and got.ordinal == expected.ordinal and
        got.family == expected.family and std.meta.eql(got.key, expected.key);
}

pub fn exchange(t: ?Transport, rank: u16, base: Frame, vote: Status, deadline: u64) error{GraphVoteFatal}!Status {
    if (base.world == 0 or rank >= base.world or vote == .none) return error.GraphVoteFatal;
    if (base.world == 1) return decide(base.stage, &.{vote});
    const io = t orelse return error.GraphVoteFatal;
    var f = base;
    var buf: [size]u8 = undefined;
    if (rank == 0) {
        f.message = .start; f.sender = 0; f.status = .none;
        const start = encode(f) catch return error.GraphVoteFatal;
        for (1..base.world) |i| io.send(io.ctx, @intCast(i), &start, deadline) catch return error.GraphVoteFatal;
        var result = decide(base.stage, &.{vote});
        for (1..base.world) |i| {
            f.message = .vote; f.sender = @intCast(i);
            const ok = if (io.recv(io.ctx, @intCast(i), &buf, deadline)) |bytes|
                if (decode(bytes)) |got| matching(got, f) and blk: {
                    result = decide(base.stage, &.{ result, got.status });
                    break :blk true;
                } else |_| false
            else |_| false;
            if (!ok) {
                // Best effort: tell the followers still listening so they end now, not at their own deadline.
                f.message = .decision; f.sender = 0; f.status = .fatal;
                if (encode(f)) |fatal| {
                    for (1..base.world) |j| io.send(io.ctx, @intCast(j), &fatal, deadline) catch {};
                } else |_| {}
                return error.GraphVoteFatal;
            }
        }
        f.message = .decision; f.sender = 0; f.status = result;
        const decision = encode(f) catch return error.GraphVoteFatal;
        for (1..base.world) |i| io.send(io.ctx, @intCast(i), &decision, deadline) catch return error.GraphVoteFatal;
        return result;
    }
    f.message = .start; f.sender = 0;
    const start_ok = if (io.recv(io.ctx, 0, &buf, deadline)) |bytes|
        if (decode(bytes)) |start| matching(start, f) else |_| false
    else |_| false;
    if (!start_ok) {
        // Best effort: a fatal vote lets rank 0 end the world now instead of at its deadline.
        f.message = .vote; f.sender = rank; f.status = .fatal;
        if (encode(f)) |fatal| io.send(io.ctx, 0, &fatal, deadline) catch {} else |_| {}
        return error.GraphVoteFatal;
    }
    f.message = .vote; f.sender = rank; f.status = vote;
    const vote_bytes = encode(f) catch return error.GraphVoteFatal;
    io.send(io.ctx, 0, &vote_bytes, deadline) catch return error.GraphVoteFatal;
    f.message = .decision; f.sender = 0;
    const result = decode(io.recv(io.ctx, 0, &buf, deadline) catch return error.GraphVoteFatal) catch return error.GraphVoteFatal;
    if (!matching(result, f)) return error.GraphVoteFatal;
    return result.status;
}

test "codec round trip and malformed frames" {
    const f: Frame = .{ .message = .vote, .stage = .reserve, .sender = 1, .world = 3, .step = 42, .ordinal = 2, .family = .round, .status = .yes, .key = .{ .a = 3, .b = 4096, .c = 2 << 8 } };
    var b = try encode(f);
    try std.testing.expectEqualDeep(f, try decode(&b));
    try std.testing.expectError(error.BadFrame, decode(b[1..]));
    for ([_]usize{0, 1, 2, 3, 20, 21, 22, 23}) |i| {
        const old = b[i]; b[i] = 255;
        try std.testing.expectError(error.BadFrame, decode(&b));
        b[i] = old;
    }
}

test "draft frames carry the keyed bit and reject unknown values" {
    for ([_]u64{ 0, 1 }) |c| for ([_]Stage{ .reserve, .ready }) |stage| {
        const f: Frame = .{ .message = .start, .stage = stage, .sender = 0, .world = 3, .step = 7, .ordinal = 0, .family = .draft, .status = .none, .key = .{ .a = 16, .b = 3, .c = c } };
        var b = try encode(f);
        try std.testing.expectEqual(@as(u8, 3), b[1]);
        try std.testing.expectEqualDeep(f, try decode(&b));
        b[1] = 2; // old keyed pass frames have no top_k.
        try std.testing.expectError(error.BadFrame, decode(&b));
    };
    for ([_]u64{ 2, 3, 1 << 63 }) |c| {
        const f: Frame = .{ .message = .start, .stage = .reserve, .sender = 0, .world = 3, .step = 7, .ordinal = 0, .family = .draft, .status = .none, .key = .{ .a = 16, .b = 3, .c = c } };
        try std.testing.expectError(error.BadFrame, encode(f));
    }
    const bad_ordinal: Frame = .{ .message = .start, .stage = .reserve, .sender = 0, .world = 3, .step = 7, .ordinal = 1, .family = .draft, .status = .none, .key = .{ .a = 16, .b = 3, .c = 1 } };
    try std.testing.expectError(error.BadFrame, encode(bad_ordinal));
}

test "all three-rank vote combinations and fatal precedence" {
    for ([_]Stage{ .reserve, .ready }) |stage| {
        for (0..64) |mask| {
            const values = [_]Status{ .yes, .no, .disable, .fatal };
            const votes = [_]Status{ values[mask & 3], values[(mask >> 2) & 3], values[(mask >> 4) & 3] };
            const got = decide(stage, &votes);
            if (std.mem.indexOfScalar(Status, &votes, .fatal) != null or (stage == .ready and (std.mem.indexOfScalar(Status, &votes, .no) != null or std.mem.indexOfScalar(Status, &votes, .disable) != null))) {
                try std.testing.expectEqual(Status.fatal, got);
            } else if (std.mem.indexOfScalar(Status, &votes, .disable) != null) {
                try std.testing.expectEqual(Status.disable, got);
            } else if (std.mem.indexOfScalar(Status, &votes, .no) != null) {
                try std.testing.expectEqual(Status.no, got);
            } else try std.testing.expectEqual(Status.yes, got);
        }
    }
}

test "graph vote gate opts in only spill without KV shard" {
    try std.testing.expect(enabled(true, 3, false, false, false));
    try std.testing.expect(!enabled(true, 3, true, false, false));
    try std.testing.expect(enabled(true, 3, true, false, true));
    try std.testing.expect(!enabled(true, 3, true, true, true));
    try std.testing.expect(!enabled(true, 3, false, true, true));
    try std.testing.expect(!enabled(false, 3, true, false, true));
    try std.testing.expect(enabled(true, 1, true, true, false));
    try std.testing.expect(!spillOptIn(null));
    try std.testing.expect(!spillOptIn("0"));
    try std.testing.expect(!spillOptIn("true"));
    try std.testing.expect(spillOptIn("1"));
}

test "fault parsing" {
    try std.testing.expectEqualDeep(Fault{ .rank = 2, .phase = .launch_first, .remaining = 4 }, parseFault("2:launch-first:4").?);
    try std.testing.expectEqualDeep(Fault{ .rank = 0, .phase = .reserve, .remaining = 1 }, parseFault("0:reserve").?);
    for ([_][]const u8{ "", "1", "x:ready", "1:no", "1:ready:0", "1:ready:1:2", "1:ready:x" }) |v| try std.testing.expect(parseFault(v) == null);
}

test "stage bound parsing" {
    try std.testing.expectEqual(@as(u64, 10_000) * std.time.ns_per_ms, timeoutNs(null));
    try std.testing.expectEqual(@as(u64, 10_000) * std.time.ns_per_ms, timeoutNs("0"));
    try std.testing.expectEqual(@as(u64, 10_000) * std.time.ns_per_ms, timeoutNs("soon"));
    try std.testing.expectEqual(@as(u64, 250) * std.time.ns_per_ms, timeoutNs(" 250 "));
}

/// Three ranks over in-memory one-way channels (one producer, one consumer each), the followers on threads.
const FakeNet = struct {
    const Chan = struct {
        slots: [8][size]u8 = undefined,
        lens: [8]usize = @splat(0),
        head: std.atomic.Value(u32) = .init(0),
        tail: std.atomic.Value(u32) = .init(0),
    };
    down: [3]Chan = .{ .{}, .{}, .{} }, // rank 0 to follower i
    up: [3]Chan = .{ .{}, .{}, .{} }, // follower i to rank 0
    dead_sender: ?u16 = null, // this rank's sends fail (link loss)

    fn now() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }
    fn push(ch: *Chan, bytes: []const u8) !void {
        const t = ch.tail.load(.monotonic);
        if (t - ch.head.load(.acquire) >= ch.slots.len or bytes.len > size) return error.Full;
        @memcpy(ch.slots[t % ch.slots.len][0..bytes.len], bytes);
        ch.lens[t % ch.slots.len] = bytes.len;
        ch.tail.store(t + 1, .release);
    }
    fn pop(ch: *Chan, buf: []u8, deadline: u64) ![]u8 {
        const h = ch.head.load(.monotonic);
        while (ch.tail.load(.acquire) == h) {
            if (now() >= deadline) return error.Timeout;
            const ts: std.c.timespec = .{ .sec = 0, .nsec = 200_000 };
            _ = std.c.nanosleep(&ts, null);
        }
        const n = ch.lens[h % ch.slots.len];
        @memcpy(buf[0..n], ch.slots[h % ch.slots.len][0..n]);
        ch.head.store(h + 1, .release);
        return buf[0..n];
    }
    const End = struct { net: *FakeNet, rank: u16 };
    fn send(ctx: *anyopaque, to: u16, bytes: []const u8, _: u64) anyerror!void {
        const e: *End = @ptrCast(@alignCast(ctx));
        if (e.net.dead_sender == e.rank) return error.ConnectionResetByPeer;
        return push(if (e.rank == 0) &e.net.down[to] else &e.net.up[e.rank], bytes);
    }
    fn recv(ctx: *anyopaque, from: u16, buf: []u8, deadline: u64) anyerror![]u8 {
        const e: *End = @ptrCast(@alignCast(ctx));
        return pop(if (e.rank == 0) &e.net.up[from] else &e.net.down[e.rank], buf, deadline);
    }
};

const Run = struct {
    end: FakeNet.End,
    vote: Status,
    key: Key = .{ .a = 3, .b = 4096, .c = 7 << 8 },
    result: anyerror!Status = error.NotRun,

    fn go(r: *Run, stage: Stage, wait_ns: u64) void {
        const t: Transport = .{ .ctx = &r.end, .send = FakeNet.send, .recv = FakeNet.recv };
        const base: Frame = .{ .message = .start, .stage = stage, .sender = 0, .world = 3, .step = 5, .ordinal = 7, .family = .round, .status = .none, .key = r.key };
        r.result = exchange(t, r.end.rank, base, r.vote, FakeNet.now() + wait_ns);
    }
};

/// Runs rank 0 here and the listed followers on threads; a follower missing from `live` never answers.
fn world3(net: *FakeNet, runs: *[3]Run, stage: Stage, live: []const u16, wait_ns: u64) !void {
    var threads: [2]?std.Thread = .{ null, null };
    for (live) |i| threads[i - 1] = try std.Thread.spawn(.{}, Run.go, .{ &runs[i], stage, wait_ns });
    runs[0].go(stage, wait_ns);
    for (threads) |t| if (t) |x| x.join();
    _ = net;
}

/// A rank ended closed: its own error, or rank 0's fatal decision.
fn expectFatal(r: anyerror!Status) !void {
    const got = r catch |err| return std.testing.expectEqual(error.GraphVoteFatal, err);
    try std.testing.expectEqual(Status.fatal, got);
}

fn fresh(net: *FakeNet, votes: [3]Status) [3]Run {
    var runs: [3]Run = undefined;
    for (&runs, 0..) |*r, i| r.* = .{ .end = .{ .net = net, .rank = @intCast(i) }, .vote = votes[i] };
    return runs;
}

test "exchange: every rank learns the same decision" {
    const cases = [_]struct { stage: Stage, votes: [3]Status, want: Status }{
        .{ .stage = .reserve, .votes = .{ .yes, .yes, .yes }, .want = .yes },
        .{ .stage = .reserve, .votes = .{ .yes, .yes, .no }, .want = .no },
        .{ .stage = .reserve, .votes = .{ .no, .yes, .disable }, .want = .disable },
        .{ .stage = .reserve, .votes = .{ .yes, .fatal, .no }, .want = .fatal },
        .{ .stage = .ready, .votes = .{ .yes, .yes, .yes }, .want = .yes },
        .{ .stage = .ready, .votes = .{ .yes, .no, .yes }, .want = .fatal },
        .{ .stage = .ready, .votes = .{ .fatal, .yes, .yes }, .want = .fatal },
    };
    for (cases) |c| {
        var net: FakeNet = .{};
        var runs = fresh(&net, c.votes);
        try world3(&net, &runs, c.stage, &.{ 1, 2 }, 2 * std.time.ns_per_s);
        for (runs) |r| try std.testing.expectEqual(c.want, try r.result);
    }
}

test "exchange: a silent follower fails rank 0 and the waiting follower closed, within the bound" {
    var net: FakeNet = .{};
    var runs = fresh(&net, .{ .yes, .yes, .yes });
    const started = FakeNet.now();
    try world3(&net, &runs, .reserve, &.{1}, 100 * std.time.ns_per_ms);
    try std.testing.expectError(error.GraphVoteFatal, runs[0].result);
    try expectFatal(runs[1].result);
    try std.testing.expect(FakeNet.now() - started < 2 * std.time.ns_per_s);
}

test "exchange: link loss and a mismatched key are fatal everywhere" {
    {
        var net: FakeNet = .{ .dead_sender = 2 };
        var runs = fresh(&net, .{ .yes, .yes, .yes });
        try world3(&net, &runs, .reserve, &.{ 1, 2 }, 200 * std.time.ns_per_ms);
        try std.testing.expectError(error.GraphVoteFatal, runs[0].result);
        try std.testing.expectError(error.GraphVoteFatal, runs[2].result);
        try expectFatal(runs[1].result); // rank 0's fatal decision, or its own equal deadline
    }
    {
        var net: FakeNet = .{};
        var runs = fresh(&net, .{ .yes, .yes, .yes });
        runs[1].key.b = 8192; // follower 1 met a different miss
        const started = FakeNet.now();
        try world3(&net, &runs, .reserve, &.{ 1, 2 }, 5 * std.time.ns_per_s);
        try std.testing.expectError(error.GraphVoteFatal, runs[0].result);
        try std.testing.expectError(error.GraphVoteFatal, runs[1].result);
        try std.testing.expectEqual(Status.fatal, try runs[2].result);
        try std.testing.expect(FakeNet.now() - started < 2 * std.time.ns_per_s); // nobody waited out the 5 s bound
    }
}

test "exchange: a malformed frame is fatal and nothing is retried" {
    var net: FakeNet = .{};
    var runs = fresh(&net, .{ .yes, .yes, .yes });
    try FakeNet.push(&net.up[1], &[_]u8{ discriminator, version, 2, 1 }); // short vote from follower 1
    try world3(&net, &runs, .reserve, &.{2}, 200 * std.time.ns_per_ms);
    try std.testing.expectError(error.GraphVoteFatal, runs[0].result);
    try std.testing.expectEqual(Status.fatal, try runs[2].result);
}

test "control: a shared step binds each exchange and one rank uses no transport" {
    const H = struct {
        fn ok(_: *anyopaque) bool { return true; }
        fn clock() u64 { return FakeNet.now(); }
    };
    var dummy: u8 = 0;
    var c: Control = .{ .rank = 0, .world = 1, .timeout_ns = std.time.ns_per_s, .transport = null, .clock = H.clock, .healthy_ctx = &dummy, .healthy = H.ok };
    try std.testing.expectEqual(Status.yes, try c.stage(.reserve, .draft, 0, .{ .a = 4, .b = 2, .c = 0 }, .yes));
    try std.testing.expectEqual(Status.no, try c.stage(.reserve, .draft, 0, .{ .a = 4, .b = 2, .c = 0 }, .no));
    try std.testing.expectEqual(@as(u64, 2), c.step);
}

/// The production link is length framed: rank_wire is an outer lanes command, while the graph vote is
/// consumed inside the verify/pass command. Exercise that boundary on the same socket in both directions.
pub const SpillLinkTest = struct {
    const rw = @import("rank_wire.zig");
    const base: Frame = .{ .message = .start, .stage = .reserve, .sender = 0, .world = 3, .step = 7, .ordinal = 2, .family = .round, .status = .none, .key = .{ .a = 12, .b = 48, .c = 2 << 8 } };

    const End = struct {
        rank: u16,
        links: [2]?link.Link,
        fn other(e: *End, rank: u16) !link.Link {
            if (e.rank == 0 and rank > 0 and rank < 3) return e.links[rank - 1] orelse error.NoLink;
            if (e.rank != 0 and rank == 0) return e.links[0] orelse error.NoLink;
            return error.BadRank;
        }
        fn send(ctx: *anyopaque, to: u16, bytes: []const u8, deadline: u64) anyerror!void {
            const e: *End = @ptrCast(@alignCast(ctx));
            return (try e.other(to)).sendWithin(bytes, deadline);
        }
        fn recv(ctx: *anyopaque, from: u16, buf: []u8, deadline: u64) anyerror![]u8 {
            const e: *End = @ptrCast(@alignCast(ctx));
            return (try e.other(from)).recvBy(buf, deadline);
        }
        fn transport(e: *End) Transport {
            return .{ .ctx = e, .send = send, .recv = recv };
        }
    };

    fn rankWire(step: u64, buf: *[128]u8) ![]const u8 {
        buf[0] = rw.lanes_kind;
        std.mem.writeInt(u64, buf[1..9], step, .little);
        const sub = try rw.encode(buf[9..], .{ .op = .touch, .from = 0, .epoch = 19, .txn = step + 1, .seq = step + 1 }, .{ .touch = .{ .point = @splat(3) } });
        return buf[0 .. 9 + sub.len];
    }
    fn readRankWire(l: link.Link, step: u64) !void {
        var buf: [128]u8 = undefined;
        const bytes = try l.recvBy(&buf, link.Link.monotonicNs() + std.time.ns_per_s);
        if (bytes.len < 9 or bytes[0] != rw.lanes_kind or std.mem.readInt(u64, bytes[1..9], .little) != step) return error.BadFrame;
        const decoded = try rw.decode(bytes[9..], &.{});
        if (decoded.head.op != .touch or decoded.head.seq != step + 1) return error.BadFrame;
    }
    fn statsRequest(buf: *[128]u8) ![]const u8 {
        buf[0] = rw.lanes_kind;
        std.mem.writeInt(u64, buf[1..9], 2, .little);
        const sub = try rw.encode(buf[9..], .{ .op = .stats, .from = 0, .epoch = 19, .txn = 0, .seq = 2 }, .{ .stats = .{ .set = .tier } });
        return buf[0 .. 9 + sub.len];
    }
    fn answerStats(l: link.Link, rank: u16) !void {
        var buf: [128]u8 = undefined;
        const bytes = try l.recvBy(&buf, link.Link.monotonicNs() + std.time.ns_per_s);
        if (bytes.len < 9 or bytes[0] != rw.lanes_kind or std.mem.readInt(u64, bytes[1..9], .little) != 2) return error.BadFrame;
        const req = try rw.decode(bytes[9..], &.{});
        if (req.head.op != .stats or req.head.seq != 2 or req.msg.stats.set != .tier) return error.BadFrame;
        var reply: [rw.max_reply]u8 = undefined;
        const encoded = try rw.encode(&reply, .{ .op = .stats_reply, .from = rank, .epoch = req.head.epoch, .txn = req.head.txn, .seq = req.head.seq }, .{ .stats_reply = .{ .set = .tier, .n = 0, .values = @splat(0) } });
        try l.sendWithin(encoded, link.Link.monotonicNs() + std.time.ns_per_s);
    }
    fn readAgreement(l: link.Link, opt_in: ?bool) !void {
        var buf: [16]u8 = undefined;
        const bytes = try l.recvBy(&buf, link.Link.monotonicNs() + std.time.ns_per_s);
        if (bytes.len != 10 or bytes[0] != 15 or std.mem.readInt(u64, bytes[1..9], .little) != 1 or bytes[9] != 1) return error.BadFrame;
        if (opt_in) |enabled_| {
            const reply = [_]u8{@intFromBool(enabled_)};
            try l.sendWithin(&reply, link.Link.monotonicNs() + std.time.ns_per_s);
        }
    }
    fn readVerify(l: link.Link) !void {
        var buf: [16]u8 = undefined;
        const bytes = try l.recvBy(&buf, link.Link.monotonicNs() + std.time.ns_per_s);
        if (bytes.len != 9 or bytes[0] != 3 or std.mem.readInt(u64, bytes[1..9], .little) != 3) return error.BadFrame;
    }

    const Peer = struct {
        end: End,
        vote: Status,
        silent: bool,
        after: bool,
        result: anyerror!Status = error.NotRun,
        err: ?anyerror = null,
        stage: Stage,
        timeout_ns: u64,
        fn run(p: *Peer) void {
            p.loop() catch |err| {
                p.err = err;
            };
        }
        fn loop(p: *Peer) !void {
            const l = try p.end.other(0);
            try readRankWire(l, 0);
            try readAgreement(l, true);
            try answerStats(l, p.end.rank);
            try readVerify(l);
            if (p.silent) {
                const pause: std.c.timespec = .{ .sec = 0, .nsec = 250 * std.time.ns_per_ms };
                _ = std.c.nanosleep(&pause, null);
                return;
            }
            var f = base;
            f.stage = p.stage;
            p.result = exchange(p.end.transport(), p.end.rank, f, p.vote, link.Link.monotonicNs() + p.timeout_ns);
            if (p.after) try readRankWire(l, 4);
        }
    };

    pub fn run(stage: Stage, votes: [3]Status, expected: ?Status, silent: bool) !void {
        var sockets: [2][2]std.c.fd_t = undefined;
        for (&sockets) |*pair| try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, pair) == 0);
        defer for (sockets) |pair| for (pair) |fd| { _ = std.c.close(fd); };
        var leader: End = .{ .rank = 0, .links = .{ .{ .fd = sockets[0][0] }, .{ .fd = sockets[1][0] } } };
        const timeout_ns: u64 = if (silent) 80 * std.time.ns_per_ms else std.time.ns_per_s;
        var peers = [2]Peer{
            .{ .end = .{ .rank = 1, .links = .{ .{ .fd = sockets[0][1] }, null } }, .vote = votes[1], .silent = false, .after = expected == .no, .stage = stage, .timeout_ns = timeout_ns },
            .{ .end = .{ .rank = 2, .links = .{ .{ .fd = sockets[1][1] }, null } }, .vote = votes[2], .silent = silent, .after = expected == .no, .stage = stage, .timeout_ns = timeout_ns },
        };
        var threads: [2]std.Thread = undefined;
        var spawned: usize = 0;
        defer for (threads[0..spawned]) |thread| thread.join();
        for (&peers, 0..) |*p, i| {
            threads[i] = try std.Thread.spawn(.{}, Peer.run, .{p});
            spawned += 1;
        }
        var command: [128]u8 = undefined;
        const wire = try rankWire(0, &command);
        const agreement = [_]u8{ 15, 1, 0, 0, 0, 0, 0, 0, 0, 1 };
        const verify = [_]u8{ 3, 3, 0, 0, 0, 0, 0, 0, 0 };
        for (leader.links) |l| try l.?.send(wire);
        try agreeSpill(&leader.links, &agreement, std.time.ns_per_s);
        const stats = try statsRequest(&command);
        for (leader.links) |l| try l.?.send(stats);
        for (leader.links, 0..) |l, i| {
            var reply: [rw.max_reply]u8 = undefined;
            const got = try l.?.recvBy(&reply, link.Link.monotonicNs() + std.time.ns_per_s);
            const decoded = try rw.decode(got, &.{});
            try std.testing.expect(decoded.head.op == .stats_reply and decoded.head.from == i + 1 and decoded.head.seq == 2 and decoded.msg.stats_reply.set == .tier);
        }
        for (leader.links) |l| try l.?.send(&verify);
        var f = base;
        f.stage = stage;
        const result = exchange(leader.transport(), 0, f, votes[0], link.Link.monotonicNs() + timeout_ns);
        if (expected) |want| {
            try std.testing.expectEqual(want, try result);
            if (want == .no) {
                const after = try rankWire(4, &command);
                for (leader.links) |l| try l.?.send(after);
            }
        } else try std.testing.expectError(error.GraphVoteFatal, result);
        // Join before reading the thread-owned results; the defer above also joins on an early test failure.
        for (threads) |thread| thread.join();
        spawned = 0;
        for (peers, 0..) |p, i| {
            try std.testing.expect(p.err == null);
            if (silent and i == 1) continue;
            if (expected) |want| try std.testing.expectEqual(want, try p.result) else try expectFatal(p.result);
        }
    }

    const AgreementPeer = struct {
        l: link.Link,
        reply: ?bool,
        err: ?anyerror = null,
        fn go(p: *AgreementPeer) void {
            readRankWire(p.l, 0) catch |err| { p.err = err; return; };
            readAgreement(p.l, p.reply) catch |err| { p.err = err; };
        }
    };

    pub fn agreementFails(replies: [2]?bool, want: anyerror) !void {
        var sockets: [2][2]std.c.fd_t = undefined;
        for (&sockets) |*pair| try std.testing.expect(std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, pair) == 0);
        defer for (sockets) |pair| for (pair) |fd| { _ = std.c.close(fd); };
        const links: [2]?link.Link = .{ .{ .fd = sockets[0][0] }, .{ .fd = sockets[1][0] } };
        var peers = [2]AgreementPeer{ .{ .l = .{ .fd = sockets[0][1] }, .reply = replies[0] }, .{ .l = .{ .fd = sockets[1][1] }, .reply = replies[1] } };
        var threads: [2]std.Thread = undefined;
        var spawned: usize = 0;
        defer for (threads[0..spawned]) |thread| thread.join();
        for (&peers, 0..) |*p, i| {
            threads[i] = try std.Thread.spawn(.{}, AgreementPeer.go, .{p});
            spawned += 1;
        }
        var command: [128]u8 = undefined;
        const wire = try rankWire(0, &command);
        for (links) |l| try l.?.send(wire);
        const agreement = [_]u8{ 15, 1, 0, 0, 0, 0, 0, 0, 0, 1 };
        const result = agreeSpill(&links, &agreement, 80 * std.time.ns_per_ms);
        for (threads) |thread| thread.join();
        spawned = 0;
        for (peers) |p| try std.testing.expect(p.err == null);
        try std.testing.expectError(want, result);
    }
};

test "spill agreement: mismatched or silent rank refuses before any graph" {
    try SpillLinkTest.agreementFails(.{ false, true }, error.GraphVoteMismatch);
    try SpillLinkTest.agreementFails(.{ true, null }, error.Timeout);
}

test "spill vote: rank-wire commands and a no vote remain ordered on the same links" {
    try SpillLinkTest.run(.reserve, .{ .yes, .yes, .no }, .no, false);
}

test "spill vote: poisoned rank makes the ready vote fatal on every link" {
    try SpillLinkTest.run(.ready, .{ .yes, .fatal, .yes }, .fatal, false);
}

test "spill vote: silent follower times out and the other follower fails closed" {
    try SpillLinkTest.run(.reserve, .{ .yes, .yes, .yes }, null, true);
}
