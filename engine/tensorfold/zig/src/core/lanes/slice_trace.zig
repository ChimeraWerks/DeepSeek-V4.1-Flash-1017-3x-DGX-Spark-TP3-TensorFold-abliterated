//! Host-only diagnostic. No clock, allocator or file operation on the null path.
const std = @import("std");

pub const Kind = enum { phase, wait, slice, park, decode, send, admission, cancels, service, idle, enqueue };
pub const Event = struct { name: []const u8, kind: Kind, begin_us: u64, end_us: u64 = std.math.maxInt(u64), parent: ?usize, lo: usize = 0, hi: usize = 0 };
pub const Identity = struct { slot: usize = 0, start: usize = 0, rows: usize = 0, replay: usize = 0, encoder_only: bool = false, layers: usize = 0 };
pub threadlocal var current: ?*Trace = null;
pub const Counters = struct { clocks: usize = 0, allocations: usize = 0, opens: usize = 0, writes: usize = 0 };
pub threadlocal var counters: Counters = .{};

pub const Scope = struct {
    trace: ?*Trace = null,
    index: usize = 0,
    generation: usize = 0,
    pub fn end(s: *Scope) void {
        const t = s.trace orelse return;
        s.trace = null;
        if (!t.enabled or !t.chunk or s.generation != t.generation) return;
        t.events[s.index].end_us = t.stamp() -| t.origin;
        t.top = t.events[s.index].parent;
    }
};

pub fn begin(name: []const u8, kind: Kind) Scope {
    const t = current orelse return .{};
    if (!t.enabled or !t.chunk) return .{};
    if (kind == .enqueue and t.quiet > 0) return .{};
    if (t.used == t.events.len) { t.disable("event buffer full"); return .{}; }
    const i = t.used;
    t.used += 1;
    t.events[i] = .{ .name = name, .kind = kind, .begin_us = t.stamp() -| t.origin, .parent = t.top };
    t.top = i;
    return .{ .trace = t, .index = i, .generation = t.generation };
}

/// A zero-length event: a fact at a point (rows dropped, rows reused), lo and hi as the site documents.
pub fn mark(name: []const u8, lo: usize, hi: usize) void {
    const s = indexed(name, .phase, lo, hi);
    const t = s.trace orelse return;
    t.events[s.index].end_us = t.events[s.index].begin_us; // one clock read
    t.top = t.events[s.index].parent;
}

/// Inside a park's decode round, per-launch and cuBLAS events are dropped: an ungraphed round launches thousands of
/// kernels, which would exhaust the event buffer at one-layer slices. The round's own scope still times it.
pub const Quiet = struct {
    trace: ?*Trace = null,
    pub fn end(q: *Quiet) void {
        const t = q.trace orelse return;
        q.trace = null;
        t.quiet -|= 1;
    }
};

pub fn quietEnqueue() Quiet {
    const t = current orelse return .{};
    if (!t.enabled or !t.chunk) return .{};
    t.quiet += 1;
    return .{ .trace = t };
}

pub fn slice(lo: usize, hi: usize) Scope {
    return indexed("model.fillSlice", .slice, lo, hi);
}

pub fn indexed(name: []const u8, kind: Kind, lo: usize, hi: usize) Scope {
    const s = begin(name, kind);
    if (s.trace) |t| { t.events[s.index].lo = lo; t.events[s.index].hi = hi; }
    return s;
}

pub const Trace = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    fd: std.c.fd_t,
    events: []Event,
    bytes: []u8,
    enabled: bool = true,
    chunk: bool = false,
    owned_by_step: bool = false,
    used: usize = 0,
    top: ?usize = null,
    origin: u64 = 0,
    generation: usize = 0,
    id: Identity = .{},
    last_end: ?u64 = null,
    gap_us: u64 = 0,
    gap: [@intFromEnum(Kind.enqueue) + 1]u64 = @splat(0),
    gap_rounds: usize = 0,
    gap_depth: usize = 0,
    quiet: usize = 0,

    pub fn fromEnv(gpa: std.mem.Allocator, io: std.Io) ?*Trace {
        const path = std.c.getenv("TF_DS_SLICE_TRACE") orelse return null;
        if (path[0] == 0) return null;
        return init(gpa, io, std.mem.span(path)) catch |err| {
            std.debug.print("slice trace disabled: {s}\n", .{@errorName(err)});
            return null;
        };
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, path: [:0]const u8) !*Trace {
        counters.opens += 1;
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.TraceOpen;
        errdefer _ = std.c.close(fd);
        counters.allocations += 1;
        const t = try gpa.create(Trace);
        errdefer gpa.destroy(t);
        counters.allocations += 1;
        const events = try gpa.alloc(Event, 65536);
        errdefer gpa.free(events);
        counters.allocations += 1;
        const bytes = try gpa.alloc(u8, 16 << 20);
        t.* = .{ .gpa = gpa, .io = io, .fd = fd, .events = events, .bytes = bytes };
        return t;
    }

    pub fn deinit(t: *Trace) void {
        _ = std.c.close(t.fd);
        t.gpa.free(t.events);
        t.gpa.free(t.bytes);
        const gpa = t.gpa;
        gpa.destroy(t);
    }

    fn stamp(t: *Trace) u64 {
        counters.clocks += 1;
        return @intCast(@max(0, std.Io.Clock.awake.now(t.io).toNanoseconds()) / 1000);
    }

    pub fn start(t: *Trace) void {
        if (!t.enabled) return;
        t.origin = t.stamp();
        t.gap_us = if (t.last_end) |last| t.origin -| last else 0;
        t.used = 0;
        t.top = null;
        t.id = .{};
        t.chunk = true;
        t.owned_by_step = false;
        t.generation +%= 1;
    }

    pub fn finish(t: *Trace) void {
        if (!t.enabled or !t.chunk) return;
        const end = t.stamp();
        const wall = end -| t.origin;
        t.chunk = false;
        // Any live scope is an error-path scope. Close it at the chunk boundary.
        for (t.events[0..t.used]) |*e| if (e.end_us == std.math.maxInt(u64)) { e.end_us = wall; };
        var covered: u64 = 0;
        for (t.events[0..t.used]) |e| if (e.parent == null) { covered +|= e.end_us -| e.begin_us; };
        var w = std.Io.Writer.fixed(t.bytes);
        const gap_sum = blk: { var n: u64 = 0; for (t.gap) |v| n +|= v; break :blk n; };
        w.print("{{\"version\":1,\"clock\":\"host_awake_us\",\"slot\":{d},\"start\":{d},\"rows\":{d},\"replay\":{d},\"encoder_only\":{},\"layers\":{d},\"wall_us\":{d},\"slack_us\":{d},\"slice_sync\":true,\"gpu_time_known\":false,\"between_chunks\":{{\"wall_us\":{d},\"decode_rounds\":{d},\"decode_us\":{d},\"admission_us\":{d},\"cancels_us\":{d},\"service_us\":{d},\"idle_us\":{d},\"send_us\":{d},\"other_us\":{d}}},\"events\":[", .{ t.id.slot, t.id.start, t.id.rows, t.id.replay, t.id.encoder_only, t.id.layers, wall, wall -| covered, t.gap_us, t.gap_rounds, t.gap[@intFromEnum(Kind.decode)], t.gap[@intFromEnum(Kind.admission)], t.gap[@intFromEnum(Kind.cancels)], t.gap[@intFromEnum(Kind.service)], t.gap[@intFromEnum(Kind.idle)], t.gap[@intFromEnum(Kind.send)], t.gap_us -| gap_sum }) catch { t.disable("serialization buffer full"); return; };
        for (t.events[0..t.used], 0..) |e, i| {
            w.print("{s}{{\"name\":\"{s}\",\"kind\":\"{s}\",\"begin_us\":{d},\"end_us\":{d},\"parent\":", .{ if (i == 0) "" else ",", e.name, @tagName(e.kind), e.begin_us, e.end_us }) catch { t.disable("serialization buffer full"); return; };
            if (e.parent) |parent| w.print("{d}", .{parent}) catch { t.disable("serialization buffer full"); return; } else w.writeAll("null") catch { t.disable("serialization buffer full"); return; };
            w.print(",\"lo\":{d},\"hi\":{d}}}", .{ e.lo, e.hi }) catch { t.disable("serialization buffer full"); return; };
        }
        w.writeAll("]}\n") catch { t.disable("serialization buffer full"); return; };
        var at: usize = 0;
        const out = w.buffered();
        while (at < out.len) {
            counters.writes += 1;
            const n = std.c.write(t.fd, out.ptr + at, out.len - at);
            if (n < 0 and std.c.errno(n) == .INTR) continue;
            if (n <= 0) { t.disable("write failed"); return; }
            at += @intCast(n);
        }
        t.last_end = end;
        t.gap = @splat(0);
        t.gap_rounds = 0;
    }

    fn disable(t: *Trace, why: []const u8) void {
        if (!t.enabled) return;
        t.enabled = false;
        std.debug.print("slice trace disabled: {s}\n", .{why});
    }
};

pub const GapScope = struct {
    scope: Scope = .{},
    trace: ?*Trace = null,
    at: u64 = 0,
    kind: Kind = .phase,
    generation: usize = 0,
    outer: bool = false,
    pub fn end(s: *GapScope) void {
        s.scope.end();
        if (s.trace) |t| {
            s.trace = null;
            t.gap_depth -= 1;
            if (s.outer and t.enabled and !t.chunk and s.generation == t.generation) {
                t.gap[@intFromEnum(s.kind)] +|= t.stamp() -| s.at;
                if (s.kind == .decode) t.gap_rounds += 1;
            }
        }
    }
};

pub fn host(name: []const u8, kind: Kind) GapScope {
    const t = current orelse return .{};
    if (!t.enabled) return .{};
    if (t.chunk) return .{ .scope = begin(name, kind) };
    if (t.last_end == null) return .{};
    const outer = t.gap_depth == 0;
    t.gap_depth += 1;
    return .{ .trace = t, .at = if (outer) t.stamp() else 0, .kind = kind, .generation = t.generation, .outer = outer };
}

test "prefill slice trace null path has no clock allocation or file access" {
    current = null;
    counters = .{};
    for (0..100) |_| {
        var s = begin("unused", .wait); s.end();
        var g = host("unused", .decode); g.end();
        var x = slice(0, 1); x.end();
        var q = quietEnqueue(); q.end();
        mark("unused", 0, 1);
    }
    try std.testing.expectEqual(Counters{}, counters);
}

test "prefill slice trace quiet drops enqueue events only and marks are zero length" {
    var events: [8]Event = undefined;
    var bytes: [4096]u8 = undefined;
    var t: Trace = .{ .gpa = std.testing.allocator, .io = std.testing.io, .fd = -1, .events = &events, .bytes = &bytes };
    const previous = current;
    current = &t;
    defer current = previous;
    t.start();
    var round = begin("test.round", .decode);
    var quiet = quietEnqueue();
    var launch = begin("test.launch", .enqueue);
    launch.end();
    var sync = begin("test.sync", .wait);
    sync.end();
    quiet.end();
    var later = begin("test.launch_after", .enqueue);
    later.end();
    round.end();
    mark("test.mark", 3, 9);
    try std.testing.expectEqual(@as(usize, 4), t.used);
    try std.testing.expectEqualStrings("test.sync", t.events[1].name);
    try std.testing.expectEqualStrings("test.launch_after", t.events[2].name);
    try std.testing.expectEqual(@as(usize, 0), t.quiet);
    try std.testing.expectEqual(t.events[3].begin_us, t.events[3].end_us);
    try std.testing.expectEqual(@as(usize, 9), t.events[3].hi);
    try std.testing.expectEqual(@as(?usize, null), t.events[3].parent);
}

test "prefill slice trace failed writes disable once and stop all tracing" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const t = try Trace.init(std.testing.allocator, std.testing.io, "/dev/full");
    defer t.deinit();
    const previous = current;
    current = t;
    defer current = previous;
    t.start();
    var scope = begin("test.write_failure", .phase);
    scope.end();
    t.finish();
    try std.testing.expect(!t.enabled);
    const after = counters;
    t.start();
    var later = begin("test.disabled", .wait);
    later.end();
    t.finish();
    try std.testing.expectEqual(after, counters);
}
