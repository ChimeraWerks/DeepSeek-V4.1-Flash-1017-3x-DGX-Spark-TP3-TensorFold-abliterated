//! Bounded, host-only graph telemetry. No CUDA handles, files, allocation or retention policy.
const std = @import("std");

pub const agreement_bit: u64 = @as(u64, 1) << 50;

pub fn enabledText(text: ?[]const u8) bool {
    return if (text) |v| std.mem.eql(u8, v, "1") else false;
}

pub fn enabled() bool {
    return enabledText(if (std.c.getenv("TF_DS_GRAPH_TELEMETRY")) |v| std.mem.span(v) else null);
}

/// Owned by one cache and touched only by its engine thread. A miss includes
/// reserve/ready voting and possible eviction, so its entire sample is dirty.
/// Tombstone/disabled eager hits do not recapture and can be timed normally.
pub const Stats = struct {
    on: bool = false,
    clean: bool = true,
    accesses: u64 = 0,
    hits: u64 = 0,
    misses: u64 = 0,
    capture_attempts: u64 = 0,
    captures: u64 = 0,
    tombstones: u64 = 0,
    eager: u64 = 0,
    failures: u64 = 0,
    peak_keys: usize = 0,

    pub fn begin(s: *Stats) void {
        s.clean = true;
    }
    pub fn timingClean(s: *const Stats) bool {
        return !s.on or s.clean;
    }
    pub fn access(s: *Stats) void {
        if (s.on) s.accesses +|= 1;
    }
    pub fn hit(s: *Stats) void {
        if (s.on) s.hits +|= 1;
    }
    pub fn miss(s: *Stats) void {
        if (!s.on) return;
        s.misses +|= 1;
        s.clean = false;
    }
    pub fn capture(s: *Stats) void {
        if (s.on) s.capture_attempts +|= 1;
    }
    pub fn fail(s: *Stats) void {
        if (!s.on) return;
        s.failures +|= 1;
        s.clean = false;
    }
    pub fn eagerRun(s: *Stats) void {
        if (s.on) s.eager +|= 1;
    }
    pub fn kept(s: *Stats, graph: bool, keys: usize) void {
        if (!s.on) return;
        if (graph) s.captures +|= 1 else s.tombstones +|= 1;
        s.peak_keys = @max(s.peak_keys, keys);
    }

    /// Logarithmically bounded in accesses (plus one teardown summary). LRU
    /// eviction totals are authoritative, including evicted tombstones.
    pub fn note(s: *Stats, kind: []const u8, keys: usize, evicted: u64, final: bool) void {
        if (!s.on or (!final and !std.math.isPowerOfTwo(s.accesses))) return;
        s.clean = false; // Diagnostic I/O is not steady-state execution time either.
        std.log.info("dsv41: graph telemetry {s}: accesses={d} hits={d} misses={d} capture_attempts={d} captures={d} tombstones={d} eager={d} failures={d} keys={d} peak_keys={d} evicted={d} final={}", .{
            kind,         s.accesses, s.hits,     s.misses, s.capture_attempts, s.captures,
            s.tombstones, s.eager,    s.failures, keys,     s.peak_keys,        evicted,
            final,
        });
    }
};

test "graph telemetry is strictly opt in and reserves only agreement bit 50" {
    try std.testing.expect(!enabledText(null));
    for ([_][]const u8{ "", "0", "true", "01", "1 " }) |v| try std.testing.expect(!enabledText(v));
    try std.testing.expect(enabledText("1"));
    try std.testing.expectEqual(@as(u64, 1) << 50, agreement_bit);
}

test "graph telemetry disabled preserves timing and counters" {
    var s: Stats = .{};
    s.access();
    s.miss();
    s.capture();
    s.kept(true, 128);
    s.fail();
    s.hit();
    s.eagerRun();
    try std.testing.expect(s.timingClean());
    try std.testing.expectEqualDeep(Stats{}, s);
}

test "graph telemetry miss is sticky through hits until next engine round" {
    var s: Stats = .{ .on = true };
    s.begin();
    s.access();
    s.miss();
    s.capture();
    s.kept(true, 1);
    s.access();
    s.hit();
    try std.testing.expect(!s.timingClean());
    s.begin();
    s.access();
    s.hit();
    try std.testing.expect(s.timingClean());
    try std.testing.expectEqual(@as(u64, 3), s.accesses);
    try std.testing.expectEqual(@as(u64, 2), s.hits);
    try std.testing.expectEqual(@as(u64, 1), s.captures);
}

test "graph telemetry failed first launch and voted tombstones are not captures" {
    var s: Stats = .{ .on = true };
    s.miss();
    s.capture();
    s.fail();
    try std.testing.expectEqual(@as(u64, 0), s.captures);
    try std.testing.expectEqual(@as(usize, 0), s.peak_keys);
    s.begin();
    s.miss();
    s.kept(false, 1);
    s.eagerRun();
    try std.testing.expect(!s.timingClean());
    try std.testing.expectEqual(@as(u64, 1), s.tombstones);
    s.begin();
    s.eagerRun();
    try std.testing.expect(s.timingClean());
}

test "graph telemetry counters saturate without wrapping" {
    var s: Stats = .{ .on = true, .accesses = std.math.maxInt(u64), .hits = std.math.maxInt(u64) };
    s.access();
    s.hit();
    try std.testing.expectEqual(std.math.maxInt(u64), s.accesses);
    try std.testing.expectEqual(std.math.maxInt(u64), s.hits);
}
