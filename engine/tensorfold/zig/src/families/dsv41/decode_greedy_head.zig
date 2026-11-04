//! Whole-round, default-off greedy head. No host-staged logits.
const std = @import("std");

/// Actual Model.agree word[6] bit, NOT prefill.Flags. Bits 20-23 are
/// reserved by other decode work; existing prefill occupies at most 19.
pub const agreement_bit: u64 = @as(u64, 1) << 24;
pub fn enabled(v: ?[]const u8) bool {
    return if (v) |s| std.mem.eql(u8, s, "1") else false;
}
pub fn agreement(on: bool) u64 {
    return if (on) agreement_bit else 0;
}
pub fn agree(a: u64, b: u64) !void {
    if (a != b) return error.PrefillSettingMismatch;
}

/// Window has masks; Sampling has seed/temp/top_k/top_p/min_p, no logit
/// bias or repetition penalty. Other full-logit callers retain Model.verify.
/// Any future logit-editing field must be added here before enabling this path.
pub fn eligible(on: bool, two: bool, windows: anytype) bool {
    if (!on or two or windows.len == 0) return false;
    for (windows) |w| {
        if (w.masks.len != 0 or w.stream.grammar != null) return false;
        if (w.stream.sampling) |s|
            if (!std.math.isFinite(s.temperature) or s.temperature > 0) return false;
    }
    return true;
}

/// Shared by the actual leader/follower before any send/parse side effect.
/// pairFault is noreturn in production: never unwind into a CUDA drain.
pub const Custody = struct {
    active: bool = false,
    pub fn failed(c: Custody, model: anytype) !void {
        if (c.active) return model.pairFault(error.GreedyHeadRankFault);
    }
};

/// The production head and CPU fault seam have exactly this stream order.
pub fn run(host: anytype) !void {
    try host.local();
    try host.gather();
    try host.merge();
}

/// The actual leader/follower execution boundary. An eligible collective failure
/// terminates immediately, rather than returning through request/spill cleanup.
pub fn verify(model: anytype, rows: anytype, absorb: []const bool, use_head: bool) !void {
    model.verifyHead(rows, absorb, use_head) catch |err| {
        if (use_head) return model.pairFault(err);
        return err;
    };
}

pub const Output = struct {
    rows: usize = 0,
    pub fn begin(o: *Output) void {
        o.rows = 0;
    }
    pub fn complete(o: *Output, rows: usize) void {
        o.rows = rows;
    }
    pub fn selected(o: Output, rows: usize) !bool {
        if (o.rows != 0 and o.rows != rows) return error.BadGreedyHeadRows;
        return o.rows != 0;
    }
    pub fn requireLogits(o: Output) !void {
        if (o.rows != 0) return error.HeadLogitsUnavailable;
    }
};

test "decode greedy head exact flag and distinct actual agreement bit" {
    try std.testing.expect(!enabled(null));
    for ([_][]const u8{ "", "0", "true", "01", " 1", "1 " }) |s| try std.testing.expect(!enabled(s));
    try std.testing.expect(enabled("1"));
    try std.testing.expectEqual(@as(u64, 0), agreement(false));
    try std.testing.expectEqual(@as(u64, 1) << 24, agreement(true));
    const occupied_and_reserved = (@as(u64, 1) << 24) - 1;
    try std.testing.expectEqual(@as(u64, 0), agreement_bit & occupied_and_reserved);
    try agree(occupied_and_reserved, occupied_and_reserved);
    try agree(occupied_and_reserved | agreement_bit, occupied_and_reserved | agreement_bit);
    try std.testing.expectError(error.PrefillSettingMismatch, agree(occupied_and_reserved, occupied_and_reserved | agreement_bit));
    try std.testing.expectError(error.PrefillSettingMismatch, agree(occupied_and_reserved | agreement_bit, occupied_and_reserved));
}

test "decode greedy head whole round eligibility leaves unsupported paths baseline" {
    const S = struct { sampling: ?struct { temperature: f64 } = null, grammar: ?*u8 = null };
    const W = struct { stream: *S, masks: []const u32 = &.{} };
    var s: S = .{};
    var w = [_]W{.{ .stream = &s }};
    try std.testing.expect(eligible(true, false, &w));
    try std.testing.expect(!eligible(false, false, &w));
    try std.testing.expect(!eligible(true, true, &w));
    try std.testing.expect(!eligible(true, false, w[0..0]));
    for ([_]f64{ 0, -0.0, -1 }) |temp| {
        s.sampling = .{ .temperature = temp };
        try std.testing.expect(eligible(true, false, &w));
    }
    for ([_]f64{ 0.1, 1, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64) }) |temp| {
        s.sampling = .{ .temperature = temp };
        try std.testing.expect(!eligible(true, false, &w));
    }
    s.sampling = null;
    w[0].masks = &.{0xffffffff};
    try std.testing.expect(!eligible(true, false, &w));
    w[0].masks = &.{};
    var grammar: u8 = 0;
    s.grammar = &grammar;
    try std.testing.expect(!eligible(true, false, &w));
    var plain: S = .{};
    const mixed = [_]W{ .{ .stream = &plain }, .{ .stream = &s } };
    try std.testing.expect(!eligible(true, false, &mixed));
    s.grammar = null;
    s.sampling = .{ .temperature = 1 };
    try std.testing.expect(!eligible(true, false, &mixed));
}

test "decode greedy head production dispatch order and each partial failure" {
    const H = struct {
        calls: u8 = 0,
        fail: u8,
        fn step(h: *@This()) !void {
            h.calls += 1;
            if (h.calls == h.fail) return error.Injected;
        }
        pub fn local(h: *@This()) !void {
            try std.testing.expectEqual(@as(u8, 0), h.calls);
            try h.step();
        }
        pub fn gather(h: *@This()) !void {
            try std.testing.expectEqual(@as(u8, 1), h.calls);
            try h.step();
        }
        pub fn merge(h: *@This()) !void {
            try std.testing.expectEqual(@as(u8, 2), h.calls);
            try h.step();
        }
    };
    for (0..4) |i| {
        var h: H = .{ .fail = @intCast(i) };
        if (i == 0) try run(&h) else try std.testing.expectError(error.Injected, run(&h));
        try std.testing.expectEqual(@as(u8, @intCast(if (i == 0) 3 else i)), h.calls);
    }
}

test "decode greedy head output readiness resets and never exposes stale full logits" {
    var out: Output = .{};
    try out.requireLogits();
    try std.testing.expect(!try out.selected(16));
    out.complete(16); // only after production verifyRun's final fence
    try std.testing.expect(try out.selected(16));
    try std.testing.expect(try out.selected(16)); // repeated read is stable
    try std.testing.expectError(error.BadGreedyHeadRows, out.selected(1));
    try std.testing.expectError(error.HeadLogitsUnavailable, out.requireLogits());
    out.begin(); // failure before completion cannot retain the preceding result
    try out.requireLogits();
    try std.testing.expect(!try out.selected(16));
    out.complete(1);
    try std.testing.expect(try out.selected(1));
    out.begin(); // next baseline round
    try out.requireLogits();
}

/// Each row count qualifies once, eagerly, before ANY stretch is captured.
/// Failed comparisons never publish qualification or selected output.
pub fn Qualification(comptime max_rows: usize) type {
    return struct {
        checked: [max_rows + 1]bool = @splat(false),
        pub fn needsOracle(q: *const @This(), rows: usize) !bool {
            if (rows == 0 or rows > max_rows) return error.BadGreedyHeadRows;
            return !q.checked[rows];
        }
        pub fn accept(q: *@This(), expected: []const u32, actual: []const u32) !void {
            _ = try q.needsOracle(expected.len);
            if (!std.mem.eql(u32, expected, actual)) return error.GreedyHeadMismatch;
            q.checked[expected.len] = true;
        }
    };
}

test "decode greedy head oracle qualifies each shape only after exact comparison" {
    var q: Qualification(48) = .{};
    try std.testing.expectError(error.BadGreedyHeadRows, q.needsOracle(0));
    try std.testing.expectError(error.BadGreedyHeadRows, q.needsOracle(49));
    var expected: [48]u32 = undefined;
    var actual: [48]u32 = undefined;
    for (1..49) |rows| {
        for (0..rows) |i| expected[i] = @intCast(i * 43136);
        @memcpy(actual[0..rows], expected[0..rows]);
        try std.testing.expect(try q.needsOracle(rows));
        actual[rows - 1] ^= 1;
        try std.testing.expectError(error.GreedyHeadMismatch, q.accept(expected[0..rows], actual[0..rows]));
        try std.testing.expect(try q.needsOracle(rows));
        actual[rows - 1] ^= 1;
        try q.accept(expected[0..rows], actual[0..rows]);
        try std.testing.expect(!try q.needsOracle(rows));
    }
    try std.testing.expectError(error.GreedyHeadMismatch, q.accept(&.{1}, &.{}));
}

/// Reserved with the ports coordinator. Requires TF_DS_SAMPLE_CAND as well.
pub const sample_agreement_bit: u64 = @as(u64, 1) << 25;
pub fn sampleCount(on: bool, unsupported: bool, vocab: usize, windows: anytype) usize {
    if (!on or unsupported or windows.len == 0) return 0;
    var count: usize = 0;
    for (windows) |w| {
        if (w.masks.len != 0 or w.stream.grammar != null) return 0;
        const sm = w.stream.sampling orelse return 0;
        if (!std.math.isFinite(sm.temperature) or sm.temperature <= 0 or sm.top_k == 0 or sm.top_k > 56) return 0;
        count = @max(count, @min(vocab, sm.top_k + 8));
    }
    return count;
}

pub fn verifySample(model: anytype, rows: anytype, absorb: []const bool, count: usize) !void {
    model.verifySampleHead(rows, absorb, count) catch |err| return model.pairFault(err);
}

test "decode greedy head sampled count matches baseline margin and exclusions" {
    const S = struct { sampling: ?struct { temperature: f64, top_k: usize } = null, grammar: ?*u8 = null };
    const W = struct { stream: *S, masks: []const u32 = &.{} };
    var s: S = .{ .sampling = .{ .temperature = 1, .top_k = 20 } };
    var w = [_]W{.{ .stream = &s }};
    try std.testing.expectEqual(@as(usize, 28), sampleCount(true, false, 129280, &w));
    try std.testing.expectEqual(@as(usize, 0), sampleCount(false, false, 129280, &w));
    try std.testing.expectEqual(@as(usize, 0), sampleCount(true, true, 129280, &w));
    for ([_]usize{0, 57, 100}) |k| {
        s.sampling.?.top_k = k;
        try std.testing.expectEqual(@as(usize, 0), sampleCount(true, false, 129280, &w));
    }
    s.sampling.?.top_k = 56;
    try std.testing.expectEqual(@as(usize, 64), sampleCount(true, false, 129280, &w));
    for ([_]f64{0, -1, std.math.nan(f64), std.math.inf(f64)}) |temp| {
        s.sampling.?.temperature = temp;
        try std.testing.expectEqual(@as(usize, 0), sampleCount(true, false, 129280, &w));
    }
    s.sampling.?.temperature = 1;
    w[0].masks = &.{1};
    try std.testing.expectEqual(@as(usize, 0), sampleCount(true, false, 129280, &w));
    try std.testing.expectEqual(@as(u64, 0), sample_agreement_bit & agreement_bit);
}

/// Raw value encodings matter: +0 and -0 tie, but the selected token's bits survive.
pub fn compareSampleRow(flag: i32, ref_flag: i32, vals: []const u32, ref_vals: []const u32, ids: []const i64, ref_ids: []const i64) !void {
    if (flag != ref_flag) return error.SampleHeadMismatch;
    if (flag != 0) return;
    if (!std.mem.eql(u32, vals, ref_vals) or !std.mem.eql(i64, ids, ref_ids)) return error.SampleHeadMismatch;
}

test "decode greedy head sampled oracle compares flags raw bits and ids" {
    try compareSampleRow(0, 0, &.{0x80000000}, &.{0x80000000}, &.{7}, &.{7});
    try std.testing.expectError(error.SampleHeadMismatch, compareSampleRow(0, 0, &.{0}, &.{0x80000000}, &.{7}, &.{7}));
    try std.testing.expectError(error.SampleHeadMismatch, compareSampleRow(0, 0, &.{0}, &.{0}, &.{7}, &.{8}));
    try std.testing.expectError(error.SampleHeadMismatch, compareSampleRow(1, 2, &.{}, &.{}, &.{}, &.{}));
    try compareSampleRow(1, 1, &.{}, &.{}, &.{}, &.{});
    try compareSampleRow(2, 2, &.{}, &.{}, &.{}, &.{});
}
