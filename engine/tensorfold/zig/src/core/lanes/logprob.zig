//! Raw temperature-one model probabilities, before sampling filters and grammar masks.
const std = @import("std");
pub const top_cap = 20;
pub const Candidate = extern struct { value: f32, id: u32 };
pub const Summary = extern struct {
    maximum: f32,
    mass: f32,
    chosen: f32,
    owns: u32,
    flags: u32,
    count: u32,
    top: [top_cap]Candidate,
};
pub const Token = struct { id: u32, logprob: f64 };
pub const Row = struct {
    chosen: Token,
    count: u8,
    top: [top_cap]Token = undefined,
};
comptime { std.debug.assert(@sizeOf(Summary) == 184); }

fn before(a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

pub fn merge(shards: []const Summary, chosen: u32, k: u8) !Row {
    if (k > top_cap or shards.len == 0) return error.BadTopLogprobs;
    var maximum = -std.math.inf(f32);
    var value = -std.math.inf(f32);
    var owners: usize = 0;
    for (shards) |s| {
        if (s.flags != 0 or s.count > top_cap or s.owns > 1 or
            std.math.isNan(s.maximum) or s.maximum == std.math.inf(f32) or
            !std.math.isFinite(s.mass) or s.mass < 0 or
            (s.maximum == -std.math.inf(f32) and s.mass != 0) or
            (std.math.isFinite(s.maximum) and s.mass < 1)) return error.BadLogprobRow;
        maximum = @max(maximum, s.maximum);
        if (s.owns == 1) {
            if (std.math.isNan(s.chosen) or s.chosen == std.math.inf(f32)) return error.BadLogprobRow;
            owners += 1;
            value = s.chosen;
        }
    }
    if (owners != 1 or maximum == -std.math.inf(f32)) return error.BadLogprobRow;
    var mass: f64 = 0;
    var candidates: [top_cap]Candidate = undefined;
    var count: usize = 0;
    for (shards) |s| {
        if (s.maximum != -std.math.inf(f32))
            mass += @as(f64, s.mass) * @exp(@as(f64, s.maximum) - @as(f64, maximum));
        for (s.top[0..s.count]) |c| {
            if (std.math.isNan(c.value) or c.value == std.math.inf(f32)) return error.BadLogprobRow;
            if (k == 0 or (count == k and !before(c, candidates[count - 1]))) continue;
            var j = @min(count, k - 1);
            while (j > 0 and before(c, candidates[j - 1])) : (j -= 1) candidates[j] = candidates[j - 1];
            candidates[j] = c;
            count = @min(count + 1, k);
        }
    }
    if (!(mass > 0) or !std.math.isFinite(mass)) return error.BadLogprobRow;
    const correction = @log(mass);
    var out: Row = .{ .chosen = .{ .id = chosen, .logprob = (@as(f64, value) - @as(f64, maximum)) - correction }, .count = @intCast(count) };
    for (candidates[0..count], out.top[0..count]) |c, *t|
        t.* = .{ .id = c.id, .logprob = (@as(f64, c.value) - @as(f64, maximum)) - correction };
    return out;
}

/// CPU oracle for arbitrary fp32 input, including unequal and empty shards.
pub fn reference(row: []const f32, chosen: u32, k: u8) !Row {
    if (row.len == 0 or chosen >= row.len or k > top_cap) return error.BadLogprobRow;
    var maximum = -std.math.inf(f64);
    for (row) |v| {
        if (std.math.isNan(v) or v == std.math.inf(f32)) return error.BadLogprobRow;
        maximum = @max(maximum, v);
    }
    if (maximum == -std.math.inf(f64)) return error.BadLogprobRow;
    var mass: f64 = 0;
    for (row) |v| mass += @exp(@as(f64, v) - maximum);
    const correction = @log(mass);
    var out: Row = .{ .chosen = .{ .id = chosen, .logprob = (@as(f64, row[chosen]) - maximum) - correction }, .count = 0 };
    for (row, 0..) |v, i| {
        const t: Token = .{ .id = @intCast(i), .logprob = (@as(f64, v) - maximum) - correction };
        if (k == 0 or (out.count == k and !(t.logprob > out.top[out.count - 1].logprob))) continue;
        var j: usize = @min(out.count, k - 1);
        while (j > 0 and t.logprob > out.top[j - 1].logprob) : (j -= 1) out.top[j] = out.top[j - 1];
        out.top[j] = t;
        out.count = @min(out.count + 1, k);
    }
    return out;
}

fn summary(row: []const f32, offset: u32, chosen: u32, k: u8) Summary {
    var s: Summary = .{ .maximum = -std.math.inf(f32), .mass = 0, .chosen = -std.math.inf(f32), .owns = 0, .flags = 0, .count = 0, .top = undefined };
    for (row, 0..) |v, i| {
        s.maximum = @max(s.maximum, v);
        if (offset + i == chosen) { s.owns = 1; s.chosen = v; }
        const c: Candidate = .{ .value = v, .id = @intCast(offset + i) };
        if (k == 0 or (s.count == k and !before(c, s.top[s.count - 1]))) continue;
        var j: usize = @min(s.count, k - 1);
        while (j > 0 and before(c, s.top[j - 1])) : (j -= 1) s.top[j] = s.top[j - 1];
        s.top[j] = c;
        s.count = @min(s.count + 1, k);
    }
    var mass: f64 = 0;
    if (s.maximum != -std.math.inf(f32)) for (row) |v| { mass += @exp(@as(f64, v) - @as(f64, s.maximum)); };
    s.mass = @floatCast(mass);
    return s;
}

test "raw logprob TP3 merge matches fp64 whole vocabulary, ties and chosen outside top" {
    var rng = std.Random.DefaultPrng.init(0x1443);
    var row: [129280]f32 = undefined;
    for (0..40) |trial| {
        for (&row) |*v| v.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -2000, 2000))) / 100;
        const chosen: u32 = @intCast(trial * 1200);
        const k: u8 = @intCast(trial % 21);
        const pieces = [_]Summary{
            summary(row[0..43136], 0, chosen, k),
            summary(row[43136..86272], 43136, chosen, k),
            summary(row[86272..], 86272, chosen, k),
        };
        const actual = try merge(&pieces, chosen, k);
        const expected = try reference(&row, chosen, k);
        try std.testing.expectApproxEqAbs(expected.chosen.logprob, actual.chosen.logprob, 1e-5);
        try std.testing.expectEqual(expected.count, actual.count);
        for (expected.top[0..k], actual.top[0..k]) |a, b| {
            try std.testing.expectEqual(a.id, b.id);
            try std.testing.expectApproxEqAbs(a.logprob, b.logprob, 1e-5);
        }
    }
}

test "raw logprob split LSE preserves large offsets and empty shards" {
    const row: [5]f32 = @splat(1e8);
    const pieces = [_]Summary{ summary(row[0..2], 0, 4, 5), summary(row[2..], 2, 4, 5), summary(&.{}, 5, 4, 5) };
    const actual = try merge(&pieces, 4, 5);
    try std.testing.expectApproxEqAbs(-@log(@as(f64, 5)), actual.chosen.logprob, 1e-12);
    for (actual.top[0..5], 0..) |v, i| try std.testing.expectEqual(i, v.id);
    try std.testing.expectError(error.BadTopLogprobs, merge(&pieces, 4, 21));
}

test "raw logprob refuses malformed rows and allows a minus-infinity shard" {
    const row = [_]f32{ -std.math.inf(f32), -std.math.inf(f32), 2, 1 };
    var pieces = [_]Summary{ summary(row[0..2], 0, 3, 4), summary(row[2..], 2, 3, 4) };
    const out = try merge(&pieces, 3, 4);
    try std.testing.expectApproxEqAbs((try reference(&row, 3, 4)).chosen.logprob, out.chosen.logprob, 1e-5);
    pieces[0].flags = 1;
    try std.testing.expectError(error.BadLogprobRow, merge(&pieces, 3, 4));
    pieces[0].flags = 0;
    pieces[0].owns = 1;
    try std.testing.expectError(error.BadLogprobRow, merge(&pieces, 3, 4));
    const bad = [_]Summary{summary(&.{-std.math.inf(f32)}, 0, 0, 1)};
    try std.testing.expectError(error.BadLogprobRow, merge(&bad, 0, 1));
}
