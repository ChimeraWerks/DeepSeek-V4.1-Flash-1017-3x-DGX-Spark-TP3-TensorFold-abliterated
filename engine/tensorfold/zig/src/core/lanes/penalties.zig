//! Request penalties. Histories are prefixes, never mutable counter state.
const std = @import("std");
pub const Settings = struct {
    presence: f64 = 0,
    frequency: f64 = 0,
    repetition: f64 = 1,
    window: u32 = 0,
    include_prompt: bool = false,
    pub fn active(s: Settings) bool {
        return s.presence != 0 or s.frequency != 0 or (s.repetition != 1 and s.window != 0);
    }
    pub fn validate(s: Settings) !void {
        if (!std.math.isFinite(s.presence) or @abs(s.presence) > 2 or
            !std.math.isFinite(s.frequency) or @abs(s.frequency) > 2 or
            !std.math.isFinite(s.repetition) or s.repetition <= 0) return error.BadPenalties;
        if (s.repetition != 1 and s.window == 0) return error.PenaltyWindowRequired;
    }
};
pub const Counts = extern struct { count: u32 = 0, repeats: u32 = 0 };
pub const Params = extern struct { presence: f64, frequency: f64, repetition: f64 };
pub fn params(s: Settings) Params {
    return .{ .presence = s.presence, .frequency = s.frequency, .repetition = s.repetition };
}
/// Prefix length is the position being sampled. Generated-only begins at prompt_len.
pub fn counts(out: []Counts, seq: []const i32, prompt_len: usize, s: Settings, col0: usize) !void {
    try s.validate();
    if (seq.len < prompt_len or seq.len > std.math.maxInt(u32)) return error.BadPenaltyHistory;
    @memset(out, .{});
    if (!s.active()) return;
    const begin = if (s.include_prompt) 0 else prompt_len;
    const repeat_begin = seq.len -| s.window;
    for (seq, 0..) |token, i| {
        if (token < 0) return error.BadPenaltyHistory;
        const id: usize = @intCast(token);
        if (id < col0 or id - col0 >= out.len) continue;
        const c = &out[id - col0];
        if (i >= begin) c.count += 1;
        if (i >= repeat_begin) c.repeats += 1;
    }
}
/// A chain's first row scans history once. Following rows add one token and expire one window token.
pub fn chain(out: []Counts, seq: []const i32, first_end: usize, n: usize, prompt_len: usize, s: Settings, col0: usize, width: usize) !void {
    if (n == 0 or first_end == 0 or first_end > seq.len or n - 1 > seq.len - first_end or out.len != n * width) return error.BadPenaltyHistory;
    try counts(out[0..width], seq[0..first_end], prompt_len, s, col0);
    for (1..n) |r| {
        const row = out[r * width ..][0..width];
        @memcpy(row, out[(r - 1) * width ..][0..width]);
        if (!s.active()) continue;
        const end = first_end + r;
        const raw = seq[end - 1];
        if (raw < 0) return error.BadPenaltyHistory;
        const id: usize = @intCast(raw);
        if (id >= col0 and id - col0 < width) {
            const c = &row[id - col0];
            if (s.include_prompt or end - 1 >= prompt_len) c.count += 1;
            if (s.window > 0) c.repeats += 1;
        }
        if (s.window > 0 and end > s.window) {
            const old: usize = @intCast(seq[end - s.window - 1]);
            if (old >= col0 and old - col0 < width) row[old - col0].repeats -= 1;
        }
    }
}
/// Preserve infinities and masks; repetition first, additive penalties second.
pub fn apply(value: f32, c: Counts, s: Settings) f32 {
    if (!s.active() or !std.math.isFinite(value)) return value;
    var x: f64 = value;
    if (c.repeats > 0 and s.repetition != 1) x = if (x > 0) x / s.repetition else x * s.repetition;
    const additive = (if (c.count > 0) s.presence else 0) + s.frequency * @as(f64, @floatFromInt(c.count));
    return @floatCast(x - additive);
}
test "penalties generated only, prompt inclusion and repetition window" {
    var c: [4]Counts = undefined;
    const h = [_]i32{ 0, 0, 1, 2, 2 };
    var s: Settings = .{ .presence = 0.5, .frequency = 0.25, .repetition = 2, .window = 2 };
    try counts(&c, &h, 3, s, 0);
    try std.testing.expectEqual(Counts{}, c[0]);
    try std.testing.expectEqual(Counts{ .count = 2, .repeats = 2 }, c[2]);
    try std.testing.expectEqual(@as(f32, 1), apply(4, c[2], s));
    try std.testing.expectEqual(@as(f32, -9), apply(-4, c[2], s));
    s.include_prompt = true;
    try counts(&c, &h, 3, s, 0);
    try std.testing.expectEqual(Counts{ .count = 2 }, c[0]);
    try std.testing.expectEqual(@as(f32, 3), apply(4, c[0], s));
}
test "penalties TP3 shard counts equal whole counts at every draft prefix" {
    var all: [12]Counts = undefined;
    var local: [4]Counts = undefined;
    const history = [_]i32{ 1, 5, 9, 1, 5, 5, 9, 10, 1 };
    const s: Settings = .{ .presence = -1, .frequency = 0.5, .repetition = 1.5, .window = 4 };
    for (4..history.len + 1) |end| {
        try counts(&all, history[0..end], 3, s, 0);
        for (0..3) |rank| {
            try counts(&local, history[0..end], 3, s, rank * 4);
            for (local, all[rank * 4 ..][0..4]) |a, b| try std.testing.expectEqual(a, b);
        }
    }
    // A rejected suffix, another stream, and the next slot occupant cannot persist.
    try counts(&local, &.{ 2, 2, 3 }, 2, s, 0);
    try std.testing.expectEqual(@as(u32, 0), local[1].count);
    try std.testing.expectEqual(@as(u32, 1), local[3].count);
}
test "penalties neutral is bit preserving and invalid settings refuse" {
    for ([_]u32{ 0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x3f812345 }) |bits|
        try std.testing.expectEqual(bits, @as(u32, @bitCast(apply(@bitCast(bits), .{ .count = 9, .repeats = 3 }, .{}))));
    for ([_]Settings{ .{ .presence = 3 }, .{ .frequency = std.math.nan(f64) }, .{ .repetition = 0 }, .{ .repetition = 2 } }) |s|
        try std.testing.expectError(if (s.repetition == 2) error.PenaltyWindowRequired else error.BadPenalties, s.validate());
}

test "penalties incremental chain matches independent prefix reconstruction" {
    const history = [_]i32{ 1, 5, 9, 1, 5, 1, 9, 1, 5, 5, 1 };
    for ([_]u32{ 0, 1, 2, 4, 8, 100 }) |window| for ([_]bool{ false, true }) |include| {
        const s: Settings = .{ .presence = 0.5, .frequency = -0.3, .window = window, .include_prompt = include };
        for (0..3) |rank| {
            var matrix: [24]Counts = undefined;
            try chain(&matrix, &history, 6, 6, 4, s, rank * 4, 4);
            for (0..6) |j| {
                var reference: [4]Counts = undefined;
                try counts(&reference, history[0 .. 6 + j], 4, s, rank * 4);
                for (reference, matrix[j * 4 ..][0..4]) |a, b| try std.testing.expectEqual(a, b);
            }
        }
    };
}
