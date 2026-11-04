//! Default-off DSpark keyed proposal policy. Target sampling and acceptance stay unchanged.
const std = @import("std");
const lanes = @import("lanes");
pub const env = "TF_DS_KEYED_DRAFT";
pub const agreement_bit: u64 = @as(u64, 1) << 63;

pub fn enabled() bool {
    const value = std.c.getenv(env) orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

/// Run on every rank before linking peers or loading weights. Off never selects or loads an image.
pub fn selectImage(on: bool, image: []const u8) !?[]const u8 {
    if (!on) return null;
    if (image.len == 0) return error.NoKeyedDraftKernel;
    return image;
}

pub fn settingBit(on: bool) u64 {
    return if (on) agreement_bit else 0;
}

pub fn checkAgreement(rank: u64, leader: u64) !void {
    if ((rank ^ leader) & agreement_bit != 0) return error.KeyedDraftMismatch;
}

/// The Markov head cannot mirror target-logit filtering. Keep those rows greedy.
pub fn eligible(s: lanes.Sampling, vocab: usize) bool {
    return std.math.isFinite(s.temperature) and s.temperature > 0 and
        (s.top_k == 0 or s.top_k >= vocab) and
        std.math.isFinite(s.top_p) and !(s.top_p > 0 and s.top_p < 1) and
        std.math.isFinite(s.min_p) and s.min_p <= 0;
}

pub fn firstPosition(q0: i64) !i64 {
    if (q0 < 0 or q0 > std.math.maxInt(i64) - 8) return error.BadDraftPosition;
    return q0 + 1;
}

test "keyed draft flag agreement includes both TP3 followers" {
    for ([_]bool{ false, true }) |leader| {
        for ([_]bool{ false, true }) |one| for ([_]bool{ false, true }) |two| {
            const settings = [_]u64{ settingBit(leader), settingBit(one), settingBit(two) };
            for (settings[1..]) |f| {
                if (f == settings[0]) try checkAgreement(f, settings[0]) else try std.testing.expectError(error.KeyedDraftMismatch, checkAgreement(f, settings[0]));
            }
        };
    }
    try std.testing.expectEqual(@as(u64, 0), settingBit(false));
}

test "keyed draft eligibility preserves greedy and filtered launches" {
    var s: lanes.Sampling = .{ .seed = 91, .temperature = 0.7, .top_k = 0, .top_p = 1 };
    try std.testing.expect(eligible(s, 129280));
    s.top_k = 129280;
    try std.testing.expect(eligible(s, 129280));
    s.top_k = 20;
    try std.testing.expect(!eligible(s, 129280));
    s.top_k = 0;
    s.top_p = 0.95;
    try std.testing.expect(!eligible(s, 129280));
    s.top_p = 1;
    s.min_p = 0.01;
    try std.testing.expect(!eligible(s, 129280));
    s.min_p = 0;
    for ([_]f64{ 0, -1, std.math.nan(f64), std.math.inf(f64) }) |t| {
        s.temperature = t;
        try std.testing.expect(!eligible(s, 129280));
    }
}

test "keyed draft position keys match real verify windows and serial draws" {
    const a = std.testing.allocator;
    var e: lanes.Engine = undefined;
    e.arena = .init(a);
    defer e.arena.deinit();
    var s: lanes.Stream = undefined;
    s.pending = 19;
    const sampler = @import("sampling.zig");
    const logits = [_]f32{ 0.2, -0.8, 1.1, 0.9, -1.2, 0.3 };
    for ([_]u64{ 0, 57, 2228, 1048574 }) |q0| {
        s.cache_len = q0;
        const w = try lanes.windows.build(&e, .{ .stream = &s, .position = q0, .kind = .head, .held = 5 }, false);
        const first = try firstPosition(@intCast(q0));
        for (0..5) |j| {
            const pos = @as(u64, @intCast(first)) + j;
            try std.testing.expectEqual(pos, w.positions[j]);
            for ([_]u64{ 0, 101, 404 }) |seed| for ([_]f64{ 0.7, 1.0 }) |temp| {
                const sm: sampler.Sampling = .{ .seed = seed, .temperature = temp, .top_k = 0, .top_p = 1 };
                try std.testing.expectEqual(try sampler.sampleRow(&logits, pos, sm), try sampler.sampleRow(&logits, w.positions[j], sm));
                for (0..logits.len) |id| {
                    try std.testing.expectEqual(lanes.sampling.keyBits(seed, pos, id), sampler.keyBits(seed, w.positions[j], id));
                }
            };
        }
    }
    try std.testing.expectError(error.BadDraftPosition, firstPosition(-1));
    try std.testing.expectError(error.BadDraftPosition, firstPosition(std.math.maxInt(i64)));
}

test "keyed draft uniform bits match the core sampler for fixed triples" {
    const sampler = @import("sampling.zig");
    for ([_][3]u64{ .{ 0, 0, 0 }, .{ 101, 2229, 2581 }, .{ 404, 59, 129279 }, .{ 0x7fffffffffffffff, 1048575, 129279 } }) |k| {
        try std.testing.expectEqual(lanes.sampling.keyBits(k[0], k[1], k[2]), sampler.keyBits(k[0], k[1], k[2]));
        try std.testing.expectEqual(@as(u64, @bitCast(lanes.sampling.uniform(k[0], k[1], k[2]))), @as(u64, @bitCast(sampler.uniform(k[0], k[1], k[2]))));
    }
}

test "keyed draft verification keeps target picks and accepts only matching prefix" {
    const a = std.testing.allocator;
    const sampler = @import("sampling.zig");
    var workspace: sampler.Workspace = .{};
    defer workspace.deinit(a);
    const vocab = 6;
    var logits: [6 * vocab]f32 = undefined;
    for (&logits, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 17 + i / vocab) % 29)) / 11;
    const parents = try lanes.accept.rowParents(a, 6, null);
    defer a.free(parents);
    for ([_]f64{ 0.7, 1.0 }) |temp| for ([_]u64{ 0, 101, 404 }) |seed| {
        const sm: sampler.Sampling = .{ .seed = seed, .temperature = temp, .top_k = 0, .top_p = 1 };
        var positions: [6]u64 = undefined;
        var serial: [6]u32 = undefined;
        for (&positions, &serial, 0..) |*pos, *pick, j| {
            pos.* = @as(u64, @intCast(try firstPosition(2228))) + j;
            pick.* = @intCast(try sampler.sampleRow(logits[j * vocab ..][0..vocab], pos.*, sm));
        }
        for (0..6) |accepted| {
            var tokens: [6]u32 = undefined;
            tokens[0] = 19;
            @memcpy(tokens[1..], serial[0..5]);
            if (accepted < 5) tokens[accepted + 1] = (serial[accepted] + 1) % vocab;
            var verified: [6]u32 = undefined;
            _ = try sampler.sampleWindow(&workspace, a, &logits, vocab, &positions, tokens[1..], sm, true, &verified);
            try std.testing.expectEqualSlices(u32, serial[0 .. accepted + 1], verified[0 .. accepted + 1]);
            const path = try lanes.accept.acceptPath(a, &tokens, parents, &verified);
            defer a.free(path);
            try std.testing.expectEqual(accepted + 1, path.len);
        }
    };
}

test "keyed module startup fails closed on every TP3 rank and off needs no image" {
    const image = "isolated-keyed-fatbin";
    for (0..3) |_| {
        try std.testing.expectEqual(@as(?[]const u8, null), try selectImage(false, &.{}));
        try std.testing.expectEqual(@as(?[]const u8, null), try selectImage(false, image));
        try std.testing.expectError(error.NoKeyedDraftKernel, selectImage(true, &.{}));
        try std.testing.expectEqualSlices(u8, image, (try selectImage(true, image)).?);
    }
}
