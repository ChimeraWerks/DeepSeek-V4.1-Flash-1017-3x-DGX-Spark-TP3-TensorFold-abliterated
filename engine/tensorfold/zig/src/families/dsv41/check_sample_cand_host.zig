// Focused host check without zig test's stripped std test fixtures.
const std = @import("std");
const sampling = @import("sampling.zig");

pub fn main() !void {
    const vocab = 128;
    const count = sampling.max_candidates;
    var logits: [4 * vocab]f32 = undefined;
    var values: [4 * count]f32 = undefined;
    var ids: [4 * count]i64 = undefined;
    for (&logits, 0..) |*v, i| v.* = if (i % vocab >= 100) -std.math.inf(f32) else @as(f32, @floatFromInt((i * 19 + i / vocab) % 37)) / 4;
    logits[3] = -0.0;
    logits[4] = 0.0;
    for (0..4) |r| _ = sampling.topCandidates(logits[r * vocab ..][0..vocab], count, values[r * count ..][0..count], ids[r * count ..][0..count]);
    const positions = [_]u64{ 1100, 1101, 1102, 1103 };
    var drafts: [3]u32 = undefined;
    var expected: [4]u32 = undefined;
    var actual: [4]u32 = undefined;
    var w: sampling.Workspace = .{};
    defer w.deinit(std.heap.page_allocator);
    var checks: usize = 0;
    for ([_]?sampling.Sampling{
        .{ .seed = 123, .temperature = 0.7, .top_k = 1, .top_p = 0.95 },
        .{ .seed = 123, .temperature = 0.7, .top_k = 20, .top_p = 0.8, .min_p = 0.1 },
        .{ .seed = 888, .temperature = 1.2, .top_k = 56, .top_p = 1, .min_p = 0.2 },
        .{ .seed = 888, .temperature = 0 },
        null,
    }) |sm| {
        _ = try sampling.sampleWindow(&w, std.heap.page_allocator, &logits, vocab, &positions, &.{ 0, 0, 0 }, sm, false, &expected);
        _ = try sampling.sampleCandidateWindow(&values, &ids, count, vocab, &positions, &.{ 0, 0, 0 }, sm, false, &actual);
        std.debug.assert(std.mem.eql(u32, &expected, &actual));
        checks += 4;
        for (0..4) |mismatch| {
            @memcpy(&drafts, expected[0..3]);
            if (mismatch < 3) drafts[mismatch] +%= 1;
            const n = try sampling.sampleWindow(&w, std.heap.page_allocator, &logits, vocab, &positions, &drafts, sm, true, &expected);
            std.debug.assert(n == try sampling.sampleCandidateWindow(&values, &ids, count, vocab, &positions, &drafts, sm, true, &actual));
            std.debug.assert(std.mem.eql(u32, &expected, &actual));
            checks += 4;
        }
    }
    std.debug.print("host candidate draw parity: {d} passed, 0 failed\n", .{checks});
}
