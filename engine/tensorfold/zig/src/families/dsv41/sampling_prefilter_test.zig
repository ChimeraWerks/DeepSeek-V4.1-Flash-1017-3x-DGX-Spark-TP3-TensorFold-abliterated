const std = @import("std");
const current = @import("sampling.zig");
const reference = @import("reference");

test "prefilter matches attested sampler over modes distributions ties and seeds" {
    const a = std.testing.allocator;
    var nw: current.Workspace = .{};
    defer nw.deinit(a);
    var old: reference.Workspace = .{};
    defer old.deinit(a);
    var row: [2048]f32 = undefined;
    for (0..5) |shape| {
        for (&row, 0..) |*v, id| {
            const u: f32 = @floatCast(current.uniform(123, shape, id));
            v.* = switch (shape) {
                0 => if (id < 16) u * 5 else u - 25,
                1 => u * 8,
                2 => 0, // equal logits and deterministic id tie-breaking
                3 => if (id < 7) @as(f32, @floatFromInt(id)) else -std.math.inf(f32),
                else => if (id == 0) 0 else -12.01, // insufficient-head fallback
            };
        }
        for ([_]f64{ 0, 0.3, 1, 2 }) |t| {
            for ([_]f64{ 0.125, 0.5, 0.95, 0.9999, 1 }) |p| {
                for ([_]usize{ 0, 20, 80, row.len }) |k| {
                    for ([_]f64{ 0, 0.01 }) |min_p| {
                        for (0..4) |seed| {
                            const want = try reference.sampleRowWithWorkspace(&old, a, &row, 399999 + seed, .{ .seed = seed, .temperature = t, .top_p = p, .top_k = k, .min_p = min_p });
                            const got = try current.sampleRowWithWorkspace(&nw, a, &row, 399999 + seed, .{ .seed = seed, .temperature = t, .top_p = p, .top_k = k, .min_p = min_p });
                            try std.testing.expectEqual(want, got);
                        }
                    }
                }
            }
        }
    }
}
test "129408 vocabulary draw matches reference without clipping or losing padding" {
    const a = std.testing.allocator;
    const row = try a.alloc(f32, 129408);
    defer a.free(row);
    for (row, 0..) |*v, id| v.* = if (id >= 129280) -std.math.inf(f32) else if (id % 4096 == 0) @as(f32, @floatCast(current.uniform(99, 0, id))) else -30;
    var nw: current.Workspace = .{};
    defer nw.deinit(a);
    var old: reference.Workspace = .{};
    defer old.deinit(a);
    for (0..12) |seed| {
        try std.testing.expectEqual(
            try reference.sampleRowWithWorkspace(&old, a, row, 12345, .{ .seed = seed, .temperature = 1, .top_p = 0.95, .top_k = 0 }),
            try current.sampleRowWithWorkspace(&nw, a, row, 12345, .{ .seed = seed, .temperature = 1, .top_p = 0.95, .top_k = 0 }),
        );
    }
}
