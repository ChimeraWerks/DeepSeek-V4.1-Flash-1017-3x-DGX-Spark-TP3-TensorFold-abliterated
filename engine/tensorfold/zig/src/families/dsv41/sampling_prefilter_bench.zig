const std = @import("std");
const current = @import("sampling.zig");
const reference = @import("reference");
extern "c" fn printf([*:0]const u8, ...) c_int;
const Timespec = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(c_int, *Timespec) c_int;
fn now() i64 {
    var t: Timespec = undefined;
    _ = clock_gettime(1, &t);
    return t.sec * 1_000_000_000 + t.nsec;
}
pub fn main() !void {
    const a = std.heap.page_allocator;
    const row = try a.alloc(f32, 129408);
    defer a.free(row);
    var nw: current.Workspace = .{};
    defer nw.deinit(a);
    var old: reference.Workspace = .{};
    defer old.deinit(a);
    _ = printf("shape,p,baseline_ms,patched_ms,ratio,matching_draws\n");
    for (0..3) |shape| {
        for (row, 0..) |*v, id| {
            const u: f32 = @floatCast(current.uniform(123, shape, id));
            v.* = switch (shape) {
                0 => if (id < 64) u * 5 else u * 5 - 25,
                1 => u * 8,
                else => 0,
            };
        }
        for ([_]f64{ 0.95, 1 }) |p| {
            // One warm-up per version. Timed trials alternate to limit order bias.
            _ = try current.sampleRowWithWorkspace(&nw, a, row, 0, .{ .seed = 0, .top_k = 0, .top_p = p });
            _ = try reference.sampleRowWithWorkspace(&old, a, row, 0, .{ .seed = 0, .top_k = 0, .top_p = p });
            var ns_old: i64 = 0;
            var ns_new: i64 = 0;
            var matches: usize = 0;
            for (0..12) |seed| {
                var got: i64 = undefined;
                var want: i64 = undefined;
                for (0..2) |trial| {
                    const start = now();
                    if ((trial + seed) % 2 == 0) {
                        want = try reference.sampleRowWithWorkspace(&old, a, row, 1000 + seed, .{ .seed = seed, .top_k = 0, .top_p = p });
                        ns_old += now() - start;
                    } else {
                        got = try current.sampleRowWithWorkspace(&nw, a, row, 1000 + seed, .{ .seed = seed, .top_k = 0, .top_p = p });
                        ns_new += now() - start;
                    }
                }
                if (got != want) return error.SamplerMismatch;
                matches += 1;
            }
            const b = @as(f64, @floatFromInt(ns_old)) / 12e6;
            const n = @as(f64, @floatFromInt(ns_new)) / 12e6;
            _ = printf("%zu,%.2f,%.6f,%.6f,%.3f,%zu\n", shape, p, b, n, b / n, matches);
        }
    }
}
