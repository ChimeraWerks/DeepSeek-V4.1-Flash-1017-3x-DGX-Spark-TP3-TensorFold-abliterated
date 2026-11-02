//! CPU-only native rank-cache index/plan gate. Does NOT verify payload bytes.
//! Usage: tf-dsv41-cache-index MODEL_DIR CACHE_FILE RANK
const std = @import("std");
const config = @import("config.zig");
const plan = @import("plan.zig");
const cache = @import("rank_cache.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) return error.Usage;
    const rank = try std.fmt.parseInt(u32, args[3], 10);
    if (rank >= 3) return error.ExpectedTP3;
    const c = try config.Config.read(a, init.io, args[1]);
    try c.checkWorld(3);
    const wants = try plan.wants(a, c, .{ .rank = rank, .world = 3 }, true);
    var ix = try cache.open(init.gpa, init.io, std.Io.Dir.cwd(), args[2]);
    defer ix.deinit();
    if (ix.entries.count() != wants.len) return error.EntryCountMismatch;
    var total: u64 = 0;
    for (wants, ix.entries.keys(), ix.entries.values()) |want, key, value| {
        if (!std.mem.eql(u8, want.key, key)) return error.KeyMismatch;
        if (want.dtype) |d| if (d != value.dtype) return error.DTypeMismatch;
        if (want.k16) |k| if (k != value.dim(0)) return error.KMismatch;
        if (want.n16) |n| if (n != value.dim(1)) return error.NMismatch;
        total += value.bytes;
    }
    if (total != ix.data_end) return error.UnindexedPayload;
    std.debug.print("INDEX_ONLY_PASS rank={d} entries={d} bytes={d}; payload NOT verified\n", .{rank, wants.len, total});
    return 0;
}
