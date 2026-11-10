//! CPU fixture, imported only by tests. No CUDA entrypoint is called.
const std = @import("std");
const model = @import("model.zig");
const prompt = @import("prompt.zig");
const engram = @import("engram.zig");
const eio = @import("engram_io.zig");
const Config = @import("config.zig").Config;

pub const Fixture = struct {
    m: model.Model,
    c: Config,
    weights: @import("weights.zig").Weights,
    tables: eio.Tables,
    pool: *eio.Pool,
    fd: std.c.fd_t,
    tmp: std.testing.TmpDir,
    ids: [8196]i32,

    pub fn create() !*Fixture {
        const a = std.testing.allocator;
        const io = std.testing.io;
        const f = try a.create(Fixture);
        errdefer a.destroy(f);
        f.tmp = std.testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        var bytes: [4096]u8 = @splat(0);
        for (0..7) |r| {
            @memset(bytes[r * 32 ..][0..32], @intCast(0x30 + r));
            bytes[224 + r] = 127;
        }
        try f.tmp.dir.writeFile(io, .{ .sub_path = "rows", .data = &bytes });
        const path = try std.fs.path.joinZ(a, &.{ ".zig-cache", "tmp", &f.tmp.sub_path, "rows" });
        defer a.free(path);
        f.fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
        if (f.fd < 0) return error.FileNotFound;
        errdefer _ = std.c.close(f.fd);
        f.tables = .{ .gpa = a };
        errdefer f.tables.layers.deinit(a);
        const table: eio.Table = .{ .fd_w = f.fd, .fd_s = f.fd, .base_w = 0, .base_s = 224, .row_w = 32, .row_s = 1, .rows = 7 };
        try f.tables.layers.put(a, 1, table);
        try f.tables.layers.put(a, 14, table);
        f.c = undefined;
        f.c.engram_layers = .{ .items = .{ 1, 14, 0, 0, 0, 0, 0, 0 }, .len = 2 };
        const Map = struct {
            const ids = [_]i32{ 0, 1, 2, 3, 4, 5, 6 };
        };
        const hasher: engram.Hasher = .{ .layers = 2, .ngram = 4, .heads = 1, .primes = @splat(@splat(2)), .offsets = @splat(.{ 0, 2, 4 } ++ @as([21]i64, @splat(0))), .mult = @splat(@splat(3)), .map = &Map.ids, .pad = 0 };
        f.pool = try eio.Pool.init(a, io, 2);
        errdefer f.pool.deinit(a);
        f.m = undefined;
        f.m.gpa = a;
        f.m.io = io;
        f.m.host = std.heap.ArenaAllocator.init(a);
        errdefer f.m.host.deinit();
        f.m.eh = try prompt.EngramHost.init(f.m.host.allocator(), &f.c, hasher, &f.tables, f.pool, 0, 1, 8192);
        f.m.keep_engram = true;
        f.m.retained = .{};
        f.m.park_nojoin = false;
        f.m.park_eh = null;
        f.m.park_pool = null;
        f.m.rs = .{};
        f.m.ahead = null;
        f.m.kvs = null;
        f.m.two = null;
        f.m.ch.spans = &.{};
        f.m.spans = .empty;
        f.m.streams = 3;
        f.m.eng.c = &f.c;
        f.weights = undefined;
        f.weights.layers = &.{};
        f.m.eng.w = &f.weights;
        f.m.caches.slots = 3;
        f.m.caches.cap = 32768;
        f.m.caches.shard = null;
        f.m.rings = &.{};
        f.m.ring_view = &.{};
        f.m.prefill = .{};
        f.m.fill_slot = 0;
        f.m.fill_seq = &f.ids;
        for (&f.ids, 0..) |*v, i| v.* = @intCast(i % 7);
        try f.m.fillBegin(0, 0);
        return f;
    }

    pub fn destroy(f: *Fixture) void {
        f.m.resetEngramAhead();
        if (f.m.park_pool) |p| p.deinit(std.testing.allocator);
        f.pool.deinit(std.testing.allocator);
        f.m.host.deinit();
        f.m.spans.deinit(std.testing.allocator);
        f.tables.layers.deinit(std.testing.allocator);
        _ = std.c.close(f.fd);
        f.tmp.cleanup();
        std.testing.allocator.destroy(f);
    }

    pub fn plan(f: *Fixture, ae: bool, n: usize) !void {
        f.m.resetEngramAhead();
        f.m.prefill.ae = ae;
        const h = &f.m.eh.?;
        h.ahead_start = 4;
        h.ahead_n = n;
        f.m.ahead = if (ae)
            try std.Thread.spawn(.{}, prompt.engramReadAhead, .{ h, &f.c, &f.ids, @as(usize, 4), n, @as(usize, 0) })
        else
            try std.Thread.spawn(.{}, prompt.engramReadAheadAll, .{ h, &f.c, &f.ids, @as(usize, 4), n });
        f.m.retainAhead(&f.ids, 4, n);
    }

    pub fn rows(f: *Fixture, layer: usize, n: usize) ![]u16 {
        return prompt.engramRows(&f.m.eh.?, layer, &f.ids, 4, n, f.c.engram_layers.slice());
    }
};
