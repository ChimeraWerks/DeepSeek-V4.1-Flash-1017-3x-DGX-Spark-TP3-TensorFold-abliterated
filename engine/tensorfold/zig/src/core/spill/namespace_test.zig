//! CPU-only namespace contract tests. TF_NAMESPACE_FIXTURE must name a new disposable directory.
const std = @import("std");
const c = std.c;
const fs = @import("fsio.zig");
const fmt = @import("format.zig");
const store = @import("store.zig");
const slots = @import("slots.zig");
const t = std.testing;
const io = t.io;
const a = t.allocator;

const Harness = struct {
    buffers: [4][]align(slots.alignment) u8 = undefined,
    lists: [4][1]slots.Buffer = undefined,
    cfg: store.Config = undefined,
    fn init(h: *Harness, root: [:0]const u8) !void {
        for (0..4) |i| {
            h.buffers[i] = try a.alignedAlloc(u8, .fromByteUnits(slots.alignment), 4 * 4096);
            h.lists[i] = .{h.buffers[i]};
        }
        h.cfg = .{ .root = root, .rank = 2, .compat = @splat(0x5a), .limits = .{ .max_tokens = 64, .block_tokens = 16, .block_bytes = 4096, .max_small_bytes = 4096 }, .cap_bytes = 16 << 20, .floor_bytes = 0, .min_tokens = 16, .direct = false, .readers = 1, .keep_builds = 10, .staging = .{ .write = &h.lists[0], .read = &h.lists[1], .small = &h.lists[2], .restore = &h.lists[3] } };
    }
    fn deinit(h: *Harness) void {
        for (h.buffers) |b| a.free(b);
    }
    fn open(h: *Harness) !*store.Store {
        return store.Store.open(a, io, h.cfg);
    }
};
fn publish(s: *store.Store, value: u32) !fmt.Digest {
    const ids: [16]u32 = @splat(value);
    const r = s.keep(.{ .keyed = &ids, .lineage = .{ .root = @splat(7) }, .sections = &.{.{ .name = "state", .len = 32 }}, .source = .{ .id = 1, .generation = 1 } });
    try t.expect(r == .queued);
    @memset(r.queued.small, @truncate(value));
    s.captured(r.queued.job, true);
    for (0..5000) |_| {
        var req: [4]store.StageReq = undefined;
        const n = s.stage(&req);
        for (req[0..n]) |q| {
            @memset(q.dst, @truncate(value));
            s.staged(q.ticket, true);
        }
        if (s.pendingStage() == 0) break;
        try std.Io.sleep(io, .fromMilliseconds(1), .awake);
    }
    try t.expect(s.drain(5 * std.time.ns_per_s).done);
    const key = fmt.pointKey(&ids, 16, &.{});
    try t.expect(s.presence(key));
    return key;
}
fn legacy(root: c.fd_t, name: [:0]const u8) !c.fd_t {
    try fs.ensureDirAt(root, name);
    const top = try fs.openDirAt(root, name);
    defer _ = c.close(top);
    try fs.ensureDirAt(top, "rank2");
    const rank = try fs.openDirAt(top, "rank2");
    try fs.ensureDirAt(rank, "points");
    return rank;
}
fn has(fd: c.fd_t, name: [:0]const u8) bool {
    _ = fs.statAt(fd, name) catch return false;
    return true;
}
test "namespace full identity isolation reopen and legacy retention" {
    const root = std.mem.span(c.getenv("TF_NAMESPACE_FIXTURE") orelse return error.MissingFixture);
    const rootfd = try fs.openDir(root);
    defer _ = c.close(rootfd);
    var initial = try fs.list(a, rootfd);
    defer fs.freeList(a, &initial);
    try t.expectEqual(@as(usize, 0), initial.items.len);
    var h: Harness = .{};
    try h.init(root);
    defer h.deinit();
    const identity_a = h.cfg.compat;
    var identity_b = identity_a;
    identity_b[31] = 0x6b; // identical first 16 hex characters, different full identity
    var sa = try h.open();
    const ka = try publish(sa, 1);
    var mb: [80]u8 = undefined;
    const man_name = try std.fmt.bufPrintSentinel(&mb, "{s}.man", .{fmt.hex(&ka)}, 0);
    const raw = try fs.readSmall(a, sa.points_fd, man_name, h.cfg.limits.maxManifest());
    defer a.free(raw);
    try t.expectError(error.Foreign, fmt.Manifest.parse(a, raw, identity_b, h.cfg.limits));
    h.cfg.compat = identity_b;
    var sb = try h.open(); // both locks can be held concurrently
    const kb = try publish(sb, 2);
    try t.expect(!sb.presence(ka));
    try t.expect(!sa.presence(kb));
    sa.close();
    sb.close();
    for (0..2) |_| {
        h.cfg.compat = identity_a;
        sa = try h.open();
        try t.expect(sa.presence(ka));
        try t.expect(!sa.presence(kb));
        sa.close();
        h.cfg.compat = identity_b;
        sb = try h.open();
        try t.expect(sb.presence(kb));
        try t.expect(!sb.presence(ka));
        sb.close();
    }
    var names = try fs.list(a, rootfd);
    defer fs.freeList(a, &names);
    try t.expectEqual(@as(usize, 2), names.items.len);
    for (names.items) |n| {
        try t.expectEqual(@as(usize, 77), n.len);
        try t.expect(std.mem.startsWith(u8, n, "native-spill-"));
        for (n[13..]) |ch| try t.expect((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'));
    }
    // Legacy16 is cold, not migrated. keep_builds=10 preserves it; 0 prunes only unlocked contents.
    const old = try legacy(rootfd, "native-spill-1111111111111111");
    defer _ = c.close(old);
    const live = try legacy(rootfd, "native-spill-2222222222222222");
    defer _ = c.close(live);
    var lock = try fs.Lock.take(live);
    defer lock.release();
    h.cfg.compat = identity_a;
    sa = try h.open();
    sa.close();
    try t.expect(has(old, "points"));
    try t.expect(has(live, "points"));
    h.cfg.keep_builds = 0;
    sa = try h.open();
    sa.close();
    try t.expect(!has(old, "points"));
    try t.expect(has(old, "owner.lock"));
    try t.expect(has(live, "points"));
    try t.expectError(error.Busy, fs.Lock.take(live));
    // Leave A's real native-produced files for the unchanged recipe consumer.
}
