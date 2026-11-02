//! Port of wojo's GLM 0110 `tools/spill_blocks_check.py` and `tools/spill_blocks_mutants.sh` (PR #124) as Zig tests,
//! plus PORT-SPILL-V2.md's stage-1 (format, lineage, dedup, references, eviction, quotas) and stage-5 (durability
//! fault campaign) cases for the native store. A real store over a scratch directory, a fake "device" of rows per
//! source the test engine packs blocks from, and the engine's side of every call (stage, staged, captured,
//! readReady, readRelease) played by the test. Each check returns error.CheckFailed when a property fails; the
//! mutant test switches each authored fix back off (`mutants.zig`) and requires some check to fail.
const std = @import("std");
const c = std.c;
const format = @import("format.zig");
const fsio = @import("fsio.zig");
const slots = @import("slots.zig");
const store = @import("store.zig");
const mutants = @import("mutants.zig");
const Allocator = std.mem.Allocator;
const Store = store.Store;
const Digest = format.Digest;
const Lineage = format.Lineage;
const testing = std.testing;
const io = std.testing.io;

const block_tokens = 16;
const bpt = 256; // bytes a token on this rank
const block_bytes = block_tokens * bpt;
const max_tokens = 1024;
const small_max = 8192;
const secs = [_]store.SectionSpec{ .{ .name = "ring", .len = 300 }, .{ .name = "raw", .len = 200 } };

var quiet = false; // a mutant run: failures are expected, not printed
var serial: u32 = 0;

const CheckFailed = error.CheckFailed;

fn ensure(ok: bool, src: std.builtin.SourceLocation) !void {
    if (ok) return;
    if (!quiet) std.debug.print("     check failed at {s}:{d}\n", .{ src.file, src.line });
    return CheckFailed;
}

const Opts = struct {
    tag: []const u8,
    write_slots: usize = 8,
    read_slots: usize = 8,
    small_slots: usize = 4,
    restore_slots: usize = 2,
    cap: u64 = 64 << 20,
    floor: u64 = 0,
    turns: u32 = 4,
    quota: f64 = 0,
    max_reads: u32 = 2,
    keep_builds: u32 = 1,
    backlog: u64 = 1 << 30,
    free: ?u64 = null,
    no_direct: bool = false,
};

const Ev = struct { ev: fsio.Event, name: [96]u8, len: usize };

/// The test's side of the hooks: event log, an action at a step, a crash point, a pwrite script, a disk-free value.
const Hk = struct {
    mutex: std.Io.Mutex = .init,
    events: std.ArrayList(Ev) = .empty,
    gpa: Allocator,
    action_step: ?fsio.Step = null,
    action_suffix: []const u8 = "",
    action: ?*const fn (h: *H) void = null,
    crash_step: ?fsio.Step = null,
    crash_suffix: []const u8 = "",
    crash_skip: u32 = 0,
    nospace_writes: u32 = 0, // the next pwrites failing with ENOSPC
    short_intr: bool = false, // every pwrite short (at most 1000 bytes), and every other one EINTR first
    flip: bool = false,
    free: ?u64 = null,
    h: *H = undefined,
    prune_probes: u32 = 0,
    prune_failed: bool = false,
    fn prune(ctx: ?*anyopaque, dir: c.fd_t) void {
        const hook: *Hk = @ptrCast(@alignCast(ctx.?));
        hook.prune_probes += 1;
        if (fsio.Lock.take(dir)) |taken| {
            var unexpected = taken;
            unexpected.release();
            hook.prune_failed = true;
        } else |err| {
            if (err != error.Busy) hook.prune_failed = true;
        }
    }

    fn crash(ctx: ?*anyopaque, step: fsio.Step, name: []const u8) bool {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (std.c.getenv("SPILL_DEBUG") != null) std.debug.print("hook {t} {s}\n", .{ step, name });
        if (k.action) |f| if (k.action_step == step and std.mem.endsWith(u8, name, k.action_suffix)) {
            k.action = null;
            f(k.h);
        };
        if (k.crash_step) |cs| if (cs == step and std.mem.endsWith(u8, name, k.crash_suffix)) {
            if (k.crash_skip > 0) {
                k.crash_skip -= 1;
                return false;
            }
            k.crash_step = null;
            return true;
        };
        return false;
    }

    fn event(ctx: ?*anyopaque, ev: fsio.Event, name: []const u8) void {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        var e: Ev = .{ .ev = ev, .name = undefined, .len = @min(name.len, 96) };
        @memcpy(e.name[0..e.len], name[0..e.len]);
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        k.events.append(k.gpa, e) catch {};
    }

    fn pwrite(ctx: ?*anyopaque, fd: c.fd_t, buf: []const u8, off: u64) fsio.Error!usize {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (k.nospace_writes > 0) {
            k.nospace_writes -= 1;
            return error.NoSpace;
        }
        var want = buf.len;
        if (k.short_intr) {
            k.flip = !k.flip;
            if (k.flip) return error.Interrupted;
            if (off % fsio.alignment == 0 and want > 1000) want = 1000;
        }
        const n = c.pwrite(fd, buf.ptr, want, @intCast(off));
        if (n < 0) return error.Io;
        return @intCast(n);
    }

    fn freeBytes(ctx: ?*anyopaque) u64 {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        return k.free orelse (1 << 40);
    }

    fn clear(k: *Hk) void {
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        k.events.clearRetainingCapacity();
    }
};

/// A store over a fresh scratch directory and the test's fake device.
const H = struct {
    gpa: Allocator,
    path: [256]u8 = undefined,
    root: [:0]const u8 = undefined,
    bufs: [4][]align(slots.alignment) u8 = @splat(&.{}),
    lists: [4][1]slots.Buffer = undefined,
    cfg: store.Config = undefined,
    s: *Store = undefined,
    rows: std.AutoHashMapUnmanaged(u64, []u8) = .empty,
    hk: Hk,
    opened: bool = false,

    fn init(gpa: Allocator, o: Opts) !*H {
        const h = try gpa.create(H);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .hk = .{ .gpa = gpa, .free = o.free } };
        h.hk.h = h;
        serial += 1;
        var tb: [64]u8 = undefined;
        const tag = try std.fmt.bufPrint(&tb, "{s}-{d}", .{ o.tag, serial });
        const r = try fsio.testRoot(&h.path, tag);
        h.root = r;
        const dfd = try fsio.freshDir(gpa, r);
        _ = c.close(dfd);
        const counts = [4]usize{ o.write_slots, o.read_slots, o.small_slots, o.restore_slots };
        const per = [4]usize{ block_bytes, block_bytes, small_max, small_max };
        for (0..4) |i| {
            h.bufs[i] = try gpa.alignedAlloc(u8, .fromByteUnits(slots.alignment), @max(counts[i] * per[i], 0));
            h.lists[i] = .{h.bufs[i]};
        }
        h.cfg = .{
            .root = h.root,
            .rank = 0,
            .compat = @splat(0x5a),
            .limits = .{ .max_tokens = max_tokens, .block_tokens = block_tokens, .block_bytes = block_bytes, .max_small_bytes = small_max },
            .cap_bytes = o.cap,
            .floor_bytes = o.floor,
            .quota = o.quota,
            .turns = o.turns,
            .min_tokens = 16,
            .backlog_bytes = o.backlog,
            .staging = .{ .write = &h.lists[0], .read = &h.lists[1], .small = &h.lists[2], .restore = &h.lists[3] },
            .readers = 3,
            .max_reads = o.max_reads,
            .keep_builds = o.keep_builds,
            .hooks = .{ .ctx = &h.hk, .crash = Hk.crash, .event = Hk.event, .pwrite = Hk.pwrite, .free = Hk.freeBytes, .no_direct = o.no_direct },
        };
        try h.open();
        return h;
    }

    fn open(h: *H) !void {
        h.s = try Store.open(h.gpa, io, h.cfg);
        h.opened = true;
    }

    fn closeStore(h: *H) void {
        if (h.opened) h.s.close();
        h.opened = false;
    }

    fn reopen(h: *H) !void {
        h.closeStore();
        try h.open();
    }

    fn deinit(h: *H) void {
        h.closeStore();
        fsio.dropDir(h.gpa, h.root);
        for (h.bufs) |b| h.gpa.free(b);
        var it = h.rows.valueIterator();
        while (it.next()) |r| h.gpa.free(r.*);
        h.rows.deinit(h.gpa);
        h.hk.events.deinit(h.gpa);
        h.gpa.destroy(h);
    }

    /// Source `id`'s rows on the fake device (deterministic bytes, made on first use).
    fn src(h: *H, id: u64) []u8 {
        const gop = h.rows.getOrPut(h.gpa, id) catch @panic("oom");
        if (!gop.found_existing) {
            const r = h.gpa.alloc(u8, max_tokens * bpt) catch @panic("oom");
            var prng = std.Random.DefaultPrng.init(id *% 0x9e3779b97f4a7c15 +% 1);
            prng.random().bytes(r);
            gop.value_ptr.* = r;
        }
        return gop.value_ptr.*;
    }

    fn dir(h: *H, which: enum { blocks, points }) c.fd_t {
        return if (which == .blocks) h.s.blocks_fd else h.s.points_fd;
    }
};

fn smallByte(source: u64, at: usize, i: usize) u8 {
    return @truncate(i *% 131 +% source *% 7 +% at);
}

const KO = struct {
    lineage: Lineage,
    source: u64,
    gen: u64 = 1,
    shared: bool = false,
    group: u64 = 0,
    media: []const format.Media = &.{},
    raw: ?[]const u32 = null, // with media: the raw ids (the same length as the keyed ones)
    capture: bool = true,
};

/// The engine at a keep: `keep`, then the small state and tail captured from the device, reported.
fn keepIds(h: *H, ids: []const u32, o: KO) store.KeepResult {
    const at = ids.len;
    const tail_tokens = at % block_tokens;
    const r = h.s.keep(.{
        .keyed = ids,
        .raw = if (o.raw) |r| r[0..ids.len] else null,
        .media = o.media,
        .lineage = o.lineage,
        .shared = o.shared,
        .group = o.group,
        .sections = &secs,
        .tail_bytes = tail_tokens * bpt,
        .source = .{ .id = o.source, .generation = o.gen },
    });
    if (r == .queued) {
        const q = r.queued;
        for (q.sections) |sec| {
            for (0..@intCast(sec.len)) |i| q.small[@intCast(sec.offset + i)] = smallByte(o.source, at, i);
        }
        if (q.tail) |t| @memcpy(t, h.src(o.source)[(at - tail_tokens) * bpt .. at * bpt]);
        if (o.capture) h.s.captured(q.job, true);
    }
    return r;
}

/// The engine between rounds: every requested block packed from its source and reported.
fn pump(h: *H) void {
    var reqs: [16]store.StageReq = undefined;
    while (true) {
        const n = h.s.stage(&reqs);
        if (n == 0) return;
        for (reqs[0..n]) |r| {
            @memcpy(r.dst, h.src(r.source.id)[r.first * bpt ..][0..r.dst.len]);
            h.s.staged(r.ticket, true);
        }
    }
}

/// The engine until nothing waits on it: blocks staged as write slots come back, then a drain.
fn settle(h: *H) store.Store.Drain {
    var i: usize = 0;
    while (i < 5000) : (i += 1) {
        pump(h);
        if (h.s.pendingStage() == 0) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return h.s.drain(5 * std.time.ns_per_s);
}

/// Keep and write `ids` at once; the point's key once it is published.
fn write(h: *H, ids: []const u32, o: KO) !Digest {
    const r = keepIds(h, ids, o);
    try ensure(r == .queued, @src());
    const d = settle(h);
    try ensure(d.done, @src());
    const key = format.pointKey(ids, block_tokens, o.media);
    try ensure(h.s.presence(key), @src());
    return key;
}

/// A restore of `key` through the engine's side: rows (at * bpt bytes, gpa-owned) or null when it fails.
fn restore(h: *H, key: Digest) !?[]u8 {
    const id = h.s.beginRead(key) catch return null;
    const p = h.s.points.get(key) orelse return null;
    const at = p.at;
    const out = try h.gpa.alloc(u8, at * bpt);
    errdefer h.gpa.free(out);
    var i: u32 = 0;
    const total: u32 = @intCast(p.blocks.len);
    while (i < total) {
        if (!h.s.readAwait(id, i, 5 * std.time.ns_per_s, .{})) break;
        var rb: [4]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, null, &rb);
        if (got.status == .failed and got.n == 0) break;
        for (rb[0..got.n]) |b| {
            @memcpy(out[b.first * bpt ..][0..b.data.len], b.data);
            h.s.readRelease(b.slot);
            i += 1;
        }
    }
    var small_ok = false;
    for (0..2500) |_| {
        if (h.s.readSmallView(id) != null) {
            small_ok = true;
            break;
        }
        if (h.s.readFailure(id).err != null) break;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    if (i < total or !small_ok) {
        _ = h.s.endRead(id, .abort) catch {};
        h.gpa.free(out);
        return null;
    }
    var done = (h.s.endRead(id, .commit) catch {
        h.gpa.free(out);
        return null;
    }) orelse {
        h.gpa.free(out);
        return null;
    };
    done.lineage.deinit(h.gpa);
    return out;
}

/// A restore that fails, its failure forgotten the way every rank forgets it: true when it failed.
fn idsOf(gpa: Allocator, n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const ids = try gpa.alloc(u32, n);
    for (ids) |*x| x.* = prng.random().intRangeAtMost(u32, 1, 150000);
    return ids;
}

fn lin(n: u8) Lineage {
    return .{ .root = @splat(n) };
}

fn blockFile(buf: []u8, name: Digest) []const u8 {
    const x = format.hex(&name);
    return std.fmt.bufPrint(buf, "{s}.blk", .{x}) catch unreachable;
}

fn pointFile(buf: []u8, key: Digest, suffix: []const u8) []const u8 {
    const x = format.hex(&key);
    return std.fmt.bufPrint(buf, "{s}{s}", .{ x, suffix }) catch unreachable;
}

fn chmodAt(dirfd: c.fd_t, name: []const u8, mode: c.mode_t) !void {
    var zb: [128]u8 = undefined;
    const z = try std.fmt.bufPrintSentinel(&zb, "{s}", .{name}, 0);
    if (c.fchmodat(dirfd, z, mode, 0) != 0) return error.Unexpected;
}

test "spill unresolved recovery inhibits GC and cap admission until reopen" {
    if (c.geteuid() == 0) return;
    const h = try H.init(testing.allocator, .{ .tag = "spill-deferred-shared", .turns = 0 });
    defer h.deinit();
    const ids = try idsOf(testing.allocator, 2 * block_tokens, 9182);
    defer testing.allocator.free(ids);
    const a = try write(h, ids[0..block_tokens], .{ .lineage = lin(1), .source = 77 });
    const b = try write(h, ids, .{ .lineage = lin(1), .source = 77 });
    const shared = h.s.points.get(a).?.blocks[0].name;
    try testing.expectEqualSlices(u8, &shared, &h.s.points.get(b).?.blocks[0].name);
    var nb: [96]u8 = undefined;
    try chmodAt(h.s.points_fd, pointFile(&nb, b, ".ids"), 0);
    try h.reopen();
    try testing.expect(h.s.unresolved_recovery);
    try testing.expect(h.s.presence(a));
    try testing.expect(!h.s.presence(b));
    h.s.forget(a, &.{}, &.{});
    _ = h.s.drain(5 * std.time.ns_per_s);
    try testing.expect(h.s.have.contains(shared));
    const fresh = try idsOf(testing.allocator, block_tokens, 9183);
    defer testing.allocator.free(fresh);
    h.s.cfg.cap_bytes = h.s.stored_bytes + 1;
    try testing.expect(keepIds(h, fresh, .{ .lineage = lin(3), .source = 88 }) == .no_staging);
    try chmodAt(h.s.points_fd, pointFile(&nb, b, ".ids"), 0o600);
    try h.reopen();
    try testing.expect(!h.s.unresolved_recovery);
    try testing.expect(h.s.presence(b));
    const got = (try restore(h, b)) orelse return error.ExpectedHealthyReopen;
    defer testing.allocator.free(got);
}

var entered = std.atomic.Value(u32).init(0);
var release_readers = std.atomic.Value(bool).init(false);
fn blockBarrier(_: Digest) void {
    _ = entered.fetchAdd(1, .seq_cst);
    while (!release_readers.load(.seq_cst)) std.atomic.spinLoopHint();
}
const FailureWait = struct {
    s: *Store,
    id: store.ReadId,
    started: std.atomic.Value(bool) = .init(false),
    finished: std.atomic.Value(bool) = .init(false),
    count: usize = 0,
    fn run(w: *FailureWait) void {
        w.started.store(true, .seq_cst);
        const f = w.s.readFailure(w.id);
        w.count = f.bad.len;
        w.finished.store(true, .seq_cst);
    }
};
test "spill multiple damaged in-flight readers settle before final invalidation" {
    const h = try H.init(testing.allocator, .{ .tag = "spill-inflight-damage", .turns = 0 });
    defer h.deinit();
    const ids = try idsOf(testing.allocator, 2 * block_tokens, 9910);
    defer testing.allocator.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(1), .source = 77 });
    const blocks = h.s.points.get(key).?.blocks;
    // Both names are damaged before either reader enters its actual disk read.
    for (blocks) |b| {
        var nb: [96]u8 = undefined;
        var z: [128]u8 = undefined;
        const name = try std.fmt.bufPrintSentinel(&z, "{s}", .{blockFile(&nb, b.name)}, 0);
        const fd = c.openat(h.s.blocks_fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true }, @as(c.mode_t, 0));
        try testing.expect(fd >= 0);
        _ = c.close(fd);
    }
    entered.store(0, .seq_cst);
    release_readers.store(false, .seq_cst);
    h.s.block_read_test_hook = blockBarrier;
    const id = try h.s.beginRead(key);
    while (entered.load(.seq_cst) < 2) std.atomic.spinLoopHint();
    // Force the first failure status while two real reads remain in flight.
    // Existing read error would otherwise depend on which disk syscall wins.
    h.s.mutex.lockUncancelable(io);
    for (h.s.reads.items) |r| if (r.id == id) {
        r.err = .damaged;
    };
    h.s.mutex.unlock(io);
    var w: FailureWait = .{ .s = h.s, .id = id };
    const thread = try std.Thread.spawn(.{}, FailureWait.run, .{&w});
    while (!w.started.load(.seq_cst)) std.atomic.spinLoopHint();
    const premature = w.finished.load(.seq_cst);
    release_readers.store(true, .seq_cst);
    thread.join();
    try testing.expect(!premature);
    try testing.expectEqual(@as(usize, 2), w.count);
    h.s.forgetRead(id);
    try testing.expect(!h.s.presence(key));
    _ = try h.s.endRead(id, .abort);
}
