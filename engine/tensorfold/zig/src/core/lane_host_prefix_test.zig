//! The prefix wait: prompts sharing a cut another prompt is filling compute it once (fake backend).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const prefix = @import("lane_host_prefix.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;

const system_len = 1030; // the shared system prompt
const cut = 1024; // its last start on the store's 16-row grid, the cut its fill keeps
const prompt_len = system_len + 16;
const unrelated = 4; // the request at this index starts with another system prompt

const Snaps = struct {
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores; // lane-core backends restore inside their own prompt pass
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .logprobs => {},
            .prefilled => |c| b.cached = c,
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
        }
    }
    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// Each prompt chunk sleeps `slow_ms`; request `id` is cancelled as the first request's fill starts its chunk `at`.
const Hook = struct {
    engine: ?api.Engine = null,
    slow_ms: i64 = 0,
    at: usize = 0,
    id: Id = 0, // 0: none
    sent: bool = false,
    host: ?*LaneHost = null,
    leader_chunk: usize = 0,
    waiter_started_at: ?usize = null,
    fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const c: *Hook = @ptrCast(@alignCast(ctx));
        if (s.prompt()[system_len] == 300) c.leader_chunk = chunk;
        if (c.waiter_started_at == null) {
            const host = c.host.?;
            host.mutex.lockUncancelable(std.testing.io);
            for (host.admitted.items) |job| if (job.id == 2) {
                c.waiter_started_at = c.leader_chunk;
                break;
            };
            host.mutex.unlock(std.testing.io);
        }
        if (c.slow_ms > 0) std.Io.sleep(std.testing.io, .fromMilliseconds(c.slow_ms), .awake) catch {};
        if (c.sent or c.id == 0 or chunk != c.at or s.prompt()[system_len] != 300) return;
        c.sent = true;
        c.engine.?.cancel(c.id);
    }
};

const Setup = struct {
    wait_ms: ?u32 = null, // null: the default bound
    stepped: bool = true, // prompts fill a chunk a step (prefill_step)
    requests: usize = 5,
    shared: u32 = system_len, // the shared block the requests name
    separate_namespace: bool = false,
    media_waiter: bool = false,
    native_grid: bool = false,
    long_leader: bool = false,
    diverge_waiter: bool = false,
    window: u32 = 0,
    hook: Hook = .{},
};

const Run = struct {
    boxes: [8]Box = @splat(.{}),
    reasons: [8]Reason = undefined,
    cached: [8]?u32 = @splat(null),
    waits: u64 = 0,
    timeouts: u64 = 0,
    queued_after: usize = 0,
    waiter_started_at: ?usize = null,

    fn deinit(r: *Run) void {
        for (&r.boxes) |*b| b.tokens.deinit(gpa);
    }
};

/// The prompt of request `i`: a system prompt (another for `unrelated`), then 16 tokens of its own.
fn promptOf(buf: []u32, i: usize) []const u32 {
    const seed: u32 = if (i == unrelated) 500 else 100;
    for (buf[0..system_len], 0..) |*t, k| t.* = seed + @as(u32, @intCast(k % 97));
    for (buf[system_len..], 0..) |*t, k| t.* = 300 + @as(u32, @intCast(i * 16 + k));
    return buf;
}

/// Every request queued before the host starts, so they arrive together, then each answered in full.
fn fanout(su: Setup, out: *Run) !void {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 13, .mark_grid = if (su.native_grid) 16 else 0 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (su.stepped) target.stepped() else target.backend(), clock.clock());
    defer core.deinit();
    const rules: pc.Rules = .{ .lookahead = 1, .min_prompt = 0, .planned = !su.native_grid, .grid = 16, .window = su.window };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, rules, 1 << 24);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 8 });
    host.cache = &store;
    if (su.wait_ms) |ms| host.prefix.ns = @as(i96, ms) * std.time.ns_per_ms;
    const e = host.engine();
    var hook = su.hook;
    hook.engine = e;
    hook.host = &host;
    target.prefill_hook = Hook.call;
    target.prefill_hook_ctx = &hook;
    var bufs: [8][2048]u32 = undefined;
    var reqs: [8]Request = undefined;
    const shared = [_]u32{su.shared};
    const media = [_]pc.Media{std.mem.zeroes(pc.Media)};
    var starts: [prompt_len / 16]u32 = undefined;
    for (&starts, 0..) |*at, i| at.* = @intCast((i + 1) * 16);
    for (0..su.requests) |i| {
        const len: usize = if (su.long_leader and i == 0) 2048 else prompt_len;
        const prompt = promptOf(bufs[i][0..len], i);
        if (su.diverge_waiter and i == 1) bufs[i][cut + 2] = 999;
        reqs[i] = .{ .prompt = prompt, .max_tokens = 6, .shared_prefixes = &shared, .chunks = &starts, .cache_namespace = if (su.separate_namespace and i == 1) pc.namespaceFor("other") else null, .cache_media = if (su.media_waiter and i == 1) &media else &.{} };
        try e.submit(@intCast(i + 1), &reqs[i], .{ .ctx = &out.boxes[i], .event = Box.event });
    }
    try host.start();
    defer host.stop();
    for (0..su.requests) |i| {
        out.reasons[i] = out.boxes[i].wait();
        out.cached[i] = out.boxes[i].cached;
    }
    host.mutex.lockUncancelable(std.testing.io);
    out.queued_after = host.queued.items.len;
    host.mutex.unlock(std.testing.io);
    out.waits = host.prefix.waits;
    out.timeouts = host.prefix.timeouts;
    out.waiter_started_at = hook.waiter_started_at;
}

test "prefix wait: requests on a system prompt another prompt is filling wait for its cut and resume it" {
    var on: Run = .{};
    defer on.deinit();
    try fanout(.{}, &on);
    var off: Run = .{};
    defer off.deinit();
    try fanout(.{ .wait_ms = 0 }, &off);
    for (0..5) |i| {
        try std.testing.expectEqual(Reason.length, on.reasons[i]);
        try std.testing.expectEqualSlices(u32, off.boxes[i].tokens.items, on.boxes[i].tokens.items); // resumed == fresh
    }
    // the first computes the system prompt and three resume its cut; the unrelated one neither waits nor is waited for
    try std.testing.expectEqualSlices(?u32, &.{ 0, cut, cut, cut, 0 }, on.cached[0..5]);
    try std.testing.expectEqual(@as(u64, 3), on.waits);
    try std.testing.expectEqual(@as(u64, 0), on.timeouts);
    try std.testing.expectEqual(@as(usize, 0), on.queued_after);
    // off (TENSORFOLD_PREFIX_WAIT_MS=0): every prompt filling side by side computes the system prompt itself
    try std.testing.expectEqualSlices(?u32, &.{ 0, 0, 0, 0, 0 }, off.cached[0..5]);
    try std.testing.expectEqual(@as(u64, 0), off.waits);
}

test "prefix wait: a cancelled leader releases its waiter, which computes the prefix itself" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .hook = .{ .at = 3, .id = 1 } }, &r);
    try std.testing.expectEqual(Reason.cancelled, r.reasons[0]);
    try std.testing.expectEqual(Reason.length, r.reasons[1]);
    try std.testing.expectEqual(@as(?u32, 0), r.cached[1]);
    try std.testing.expectEqual(@as(u64, 1), r.waits);
    try std.testing.expectEqual(@as(u64, 0), r.timeouts);
}

test "prefix wait: a waiter cancelled while it waits leaves the queue once; the fill runs on" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .hook = .{ .at = 3, .id = 2 } }, &r);
    try std.testing.expectEqual(Reason.length, r.reasons[0]);
    try std.testing.expectEqual(Reason.cancelled, r.reasons[1]);
    try std.testing.expectEqual(@as(u64, 1), r.waits);
    try std.testing.expectEqual(@as(usize, 0), r.queued_after);
}

test "prefix wait: at the bound a waiter is released, counted, and fills beside the leader" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .wait_ms = 1, .hook = .{ .slow_ms = 3 } }, &r);
    try std.testing.expectEqual(Reason.length, r.reasons[1]);
    try std.testing.expectEqual(@as(u64, 1), r.waits);
    try std.testing.expectEqual(@as(u64, 1), r.timeouts);
}

test "prefix wait: whole prompt passes never wait, and the kept cut still serves the later requests" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .stepped = false }, &r);
    try std.testing.expectEqualSlices(?u32, &.{ 0, cut, cut, cut, 0 }, r.cached[0..5]);
    try std.testing.expectEqual(@as(u64, 0), r.waits);
}

test "prefix wait: a shared cut under the store's min_gap is not waited for" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .shared = 130 }, &r); // its cut, 128, adds under 256 tokens past the waiter's hit
    try std.testing.expectEqual(@as(u64, 0), r.waits);
}

test "prefix wait: namespaces and media exclude prospective peers" {
    for ([_]Setup{ .{ .requests = 2, .separate_namespace = true }, .{ .requests = 2, .media_waiter = true } }) |setup| {
        var r: Run = .{};
        defer r.deinit();
        try fanout(setup, &r);
        try std.testing.expectEqual(@as(u64, 0), r.waits);
        try std.testing.expectEqual(Reason.length, r.reasons[1]);
    }
}

test "prefix wait: native off-grid marks release at the physical cut" {
    for ([_]bool{ false, true }) |diverge| {
        var r: Run = .{};
        defer r.deinit();
        try fanout(.{ .requests = 2, .shared = 1029, .native_grid = true, .long_leader = true, .diverge_waiter = diverge }, &r);
        try std.testing.expectEqual(@as(u64, 1), r.waits);
        try std.testing.expectEqual(@as(?u32, cut), r.cached[1]);
        try std.testing.expectEqual(@as(u64, 0), r.timeouts);
        try std.testing.expect(r.waiter_started_at != null and r.waiter_started_at.? < 12);
    }
}

test "prefix wait: a native cut inside the follower replay window is not waited for" {
    var r: Run = .{};
    defer r.deinit();
    try fanout(.{ .requests = 2, .shared = 1029, .native_grid = true, .long_leader = true, .window = 128 }, &r);
    try std.testing.expectEqual(@as(u64, 0), r.waits);
    try std.testing.expectEqual(@as(?u32, 0), r.cached[1]);
}

test "prefix wait: prospective native eligibility matches lookup rules" {
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .grid = 2, .window = 2 }, 1024);
    defer store.deinit();
    const a = [_]u32{ 1, 2, 3, 4, 5, 6 };
    try std.testing.expect(store.wouldServe(null, &a, &.{}, null, &a, 4));
    try std.testing.expect(!store.wouldServe(null, a[0..5], &.{}, null, &a, 4));
    try std.testing.expect(!store.wouldServe(null, &a, &.{}, null, &a, 3));
    store.rules.exact_end = true;
    try std.testing.expect(!store.wouldServe(null, a[0..4], &.{}, null, &a, 4));
    store.family.vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop, .compatible = struct {
        fn call(_: *anyopaque, _: pc.Saved, _: usize) bool { return true; }
    }.call };
    try std.testing.expect(store.wouldServe(null, a[0..4], &.{}, null, &a, 4));
}

test "prefix wait: prospective lookup checks namespace lookahead and chunk starts" {
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .lookahead = 1, .planned = true }, 1024);
    defer store.deinit();
    const a = [_]u32{ 1, 2, 3, 4, 5 };
    const b = [_]u32{ 1, 2, 3, 9, 5 };
    const ns = pc.namespaceFor("scope");
    try std.testing.expect(store.wouldServe(ns, &a, &.{2}, ns, &a, 2));
    try std.testing.expect(!store.wouldServe(ns, &a, &.{2}, null, &a, 2));
    try std.testing.expect(!store.wouldServe(ns, &a, &.{1}, ns, &a, 2));
    try std.testing.expect(!store.wouldServe(ns, &a, &.{3}, ns, &b, 3));
    try std.testing.expect(!store.wouldServe(ns, &a, &.{4}, ns, a[0..4], 4));
    try std.testing.expect(!store.wouldServe(ns, a[0..2], &.{2}, ns, &a, 2));
}

test "prefix wait: TENSORFOLD_PREFIX_WAIT_MS is whole milliseconds, 0 off, anything else the default" {
    const default_ns = @as(i96, prefix.Wait.default_ms) * std.time.ns_per_ms;
    try std.testing.expectEqual(default_ns, prefix.Wait.fromText("").ns);
    try std.testing.expectEqual(default_ns, prefix.Wait.fromText("x").ns);
    try std.testing.expectEqual(default_ns, prefix.Wait.fromText("-5").ns);
    try std.testing.expectEqual(default_ns, prefix.Wait.fromText("1.5").ns);
    try std.testing.expectEqual(@as(i96, 0), prefix.Wait.fromText("0").ns);
    try std.testing.expectEqual(@as(i96, 250) * std.time.ns_per_ms, prefix.Wait.fromText("250").ns);
}
