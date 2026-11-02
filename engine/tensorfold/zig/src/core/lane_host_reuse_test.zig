//! LaneHost with the prompt cache: a conversation's next turn resumes from its kept prompt state (the fake backend).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;

test "cancelled reply resends reuse only a matching kept prefix" {
    const gpa = std.testing.allocator;
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
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
        cancel_engine: ?api.Engine = null,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| {
                    b.tokens.appendSlice(gpa, t) catch unreachable;
                    if (b.tokens.items.len >= 3) if (b.cancel_engine) |engine| {
                        b.cancel_engine = null;
                        engine.cancel(id);
                    };
                },
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    // Isolate the three resends so none can warm another's cache.
    for (0..3) |scenario| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        defer store.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        host.cache = &store;
        try host.start();
        defer host.stop();
        const engine = host.engine();
        const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
        var cancelled: Box = .{ .cancel_engine = engine };
        defer cancelled.tokens.deinit(gpa);
        const first: Request = .{ .prompt = &prompt, .max_tokens = 100, .history_len = 8 };
        try engine.submit(1, &first, .{ .ctx = &cancelled, .event = Box.event });
        try std.testing.expectEqual(Reason.cancelled, cancelled.wait());
        try std.testing.expect(cancelled.tokens.items.len >= 3 and cancelled.tokens.items.len < 100);
        try std.testing.expectEqual(@as(?u32, 0), cancelled.cached);
        try std.testing.expectEqual(@as(u64, 1), store.counts.misses);
        try std.testing.expectEqual(@as(u64, 0), store.counts.hits);
        try std.testing.expectEqual(@as(u64, 1), store.counts.kept);
        try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
        try std.testing.expectEqual(@as(u64, 96), store.held);

        var resend: std.ArrayList(u32) = .empty;
        defer resend.deinit(gpa);
        try resend.appendSlice(gpa, &prompt);
        switch (scenario) {
            0 => try resend.appendSlice(gpa, cancelled.tokens.items[0..1]), // shortened reply
            1 => { // divergence after one emitted token
                try resend.append(gpa, cancelled.tokens.items[0]);
                try resend.append(gpa, cancelled.tokens.items[1] ^ 1);
            },
            2 => { // extended reply
                try resend.appendSlice(gpa, cancelled.tokens.items);
                try resend.appendSlice(gpa, &.{ 7, 11 });
            },
            else => unreachable,
        }
        const kept = store.find(resend.items, &.{}).?;
        try std.testing.expectEqual(@as(u32, 8), kept.at);
        try std.testing.expect(kept.at < resend.items.len);
        try std.testing.expectEqualSlices(u32, kept.tokens, resend.items[0..kept.tokens.len]);
        var answer: Box = .{};
        defer answer.tokens.deinit(gpa);
        const second: Request = .{ .prompt = resend.items, .max_tokens = 4, .history_len = 8 };
        try engine.submit(2, &second, .{ .ctx = &answer, .event = Box.event });
        try std.testing.expectEqual(Reason.length, answer.wait());
        try std.testing.expectEqual(@as(?u32, 8), answer.cached);
        for (answer.tokens.items) |token| {
            try std.testing.expectEqual(lanes.fake.next(resend.items, null, resend.items.len), token);
            try resend.append(gpa, token);
        }
        try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
        try std.testing.expectEqual(@as(u64, 1), store.counts.misses);
        try std.testing.expectEqual(@as(u64, 0), store.counts.failed);
        try std.testing.expectEqual(@as(u64, 1), store.counts.kept);
        try std.testing.expectEqual(@as(u64, 96), store.held);
    }
}

test "a lane host resumes a conversation from its kept prompt state and reports the cached tokens" {
    try resumed(false);
    try resumed(true);
}

fn resumed(partial: bool) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = if (partial) 3 else 0 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (partial) target.stepped() else target.backend(), clock.clock());
    defer core.deinit();
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
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(?u32, 0), b1.cached);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    var t2: std.ArrayList(u32) = .empty; // the next turn: the history, the reply, a tool result, a new generation prompt
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached);
    var history: std.ArrayList(u32) = .empty; // the fake's tokens read the whole history: any slip in it changes them
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
    try std.testing.expect(e.info().cache_salt);
    // Exercise the real host -> stream keep callback -> store lookup path.
    // Identical model inputs are cold across salts and hot within each salt;
    // generated tokens must be identical in all five passes.
    const scopes = [_]pc.Namespace{ pc.namespaceFor("A"), pc.namespaceFor("B"), pc.namespaceFor("A"), null, pc.namespaceFor("B") };
    for (scopes, 0..) |scope, i| {
        var box: Box = .{};
        defer box.tokens.deinit(gpa);
        const r: Request = .{ .prompt = &t1, .max_tokens = 4, .history_len = 8, .cache_namespace = scope };
        try e.submit(@intCast(10 + i), &r, .{ .ctx = &box, .event = Box.event });
        try std.testing.expectEqual(Reason.length, box.wait());
        try std.testing.expectEqual(@as(?u32, if (i < 2) 0 else 8), box.cached);
        try std.testing.expectEqualSlices(u32, b1.tokens.items[0..4], box.tokens.items);
    }
}

test "a turn's mark evicting the state it resumed from: the hit is reported first and the entry freed once" {
    try evicted(false);
    try evicted(true);
}

fn evicted(partial: bool) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = if (partial) 3 else 0 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (partial) target.stepped() else target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct { // a budget of one state: each keep records the hits the store had counted by then
        f: *lanes.fake.Fake,
        store: ?*pc.Store = null,
        hits_at_save: [4]u64 = @splat(0),
        saves: usize = 0,
        live: usize = 0,
        fn of(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const x = of(ptr);
            x.hits_at_save[x.saves] = x.store.?.counts.hits;
            x.saves += 1;
            x.live += 1;
            return x.f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            of(ptr).live -= 1;
            of(ptr).f.drop(saved);
        }
    };
    var snaps: Snaps = .{ .f = &target };
    var store = pc.Store.init(gpa, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 150);
    defer store.deinit();
    snaps.store = &store;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    var t2: std.ArrayList(u32) = .empty; // 18 tokens: its history's state (128 bytes) and turn 1's (96) pass the budget together
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached);
    try std.testing.expectEqual(@as(usize, 2), snaps.saves);
    try std.testing.expectEqual(@as(u64, 1), snaps.hits_at_save[1]); // the resume was reported before the keep evicted it
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expectEqual(@as(u32, 16), store.entries.items[0].at);
    try std.testing.expectEqual(@as(usize, 1), snaps.live);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

test "image cache resumes when pixels match, refuses different pixels or no key, and never past a changed image" {
    try imageReuse(false);
    try imageReuse(true);
}

fn imageReuse(partial: bool) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = if (partial) 3 else 0 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (partial) target.stepped() else target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const image = [_]api.Image{.{ .at = 4, .tokens = 3, .bytes = "pixels" }};
    // an image span (positions 4 to 6, each the image token 99) that the key spells by the image's digest
    const t1 = [_]u32{ 3, 1, 4, 1, 99, 99, 99, 5, 9, 2 };
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(image[0].bytes, &digest, .{});
    var k1 = t1;
    for (0..3) |j| k1[4 + j] = pc.pictureId(&digest, @intCast(j));
    const media = [_]pc.Media{.{ .start = 4, .len = 3, .digest = digest }};
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .images = &image, .cache_key = &k1, .cache_media = &media, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    var t2: std.ArrayList(u32) = .empty; // the next turn: the history, the reply, a new user message
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var k2: std.ArrayList(u32) = .empty;
    defer k2.deinit(gpa);
    try k2.appendSlice(gpa, &k1);
    try k2.appendSlice(gpa, t2.items[k1.len..]);
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .images = &image, .cache_key = k2.items, .cache_media = &media, .max_tokens = 4, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached); // the same image: resumed past it
    // The same prompt with another picture has a valid different content key: never restore the first image.
    const other = [_]api.Image{.{ .at = 4, .tokens = 3, .bytes = "other pixels" }};
    var other_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(other[0].bytes, &other_digest, .{});
    const other_media = [_]pc.Media{.{ .start = 4, .len = 3, .digest = other_digest }};
    var k3: std.ArrayList(u32) = .empty;
    defer k3.deinit(gpa);
    try k3.appendSlice(gpa, k2.items);
    for (0..3) |j| k3.items[4 + j] = pc.pictureId(&other_digest, @intCast(j));
    var b3: Box = .{};
    defer b3.tokens.deinit(gpa);
    const r3: Request = .{ .prompt = t2.items, .images = &other, .cache_key = k3.items, .cache_media = &other_media, .max_tokens = 4, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(3, &r3, .{ .ctx = &b3, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b3.wait());
    try std.testing.expectEqual(@as(?u32, 0), b3.cached);
    // The off switch clears the content key: the same picture must neither find nor keep a state.
    const before = store.entries.items.len;
    var b4: Box = .{};
    defer b4.tokens.deinit(gpa);
    const unkeyed: Request = .{ .prompt = t2.items, .images = &image, .max_tokens = 4, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(4, &unkeyed, .{ .ctx = &b4, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b4.wait());
    try std.testing.expectEqual(@as(?u32, 0), b4.cached);
    try std.testing.expectEqual(before, store.entries.items.len);
    // A fabricated key that names the first picture but carries different pixels is refused too.
    var b5: Box = .{};
    defer b5.tokens.deinit(gpa);
    const mismatched: Request = .{ .prompt = t2.items, .images = &other, .cache_key = k2.items, .cache_media = &media, .max_tokens = 4, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(5, &mismatched, .{ .ctx = &b5, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b5.wait());
    try std.testing.expectEqual(@as(?u32, 0), b5.cached);
    try std.testing.expectEqual(before, store.entries.items.len);

    // Two equal-length spans with distinct payloads: changing the later one may reuse only a prefix before it;
    // swapping their order invalidates even the first image despite identical placeholder tokens.
    var duo: [24]u32 = @splat(17);
    @memset(duo[4..6], 99);
    @memset(duo[12..14], 99);
    var da: [32]u8 = undefined;
    var db: [32]u8 = undefined;
    var dc: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("image A", &da, .{});
    std.crypto.hash.sha2.Sha256.hash("image B", &db, .{});
    std.crypto.hash.sha2.Sha256.hash("image C", &dc, .{});
    var keyed = duo;
    for (0..2) |j| {
        keyed[4 + j] = pc.pictureId(&da, @intCast(j));
        keyed[12 + j] = pc.pictureId(&db, @intCast(j));
    }
    const pair_media = [_]pc.Media{ .{ .start = 4, .len = 2, .digest = da }, .{ .start = 12, .len = 2, .digest = db } };
    const pair = [_]api.Image{ .{ .at = 4, .tokens = 2, .bytes = "image A" }, .{ .at = 12, .tokens = 2, .bytes = "image B" } };
    var original: Box = .{};
    defer original.tokens.deinit(gpa);
    const rp: Request = .{ .prompt = &duo, .images = &pair, .cache_key = &keyed, .cache_media = &pair_media, .max_tokens = 2, .history_len = 20 };
    try e.submit(4, &rp, .{ .ctx = &original, .event = Box.event });
    try std.testing.expectEqual(Reason.length, original.wait());
    const saved_pair = store.findMediaScoped(null, &keyed, &.{}, &pair_media).?;
    try std.testing.expect(saved_pair.at > 14);

    var changed = keyed;
    for (0..2) |j| changed[12 + j] = pc.pictureId(&dc, @intCast(j));
    const changed_media = [_]pc.Media{ pair_media[0], .{ .start = 12, .len = 2, .digest = dc } };
    const changed_pair = [_]api.Image{ pair[0], .{ .at = 12, .tokens = 2, .bytes = "image C" } };
    var different: Box = .{};
    defer different.tokens.deinit(gpa);
    const rc: Request = .{ .prompt = &duo, .images = &changed_pair, .cache_key = &changed, .cache_media = &changed_media, .max_tokens = 2, .history_len = 20 };
    try e.submit(5, &rc, .{ .ctx = &different, .event = Box.event });
    try std.testing.expectEqual(Reason.length, different.wait());
    try std.testing.expect(different.cached.? <= 12);

    var swapped = keyed;
    for (0..2) |j| {
        swapped[4 + j] = pc.pictureId(&db, @intCast(j));
        swapped[12 + j] = pc.pictureId(&da, @intCast(j));
    }
    const reversed_media = [_]pc.Media{ .{ .start = 4, .len = 2, .digest = db }, .{ .start = 12, .len = 2, .digest = da } };
    const reversed = [_]api.Image{ .{ .at = 4, .tokens = 2, .bytes = "image B" }, .{ .at = 12, .tokens = 2, .bytes = "image A" } };
    var order: Box = .{};
    defer order.tokens.deinit(gpa);
    const rs: Request = .{ .prompt = &duo, .images = &reversed, .cache_key = &swapped, .cache_media = &reversed_media, .max_tokens = 2, .history_len = 20 };
    try e.submit(6, &rs, .{ .ctx = &order, .event = Box.event });
    try std.testing.expectEqual(Reason.length, order.wait());
    try std.testing.expect(order.cached.? <= 4);
}

/// A fake snapshot vtable over lanes.fake.Fake (states at any position) and a sink that records one reply.
const AheadKit = struct {
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
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
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(std.testing.allocator, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
};

test "step-ahead: a reply's own tokens are read while the client works, and its next step resumes past them with the same tokens" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = AheadKit.Snaps.bytes, .save = AheadKit.Snaps.save, .restore = AheadKit.Snaps.restore, .drop = AheadKit.Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    host.ahead_on = true;
    host.ahead_grid = .{ .chunk = 4, .tail = 2 };
    var warm: AheadKit.Box = .{};
    defer warm.tokens.deinit(gpa);
    host.ahead_sink = .{ .ctx = &warm, .event = AheadKit.Box.event };
    try host.start();
    defer host.stop();
    const e = host.engine();
    // step 1: a 10-token prompt kept at its history (8); the reply's 6 tokens make 16, so the pass keeps 12 (16 - 2 down to 4s)
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: AheadKit.Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = AheadKit.Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(?u32, 0), b1.cached);
    // the step-ahead pass: one token past a pass that resumed at 8
    try std.testing.expectEqual(Reason.length, warm.wait());
    try std.testing.expectEqual(@as(?u32, 8), warm.cached);
    try std.testing.expectEqual(@as(usize, 0), warm.tokens.items.len);
    try std.testing.expect(target.warm_steps > 1);
    try std.testing.expectEqual(@as(u32, 0), target.warm_budget);
    for ([_]usize{ target.warm_first, target.warm_draft, target.warm_verify, target.warm_sample, target.warm_commit }) |calls| try std.testing.expectEqual(@as(usize, 0), calls);
    var found = false;
    for (store.entries.items) |x| found = found or x.at == 12;
    try std.testing.expect(found);
    // step 2: the history, the whole reply, a tool result: resumes at 12, not 8, and reads the history exactly
    var t2: std.ArrayList(u32) = .empty;
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: AheadKit.Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2), .drafts = true };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = AheadKit.Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 12), b2.cached);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

test "step-ahead: nothing runs when off, or when the reply moves no kept state a chunk on" {
    const gpa = std.testing.allocator;
    for ([_]bool{ false, true }) |on| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa };
        defer target.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = AheadKit.Snaps.bytes, .save = AheadKit.Snaps.save, .restore = AheadKit.Snaps.restore, .drop = AheadKit.Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        defer store.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        host.cache = &store;
        host.ahead_on = on;
        // with the grid's 8-row chunks, 10 + 2 tokens reach no chunk end past the kept 8: no pass when on either
        host.ahead_grid = .{ .chunk = 8, .tail = 2 };
        try host.start();
        const e = host.engine();
        const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
        var b1: AheadKit.Box = .{};
        defer b1.tokens.deinit(gpa);
        const r1: Request = .{ .prompt = &t1, .max_tokens = 2, .history_len = 8 };
        try e.submit(1, &r1, .{ .ctx = &b1, .event = AheadKit.Box.event });
        try std.testing.expectEqual(Reason.length, b1.wait());
        host.stop();
        try std.testing.expectEqual(@as(u64, 0), host.ahead_counts.queued);
    }
}
