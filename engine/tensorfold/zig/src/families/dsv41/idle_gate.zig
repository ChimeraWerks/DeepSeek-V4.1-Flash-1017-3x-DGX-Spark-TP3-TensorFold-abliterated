//! CPU-owned idle gate. A GPU doorbell cannot wake a Linux futex, so only the
//! engine's coordinated, globally quiescent idle boundary may arm this gate.
const std = @import("std");
const linux = std.os.linux;

pub const Gate = struct {
    parked: std.atomic.Value(u32) = .init(0),

    pub fn park(g: *Gate) void {
        g.parked.store(1, .release);
    }

    pub fn wake(g: *Gate) void {
        if (g.parked.swap(0, .acq_rel) == 0) return;
        const rc = linux.futex_3arg(&g.parked.raw, .{ .cmd = .WAKE, .private = true }, 1);
        if (linux.errno(rc) != .SUCCESS) @panic("RDMA idle futex wake failed");
    }

    pub fn wait(g: *Gate) error{FutexWaitFailed}!void {
        while (g.parked.load(.acquire) != 0) {
            // The kernel compares and sleeps atomically; wake-before-wait
            // yields AGAIN rather than losing a wake-up.
            const rc = linux.futex_4arg(&g.parked.raw, .{ .cmd = .WAIT, .private = true }, 1, null);
            switch (linux.errno(rc)) {
                .SUCCESS, .INTR, .AGAIN => {},
                else => return error.FutexWaitFailed,
            }
        }
    }
};

test "parked proxy wakes and wake-before-wait is not lost" {
    const Context = struct {
        gate: Gate = .{},
        entered: std.atomic.Value(bool) = .init(false),
        finished: std.atomic.Value(bool) = .init(false),
        fn run(c: *@This()) void {
            c.entered.store(true, .release);
            c.gate.wait() catch @panic("futex wait failed");
            c.finished.store(true, .release);
        }
    };
    var ctx: Context = .{};
    ctx.gate.park();
    const thread = try std.Thread.spawn(.{}, Context.run, .{&ctx});
    defer { ctx.gate.wake(); thread.join(); }
    while (!ctx.entered.load(.acquire)) std.atomic.spinLoopHint();
    try std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake);
    try std.testing.expect(!ctx.finished.load(.acquire));
    ctx.gate.wake();
    for (0..1000) |_| {
        ctx.gate.park();
        ctx.gate.wake();
        try ctx.gate.wait();
    }
}

test "repeated cross-thread park and resume preserve wake-ups" {
    const Context = struct {
        work: Gate = .{},
        done: Gate = .{},
        fn run(c: *@This()) void {
            for (0..1000) |_| {
                c.work.wait() catch @panic("work wait failed");
                c.work.park();
                c.done.wake();
            }
        }
    };
    var ctx: Context = .{};
    ctx.work.park();
    const thread = try std.Thread.spawn(.{}, Context.run, .{&ctx});
    defer { ctx.work.wake(); thread.join(); }
    for (0..1000) |_| {
        ctx.done.park();
        ctx.work.wake();
        try ctx.done.wait();
    }
}
