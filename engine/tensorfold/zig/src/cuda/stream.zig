//! Streams and events: ordering, waits and GPU-side timing.

const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

pub const Stream = struct {
    d: *const Driver,
    owner: @import("context_health.zig").Owner = .{},
    handle: abi.Stream,

    /// A non-blocking stream never waits on the legacy default stream (what every engine stream should be).
    pub fn init(d: *const Driver, non_blocking: bool) Error!Stream {
        try d.ensureHealthy();
        var s: abi.Stream = null;
        try d.check(d.invoke(.cuStreamCreate, .{&s, if (non_blocking) abi.stream_non_blocking else 0}), "cuStreamCreate");
        return .{ .d = d, .owner = @import("context_health.zig").Owner.capture(), .handle = s };
    }

    pub fn deinit(self: *Stream) void {
        self.owner.ensure() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.ensureHealthy() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.result(self.d.invoke(.cuStreamDestroy_v2, .{self.handle})) catch { @import("context_health.zig").cleanupRetained(); return; };
        self.owner.release();
        self.* = undefined;
    }

    pub fn synchronize(self: Stream) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        try self.d.check(self.d.invoke(.cuStreamSynchronize, .{self.handle}), "cuStreamSynchronize");
    }

    /// True once every queued item has finished.
    pub fn done(self: Stream) Error!bool {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        self.d.check(self.d.invoke(.cuStreamQuery, .{self.handle}), "cuStreamQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    pub fn wait(self: Stream, event: Event) Error!void {
        try event.owner.ensure();
        try self.owner.ensure();
        try self.d.ensureHealthy();
        try self.d.check(self.d.invoke(.cuStreamWaitEvent, .{self.handle, event.handle, 0}), "cuStreamWaitEvent");
    }
};

pub const Event = struct {
    d: *const Driver,
    owner: @import("context_health.zig").Owner = .{},
    handle: abi.Event,

    /// Timing events cost a little more to record; ordering-only events skip the timestamp.
    pub fn init(d: *const Driver, timing: bool) Error!Event {
        try d.ensureHealthy();
        var e: abi.Event = null;
        try d.check(d.invoke(.cuEventCreate, .{&e, if (timing) 0 else abi.event_disable_timing}), "cuEventCreate");
        return .{ .d = d, .owner = @import("context_health.zig").Owner.capture(), .handle = e };
    }

    pub fn deinit(self: *Event) void {
        if (self.handle == null) return;
        self.owner.ensure() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.ensureHealthy() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.result(self.d.invoke(.cuEventDestroy_v2, .{self.handle})) catch { @import("context_health.zig").cleanupRetained(); return; };
        self.owner.release();
        self.handle = null;
        self.owner = .{};
    }

    pub fn record(self: Event, stream: Stream) Error!void {
        try stream.owner.ensure();
        try self.owner.ensure();
        try self.d.ensureHealthy();
        try self.d.check(self.d.invoke(.cuEventRecord, .{self.handle, stream.handle}), "cuEventRecord");
    }

    pub fn synchronize(self: Event) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        try self.d.check(self.d.invoke(.cuEventSynchronize, .{self.handle}), "cuEventSynchronize");
    }

    pub fn done(self: Event) Error!bool {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        self.d.check(self.d.invoke(.cuEventQuery, .{self.handle}), "cuEventQuery") catch |e| switch (e) {
            error.NotReady => return false,
            else => return e,
        };
        return true;
    }

    /// Milliseconds between two recorded timing events (about 0.5 us resolution).
    pub fn elapsedMs(start: Event, end: Event) Error!f32 {
        try start.owner.ensure();
        try end.owner.ensure();
        try start.d.ensureHealthy();
        var ms: f32 = 0;
        try start.d.check(start.d.invoke(.cuEventElapsedTime, .{&ms, start.handle, end.handle}), "cuEventElapsedTime");
        return ms;
    }
};


test "poison first cleanup result fences following destructor" {
    const std = @import("std");
    const health = @import("context_health.zig");
    const Mock = struct {
        var calls: usize = 0;
        fn destroy(_: abi.Stream) callconv(.c) abi.Result { calls += 1; return 700; }
    };
    var d: Driver = undefined;
    d.api.cuStreamDestroy_v2 = Mock.destroy;
    const h: abi.Context = @ptrFromInt(0xf912);
    try health.retain(@intFromPtr(&d), h);
    health.setCurrent(@intFromPtr(&d), h);
    defer health.clearCurrent(@intFromPtr(&d), h);
    Mock.calls = 0;
    var a: Stream = .{ .d = &d, .owner = health.Owner.capture(), .handle = null };
    var b = a;
    a.deinit();
    b.deinit();
    try std.testing.expectEqual(@as(usize, 1), Mock.calls);
}
