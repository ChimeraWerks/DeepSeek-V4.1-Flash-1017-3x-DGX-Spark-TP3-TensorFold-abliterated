//! One device's primary context, the one CUDA's runtime and PyTorch share, made current on the calling thread.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const health = @import("context_health.zig");

pub const Context = struct {
    d: *const Driver,
    device: abi.Device,
    handle: abi.Context,

    pub fn init(d: *const Driver, ordinal: c_int) Error!Context {
        if (!health.primaryHealthy(ordinal)) return error.CudaContextLost;
        var dev: abi.Device = 0;
        try d.check(d.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var ctx: abi.Context = null;
        try d.check(d.api.cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
        // Register before binding: a failed bind belongs to the target, not the old current context.
        health.retain(@intFromPtr(d), ctx) catch |err| {
            if (err != error.CudaContextLost) _ = d.api.cuDevicePrimaryCtxRelease_v2(dev);
            return err;
        };
        health.primary(@intFromPtr(d), ctx, ordinal);
        // Failure retains the primary reference and library if release is uncertain.
        const admission = try health.admitContext(@intFromPtr(d), ctx);
        defer admission.end();
        const res = d.api.cuCtxSetCurrent(ctx);
        admission.observe(res);
        if (res != abi.success) {
            if (!health.sticky(res)) {
                const released = d.api.cuDevicePrimaryCtxRelease_v2(dev);
                admission.observe(released);
                if (released == 0) health.release(@intFromPtr(d), ctx);
            }
            return Driver.errorOf(res);
        }
        health.setCurrent(@intFromPtr(d), ctx);
        return .{ .d = d, .device = dev, .handle = ctx };
    }

    /// Waits for all work, then drops this retain; resources made in it must be released first.
    pub fn deinit(self: *Context) void { _ = self.close(); }

    /// Healthy is not quiescent. Only successful synchronization AND release relinquish custody.
    pub fn close(self: *Context) bool {
        self.synchronize() catch return false;
        const key = @intFromPtr(self.d);
        const admission = health.admitContext(key, self.handle) catch return false;
        defer admission.end();
        if (!health.soleReference(key, self.handle)) return false;
        const res = self.d.api.cuDevicePrimaryCtxRelease_v2(self.device);
        admission.observe(res);
        if (res != 0) return false;
        health.clearCurrent(key, self.handle);
        health.release(key, self.handle);
        self.* = undefined;
        return true;
    }

    /// Failure never publishes a bound context; callers must also leave their bound cache unchanged.
    pub fn makeCurrent(self: *const Context) Error!void {
        try self.ensureHealthy();
        const admission = try health.admitContext(@intFromPtr(self.d), self.handle);
        defer admission.end();
        const res = self.d.api.cuCtxSetCurrent(self.handle);
        admission.observe(res);
        if (res != abi.success) return Driver.errorOf(res);
        health.setCurrent(@intFromPtr(self.d), self.handle);
    }

    /// Host-only, independent of peer probing. A false result requires HTTP readiness 503.
    pub fn healthy(self: *const Context) bool {
        return health.healthy(@intFromPtr(self.d), self.handle);
    }

    pub fn fault(self: *const Context) abi.Result {
        return health.fault(@intFromPtr(self.d), self.handle);
    }

    pub fn ensureHealthy(self: *const Context) Error!void {
        if (!self.healthy()) return error.CudaContextLost;
    }

    /// Observe errors returned by a family/backend that uses this context on another thread.
    pub fn observe(self: *const Context, res: abi.Result) Error!void {
        health.observe(@intFromPtr(self.d), self.handle, res);
        if (res != abi.success) return Driver.errorOf(res);
    }

    pub fn synchronize(self: *const Context) Error!void {
        try self.makeCurrent();
        try self.observe(self.d.invoke(.cuCtxSynchronize, .{}));
    }

    pub fn attribute(self: *const Context, a: abi.DeviceAttribute) Error!c_int {
        try self.ensureHealthy();
        var v: c_int = 0;
        try self.observe(self.d.invoke(.cuDeviceGetAttribute, .{&v, a, self.device}));
        return v;
    }

    /// Compute capability as 10 * major + minor (GB10: 121).
    pub fn capability(self: *const Context) Error!u32 {
        const major = try self.attribute(.compute_capability_major);
        const minor = try self.attribute(.compute_capability_minor);
        return @intCast(10 * major + minor);
    }

    pub fn name(self: *const Context, buf: []u8) Error![]const u8 {
        try self.ensureHealthy();
        if (buf.len < 2) return error.Invalid;
        try self.observe(self.d.invoke(.cuDeviceGetName, .{buf.ptr, @as(c_int, @intCast(buf.len)), self.device}));
        return std.mem.sliceTo(buf, 0);
    }

    pub const MemInfo = struct { free: usize, total: usize };

    pub fn memInfo(self: *const Context) Error!MemInfo {
        try self.makeCurrent();
        var m: MemInfo = .{ .free = 0, .total = 0 };
        try self.observe(self.d.invoke(.cuMemGetInfo_v2, .{&m.free, &m.total}));
        return m;
    }
};


test "poison rejects binding work and teardown without touching a driver" {
    // Undefined driver is intentional: any accidental CUDA access makes this CPU-only test fail.
    var d: Driver = undefined;
    const handle: abi.Context = @ptrFromInt(0x9c02);
    try health.retain(@intFromPtr(&d), handle);
    var ctx: Context = .{ .d = &d, .device = 0, .handle = handle };
    try std.testing.expectError(error.CudaContextLost, ctx.observe(700));
    try std.testing.expect(!ctx.healthy());
    try std.testing.expectError(error.CudaContextLost, ctx.makeCurrent());
    try std.testing.expectError(error.CudaContextLost, ctx.synchronize());
    try std.testing.expectError(error.CudaContextLost, ctx.memInfo());
    try std.testing.expectError(error.CudaContextLost, ctx.attribute(.compute_capability_major));
    var buf: [128]u8 = undefined;
    try std.testing.expectError(error.CudaContextLost, ctx.name(&buf));
    ctx.deinit();
}

test "known poisoned primary refuses reopen before any CUDA API" {
    var d: Driver = undefined;
    const handle: abi.Context = @ptrFromInt(0x9e02);
    try health.retain(@intFromPtr(&d), handle);
    health.primary(@intFromPtr(&d), handle, 197);
    health.observe(@intFromPtr(&d), handle, 700);
    try std.testing.expectError(error.CudaContextLost, Context.init(&d, 197));
    // A new wrapper also refuses without dereferencing its undefined API.
    var other: Driver = undefined;
    try std.testing.expectError(error.CudaContextLost, Context.init(&other, 197));
}

test "current-context operations bind their receiver before attributing a sticky fault" {
    const Mock = struct {
        var active: abi.Context = null;
        var calls: usize = 0;
        fn bind(ctx: abi.Context) callconv(.c) abi.Result { active = ctx; return 0; }
        fn sync() callconv(.c) abi.Result { calls += 1; return 700; }
        fn mem(_: *usize, _: *usize) callconv(.c) abi.Result { calls += 1; return 700; }
    };
    var d: Driver = undefined;
    d.api.cuCtxSetCurrent = Mock.bind;
    d.api.cuCtxSynchronize = Mock.sync;
    d.api.cuMemGetInfo_v2 = Mock.mem;
    const ah: abi.Context = @ptrFromInt(0x9f01);
    const bh: abi.Context = @ptrFromInt(0x9f02);
    try health.retain(@intFromPtr(&d), ah);
    try health.retain(@intFromPtr(&d), bh);
    const a: Context = .{ .d = &d, .device = 0, .handle = ah };
    const b: Context = .{ .d = &d, .device = 1, .handle = bh };
    try b.makeCurrent();
    Mock.calls = 0;
    try std.testing.expectError(error.CudaContextLost, a.synchronize());
    try std.testing.expect(Mock.active == ah);
    try std.testing.expect(!a.healthy() and b.healthy());
    try std.testing.expectError(error.CudaContextLost, a.memInfo());
    try std.testing.expectEqual(@as(usize, 1), Mock.calls);
    try std.testing.expectError(error.CudaContextLost, b.memInfo());
    try std.testing.expect(Mock.active == bh);
    health.clearCurrent(@intFromPtr(&d), bh);
}

test "custody stage sync and release failures preserve context without poison" {
    const Mock = struct {
        var sync_code: abi.Result = 1;
        var release_code: abi.Result = 0;
        var releases: usize = 0;
        fn bind(_: abi.Context) callconv(.c) abi.Result { return 0; }
        fn sync() callconv(.c) abi.Result { return sync_code; }
        fn drop(_: abi.Device) callconv(.c) abi.Result { releases += 1; return release_code; }
    };
    var d: Driver = undefined;
    d.api.cuCtxSetCurrent = Mock.bind;
    d.api.cuCtxSynchronize = Mock.sync;
    d.api.cuDevicePrimaryCtxRelease_v2 = Mock.drop;
    const h: abi.Context = @ptrFromInt(0xbc01);
    try health.retain(@intFromPtr(&d), h);
    var c: Context = .{ .d = &d, .handle = h, .device = 0 };
    Mock.releases = 0;
    Mock.sync_code = 1;
    try std.testing.expect(!c.close());
    try std.testing.expect(c.healthy());
    try std.testing.expectEqual(@as(usize, 0), Mock.releases);
    Mock.sync_code = 0;
    Mock.release_code = 1;
    try std.testing.expect(!c.close());
    try std.testing.expect(c.healthy());
    try std.testing.expect(c.handle == h);
    Mock.release_code = 0;
    try std.testing.expect(c.close());
    try std.testing.expectEqual(@as(usize, 2), Mock.releases);
}
