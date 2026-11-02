//! Device memory and pinned host memory, each owned by one value; copies and fills, blocking or on a stream.

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;

/// Bytes this process holds in DeviceBuffers and HostBuffers, and the device peak since the last reset.
pub const Usage = struct { device: u64, host: u64, peak: u64 };

var device_bytes: std.atomic.Value(u64) = .init(0);
var host_bytes: std.atomic.Value(u64) = .init(0);
var peak_bytes: std.atomic.Value(u64) = .init(0);
var counts_mutex: std.atomic.Mutex = .unlocked;

/// The counts now; `reset_peak` starts a new peak at the current device bytes. Safe from any thread.
pub fn usage(reset_peak: bool) Usage {
    const NoHook = struct {
        fn afterRead(_: @This()) void {}
    };
    return usageWithHook(reset_peak, NoHook{});
}

fn usageWithHook(reset_peak: bool, hook: anytype) Usage {
    lockCounts();
    defer counts_mutex.unlock();
    const now = device_bytes.load(.monotonic);
    hook.afterRead();
    if (reset_peak) peak_bytes.store(now, .monotonic);
    return .{ .device = now, .host = host_bytes.load(.monotonic), .peak = @max(now, peak_bytes.load(.monotonic)) };
}

fn held(counter: *std.atomic.Value(u64), n: usize) void {
    const device = counter == &device_bytes;
    if (device) lockCounts();
    defer if (device) counts_mutex.unlock();
    const now = counter.fetchAdd(n, .monotonic) + n;
    if (counter == &device_bytes) _ = peak_bytes.fetchMax(now, .monotonic);
}

fn freed(counter: *std.atomic.Value(u64), n: usize) void {
    const device = counter == &device_bytes;
    if (device) lockCounts();
    defer if (device) counts_mutex.unlock();
    _ = counter.fetchSub(n, .monotonic);
}

fn lockCounts() void {
    while (!counts_mutex.tryLock()) std.Thread.yield() catch {};
}

pub const DeviceBuffer = struct {
    d: *const Driver,
    owner: @import("context_health.zig").Owner = .{},
    ptr: abi.DevicePtr,
    len: usize,

    /// `len` bytes, 256-byte aligned by the driver; zero bytes allocate nothing and hold address 0.
    pub fn alloc(d: *const Driver, len: usize) Error!DeviceBuffer {
        const slot = try std.heap.page_allocator.create(?DeviceBuffer);
        const custody = @import("context_health.zig").Custody.init(slot) catch |err| {
            std.heap.page_allocator.destroy(slot);
            return err;
        };
        slot.* = null;
        allocInto(d, len, slot) catch |err| {
            // A failing driver may already have published a live output. Keep
            // its actual owner ledger, not a copy on this returning stack.
            if (slot.* != null) custody.retain() else {
                custody.release();
                std.heap.page_allocator.destroy(slot);
            }
            return err;
        };
        const result = slot.*.?;
        custody.release();
        std.heap.page_allocator.destroy(slot);
        return result;
    }

    /// Constructor-owned slot records even outputs published on a failing call.
    /// The caller must unwind out on every error, before disposing its ledger.
    pub fn allocInto(d: *const Driver, len: usize, out: *?DeviceBuffer) Error!void {
        try d.ensureHealthy();
        var p: abi.DevicePtr = 0;
        const status = if (len > 0) d.invoke(.cuMemAlloc_v2, .{ &p, len }) else abi.success;
        if (p != 0 or status == abi.success) {
            held(&device_bytes, len);
            out.* = .{ .d = d, .owner = @import("context_health.zig").Owner.capture(), .ptr = p, .len = len };
        }
        try d.check(status, "cuMemAlloc");
    }

    /// Allocates and fills from host bytes in one call.
    pub fn fromHost(d: *const Driver, bytes: []const u8) Error!DeviceBuffer {
        const health = @import("context_health.zig");
        const slot = try std.heap.page_allocator.create(?DeviceBuffer);
        slot.* = null;
        const custody = health.Custody.init(slot) catch |err| {
            std.heap.page_allocator.destroy(slot);
            return err;
        };
        errdefer {
            const mark = health.cleanupMark();
            if (slot.*) |*buffer| buffer.free();
            if (!health.cleanupReleased(mark)) custody.retain() else {
                custody.release();
                std.heap.page_allocator.destroy(slot);
            }
        }
        try allocInto(d, bytes.len, slot);
        try slot.*.?.upload(0, bytes);
        const result = slot.*.?;
        custody.release();
        std.heap.page_allocator.destroy(slot);
        return result;
    }

    pub fn free(self: *DeviceBuffer) void {
        self.owner.ensure() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.ensureHealthy() catch { @import("context_health.zig").cleanupRetained(); return; };
        if (self.ptr != 0) self.d.result(self.d.invoke(.cuMemFree_v2, .{self.ptr})) catch { @import("context_health.zig").cleanupRetained(); return; };
        freed(&device_bytes, self.len);
        self.owner.release();
        self.* = undefined;
    }

    /// The device address `offset` bytes in; out of range is refused.
    pub fn at(self: DeviceBuffer, offset: usize) Error!abi.DevicePtr {
        if (offset > self.len) return error.Invalid;
        return self.ptr + offset;
    }

    fn span(self: DeviceBuffer, offset: usize, n: usize) Error!abi.DevicePtr {
        if (offset > self.len or n > self.len - offset) return error.Invalid;
        return self.ptr + offset;
    }

    pub fn upload(self: DeviceBuffer, offset: usize, bytes: []const u8) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try self.d.check(self.d.invoke(.cuMemcpyHtoD_v2, .{dst, bytes.ptr, bytes.len}), "cuMemcpyHtoD");
    }

    pub fn download(self: DeviceBuffer, offset: usize, out: []u8) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.invoke(.cuMemcpyDtoH_v2, .{out.ptr, src, out.len}), "cuMemcpyDtoH");
    }

    /// Asynchronous only when `bytes` is pinned (HostBuffer); pageable memory makes the driver stage it.
    pub fn uploadAsync(self: DeviceBuffer, offset: usize, bytes: []const u8, stream: abi.Stream) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const dst = try self.span(offset, bytes.len);
        if (bytes.len == 0) return;
        try self.d.check(self.d.invoke(.cuMemcpyHtoDAsync_v2, .{dst, bytes.ptr, bytes.len, stream}), "cuMemcpyHtoDAsync");
    }

    pub fn downloadAsync(self: DeviceBuffer, offset: usize, out: []u8, stream: abi.Stream) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const src = try self.span(offset, out.len);
        if (out.len == 0) return;
        try self.d.check(self.d.invoke(.cuMemcpyDtoHAsync_v2, .{out.ptr, src, out.len, stream}), "cuMemcpyDtoHAsync");
    }

    pub fn copyFrom(self: DeviceBuffer, offset: usize, src: abi.DevicePtr, n: usize, stream: ?abi.Stream) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        const dst = try self.span(offset, n);
        if (n == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.invoke(.cuMemcpyDtoDAsync_v2, .{dst, src, n, s}), "cuMemcpyDtoDAsync");
        } else {
            try self.d.check(self.d.invoke(.cuMemcpyDtoD_v2, .{dst, src, n}), "cuMemcpyDtoD");
        }
    }

    pub fn fill8(self: DeviceBuffer, value: u8, stream: ?abi.Stream) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        if (self.len == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.invoke(.cuMemsetD8Async, .{self.ptr, value, self.len, s}), "cuMemsetD8Async");
        } else {
            try self.d.check(self.d.invoke(.cuMemsetD8_v2, .{self.ptr, value, self.len}), "cuMemsetD8");
        }
    }

    /// Fills whole 32-bit words; the length must be a multiple of four.
    pub fn fill32(self: DeviceBuffer, value: u32, stream: ?abi.Stream) Error!void {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        if (self.len % 4 != 0) return error.Invalid;
        if (self.len == 0) return;
        if (stream) |s| {
            try self.d.check(self.d.invoke(.cuMemsetD32Async, .{self.ptr, value, self.len / 4, s}), "cuMemsetD32Async");
        } else {
            try self.d.check(self.d.invoke(.cuMemsetD32_v2, .{self.ptr, value, self.len / 4}), "cuMemsetD32");
        }
    }
};

/// Page-locked host memory: the only host memory an asynchronous copy can read or write without staging.
pub const HostBuffer = struct {
    d: *const Driver,
    owner: @import("context_health.zig").Owner = .{},
    bytes: []align(16) u8,

    pub fn alloc(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, abi.host_alloc_portable);
    }

    pub fn allocInto(d: *const Driver, len: usize, out: *?HostBuffer) Error!void {
        return allocFlagsInto(d, len, abi.host_alloc_portable, out);
    }

    fn allocFlagsInto(d: *const Driver, len: usize, flags: c_uint, out: *?HostBuffer) Error!void {
        try d.ensureHealthy();
        if (len == 0) return error.Invalid;
        var p: ?*anyopaque = null;
        const status = d.invoke(.cuMemHostAlloc, .{ &p, len, flags });
        if (p) |base| {
            held(&host_bytes, len);
            const bytes: [*]align(16) u8 = @ptrCast(@alignCast(base));
            out.* = .{ .d = d, .owner = @import("context_health.zig").Owner.capture(), .bytes = bytes[0..len] };
        }
        try d.check(status, "cuMemHostAlloc");
        if (p == null) return error.Invalid;
    }

    /// Pinned memory kernels write directly (mapped into the device's address space): no copy reads it back.
    pub fn allocMapped(d: *const Driver, len: usize) Error!HostBuffer {
        return allocFlags(d, len, abi.host_alloc_portable | abi.host_alloc_devicemap);
    }

    /// The device address of a mapped buffer.
    pub fn device(self: HostBuffer) Error!abi.DevicePtr {
        try self.owner.ensure();
        try self.d.ensureHealthy();
        var p: abi.DevicePtr = 0;
        try self.d.check(self.d.invoke(.cuMemHostGetDevicePointer_v2, .{&p, self.bytes.ptr, 0}), "cuMemHostGetDevicePointer");
        return p;
    }

    fn allocFlags(d: *const Driver, len: usize, flags: c_uint) Error!HostBuffer {
        const slot = try std.heap.page_allocator.create(?HostBuffer);
        slot.* = null;
        const custody = @import("context_health.zig").Custody.init(slot) catch |err| {
            std.heap.page_allocator.destroy(slot);
            return err;
        };
        allocFlagsInto(d, len, flags, slot) catch |err| {
            if (slot.* != null) custody.retain() else {
                custody.release();
                std.heap.page_allocator.destroy(slot);
            }
            return err;
        };
        const result = slot.*.?;
        custody.release();
        std.heap.page_allocator.destroy(slot);
        return result;
    }

    pub fn free(self: *HostBuffer) void {
        self.owner.ensure() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.ensureHealthy() catch { @import("context_health.zig").cleanupRetained(); return; };
        self.d.result(self.d.invoke(.cuMemFreeHost, .{self.bytes.ptr})) catch { @import("context_health.zig").cleanupRetained(); return; };
        freed(&host_bytes, self.bytes.len);
        self.owner.release();
        self.* = undefined;
    }

    pub fn slice(self: HostBuffer, comptime T: type) []T {
        return std.mem.bytesAsSlice(T, self.bytes[0 .. self.bytes.len / @sizeOf(T) * @sizeOf(T)]);
    }
};

test "usage counts held bytes and the device peak" {
    const before = usage(true);
    held(&device_bytes, 1000);
    held(&host_bytes, 24);
    freed(&device_bytes, 1000);
    const after = usage(false);
    try std.testing.expectEqual(before.device, after.device);
    try std.testing.expectEqual(before.host + 24, after.host);
    try std.testing.expectEqual(before.device + 1000, after.peak);
    try std.testing.expectEqual(before.device, usage(true).peak);
    freed(&host_bytes, 24);
}

test "peak reset preserves an allocation between its read and store" {
    const before = usage(true);
    held(&device_bytes, 100);
    defer freed(&device_bytes, 100);
    const Probe = struct {
        read: std.atomic.Value(bool) = .init(false),
        attempted: std.atomic.Value(bool) = .init(false),
        blocked: std.atomic.Value(bool) = .init(false),
        allocated: std.atomic.Value(bool) = .init(false),
        finish: std.atomic.Value(bool) = .init(false),

        fn wait(flag: *std.atomic.Value(bool)) void {
            while (!flag.load(.acquire)) std.Thread.yield() catch {};
        }

        fn afterRead(probe: *@This()) void {
            probe.read.store(true, .release);
            wait(&probe.attempted);
            if (!probe.blocked.load(.acquire)) wait(&probe.allocated);
        }

        fn allocate(probe: *@This()) void {
            wait(&probe.read);
            const acquired = counts_mutex.tryLock();
            if (acquired) counts_mutex.unlock();
            probe.blocked.store(!acquired, .release);
            probe.attempted.store(true, .release);
            held(&device_bytes, 300);
            probe.allocated.store(true, .release);
            wait(&probe.finish);
            freed(&device_bytes, 300);
        }
    };
    var probe: Probe = .{};
    const thread = try std.Thread.spawn(.{}, Probe.allocate, .{&probe});
    _ = usageWithHook(true, &probe);
    probe.finish.store(true, .release);
    thread.join();
    const after = usage(false);
    try std.testing.expectEqual(before.device + 100, after.device);
    try std.testing.expectEqual(before.device + 400, after.peak);
}

test "custody stage convenience buffers actual published output and cleanup matrix" {
    const health = @import("context_health.zig");
    const Mock = struct {
        var status: abi.Result = 0;
        var publish: bool = true;
        var fail_free: bool = false;
        var frees: usize = 0;
        var flags: c_uint = 0;
        var storage: [16]u8 align(16) = undefined;
        fn host(p: *?*anyopaque, _: usize, f: c_uint) callconv(.c) abi.Result {
            flags = f; p.* = if (publish) &storage else null; return status;
        }
        fn device(p: *abi.DevicePtr, _: usize) callconv(.c) abi.Result {
            p.* = if (publish) 0x123400 else 0; return status;
        }
        fn upload(_: abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) abi.Result { return abi.error_not_ready; }
        fn freeHost(_: ?*anyopaque) callconv(.c) abi.Result { frees += 1; return if (fail_free) 1 else 0; }
        fn freeDevice(_: abi.DevicePtr) callconv(.c) abi.Result { frees += 1; return if (fail_free) 1 else 0; }
    };
    var d: Driver = undefined;
    d.api.cuMemHostAlloc = Mock.host;
    d.api.cuMemFreeHost = Mock.freeHost;
    d.api.cuMemAlloc_v2 = Mock.device;
    d.api.cuMemFree_v2 = Mock.freeDevice;
    d.api.cuMemcpyHtoD_v2 = Mock.upload;
    const ctx: abi.Context = @ptrFromInt(0xcbb1);
    try health.retain(@intFromPtr(&d), ctx);
    defer health.release(@intFromPtr(&d), ctx);
    health.setCurrent(@intFromPtr(&d), ctx);
    defer health.clearCurrent(@intFromPtr(&d), ctx);
    for ([_]bool{ false, true }) |mapped| for ([_]bool{ false, true }) |published| for ([_]abi.Result{ 0, abi.error_not_ready }) |status| for ([_]bool{ false, true }) |fail_free| {
        const before = usage(false).host;
        const previous = health.latestTestCustody();
        Mock.status = status; Mock.publish = published; Mock.fail_free = fail_free; Mock.frees = 0;
        const result = if (mapped) HostBuffer.allocMapped(&d, 16) else HostBuffer.alloc(&d, 16);
        try std.testing.expectEqual(abi.host_alloc_portable | (if (mapped) abi.host_alloc_devicemap else @as(c_uint, 0)), Mock.flags);
        if (status != 0) try std.testing.expectError(error.NotReady, result) else if (!published) try std.testing.expectError(error.Invalid, result);
        if (published) {
            var value: HostBuffer = undefined;
            var slot: ?*?HostBuffer = null;
            if (status == 0) value = try result else {
                const raw = health.latestTestCustody().?;
                try std.testing.expect(raw != previous);
                slot = @ptrCast(@alignCast(raw));
                try std.testing.expectEqual(before + 16, usage(false).host);
            }
            const buffer = if (slot) |v| &v.*.? else &value;
            const mark = health.cleanupMark();
            buffer.free();
            try std.testing.expectEqual(!fail_free, health.cleanupReleased(mark));
            if (fail_free) {
                try buffer.owner.ensure();
                try std.testing.expectEqual(before + 16, usage(false).host);
                Mock.fail_free = false; buffer.free();
            }
            if (slot) |v| { health.releaseTestCustody(v); std.heap.page_allocator.destroy(v); }
        } else try std.testing.expectEqual(previous, health.latestTestCustody());
        try std.testing.expectEqual(before, usage(false).host);
    };
    for ([_]bool{ false, true }) |alloc_fail| for ([_]bool{ false, true }) |fail_free| {
        const before = usage(false).device;
        const previous = health.latestTestCustody();
        Mock.status = if (alloc_fail) abi.error_not_ready else 0;
        Mock.publish = true; Mock.fail_free = fail_free; Mock.frees = 0;
        try std.testing.expectError(error.NotReady, DeviceBuffer.fromHost(&d, "abcd"));
        try std.testing.expectEqual(@as(usize, 1), Mock.frees);
        if (fail_free) {
            const raw = health.latestTestCustody().?;
            try std.testing.expect(raw != previous);
            const slot: *?DeviceBuffer = @ptrCast(@alignCast(raw));
            try slot.*.?.owner.ensure();
            try std.testing.expectEqual(before + 4, usage(false).device);
            Mock.fail_free = false; slot.*.?.free();
            health.releaseTestCustody(slot); std.heap.page_allocator.destroy(slot);
        } else try std.testing.expectEqual(previous, health.latestTestCustody());
        try std.testing.expectEqual(before, usage(false).device);
    };
}
