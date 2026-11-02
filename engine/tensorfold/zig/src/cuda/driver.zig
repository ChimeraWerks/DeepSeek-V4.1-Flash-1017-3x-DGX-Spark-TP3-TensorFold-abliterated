//! The CUDA driver opened at run time: libcuda.so.1's versioned entry points in one table, and checked calls.

const std = @import("std");
const abi = @import("abi.zig");
const health = @import("context_health.zig");

pub const Error = error{ DriverUnavailable, MissingSymbol, CudaFailed, CudaContextLost, OutOfMemory, OutOfDeviceMemory, NotReady, NotFound, Invalid };

pub const Driver = struct {
    lib: std.DynLib,
    api: abi.Api,

    pub fn open() Error!Driver {
        return openPath("libcuda.so.1");
    }

    /// Resolves every field of `abi.Api` by its exact name; a missing symbol refuses the whole driver. cuInit runs
    /// first, so a profiler's injection (Nsight Systems' CUPTI, loaded by cuInit) is in place when the entry points are
    /// resolved. TF_CUDA_PROC_ADDRESS=1: the launches (cuLaunchKernel, cuLaunchKernelEx, cuGraphLaunch) through
    /// cuGetProcAddress at CUDA 12.0's versions, the table such an injection hooks.
    pub fn openPath(path: []const u8) Error!Driver {
        var lib = std.DynLib.open(path) catch return error.DriverUnavailable;
        errdefer lib.close();
        var api: abi.Api = undefined;
        api.cuInit = lib.lookup(@TypeOf(api.cuInit), "cuInit") orelse return error.MissingSymbol;
        api.cuGetErrorName = lib.lookup(@TypeOf(api.cuGetErrorName), "cuGetErrorName") orelse return error.MissingSymbol;
        api.cuGetErrorString = lib.lookup(@TypeOf(api.cuGetErrorString), "cuGetErrorString") orelse return error.MissingSymbol;
        try (Driver{ .lib = lib, .api = api }).check(api.cuInit(0), "cuInit");
        const info = @typeInfo(abi.Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        if (std.c.getenv("TF_CUDA_PROC_ADDRESS")) |v| if (v[0] == '1') {
            const GetProc = *const fn ([*:0]const u8, *?*anyopaque, c_int, u64, ?*c_int) callconv(.c) abi.Result;
            const get = lib.lookup(GetProc, "cuGetProcAddress_v2") orelse return error.MissingSymbol;
            inline for (.{ "cuLaunchKernel", "cuLaunchKernelEx", "cuGraphLaunch" }) |name| {
                var p: ?*anyopaque = null;
                var status: c_int = 0;
                if (get(name, &p, 12000, 0, &status) == abi.success and status == 0 and p != null) @field(api, name) = @ptrCast(@alignCast(p.?));
            }
        };
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Driver) void {
        if (!health.driverHealthy(@intFromPtr(self))) return;
        self.lib.close();
    }

    /// All context-dependent raw calls cross one shared context boundary.
    /// Rejected admission is represented by the sticky-context result, without a foreign call.
    pub fn invoke(self: *const Driver, comptime name: std.meta.FieldEnum(abi.Api), args: anytype) abi.Result {
        const admission = health.admitCurrent() catch return 709;
        defer admission.end();
        const res = @call(.auto, @field(self.api, @tagName(name)), args);
        admission.observe(res);
        return res;
    }

    /// Log without querying CUDA after a fault; NOT_READY is a state, not a failure.
    pub fn check(self: *const Driver, res: abi.Result, what: []const u8) Error!void {
        if (res == abi.success) return;
        self.result(res) catch |err| {
            if (err != error.NotReady) std.log.err("{s}: CUDA result {d} ({s})", .{ what, res, @errorName(err) });
            return err;
        };
    }

    /// Silent classification publishes a fault before any caller can start cleanup.
    pub fn result(self: *const Driver, res: abi.Result) Error!void {
        health.observeCurrent(@intFromPtr(self), res);
        if (res != abi.success) return errorOf(res);
    }

    /// Call BEFORE evaluating a GPU API call, not around its already-evaluated result.
    pub fn ensureHealthy(self: *const Driver) Error!void {
        if (!health.currentHealthy(@intFromPtr(self))) return error.CudaContextLost;
    }

    pub fn errorOf(res: abi.Result) Error {
        if (health.sticky(res)) return error.CudaContextLost;
        return switch (res) {
            2 => error.OutOfDeviceMemory,
            abi.error_not_found => error.NotFound,
            abi.error_not_ready => error.NotReady,
            else => error.CudaFailed,
        };
    }

    pub fn errorName(self: *const Driver, res: abi.Result) []const u8 {
        self.ensureHealthy() catch return "CUDA_CONTEXT_LOST";
        const admission = health.admitCurrent() catch return "CUDA_CONTEXT_LOST";
        defer admission.end();
        var s: ?[*:0]const u8 = null;
        if (self.api.cuGetErrorName(res, &s) != abi.success) return "CUDA_ERROR_UNKNOWN_CODE";
        return if (s) |p| std.mem.span(p) else "CUDA_ERROR_UNKNOWN_CODE";
    }

    pub fn errorText(self: *const Driver, res: abi.Result) []const u8 {
        self.ensureHealthy() catch return "CUDA_CONTEXT_LOST";
        const admission = health.admitCurrent() catch return "";
        defer admission.end();
        var s: ?[*:0]const u8 = null;
        if (self.api.cuGetErrorString(res, &s) != abi.success) return "";
        return if (s) |p| std.mem.span(p) else "";
    }

    /// The driver's CUDA version as 1000 * major + 10 * minor.
    pub fn version(self: *const Driver) Error!c_int {
        try self.ensureHealthy();
        var v: c_int = 0;
        try self.check(self.api.cuDriverGetVersion(&v), "cuDriverGetVersion");
        return v;
    }

    pub fn deviceCount(self: *const Driver) Error!c_int {
        try self.ensureHealthy();
        var n: c_int = 0;
        try self.check(self.api.cuDeviceGetCount(&n), "cuDeviceGetCount");
        return n;
    }

    /// Device `ordinal`'s compute capability as 10 * major + minor, read without making a context.
    pub fn capability(self: *const Driver, ordinal: c_int) Error!u32 {
        try self.ensureHealthy();
        var dev: abi.Device = 0;
        try self.check(self.api.cuDeviceGet(&dev, ordinal), "cuDeviceGet");
        var major: c_int = 0;
        var minor: c_int = 0;
        try self.check(self.api.cuDeviceGetAttribute(&major, .compute_capability_major, dev), "cuDeviceGetAttribute");
        try self.check(self.api.cuDeviceGetAttribute(&minor, .compute_capability_minor, dev), "cuDeviceGetAttribute");
        return @intCast(10 * major + minor);
    }
};


test "driver classification preserves recoverable errors and exact sticky results" {
    for ([_]abi.Result{ 214, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719 }) |code| {
        try std.testing.expectEqual(error.CudaContextLost, Driver.errorOf(code));
    }
    try std.testing.expectEqual(error.OutOfDeviceMemory, Driver.errorOf(2));
    try std.testing.expectEqual(error.NotReady, Driver.errorOf(600));
    try std.testing.expectEqual(error.NotFound, Driver.errorOf(500));
    for ([_]abi.Result{ 1, 3, 4, 201, 701, 800, 801, 999, 12345 }) |code| {
        try std.testing.expectEqual(error.CudaFailed, Driver.errorOf(code));
    }
}
