//! cuBLASLt opened at run time: descriptors, layouts, heuristics and matmul, with fp32 compute only (no TF32 or fast modes).

const std = @import("std");
threadlocal var expected_test_error = false;
const abi = @import("abi.zig");
const health = @import("context_health.zig");

pub const Status = c_int;
pub const Handle = ?*opaque {};
pub const MatmulDesc = ?*opaque {};
pub const Layout = ?*opaque {};
pub const Preference = ?*opaque {};

pub const Algo = extern struct { data: [8]u64 };

pub const Heuristic = extern struct {
    algo: Algo,
    workspace_size: usize,
    state: Status,
    waves_count: f32,
    reserved: [4]c_int,
};

pub const DataType = enum(c_int) { f32 = 0, f16 = 2, bf16 = 14 };
pub const compute_32f: c_int = 68;
pub const Op = enum(c_int) { n = 0, t = 1 };
pub const Order = enum(c_int) { col = 0, row = 1 };

pub const desc_transa: c_int = 3;
pub const desc_transb: c_int = 4;
pub const layout_order: c_int = 1;
pub const pref_max_workspace_bytes: c_int = 1;

pub const Error = error{ LibraryUnavailable, MissingSymbol, CublasFailed, NoAlgorithm, Invalid, CudaContextLost, OutOfMemory };

const S = Status;
const CPtr = ?*const anyopaque;

/// Device pointers are declared as u64: the same register class as `void*` on every LP64 target.
pub const Api = struct {
    cublasLtCreate: *const fn (*Handle) callconv(.c) S,
    cublasLtDestroy: *const fn (Handle) callconv(.c) S,
    cublasLtGetVersion: *const fn () callconv(.c) usize,
    cublasLtGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasLtMatmulDescCreate: *const fn (*MatmulDesc, c_int, DataType) callconv(.c) S,
    cublasLtMatmulDescDestroy: *const fn (MatmulDesc) callconv(.c) S,
    cublasLtMatmulDescSetAttribute: *const fn (MatmulDesc, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatrixLayoutCreate: *const fn (*Layout, DataType, u64, u64, i64) callconv(.c) S,
    cublasLtMatrixLayoutDestroy: *const fn (Layout) callconv(.c) S,
    cublasLtMatrixLayoutSetAttribute: *const fn (Layout, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulPreferenceCreate: *const fn (*Preference) callconv(.c) S,
    cublasLtMatmulPreferenceDestroy: *const fn (Preference) callconv(.c) S,
    cublasLtMatmulPreferenceSetAttribute: *const fn (Preference, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulAlgoGetHeuristic: *const fn (Handle, MatmulDesc, Layout, Layout, Layout, Layout, Preference, c_int, [*]Heuristic, *c_int) callconv(.c) S,
    cublasLtMatmul: *const fn (Handle, MatmulDesc, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, CPtr, abi.DevicePtr, Layout, abi.DevicePtr, Layout, ?*const Algo, abi.DevicePtr, usize, abi.Stream) callconv(.c) S,
};

pub const Library = struct {
    lib: std.DynLib,
    api: Api,
    owner: health.Owner = .{},
    resources: std.atomic.Value(usize) = .init(0),

    /// The CUDA 13 soname first, then the unversioned link name.
    pub fn open() Error!Library {
        for ([_][]const u8{ "libcublasLt.so.13", "libcublasLt.so" }) |path| {
            return openPath(path) catch |e| switch (e) {
                error.LibraryUnavailable => continue,
                else => return e,
            };
        }
        return error.LibraryUnavailable;
    }

    pub fn openPath(path: []const u8) Error!Library {
        var lib = std.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        return .{ .lib = lib, .api = api, .owner = @import("context_health.zig").Owner.capture() };
    }

    pub fn close(self: *Library) void {
        if (!@import("context_health.zig").boundHealthy()) return;
        self.owner.ensure() catch return;
        if (self.resources.load(.acquire) != 0) return;
        self.lib.close();
        self.owner.release();
    }

    pub fn check(self: *const Library, s: Status, what: []const u8) Error!void {
        if (s == 0) return;
        _ = self;
        if (!(@import("builtin").is_test and expected_test_error)) std.log.err("{s}: status {d}", .{ what, s });
        return error.CublasFailed;
    }
};

/// One row-major bf16 product D[m,n] = X[m,k] . W[n,k]^T (a torch Linear), fp32 accumulate, bf16 or fp32 out.
pub const Linear = struct {
    lt: *const Library,
    owner: @import("context_health.zig").Owner = .{},
    handle: Handle = null,
    desc: MatmulDesc = null,
    a: Layout = null,
    b: Layout = null,
    c: Layout = null,
    pref: Preference = null,
    algo: Algo,
    workspace_need: usize,

    /// Picks the heuristic's first algorithm for this shape once; `workspace_limit` bounds what it may ask for.
    pub fn init(lt: *const Library, m: u64, n: u64, k: u64, out: DataType, workspace_limit: usize) Error!Linear {
        try @import("context_health.zig").ensureBoundHealthy();
        const custody = std.heap.page_allocator.create(Quarantine) catch return error.OutOfMemory;
        var transferred = false;
        defer if (!transferred) std.heap.page_allocator.destroy(custody);
        var self: Linear = .{ .lt = lt, .owner = try lt.owner.retain(), .algo = undefined, .workspace_need = 0 };
        _ = @constCast(&lt.resources).fetchAdd(1, .acq_rel);
        errdefer if (!self.close()) {
            // The caller never receives self on an error. Move the actual acquisition
            // ledger and loader/API lifetime off this stack before returning.
            custody.* = .{ .library = lt.*, .linear = self };
            custody.linear.lt = &custody.library;
            while (!quarantine_mutex.tryLock()) std.Thread.yield() catch {};
            custody.next = quarantine;
            quarantine = custody;
            quarantine_mutex.unlock();
            transferred = true;
            health.cleanupRetained();
        };
        const admission = try self.owner.admit();
        defer admission.end();
        try lt.check(lt.api.cublasLtCreate(&self.handle), "cublasLtCreate");
        try lt.check(lt.api.cublasLtMatmulDescCreate(&self.desc, compute_32f, .f32), "cublasLtMatmulDescCreate");
        const ta: c_int = @intFromEnum(Op.t);
        const tb: c_int = @intFromEnum(Op.n);
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transa, &ta, @sizeOf(c_int)), "set transa");
        try lt.check(lt.api.cublasLtMatmulDescSetAttribute(self.desc, desc_transb, &tb, @sizeOf(c_int)), "set transb");
        // column-major view: D^T[n,m] = W[n,k] (stored k x n, transposed) . X^T[k,m]
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.a, .bf16, k, n, @intCast(k)), "layout W");
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.b, .bf16, k, m, @intCast(k)), "layout X");
        try lt.check(lt.api.cublasLtMatrixLayoutCreate(&self.c, out, n, m, @intCast(n)), "layout D");
        try lt.check(lt.api.cublasLtMatmulPreferenceCreate(&self.pref), "cublasLtMatmulPreferenceCreate");
        const limit: u64 = workspace_limit;
        try lt.check(lt.api.cublasLtMatmulPreferenceSetAttribute(self.pref, pref_max_workspace_bytes, &limit, @sizeOf(u64)), "set workspace");
        var found: [1]Heuristic = undefined;
        var count: c_int = 0;
        try lt.check(lt.api.cublasLtMatmulAlgoGetHeuristic(self.handle, self.desc, self.a, self.b, self.c, self.c, self.pref, 1, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        self.algo = found[0].algo;
        self.workspace_need = found[0].workspace_size;
        return self;
    }

    pub fn deinit(self: *Linear) void { _ = self.close(); }

    pub fn close(self: *Linear) bool {
        const admission = self.owner.admit() catch return false;
        defer admission.end();
        const api = self.lt.api;
        // Stop at the first uncertain release; keep the object, owner, and library count.
        if (self.pref != null) { if (api.cublasLtMatmulPreferenceDestroy(self.pref) != 0) return false; self.pref = null; }
        if (self.c != null) { if (api.cublasLtMatrixLayoutDestroy(self.c) != 0) return false; self.c = null; }
        if (self.b != null) { if (api.cublasLtMatrixLayoutDestroy(self.b) != 0) return false; self.b = null; }
        if (self.a != null) { if (api.cublasLtMatrixLayoutDestroy(self.a) != 0) return false; self.a = null; }
        if (self.desc != null) { if (api.cublasLtMatmulDescDestroy(self.desc) != 0) return false; self.desc = null; }
        if (self.handle != null) { if (api.cublasLtDestroy(self.handle) != 0) return false; self.handle = null; }
        _ = @constCast(&self.lt.resources).fetchSub(1, .acq_rel);
        self.owner.release();
        self.* = undefined;
        return true;
    }

    /// D = X . W^T on `stream`; the workspace must hold `workspace_need` bytes.
    pub fn run(self: *const Linear, x: abi.DevicePtr, w: abi.DevicePtr, d: abi.DevicePtr, workspace: abi.DevicePtr, workspace_len: usize, stream: @import("stream.zig").Stream) Error!void {
        try @import("context_health.zig").ensureBoundHealthy();
        try stream.owner.ensure();
        const admission = try self.owner.admit();
        defer admission.end();
        if (workspace_len < self.workspace_need) return error.Invalid;
        const one: f32 = 1;
        const zero: f32 = 0;
        try self.lt.check(self.lt.api.cublasLtMatmul(self.handle, self.desc, &one, w, self.a, x, self.b, &zero, d, self.c, d, self.c, &self.algo, workspace, workspace_len, stream.handle), "cublasLtMatmul");
    }
};

comptime {
    std.debug.assert(@sizeOf(Algo) == 64 and @sizeOf(Heuristic) == 96);
}

test "Lt custody stage failures retain descriptors owner and library" {
    const Mock = struct {
        var fail: bool = true;
        var calls: usize = 0;
        fn pref(_: Preference) callconv(.c) Status { calls += 1; return if (fail) 7 else 0; }
        fn layout(_: Layout) callconv(.c) Status { calls += 1; return 0; }
        fn desc(_: MatmulDesc) callconv(.c) Status { calls += 1; return 0; }
        fn handle(_: Handle) callconv(.c) Status { calls += 1; return 0; }
    };
    const ctx: abi.Context = @ptrFromInt(0xbd01);
    try health.retain(0xbd02, ctx);
    defer health.release(0xbd02, ctx);
    health.setCurrent(0xbd02, ctx);
    defer health.clearCurrent(0xbd02, ctx);
    var lib: Library = undefined;
    lib.owner = health.Owner.capture();
    defer lib.owner.release();
    lib.resources = .init(1);
    lib.api.cublasLtMatmulPreferenceDestroy = Mock.pref;
    lib.api.cublasLtMatrixLayoutDestroy = Mock.layout;
    lib.api.cublasLtMatmulDescDestroy = Mock.desc;
    lib.api.cublasLtDestroy = Mock.handle;
    var linear: Linear = .{ .lt = &lib, .owner = try lib.owner.retain(), .handle = @ptrFromInt(1), .desc = @ptrFromInt(2), .a = @ptrFromInt(3), .b = @ptrFromInt(4), .c = @ptrFromInt(5), .pref = @ptrFromInt(6), .algo = undefined, .workspace_need = 0 };
    Mock.calls = 0;
    Mock.fail = true;
    linear.deinit();
    try std.testing.expectEqual(@as(usize, 1), Mock.calls);
    try std.testing.expectEqual(@as(usize, 1), lib.resources.load(.acquire));
    try std.testing.expect(linear.pref != null);
    try linear.owner.ensure();
    Mock.fail = false;
    linear.deinit();
    try std.testing.expectEqual(@as(usize, 7), Mock.calls);
    try std.testing.expectEqual(@as(usize, 0), lib.resources.load(.acquire));
}

/// Process-lifetime quarantine owns the entire failed constructor, not just a
/// reference count. Never retry uncertain foreign destruction automatically.
const Quarantine = struct {
    next: ?*Quarantine = null,
    library: Library,
    linear: Linear,
};
var quarantine_mutex: std.atomic.Mutex = .unlocked;
var quarantine: ?*Quarantine = null;

test "Lt custody constructor acquisition by cleanup matrix" {
    expected_test_error = true;
    defer expected_test_error = false;
    const Mock = struct {
        var stage: usize = 0;
        var fail_at: usize = 0;
        var publish: bool = false;
        var cleanup_fail: bool = false;
        var destroys: usize = 0;
        fn acquire(comptime T: type, out: *T) Status {
            stage += 1;
            out.* = if (stage != fail_at or publish) @ptrFromInt(0x100 + stage) else null;
            return if (stage == fail_at) 7 else 0;
        }
        fn create(p: *Handle) callconv(.c) Status { return acquire(Handle, p); }
        fn desc(p: *MatmulDesc, _: c_int, _: DataType) callconv(.c) Status { return acquire(MatmulDesc, p); }
        fn layout(p: *Layout, _: DataType, _: u64, _: u64, _: i64) callconv(.c) Status { return acquire(Layout, p); }
        fn pref(p: *Preference) callconv(.c) Status { return acquire(Preference, p); }
        fn attr(_: MatmulDesc, _: c_int, _: CPtr, _: usize) callconv(.c) Status { return 0; }
        fn destroyHandle(_: Handle) callconv(.c) Status { destroys += 1; return if (cleanup_fail) 7 else 0; }
        fn destroyDesc(_: MatmulDesc) callconv(.c) Status { destroys += 1; return if (cleanup_fail) 7 else 0; }
        fn destroyLayout(_: Layout) callconv(.c) Status { destroys += 1; return if (cleanup_fail) 7 else 0; }
        fn destroyPref(_: Preference) callconv(.c) Status { destroys += 1; return if (cleanup_fail) 7 else 0; }
    };
    const ctx: abi.Context = @ptrFromInt(0xbe01);
    try health.retain(0xbe02, ctx);
    defer health.release(0xbe02, ctx);
    health.setCurrent(0xbe02, ctx);
    defer health.clearCurrent(0xbe02, ctx);
    for (1..7) |stage| for ([_]bool{ false, true }) |publish| for ([_]bool{ false, true }) |fail_cleanup| {
        var lib: Library = undefined;
        lib.owner = health.Owner.capture();
        defer lib.owner.release();
        lib.resources = .init(0);
        lib.api.cublasLtCreate = Mock.create;
        lib.api.cublasLtMatmulDescCreate = Mock.desc;
        lib.api.cublasLtMatrixLayoutCreate = Mock.layout;
        lib.api.cublasLtMatmulPreferenceCreate = Mock.pref;
        lib.api.cublasLtMatmulDescSetAttribute = Mock.attr;
        lib.api.cublasLtDestroy = Mock.destroyHandle;
        lib.api.cublasLtMatmulDescDestroy = Mock.destroyDesc;
        lib.api.cublasLtMatrixLayoutDestroy = Mock.destroyLayout;
        lib.api.cublasLtMatmulPreferenceDestroy = Mock.destroyPref;
        Mock.stage = 0; Mock.fail_at = stage; Mock.publish = publish;
        Mock.cleanup_fail = fail_cleanup; Mock.destroys = 0;
        const prior = quarantine;
        try std.testing.expectError(error.CublasFailed, Linear.init(&lib, 2, 2, 2, .bf16, 0));
        const acquired = stage - @as(usize, if (publish) 0 else 1);
        if (fail_cleanup and acquired > 0) {
            try std.testing.expect(quarantine != prior);
            const q = quarantine.?;
            try std.testing.expect(q.linear.lt == &q.library);
            try q.linear.owner.ensure();
            try std.testing.expectEqual(@as(usize, 1), q.library.resources.load(.acquire));
            try std.testing.expect(q.linear.handle != null);
            // Test-only verified cleanup returns custody, avoiding a dangling mock.
            Mock.cleanup_fail = false;
            try std.testing.expect(q.linear.close());
            quarantine = q.next;
            std.heap.page_allocator.destroy(q);
        } else {
            try std.testing.expect(quarantine == prior);
            try std.testing.expectEqual(@as(usize, 0), lib.resources.load(.acquire));
            try std.testing.expectEqual(acquired, Mock.destroys);
        }
    };
}
