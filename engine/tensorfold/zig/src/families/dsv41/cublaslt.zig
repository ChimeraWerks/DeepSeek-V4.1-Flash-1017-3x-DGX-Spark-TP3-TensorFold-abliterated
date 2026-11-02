//! cuBLASLt by dlopen, for the bf16 linears with a bias the served build runs through torch (nn.Linear's addmm: torch's
//! gemm_and_bias): the vision tower's patch embed, wqkv and wo, the aligner's two linears. torch hands cuBLASLt its
//! cuBLAS handle as the Lt handle, a 1 MiB workspace, COMPUTE_32F with fp32 scale, TRANSA T (TRANSB T for an input that
//! is a transposed view), EPILOGUE_BIAS with the bias pointer, alignment preferences of each pointer, and takes the
//! first heuristic result (the recording's cuBLASLt log, level 5); the same calls on the same library pick the same
//! kernel, so the bytes are torch's.
const std = @import("std");
threadlocal var expected_test_error = false;
const cuda = @import("cuda");
const cublas = @import("cublas.zig");

pub const Status = c_int;
const Ptr = ?*anyopaque;
const S = Status;

pub const Error = error{ LibraryUnavailable, MissingSymbol, CublasLtFailed, NoAlgorithm, CudaContextLost };

const r_16bf: c_int = 14; // CUDA_R_16BF
const r_32f: c_int = 0; // CUDA_R_32F
const compute_32f: c_int = 68; // CUBLAS_COMPUTE_32F

const desc_transa: c_int = 3;
const desc_transb: c_int = 4;
const desc_epilogue: c_int = 7;
const desc_bias_pointer: c_int = 8;
const epilogue_bias: u32 = 4;

const pref_max_workspace: c_int = 1;
const pref_align_a: c_int = 5;
const pref_align_b: c_int = 6;
const pref_align_c: c_int = 7;
const pref_align_d: c_int = 8;

/// cublasLtMatmulHeuristicResult_t: the algorithm (64 bytes), its workspace, status, waves.
const Heuristic = extern struct { algo: [8]u64, workspace: usize, state: c_int, waves: f32, reserved: [4]c_int };

pub const Api = struct {
    cublasLtMatmulDescCreate: *const fn (*Ptr, c_int, c_int) callconv(.c) S,
    cublasLtMatmulDescDestroy: *const fn (Ptr) callconv(.c) S,
    cublasLtMatmulDescSetAttribute: *const fn (Ptr, c_int, *const anyopaque, usize) callconv(.c) S,
    cublasLtMatrixLayoutCreate: *const fn (*Ptr, c_int, u64, u64, i64) callconv(.c) S,
    cublasLtMatrixLayoutDestroy: *const fn (Ptr) callconv(.c) S,
    cublasLtMatmulPreferenceCreate: *const fn (*Ptr) callconv(.c) S,
    cublasLtMatmulPreferenceDestroy: *const fn (Ptr) callconv(.c) S,
    cublasLtMatmulPreferenceSetAttribute: *const fn (Ptr, c_int, *const anyopaque, usize) callconv(.c) S,
    cublasLtMatmulAlgoGetHeuristic: *const fn (Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, c_int, *Heuristic, *c_int) callconv(.c) S,
    cublasLtMatmul: *const fn (Ptr, Ptr, *const f32, u64, Ptr, u64, Ptr, *const f32, u64, Ptr, u64, Ptr, *const [8]u64, u64, usize, cuda.abi.Stream) callconv(.c) S,
};

/// torch's cuBLASLt workspace for these calls (the log's workSpaceSizeInBytes).
pub const workspace_bytes: usize = 1 << 20;

pub const Lt = struct {
    lib: std.DynLib,
    api: Api,
    handle: Ptr, // the cuBLAS handle, as torch passes it
    stream: cuda.Stream,
    workspace: u64,
    owner: cuda.context_health.Owner = .{},
    blas: *const cublas.Blas,
    uncertain: std.atomic.Value(bool) = .init(false),
    descriptors: [5]Ptr = @splat(null),

    pub fn open(b: *const cublas.Blas, stream: cuda.Stream, workspace: u64) Error!Lt {
        try cuda.context_health.ensureBoundHealthy();
        var lib = std.DynLib.open("libcublasLt.so.13") catch std.DynLib.open("libcublasLt.so") catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        try b.owner.ensure();
        try stream.owner.ensure();
        const owner = try b.owner.retain();
        _ = @constCast(&b.borrowers).fetchAdd(1, .acq_rel);
        return .{ .lib = lib, .api = api, .handle = @ptrCast(b.handle), .stream = stream, .workspace = workspace, .owner = owner, .blas = b };
    }

    pub fn close(l: *Lt) void {
        if (!cuda.context_health.boundHealthy()) { cuda.context_health.cleanupRetained(); return; }
        l.owner.ensure() catch { cuda.context_health.cleanupRetained(); return; };
        if (l.uncertain.load(.acquire)) { cuda.context_health.cleanupRetained(); return; }
        l.lib.close();
        _ = @constCast(&l.blas.borrowers).fetchSub(1, .acq_rel);
        l.owner.release();
        l.* = undefined;
    }

    fn check(s: Status) Error!void {
        if (s != 0) {
            if (!(@import("builtin").is_test and expected_test_error)) std.log.err("cuBLASLt status {d}", .{s});
            return error.CublasLtFailed;
        }
    }

    /// torch's _getAlignment: the largest power of two up to 256 dividing the address.
    fn alignment(p: u64) u32 {
        var a: u32 = 256;
        while (p % a != 0) a /= 2;
        return a;
    }

    /// nn.Linear of x bf16 and w bf16 [n, k] with bias b bf16 [n]: out bf16 [rows, n]. x is [rows, k] row-major
    /// (x_t false: B rows k, cols rows, ld k, TRANSB N) or a transposed view of a [k, rows] buffer (x_t true: B rows
    /// rows, cols k, ld rows, TRANSB T).
    pub fn linearBias(l: *const Lt, x: u64, x_t: bool, w: u64, b: u64, out: u64, rows: usize, k: usize, n: usize) Error!void {
        try cuda.context_health.ensureBoundHealthy();
        try l.owner.ensure();
        try l.stream.owner.ensure();
        const admission = try l.owner.admit();
        defer admission.end();
        if (l.uncertain.load(.acquire)) return error.CublasLtFailed;
        const a = &l.api;
        const held = &@constCast(l).descriptors;
        defer {
            var clean = true;
            var i: usize = held.len;
            while (i > 0) {
                i -= 1;
                if (!clean or held[i] == null) continue;
                const status = switch (i) {
                    0 => a.cublasLtMatmulPreferenceDestroy(held[i]),
                    1 => a.cublasLtMatmulDescDestroy(held[i]),
                    else => a.cublasLtMatrixLayoutDestroy(held[i]),
                };
                if (status == 0) held[i] = null else clean = false;
            }
            if (!clean) @constCast(&l.uncertain).store(true, .release);
        }
        var pref: Ptr = null;
        const pref_status = a.cublasLtMatmulPreferenceCreate(&pref);
        held[0] = pref;
        try check(pref_status);
        var desc: Ptr = null;
        const desc_status = a.cublasLtMatmulDescCreate(&desc, compute_32f, r_32f);
        held[1] = desc;
        try check(desc_status);
        const ta: c_int = 1;
        const tb: c_int = if (x_t) 1 else 0;
        try check(a.cublasLtMatmulDescSetAttribute(desc, desc_transa, &ta, 4));
        try check(a.cublasLtMatmulDescSetAttribute(desc, desc_transb, &tb, 4));
        try check(a.cublasLtMatmulDescSetAttribute(desc, desc_epilogue, &epilogue_bias, 4));
        try check(a.cublasLtMatmulDescSetAttribute(desc, desc_bias_pointer, &b, 8));
        var la: Ptr = null;
        const la_status = a.cublasLtMatrixLayoutCreate(&la, r_16bf, k, n, @intCast(k));
        held[2] = la;
        try check(la_status);
        var lb: Ptr = null;
        const lb_status = if (x_t) a.cublasLtMatrixLayoutCreate(&lb, r_16bf, rows, k, @intCast(rows)) else a.cublasLtMatrixLayoutCreate(&lb, r_16bf, k, rows, @intCast(k));
        held[3] = lb;
        try check(lb_status);
        var lc: Ptr = null;
        const lc_status = a.cublasLtMatrixLayoutCreate(&lc, r_16bf, n, rows, @intCast(n));
        held[4] = lc;
        try check(lc_status);
        const ws: u64 = workspace_bytes;
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_max_workspace, &ws, 8));
        const aa = alignment(w);
        const ab = alignment(x);
        const ac = alignment(out);
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_align_a, &aa, 4));
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_align_b, &ab, 4));
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_align_c, &ac, 4));
        try check(a.cublasLtMatmulPreferenceSetAttribute(pref, pref_align_d, &ac, 4));
        var h: Heuristic = undefined;
        var got: c_int = 0;
        try check(a.cublasLtMatmulAlgoGetHeuristic(l.handle, desc, la, lb, lc, lc, pref, 1, &h, &got));
        if (got == 0) return error.NoAlgorithm;
        const one: f32 = 1;
        const zero: f32 = 0;
        try check(a.cublasLtMatmul(l.handle, desc, &one, w, la, x, lb, &zero, out, lc, out, lc, &h.algo, l.workspace, workspace_bytes, l.stream.handle));
    }
};

test "Lt custody DS acquisition outputs every stage null and published" {
    expected_test_error = true;
    defer expected_test_error = false;
    const Mock = struct {
        var stage: usize = 0;
        var fail_at: usize = 0;
        var publish: bool = false;
        var cleanup_fail: bool = false;
        var destroys: usize = 0;
        fn acquire(p: *Ptr) Status {
            stage += 1;
            p.* = if (stage != fail_at or publish) @ptrFromInt(0x200 + stage) else null;
            return if (stage == fail_at) 7 else 0;
        }
        fn pref(p: *Ptr) callconv(.c) Status { return acquire(p); }
        fn desc(p: *Ptr, _: c_int, _: c_int) callconv(.c) Status { return acquire(p); }
        fn layout(p: *Ptr, _: c_int, _: u64, _: u64, _: i64) callconv(.c) Status { return acquire(p); }
        fn attr(_: Ptr, _: c_int, _: *const anyopaque, _: usize) callconv(.c) Status { return 0; }
        fn destroy(_: Ptr) callconv(.c) Status { destroys += 1; return if (cleanup_fail) 7 else 0; }
    };
    const ctx: cuda.abi.Context = @ptrFromInt(0xbf01);
    try cuda.context_health.retain(0xbf02, ctx);
    defer cuda.context_health.release(0xbf02, ctx);
    cuda.context_health.setCurrent(0xbf02, ctx);
    defer cuda.context_health.clearCurrent(0xbf02, ctx);
    for (1..6) |stage| for ([_]bool{ false, true }) |publish| for ([_]bool{ false, true }) |fail_cleanup| {
        var lt: Lt = undefined;
        lt.owner = cuda.context_health.Owner.capture();
        defer lt.owner.release();
        lt.stream.owner = lt.owner;
        lt.uncertain = .init(false);
        lt.descriptors = @splat(null);
        lt.api.cublasLtMatmulPreferenceCreate = Mock.pref;
        lt.api.cublasLtMatmulDescCreate = Mock.desc;
        lt.api.cublasLtMatrixLayoutCreate = Mock.layout;
        lt.api.cublasLtMatmulDescSetAttribute = Mock.attr;
        lt.api.cublasLtMatmulPreferenceDestroy = Mock.destroy;
        lt.api.cublasLtMatmulDescDestroy = Mock.destroy;
        lt.api.cublasLtMatrixLayoutDestroy = Mock.destroy;
        Mock.stage = 0; Mock.fail_at = stage; Mock.publish = publish;
        Mock.cleanup_fail = fail_cleanup; Mock.destroys = 0;
        try std.testing.expectError(error.CublasLtFailed, lt.linearBias(256, false, 512, 768, 1024, 2, 2, 2));
        const acquired = stage - @as(usize, if (publish) 0 else 1);
        try std.testing.expectEqual(fail_cleanup and acquired > 0, lt.uncertain.load(.acquire));
        var held: usize = 0;
        for (lt.descriptors) |p| { if (p != null) held += 1; }
        try std.testing.expectEqual(if (fail_cleanup) acquired else 0, held);
        if (fail_cleanup and acquired > 0) {
            const cleanup = cuda.context_health.cleanupMark();
            lt.close();
            try std.testing.expect(!cuda.context_health.cleanupReleased(cleanup));
            try lt.owner.ensure();
        }
    };
}
