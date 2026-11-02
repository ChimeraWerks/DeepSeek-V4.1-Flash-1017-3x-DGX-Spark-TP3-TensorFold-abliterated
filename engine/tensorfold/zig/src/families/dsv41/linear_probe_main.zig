//! Independent NumPy-oracle EXL3 probes through the actual native wrappers.
//! Synthetic weights only; no model loading or distributed qualification.
const std = @import("std");
const cuda = @import("cuda");
const ds = @import("dsv41");
const File = struct { name: []const u8, bytes: usize, sha256: []const u8 };
const Case = struct { id: []const u8, k: u32, n: u32, k2: u32, words: File, suh: File, svh: File, x: File, unpacked: File, expected: File };
const Manifest = struct { schema: []const u8, rows: usize, source_kind: []const u8, cases: []Case };

fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, f: File, want: usize) ![]align(16) u8 {
    if (f.bytes != want or std.fs.path.isAbsolute(f.name) or std.mem.indexOf(u8, f.name, "..") != null) return error.InvalidFixture;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ dir, f.name }), a, .limited(1 << 29), .@"16", null);
    var hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
    if (bytes.len != want or !std.mem.eql(u8, &std.fmt.bytesToHex(hash, .lower), f.sha256)) return error.FixtureHashMismatch;
    return bytes;
}

fn image(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]align(16) u8 {
    return std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ dir, "cubins", name }), a, .limited(1 << 28), .@"16", null);
}

const ErrorMetrics = struct { relative_l2: f64, max_abs_over_reference_rms: f64 };
fn metrics(got: []const f32, expected: []const f32) !ErrorMetrics {
    if (got.len != expected.len or got.len == 0) return error.InvalidFixture;
    var sum: f64 = 0;
    var ref: f64 = 0;
    var max: f64 = 0;
    for (got, expected) |g, e| {
        if (!std.math.isFinite(g) or !std.math.isFinite(e)) return error.NonFiniteOutput;
        const d = @as(f64, g) - @as(f64, e);
        sum += d * d;
        ref += @as(f64, e) * @as(f64, e);
        max = @max(max, @abs(d));
    }
    if (ref == 0) return error.DegenerateReference;
    return .{ .relative_l2 = @sqrt(sum / ref), .max_abs_over_reference_rms = max / @sqrt(ref / @as(f64, @floatFromInt(got.len))) };
}

fn probe(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, linear: *const ds.exl3_linear.Kernels, pf: *const ds.exl3_prefill.Kernels, dir: []const u8, c: Case, out: *std.Io.Writer) !void {
    if (c.k == 0 or c.n == 0 or c.k % 128 != 0 or c.n % 128 != 0 or c.k2 < 2 or c.k2 > 16) return error.InvalidFixture;
    const k: usize = c.k;
    const n: usize = c.n;
    const words = try read(a, io, dir, c.words, k * n * c.k2 / 16);
    const suh = try read(a, io, dir, c.suh, k * 2);
    const svh = try read(a, io, dir, c.svh, n * 2);
    const x = try read(a, io, dir, c.x, 128 * k * 2);
    const wq = try read(a, io, dir, c.unpacked, k * n * 2);
    const expected_bytes = try read(a, io, dir, c.expected, 128 * n * 4);
    const expected = std.mem.bytesAsSlice(f32, expected_bytes);
    var dw = try cuda.DeviceBuffer.fromHost(d, words);
    defer dw.free();
    var dsu = try cuda.DeviceBuffer.fromHost(d, suh);
    defer dsu.free();
    var dsv = try cuda.DeviceBuffer.fromHost(d, svh);
    defer dsv.free();
    var dx = try cuda.DeviceBuffer.fromHost(d, x);
    defer dx.free();
    var unpacked = try cuda.DeviceBuffer.alloc(d, wq.len);
    defer unpacked.free();
    const layer: ds.weights.Linear = .{ .words = dw.ptr, .suh = dsu.ptr, .svh = dsv.ptr, .k = c.k, .n = c.n, .k2 = c.k2 };
    try ds.exl3_prefill.unpack(pf, stream, layer, unpacked.ptr);
    try stream.synchronize();
    const decoded = try a.alloc(u8, wq.len);
    try unpacked.download(0, decoded);
    if (!std.mem.eql(u8, decoded, wq)) return error.UnpackNotBitwiseEqual;
    var xh = try cuda.DeviceBuffer.alloc(d, 128 * k * 2);
    defer xh.free();
    var y = try cuda.DeviceBuffer.alloc(d, 128 * n * 4);
    defer y.free();
    var counts = try cuda.DeviceBuffer.alloc(d, 8 * n / 128 * 4);
    defer counts.free();
    var z = try cuda.DeviceBuffer.alloc(d, @max(1, ds.exl3_linear.zFloats(&.{layer}, 128)) * 4);
    defer z.free();
    const calls = [_]ds.exl3_linear.Call{.{ .layer = layer, .x = dx.ptr, .ldx = @intCast(k), .x_dtype = .bf16, .xh = xh.ptr, .y = y.ptr, .ldy = @intCast(n), .y_dtype = .f32, .counters = counts.ptr }};
    const got = try a.alloc(f32, 128 * n);
    const first = try a.alloc(f32, n);
    var worst: ErrorMetrics = .{ .relative_l2 = 0, .max_abs_over_reference_rms = 0 };
    for ([_]usize{ 1, 8, 48, 128 }) |rows| {
        try counts.fill32(0, stream.handle);
        try ds.exl3_linear.rotMany(linear, stream, &calls, rows, true);
        try ds.exl3_linear.glinear(linear, stream, &calls, rows, z.ptr, true, true);
        try stream.synchronize();
        try y.download(0, std.mem.sliceAsBytes(got[0 .. rows * n]));
        const e = try metrics(got[0 .. rows * n], expected[0 .. rows * n]);
        worst.relative_l2 = @max(worst.relative_l2, e.relative_l2);
        worst.max_abs_over_reference_rms = @max(worst.max_abs_over_reference_rms, e.max_abs_over_reference_rms);
        // Fixed thresholds, never taken from an editable manifest.
        if (e.relative_l2 > 0.002 or e.max_abs_over_reference_rms > 0.02) return error.NumericToleranceExceeded;
        if (rows == 1) {
            @memcpy(first, got[0..n]);
        } else if (!std.mem.eql(u8, std.mem.sliceAsBytes(first), std.mem.sliceAsBytes(got[0..n]))) return error.RowInvarianceFailed;
    }
    const eager = try a.dupe(f32, got);
    try cuda.graph.beginCapture(stream, .thread_local);
    try ds.exl3_linear.rotMany(linear, stream, &calls, 128, true);
    try ds.exl3_linear.glinear(linear, stream, &calls, 128, z.ptr, true, true);
    var graph = try cuda.graph.endCapture(stream);
    defer graph.deinit();
    var executable = try graph.instantiate();
    defer executable.deinit();
    for (0..3) |_| {
        try executable.launchOn(stream);
        try stream.synchronize();
        try y.download(0, std.mem.sliceAsBytes(got));
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(eager), std.mem.sliceAsBytes(got))) return error.GraphReplayMismatch;
    }
    try out.print("{{\"case\":\"{s}\",\"unpack_bitwise_equal\":true,\"first_row_invariant\":true,\"graph_replay_equal\":true,\"relative_l2\":{d},\"max_abs_over_reference_rms\":{d}}}\n", .{ c.id, worst.relative_l2, worst.max_abs_over_reference_rms });
    try out.flush();
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("usage: tf-dsv41-linear-probe KERNEL_KIT FIXTURE_DIR\n", .{});
        return 2;
    }
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ args[2], "manifest.json" }), a, .limited(1 << 24));
    const manifest = try std.json.parseFromSlice(Manifest, a, raw, .{ .ignore_unknown_fields = true });
    defer manifest.deinit();
    const m = manifest.value;
    if (!std.mem.eql(u8, m.schema, "ds41f.exl3-linear-oracle.v1") or !std.mem.eql(u8, m.source_kind, "synthetic") or m.rows != 128 or m.cases.len == 0) return error.InvalidManifest;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var linear = try ds.exl3_linear.Kernels.load(&driver, try image(a, io, args[1], "linear_grouped.cubin"));
    defer linear.unload();
    var pf = try ds.exl3_prefill.Kernels.load(&driver, try image(a, io, args[1], "linear.cubin"));
    defer pf.unload();
    var buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &buf);
    for (m.cases) |c| {
        var arena = std.heap.ArenaAllocator.init(init.gpa);
        defer arena.deinit();
        try probe(arena.allocator(), io, &driver, stream, &linear, &pf, args[2], c, &out.interface);
    }
    try out.interface.print("{{\"status\":\"LINEAR_PROBES_PASSED_NOT_MODEL_QUALIFIED\",\"cases\":{d},\"model_qualified\":false}}\n", .{m.cases.len});
    try out.interface.flush();
    return 0;
}
