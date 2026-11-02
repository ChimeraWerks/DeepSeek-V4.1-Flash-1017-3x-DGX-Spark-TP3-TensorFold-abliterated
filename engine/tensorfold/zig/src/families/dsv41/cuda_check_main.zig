//! CPU-only: real native loaders and wrappers, real cubin ELF symbols, mocked
//! CUDA driver. Proves symbol selection and launch construction, not GPU ABI
//! execution, numerical output, scratch sizing, or whole-model coverage.
const std = @import("std");
const cuda = @import("cuda");
const ds = @import("dsv41");
const abi = cuda.abi;
const ex = ds.exl3_experts;
var images: [4][]const u8 = undefined;
var function_names: [1024][]const u8 = undefined;
var nfunctions: usize = 0;
var launch_counts: [1024]usize = @splat(0);
var allocator: std.mem.Allocator = undefined;

fn moduleLoad(out: *abi.Module, data: ?*const anyopaque) callconv(.c) abi.Result {
    for (images, 0..) |bytes, i| if (@intFromPtr(data) == @intFromPtr(bytes.ptr)) {
        out.* = @ptrFromInt(i + 1);
        return 0;
    };
    return abi.error_not_found;
}
fn moduleUnload(_: abi.Module) callconv(.c) abi.Result { return 0; }
fn function(out: *abi.Function, module: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
    const index = @intFromPtr(module);
    if (index == 0 or index > images.len or nfunctions == function_names.len) return abi.error_not_found;
    var symbols = ds.elf.Symbols.init(images[index - 1]) orelse return abi.error_not_found;
    const requested = std.mem.span(name);
    while (symbols.next()) |symbol| if (std.mem.eql(u8, symbol, requested)) {
        function_names[nfunctions] = allocator.dupe(u8, symbol) catch return abi.error_not_found;
        nfunctions += 1;
        out.* = @ptrFromInt(nfunctions);
        return 0;
    };
    return abi.error_not_found;
}
fn errorText(_: abi.Result, out: *?[*:0]const u8) callconv(.c) abi.Result {
    out.* = "CPU_AUDIT_SYMBOL_NOT_FOUND";
    return 0;
}
fn record(f: abi.Function) abi.Result {
    const index = @intFromPtr(f);
    if (index == 0 or index > nfunctions) return abi.error_not_found;
    launch_counts[index - 1] += 1;
    return 0;
}
fn launch(f: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result { return record(f); }
fn launchEx(_: *const abi.LaunchConfig, f: abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result { return record(f); }

fn driver() cuda.Driver {
    var d: cuda.Driver = undefined; // Only the explicit CPU mocks below may be called.
    d.api.cuModuleLoadData = moduleLoad;
    d.api.cuModuleUnload = moduleUnload;
    d.api.cuModuleGetFunction = function;
    d.api.cuGetErrorName = errorText;
    d.api.cuGetErrorString = errorText;
    d.api.cuLaunchKernel = launch;
    d.api.cuLaunchKernelEx = launchEx;
    return d;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    allocator = a;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) return error.ExpectedCubinDirectoryAndProjectionShapes;
    for ([_][]const u8{ "experts.cubin", "experts_cb.cubin", "linear_grouped.cubin", "linear.cubin" }, 0..) |name, i| {
        const path = try std.fs.path.join(a, &.{ args[1], name });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(1 << 28));
        const aligned = try a.alignedAlloc(u8, .@"8", bytes.len);
        @memcpy(aligned, bytes);
        images[i] = aligned;
    }
    const d = driver();
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var experts = try ex.Kernels.load(&d, images[0], images[1], 2);
    defer experts.unload();
    var linear = try ds.exl3_linear.Kernels.load(&d, images[2]);
    defer linear.unload();
    var prefill = try ds.exl3_prefill.Kernels.load(&d, images[3]);
    defer prefill.unload();

    var weights: ds.weights.Experts = .{ .trellis = 0, .gate_ptr = 0, .up_ptr = 0, .down_ptr = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .count = 385, .dims = 5120, .width = 768, .down_k = 768, .down_n = 5120, .k2_gu = .{ 2, 16 }, .k2_d = .{ 2, 16 } };
    var ps: ex.Scratch = .{ .rows = 4096, .slots = 7, .xg = 0, .xu = 0, .xd = 0, .z = 0, .no_y = 0, .ids = 0, .count = 0, .counts = 0, .members = 0, .work_gu = 0, .work_d = 0 };
    var dec: ex.DecodeScratch = .{ .rows = 64, .slots = 7, .xg = 0, .xu = 0, .xd = 0, .z = 0, .y = 0, .cnt_gu = 0, .cnt_d = 0, .epoch = 0, .ready = 0, .ready_cnt = 0, .ids = 0, .count = 0, .members = 0 };
    var expert_calls: usize = 0;
    for ([_]u32{ 768, 1152 }) |width| {
        weights.width = width;
        weights.down_k = width;
        for ([_]usize{ 4, 7 }) |slots| {
            weights.count = if (slots == 4) 129 else 385;
            ps.slots = slots;
            dec.slots = slots;
            for ([_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 16 } }) |bits| {
                weights.k2_gu = bits;
                weights.k2_d = bits;
                for ([_]usize{ 1, 8, 24, 48, 63 }) |rows| {
                    try ex.decode(&experts, stream, weights, dec, 0, 5120, 0, 0, 0, rows, 10);
                    expert_calls += 1;
                }
                for ([_]usize{ 64, 128, 4096 }) |rows| {
                    try ex.prompt(&experts, stream, weights, ps, 0, 5120, 0, 0, 0, rows, 10);
                    expert_calls += 1;
                }
            }
        }
    }
    const ShapeFile = struct { projections: []struct { k: u32, n: u32, k2: []u32 } };
    const shape_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], a, .limited(1 << 22));
    const shapes = try std.json.parseFromSlice(ShapeFile, a, shape_bytes, .{ .ignore_unknown_fields = true });
    defer shapes.deinit();
    if (shapes.value.projections.len == 0) return error.EmptyProjections;
    var dense_calls: usize = 0;
    for (shapes.value.projections) |shape| {
        if (shape.k == 0 or shape.n == 0 or shape.k % 128 != 0 or shape.n % 128 != 0 or shape.k2.len == 0) return error.BadShape;
        for (shape.k2) |k2| {
            if (k2 < 2 or k2 > 16) return error.BadBits;
            const layer: ds.weights.Linear = .{ .words = 0, .suh = 0, .svh = 0, .k = shape.k, .n = shape.n, .k2 = k2 };
            for ([_]usize{ 1, 8, 48, 128 }) |rows| {
                const call: ds.exl3_linear.Call = .{ .layer = layer, .x = 0, .ldx = shape.k, .x_dtype = .bf16, .xh = 0, .y = 0, .ldy = shape.n, .y_dtype = .bf16, .counters = 0 };
                try ds.exl3_linear.rotMany(&linear, stream, &.{call}, rows, true);
                try ds.exl3_linear.glinear(&linear, stream, &.{call}, rows, 0, true, true);
                dense_calls += 1;
            }
            for ([_]usize{ 129, 4096 }) |rows| {
                try ds.exl3_prefill.rotIn(&prefill, stream, 0, .bf16, shape.k, 0, 0, rows, shape.k);
                try ds.exl3_prefill.unpack(&prefill, stream, layer, 0);
                dense_calls += 1;
            }
        }
    }
    const Row = struct { name: []const u8, launches: usize };
    var used: std.ArrayList(Row) = .empty;
    var count: usize = 0;
    for (launch_counts[0..nfunctions], 0..) |n, i| {
        if (n == 0) continue;
        try used.append(a, .{ .name = function_names[i], .launches = n });
        count += n;
    }
    var buf: [65536]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &buf);
    const json = try std.json.Stringify.valueAlloc(a, .{ .status = "CPU_SYMBOL_AND_LAUNCH_AUDIT_NOT_GPU_QUALIFIED", .expert_calls = expert_calls, .dense_calls = dense_calls, .cuda_launches = count, .symbols = used.items, .limitations = "mock driver; excludes kernel execution, numeric equivalence, full-model and top-k/RDMA coverage" }, .{});
    try out.interface.writeAll(json);
    try out.interface.writeAll("\n");
    try out.interface.flush();
}
