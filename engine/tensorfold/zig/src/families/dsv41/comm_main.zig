//! Native communication qualification, without loading model weights.
//! Default TP2 CLI is preserved; --world 3 uses the served multi-peer rendezvous.
//! Optional --rdma DEVICES checks both contiguous and strided gathers, including
//! the TP3 C8/48-row padded vocabulary payload. Run only in a closed GPU window.
const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41");

const hidden = 5120;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4) {
        std.debug.print("usage: tf-dsv41-comm RANK MASTER_IP PORT [--world 2|3] [--rdma DEVICES]\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[1], 10);
    const ip = try dsv41.link.parseIp(args[2]);
    const port = try std.fmt.parseInt(u16, args[3], 10);
    var world: u32 = 2;
    var devices: std.ArrayList([]const u8) = .empty;
    if ((args.len - 4) % 2 != 0) return error.BadArguments;
    var arg: usize = 4;
    while (arg < args.len) : (arg += 2) {
        if (std.mem.eql(u8, args[arg], "--world")) {
            world = try std.fmt.parseInt(u32, args[arg + 1], 10);
        } else if (std.mem.eql(u8, args[arg], "--rdma")) {
            if (devices.items.len != 0) return error.BadArguments;
            var it = std.mem.splitScalar(u8, args[arg + 1], ',');
            while (it.next()) |device| {
                if (device.len == 0) return error.BadArguments;
                try devices.append(a, device);
            }
        } else return error.BadArguments;
    }
    if ((world != 2 and world != 3) or rank >= world) return error.BadWorld;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var fan = try dsv41.prompt2d.Fan.openTimed(io, ip, port, rank, world, 120);
    defer fan.close();
    var comm = try fan.comm(rank, world);
    defer comm.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    const max_rows = 2048;
    var send = try cuda.DeviceBuffer.alloc(&driver, max_rows * hidden * 4);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(&driver, world * max_rows * hidden * 4);
    defer recv.free();
    const host = try a.alloc(f32, world * max_rows * hidden);
    for (host[0 .. max_rows * hidden], 0..) |*x, i| x.* = @floatFromInt((rank + 1) * 1000003 +% @as(u32, @intCast(i % 65521)));
    try send.upload(0, std.mem.sliceAsBytes(host[0 .. max_rows * hidden]));
    // correctness at 24 rows: rank r's slot holds rank r's pattern
    try comm.allGather(send.ptr, recv.ptr, 24 * hidden, .f32, stream);
    try stream.synchronize();
    try recv.download(0, std.mem.sliceAsBytes(host[0 .. world * 24 * hidden]));
    var ok = true;
    for (0..world) |r| for (0..24 * hidden) |i| {
        const want: f32 = @floatFromInt(@as(u32, @intCast(r + 1)) * 1000003 +% @as(u32, @intCast(i % 65521)));
        if (host[r * 24 * hidden + i] != want) ok = false;
    };
    var start = try cuda.Event.init(&driver, true);
    defer start.deinit();
    var end = try cuda.Event.init(&driver, true);
    defer end.deinit();
    var out_buf: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    try out.interface.print("{{\"rank\": {d}, \"world\": {d}, \"gather_ok\": {}", .{ rank, world, ok });
    for ([_]usize{ 1, 6, 24, 48, 2048 }) |rows| {
        const reps: usize = if (rows > 64) 50 else 500;
        for (0..10) |_| try comm.allGather(send.ptr, recv.ptr, rows * hidden, .f32, stream);
        try start.record(stream);
        for (0..reps) |_| try comm.allGather(send.ptr, recv.ptr, rows * hidden, .f32, stream);
        try end.record(stream);
        try end.synchronize();
        const ms = try cuda.Event.elapsedMs(start, end);
        try recv.download(0, std.mem.sliceAsBytes(host[0 .. world * rows * hidden]));
        for (0..world) |r| for (0..rows * hidden) |i| {
            const want: f32 = @floatFromInt(@as(u32, @intCast(r + 1)) * 1000003 +% @as(u32, @intCast(i % 65521)));
            if (host[r * rows * hidden + i] != want) ok = false;
        };
        try out.interface.print(", \"us_{d}_rows\": {d:.1}", .{ rows, ms * 1000 / @as(f32, @floatFromInt(reps)) });
    }
    var rdma_ok = true;
    if (devices.items.len != 0) {
        var module = try cuda.Module.load(&driver, cuda.kernels.dsv41_rdma);
        defer module.unload();
        // Same peer masks, GID, sizing and ring implementation as the served model.
        const width: usize = if (world == 3) 43136 else 64640;
        const ring = try dsv41.model.Model.openRingSized(init.gpa, &driver, try dsv41.rdma.Kernels.load(module), devices.items, rank, &comm, stream, 48 * width * 4);
        defer ring.destroy(init.gpa);
        for ([_]usize{ hidden, width }) |cols| {
            for ([_]usize{ 1, 6, 24, 48 }) |rows| {
                const n = rows * cols;
                // Change payload between cases and passes so stale ring contents fail.
                for (0..3) |pass| {
                    for (host[0..n], 0..) |*x, j| x.* = probeValue(rank, j, pass, rows);
                    try send.upload(0, std.mem.sliceAsBytes(host[0..n]));
                    try ring.gather(stream, send.ptr, @intCast(n / 4), recv.ptr);
                    try stream.synchronize();
                    try recv.download(0, std.mem.sliceAsBytes(host[0 .. world * n]));
                    for (0..world) |r| for (0..n) |j| {
                        if (host[r * n + j] != probeValue(@intCast(r), j, pass, rows)) rdma_ok = false;
                    };
                    var dsts: [dsv41.rdma.max_ranks]dsv41.rdma.Ring.Dst = undefined;
                    for (0..world) |r| dsts[r] = .{ .ptr = recv.ptr + r * cols * 4, .row4 = @intCast(cols / 4), .stride4 = world * cols / 4 };
                    try ring.gatherInto(stream, send.ptr, @intCast(rows), @intCast(cols / 4), dsts[0..world]);
                    try stream.synchronize();
                    // Capture and replay the same communication operations used in
                    // decode. Both layouts are checked separately after replay.
                    if (pass == 2) {
                        try cuda.graph.beginCapture(stream, .thread_local);
                        try ring.gather(stream, send.ptr, @intCast(n / 4), recv.ptr);
                        var graph = try cuda.graph.endCapture(stream);
                        defer graph.deinit();
                        var executable = try graph.instantiate();
                        defer executable.deinit();
                        try executable.launchOn(stream);
                        try stream.synchronize();
                        try recv.download(0, std.mem.sliceAsBytes(host[0 .. world * n]));
                        for (0..world) |r| for (0..n) |j| {
                            if (host[r * n + j] != probeValue(@intCast(r), j, pass, rows)) rdma_ok = false;
                        };
                        try cuda.graph.beginCapture(stream, .thread_local);
                        try ring.gatherInto(stream, send.ptr, @intCast(rows), @intCast(cols / 4), dsts[0..world]);
                        var into_graph = try cuda.graph.endCapture(stream);
                        defer into_graph.deinit();
                        var into_executable = try into_graph.instantiate();
                        defer into_executable.deinit();
                        try into_executable.launchOn(stream);
                        try stream.synchronize();
                    }
                    try recv.download(0, std.mem.sliceAsBytes(host[0 .. world * n]));
                    for (0..rows) |row| for (0..world) |r| for (0..cols) |col| {
                        if (host[row * world * cols + r * cols + col] != probeValue(@intCast(r), row * cols + col, pass, rows)) rdma_ok = false;
                    };
                }
            }
        }
        if (ring.failure() != null) rdma_ok = false;
    }
    try out.interface.print(", \"nccl_all_sizes_ok\": {}, \"rdma_checked\": {}, \"rdma_ok\": {}, \"model_numerics_qualified\": false", .{ ok, devices.items.len != 0, rdma_ok });
    try out.interface.print("}}\n", .{});
    try out.interface.flush();
    return if (ok and rdma_ok) 0 else 1;
}

fn probeValue(rank: u32, index: usize, pass: usize, rows: usize) f32 {
    return @floatFromInt((rank + 1) * 1000003 + @as(u32, @intCast(index % 65521 + pass * 70001 + rows * 997)));
}
