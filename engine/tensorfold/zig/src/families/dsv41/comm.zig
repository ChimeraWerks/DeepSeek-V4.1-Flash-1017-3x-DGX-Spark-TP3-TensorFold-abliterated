//! The TP ranks' collectives: NCCL opened at run time, its unique id passed from rank 0 over the link. Every model
//! reduction is an all-gather of fp32 partials that each rank then adds in rank order (DESIGN.md section 3), so the
//! ranks hold the same bits without a broadcast.
const std = @import("std");
const trace = @import("lanes").slice_trace;
const cuda = @import("cuda");
const Link = @import("link.zig").Link;

const nccl = cuda.nccl;

// Failed initialization has no documented "nothing allocated" result. Keep a durable
// in-process custody record, including null outputs, instead of discarding the handle.
const Quarantine = struct { lib: nccl.Library, owner: cuda.context_health.Owner, comm: nccl.Comm = null, next: ?*Quarantine = null };
var quarantined: ?*Quarantine = null;
var quarantine_mutex: std.atomic.Mutex = .unlocked;
fn quarantine(q: *Quarantine) void {
    while (!quarantine_mutex.tryLock()) std.Thread.yield() catch {};
    defer quarantine_mutex.unlock();
    q.next = quarantined;
    quarantined = q;
}


pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    owner: cuda.context_health.Owner = .{},
    /// TF_DS_DIRECT_GATHER (set after the start-up agreement, every rank alike): all-gathers as one group of sends and
    /// receives to every peer instead of NCCL's ring, so on the triangle each rank's part crosses both of its links at
    /// once, both directions busy, one hop. The bytes land where the ring puts them.
    direct: bool = false,

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        try cuda.context_health.ensureBoundHealthy();
        var lib = try nccl.Library.open();
        var close_lib = true;
        errdefer if (close_lib) lib.close();
        var owner = cuda.context_health.Owner.capture();
        try owner.ensure();
        errdefer if (close_lib) owner.release();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        const custody = try std.heap.page_allocator.create(Quarantine);
        custody.* = .{ .lib = lib, .owner = owner };
        var keep_custody = false;
        defer if (!keep_custody) std.heap.page_allocator.destroy(custody);
        const outcome = try initialize(&lib, owner, rank, world, id);
        close_lib = outcome.released;
        const comm = outcome.comm;
        if (outcome.result != 0) {
            custody.comm = comm;
            quarantine(custody);
            keep_custody = true;
            cuda.context_health.cleanupRetained();
            return error.NcclFailed;
        }
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world, .owner = owner };
    }

    const InitOutcome = struct { comm: nccl.Comm = null, result: nccl.Result, released: bool = false };
    fn initialize(lib: *const nccl.Library, owner: cuda.context_health.Owner, rank: u32, world: u32, id: nccl.UniqueId) !InitOutcome {
        // Admission is after ID exchange. No registry lock is held across NCCL.
        const admission = try owner.admit();
        defer admission.end();
        var outcome: InitOutcome = .{ .result = 0 };
        outcome.result = lib.api.ncclCommInitRank(&outcome.comm, @intCast(world), id, @intCast(rank));
        // Even a null failed output may conceal allocated partial state. Never unload it.
        return outcome;
    }

    pub fn deinit(c: *Comm) void {
        if (!cuda.context_health.boundHealthy()) { cuda.context_health.cleanupRetained(); return; }
        const admission = c.owner.admit() catch { cuda.context_health.cleanupRetained(); return; };
        defer admission.end();
        if (blk: { var ts = trace.begin("comm.deinit.ncclCommDestroy.site_86", .wait); defer ts.end(); break :blk c.lib.api.ncclCommDestroy(c.comm); } != 0) { cuda.context_health.cleanupRetained(); return; }
        c.lib.close();
        c.owner.release();
        c.* = undefined;
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        var trace_scope = trace.begin("comm.allGather", .wait);
        defer trace_scope.end();
        try cuda.context_health.ensureBoundHealthy();
        try c.owner.ensure();
        try stream.owner.ensure();
        const admission = try c.owner.admit();
        defer admission.end();
        if (c.direct) {
            const bytes = count * dtypeBytes(dtype);
            try c.lib.check(blk: { var ts = trace.begin("comm.allGather.ncclGroupStart.site_103", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupStart(); }, "ncclGroupStart");
            const first = c.gatherDirect(send, recv, bytes, stream);
            try c.lib.check(blk: { var ts = trace.begin("comm.allGather.ncclGroupEnd.site_105", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupEnd(); }, "ncclGroupEnd");
            return c.lib.check(first, "ncclSend/ncclRecv");
        }
        try c.lib.check(blk: { var ts = trace.begin("comm.allGather.ncclAllGather.site_108", .wait); defer ts.end(); break :blk c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle); }, "ncclAllGather");
    }

    /// One all-gather's sends and receives (inside the caller's group): this rank's `bytes` from `send` to every peer and
    /// into its own slot of `recv` when `send` is not that slot, every peer's into its slot. The first failure, or 0.
    fn gatherDirect(c: Comm, send: u64, recv: u64, bytes: usize, stream: cuda.Stream) nccl.Result {
        var first: nccl.Result = 0;
        if (bytes == 0) return first;
        for (0..c.world) |j| {
            if (j == c.rank and send == recv + j * bytes) continue;
            const s = blk: { var ts = trace.begin("comm.gatherDirect.ncclSend.site_118", .wait); defer ts.end(); break :blk c.lib.api.ncclSend(send, bytes, .u8, @intCast(j), c.comm, stream.handle); };
            const r = blk: { var ts = trace.begin("comm.gatherDirect.ncclRecv.site_119", .wait); defer ts.end(); break :blk c.lib.api.ncclRecv(recv + j * bytes, bytes, .u8, @intCast(j), c.comm, stream.handle); };
            if (first == 0) first = s;
            if (first == 0) first = r;
        }
        return first;
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: cuda.Stream) !void {
        var trace_scope = trace.begin("comm.exchange", .wait);
        defer trace_scope.end();
        try cuda.context_health.ensureBoundHealthy();
        try c.owner.ensure();
        try stream.owner.ensure();
        const admission = try c.owner.admit();
        defer admission.end();
        try c.lib.check(blk: { var ts = trace.begin("comm.exchange.ncclGroupStart.site_135", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupStart(); }, "ncclGroupStart");
        const s = blk: { var ts = trace.begin("comm.exchange.ncclSend.site_136", .wait); defer ts.end(); break :blk c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle); };
        const r = blk: { var ts = trace.begin("comm.exchange.ncclRecv.site_137", .wait); defer ts.end(); break :blk c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle); };
        try c.lib.check(blk: { var ts = trace.begin("comm.exchange.ncclGroupEnd.site_138", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupEnd(); }, "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
    }

    /// TF_DS_KV_SHARD (GLM 0118's send_rows / recv_rows): device byte spans one way, `spans` to rank `peer` in one NCCL
    /// group (the peer calls recvSpans with spans of the same sizes in the same order; each send meets its receive).
    /// A plane's rows go straight from the sender's plane into the receiver's: no staging.
    pub fn sendSpans(c: Comm, spans: []const Span, peer: u32, stream: cuda.Stream) !void {
        try c.p2p(spans, peer, stream, true);
    }

    pub fn recvSpans(c: Comm, spans: []const Span, peer: u32, stream: cuda.Stream) !void {
        try c.p2p(spans, peer, stream, false);
    }

    fn p2p(c: Comm, spans: []const Span, peer: u32, stream: cuda.Stream, out: bool) !void {
        try cuda.context_health.ensureBoundHealthy();
        try c.owner.ensure();
        try stream.owner.ensure();
        const admission = try c.owner.admit();
        defer admission.end();
        if (peer == c.rank or peer >= c.world) return error.BadPeer;
        var any = false;
        for (spans) |sp| any = any or sp.bytes > 0;
        if (!any) return;
        try c.lib.check(blk: { var ts = trace.begin("comm.p2p.ncclGroupStart.site_164", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupStart(); }, "ncclGroupStart");
        var first: nccl.Result = 0;
        for (spans) |sp| {
            if (sp.bytes == 0) continue;
            const r = if (out)
                blk: { var ts = trace.begin("comm.p2p.ncclSend.site_169", .wait); defer ts.end(); break :blk c.lib.api.ncclSend(sp.ptr, sp.bytes, .u8, @intCast(peer), c.comm, stream.handle); }
            else
                blk: { var ts = trace.begin("comm.p2p.ncclRecv.site_171", .wait); defer ts.end(); break :blk c.lib.api.ncclRecv(sp.ptr, sp.bytes, .u8, @intCast(peer), c.comm, stream.handle); };
            if (first == 0) first = r;
        }
        try c.lib.check(blk: { var ts = trace.begin("comm.p2p.ncclGroupEnd.site_174", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupEnd(); }, "ncclGroupEnd");
        try c.lib.check(first, if (out) "ncclSend" else "ncclRecv");
    }

    /// A reduce-scatter's exchange without its sum (TF_DS_PREFILL_OWNED): rank j's rows of `send` (`bytes[j]` at
    /// `send + offs[j]`) to rank j, and from every other rank its rows for this one into `recv`'s slot of that rank
    /// (recv + j * bytes[rank]), in one NCCL group, so both links of the triangle carry at once. This rank's own slot
    /// is the caller's to copy.
    pub fn scatterRows(c: Comm, send: u64, offs: []const usize, bytes: []const usize, recv: u64, stream: cuda.Stream) !void {
        var trace_scope = trace.begin("comm.scatterRows", .wait);
        defer trace_scope.end();
        try cuda.context_health.ensureBoundHealthy();
        try c.owner.ensure();
        try stream.owner.ensure();
        const admission = try c.owner.admit();
        defer admission.end();
        if (offs.len != c.world or bytes.len != c.world) return error.BadPeer;
        const mine = bytes[c.rank];
        try c.lib.check(blk: { var ts = trace.begin("comm.scatterRows.ncclGroupStart.site_192", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupStart(); }, "ncclGroupStart");
        var first: nccl.Result = 0;
        for (0..c.world) |j| {
            if (j == c.rank) continue;
            if (bytes[j] > 0) {
                const r = blk: { var ts = trace.begin("comm.scatterRows.ncclSend.site_197", .wait); defer ts.end(); break :blk c.lib.api.ncclSend(send + offs[j], bytes[j], .u8, @intCast(j), c.comm, stream.handle); };
                if (first == 0) first = r;
            }
            if (mine > 0) {
                const r = blk: { var ts = trace.begin("comm.scatterRows.ncclRecv.site_201", .wait); defer ts.end(); break :blk c.lib.api.ncclRecv(recv + j * mine, mine, .u8, @intCast(j), c.comm, stream.handle); };
                if (first == 0) first = r;
            }
        }
        try c.lib.check(blk: { var ts = trace.begin("comm.scatterRows.ncclGroupEnd.site_205", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupEnd(); }, "ncclGroupEnd");
        try c.lib.check(first, "ncclSend/ncclRecv");
    }

    /// `count` bytes from this rank's slot of `buf` (buf + rank * count) to every rank's slot, in place (NCCL's
    /// in-place all-gather: the send buffer is the receive buffer's own slot).
    pub fn allGatherInPlace(c: Comm, buf: u64, count: usize, stream: cuda.Stream) !void {
        var trace_scope = trace.begin("comm.allGatherInPlace", .wait);
        defer trace_scope.end();
        try c.allGather(buf + c.rank * count, buf, count, .u8, stream);
    }
    pub fn allGatherInPlace2(c: Comm, a: u64, a_count: usize, b: u64, b_count: usize, stream: cuda.Stream) !void {
        var trace_scope = trace.begin("comm.allGatherInPlace2", .wait);
        defer trace_scope.end();
        try cuda.context_health.ensureBoundHealthy();
        try c.owner.ensure();
        try stream.owner.ensure();
        const admission = try c.owner.admit();
        defer admission.end();
        try c.lib.check(blk: { var ts = trace.begin("comm.allGatherInPlace2.ncclGroupStart.site_224", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupStart(); }, "ncclGroupStart");
        const x = if (c.direct) c.gatherDirect(a + c.rank * a_count, a, a_count, stream) else blk: { var ts = trace.begin("comm.allGatherInPlace2.ncclAllGather.site_225", .wait); defer ts.end(); break :blk c.lib.api.ncclAllGather(a + c.rank * a_count, a, a_count, .u8, c.comm, stream.handle); };
        const y = if (c.direct) c.gatherDirect(b + c.rank * b_count, b, b_count, stream) else blk: { var ts = trace.begin("comm.allGatherInPlace2.ncclAllGather.site_226", .wait); defer ts.end(); break :blk c.lib.api.ncclAllGather(b + c.rank * b_count, b, b_count, .u8, c.comm, stream.handle); };
        try c.lib.check(blk: { var ts = trace.begin("comm.allGatherInPlace2.ncclGroupEnd.site_227", .wait); defer ts.end(); break :blk c.lib.api.ncclGroupEnd(); }, "ncclGroupEnd");
        try c.lib.check(x, "ncclAllGather");
        try c.lib.check(y, "ncclAllGather");
    }

};

fn dtypeBytes(d: nccl.DataType) usize {
    return switch (d) {
        .i8, .u8 => 1,
        .f16, .bf16 => 2,
        .i32, .u32, .f32 => 4,
        .i64, .u64, .f64 => 8,
    };
}

/// A device byte span (Comm.sendSpans / recvSpans).
pub const Span = struct { ptr: u64, bytes: usize };

test "NCCL custody quarantines null and published partial init and rejects post exchange poison" {
    const Mock = struct {
        var calls: usize = 0;
        var partial: bool = false;
        fn init(out: *nccl.Comm, _: c_int, _: nccl.UniqueId, _: c_int) callconv(.c) nccl.Result {
            calls += 1;
            out.* = if (partial) @ptrFromInt(1) else null;
            return 2;
        }
    };
    const ctx: cuda.abi.Context = @ptrFromInt(0xbe01);
    try cuda.context_health.retain(0xbe02, ctx);
    cuda.context_health.setCurrent(0xbe02, ctx);
    defer cuda.context_health.clearCurrent(0xbe02, ctx);
    var owner = cuda.context_health.Owner.capture();
    defer owner.release();
    var lib: nccl.Library = undefined;
    lib.api.ncclCommInitRank = Mock.init;
    Mock.calls = 0;
    for ([_]bool{ false, true }) |partial| {
        Mock.partial = partial;
        const result = try Comm.initialize(&lib, owner, 0, 1, .{ .internal = @splat(0) });
        try std.testing.expect(!result.released);
        try std.testing.expectEqual(@as(nccl.Result, 2), result.result);
        try std.testing.expect((result.comm != null) == partial);
    }
    cuda.context_health.observe(0xbe02, ctx, 700);
    try std.testing.expectError(error.CudaContextLost, Comm.initialize(&lib, owner, 0, 1, .{ .internal = @splat(0) }));
    try std.testing.expectEqual(@as(usize, 2), Mock.calls);
}

test "a direct all-gather sends this rank's part to every peer and fills every slot" {
    const Mock = struct {
        var sends: [8][2]u64 = undefined;
        var recvs: [8][2]u64 = undefined;
        var ns: usize = 0;
        var nr: usize = 0;
        fn send(buf: cuda.abi.DevicePtr, _: usize, _: nccl.DataType, peer: c_int, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) nccl.Result {
            sends[ns] = .{ buf, @intCast(peer) };
            ns += 1;
            return 0;
        }
        fn recv(buf: cuda.abi.DevicePtr, _: usize, _: nccl.DataType, peer: c_int, _: nccl.Comm, _: cuda.abi.Stream) callconv(.c) nccl.Result {
            recvs[nr] = .{ buf, @intCast(peer) };
            nr += 1;
            return 0;
        }
    };
    var lib: nccl.Library = undefined;
    lib.api.ncclSend = Mock.send;
    lib.api.ncclRecv = Mock.recv;
    const c: Comm = .{ .lib = lib, .comm = null, .rank = 1, .world = 3, .direct = true };
    const stream: cuda.Stream = undefined;
    // in place: the own slot is skipped
    Mock.ns = 0;
    Mock.nr = 0;
    try std.testing.expectEqual(@as(nccl.Result, 0), c.gatherDirect(1000 + 100, 1000, 100, stream));
    try std.testing.expectEqual(@as(usize, 2), Mock.ns);
    try std.testing.expectEqualSlices(u64, &.{ 1100, 0, 1100, 2 }, std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(Mock.sends[0..2])));
    try std.testing.expectEqualSlices(u64, &.{ 1000, 0, 1200, 2 }, std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(Mock.recvs[0..2])));
    // separate send buffer: the own slot is filled by a send to self
    Mock.ns = 0;
    Mock.nr = 0;
    try std.testing.expectEqual(@as(nccl.Result, 0), c.gatherDirect(5000, 1000, 100, stream));
    try std.testing.expectEqual(@as(usize, 3), Mock.nr);
    try std.testing.expectEqualSlices(u64, &.{ 1000, 0, 1100, 1, 1200, 2 }, std.mem.bytesAsSlice(u64, std.mem.sliceAsBytes(Mock.recvs[0..3])));
    try std.testing.expectEqual(@as(usize, 2), dtypeBytes(.bf16));
    try std.testing.expectEqual(@as(usize, 8), dtypeBytes(.u64));
}
