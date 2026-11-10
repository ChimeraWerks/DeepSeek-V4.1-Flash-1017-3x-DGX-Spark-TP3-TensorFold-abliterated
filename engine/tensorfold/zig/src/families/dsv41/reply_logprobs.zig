//! Opt-in compact shard exchange. Prompt copies live by slot, decode rows remain in the existing head buffer.
const std = @import("std");
const cuda = @import("cuda");
const lp = @import("lanes").logprob;
/// Startup agreement word 6 bit 58 (54 compact draft head, 55 round timing, 56 prefill checkpoints, 57 penalties).
pub const agreement_bit: u64 = @as(u64, 1) << 58;
/// The outer lanes frame kind (19 and 20 are the penalties' frames).
pub const frame_kind: u8 = 21;
pub fn enabled() bool {
    return if (std.c.getenv("TF_DS_REPLY_LOGPROBS")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
}
pub fn agree(a: u64, b: u64) !void {
    if ((a ^ b) & agreement_bit != 0) return error.ReplyLogprobsMismatch;
}
pub const Frame = struct {
    op: u32, // 0 save prompt, 1 score prompt, 2 score decode, 3 release prompt
    slot: u32,
    row: u32 = 0,
    token: u32 = 0,
    k: u32 = 0,
    pub fn validate(f: Frame) !void {
        if (f.op > 3 or f.slot >= 16 or f.k > 20 or f.row >= 48) return error.BadLogprobFrame;
    }
};
/// Rank 0's host-only merge, after every collective of the frame has completed on every rank: a non-finite or empty
/// summary fails only the request that asked (Stream.commit ends it with an error), never the rank group.
pub fn mergeRow(shards: ?[3]lp.Summary, token: u32, k: u32) error{ LogprobRowRejected, NoLogprobRow }!lp.Row {
    const got = shards orelse return error.NoLogprobRow;
    return lp.merge(&got, token, @intCast(k)) catch error.LogprobRowRejected;
}
/// Prompt snapshots and flattened verify windows use distinct storage.
pub fn rowFor(prompt: u64, position: u64, start: u64, base: u32, width: u64) !?u32 {
    if (position == prompt) return null;
    if (position < start or position - start >= width) return error.LogprobPosition;
    const row = @as(u64, base) + position - start;
    if (row >= 48) return error.LogprobPosition;
    return @intCast(row);
}
pub const State = struct {
    module: ?cuda.Module = null,
    kernel: ?cuda.Function = null,
    work: ?cuda.DeviceBuffer = null,
    prompts: [16]?cuda.DeviceBuffer = @splat(null),
    saved: [16]bool = @splat(false),
    /// TF_DS_REPLY_LOGPROBS: the module, the exchange buffer and every slot's prompt copy, once at open, so no score
    /// frame allocates or loads after it may have reached a follower (off: never called, nothing allocated).
    pub fn init(s: *State, d: *const cuda.Driver, width: usize) !void {
        errdefer s.deinit();
        if (cuda.kernels.dsv41_logprobs.len == 0) return error.NoLogprobKernel;
        s.module = try cuda.Module.load(d, cuda.kernels.dsv41_logprobs);
        s.kernel = try s.module.?.function("tf_ds_logprob_shard_kernel");
        s.work = try cuda.DeviceBuffer.alloc(d, 8 + @sizeOf(lp.Summary) * 4);
        for (&s.prompts) |*p| p.* = try cuda.DeviceBuffer.alloc(d, width * 4);
    }
    /// Teardown under the cleanup gate: a failed free or unload stops here, as Ops.unload does.
    pub fn deinit(s: *State) void {
        const cleanup = cuda.context_health.cleanupMark();
        for (&s.prompts) |*p| if (p.*) |*b| {
            b.free();
            p.* = null;
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
        };
        if (s.work) |*b| {
            b.free();
            s.work = null;
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
        }
        if (s.module) |*m| {
            m.unload();
            s.module = null;
            if (!cuda.context_health.cleanupReleased(cleanup)) return;
        }
        s.* = .{};
    }
    /// The prompt copies' custody, shared by every rank through the mirrored frames: op 0 saves a slot's copy, op 1
    /// scores it, op 3 (the stream's release, whatever ended it) frees the slot for its next stream.
    pub fn bookkeep(s: *State, f: Frame) error{NoPromptLogits}!void {
        switch (f.op) {
            0 => s.saved[f.slot] = true,
            1 => if (!s.saved[f.slot]) return error.NoPromptLogits,
            3 => s.saved[f.slot] = false,
            else => {},
        }
    }
    /// Every collective of a frame (upload, launch, all-gather, synchronize, download): a failure here may leave the
    /// ranks out of step, so the caller ends the group. Rank 0 gets the three summaries; mergeRow decides the row.
    pub fn run(s: *State, m: anytype, f: Frame) !?[3]lp.Summary {
        try f.validate();
        if (!m.reply_logprobs or m.two != null or m.eng.world != 3) return error.LogprobsUnsupported;
        const d = m.ctx.d;
        const width: usize = m.eng.w.head.n;
        if (s.module == null or s.work == null) return error.NoLogprobKernel; // (loaded at open: never here)
        if (f.op == 3) {
            try s.bookkeep(f);
            return null;
        }
        if (f.op == 0) {
            const copy = s.prompts[f.slot] orelse return error.NoPromptLogits;
            // Complete-prompt snapshots restore head_g, not the transient local head. head_g stays raw under
            // penalties (penalizePrompt writes its own scratch).
            try d.check(d.api.cuMemcpyDtoDAsync_v2(copy.ptr, m.promptLogits() + m.rank * width * 4, width * 4, m.stream.handle), "cuMemcpyDtoDAsync");
            try m.stream.synchronize();
            try s.bookkeep(f);
            return null;
        }
        if (f.token >= m.cfg.tokenVocab()) return error.BadLogprobFrame;
        const source = if (f.op == 1) blk: {
            try s.bookkeep(f);
            break :blk s.prompts[f.slot].?.ptr;
        } else blk: {
            if (f.row >= m.rd.r) return error.BadLogprobFrame;
            // a penalized round's raw rows (copied before the penalty kernel), else the head rows themselves
            break :blk (if (m.score_raw != 0) m.score_raw else m.rd.hl) + @as(u64, f.row) * width * 4;
        };
        const bytes = @sizeOf(lp.Summary);
        const work = &s.work.?;
        const token = [_]u32{f.token};
        try work.upload(0, std.mem.sliceAsBytes(&token));
        const local = work.ptr + 8;
        const all = local + bytes;
        const offset = m.rank * width;
        const n = @min(width, m.cfg.tokenVocab() -| offset);
        var args: cuda.Args = .{};
        args.add(source);
        args.add(@as(u64, width));
        args.add(@as(u32, @intCast(n)));
        args.add(@as(u32, @intCast(offset)));
        args.add(work.ptr);
        args.add(f.k);
        args.add(local);
        try cuda.launch.launch(s.kernel.?, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, m.stream, &args);
        // The same small allGather used by the reduced candidate head, never full logits.
        try m.comm.allGather(local, all, bytes / 8, .u64, m.stream);
        try m.stream.synchronize();
        if (m.rank != 0) return null;
        var summaries: [3]lp.Summary = undefined;
        try work.download(8 + bytes, std.mem.sliceAsBytes(&summaries));
        return summaries;
    }
};

test "reply logprobs agreement remains unchanged off and refuses mismatch" {
    try std.testing.expectEqual(@as(u64, 1) << 58, agreement_bit);
    const taken = @import("draft_head.zig").agreement_bit | @import("round_timing.zig").agreement_bit | @import("keyed_draft.zig").agreement_bit | @import("logit_penalties.zig").agreement_bit | (@as(u64, 1) << 56) | (@as(u64, 1) << 32);
    try std.testing.expectEqual(@as(u64, 0), taken & agreement_bit);
    try std.testing.expectEqual(@as(u8, 21), frame_kind);
    try agree(0, 0);
    try agree(agreement_bit, agreement_bit);
    try std.testing.expectError(error.ReplyLogprobsMismatch, agree(0, agreement_bit));
    try std.testing.expectError(error.ReplyLogprobsMismatch, agree(agreement_bit, 0));
}
test "reply logprobs frame bounds" {
    try (Frame{ .op = 2, .slot = 15, .row = 47, .token = 129279, .k = 20 }).validate();
    try std.testing.expectError(error.BadLogprobFrame, (Frame{ .op = 4, .slot = 0 }).validate());
    try std.testing.expectError(error.BadLogprobFrame, (Frame{ .op = 0, .slot = 16 }).validate());
    try std.testing.expectError(error.BadLogprobFrame, (Frame{ .op = 2, .slot = 0, .row = 48 }).validate());
    try std.testing.expectError(error.BadLogprobFrame, (Frame{ .op = 2, .slot = 0, .k = 21 }).validate());
}

test "reply logprobs commit positions select prompt or each stream's own verify row" {
    try std.testing.expectEqual(@as(?u32, null), try rowFor(123, 123, 0, 0, 6));
    for (0..6) |i| try std.testing.expectEqual(@as(?u32, @intCast(17 + i)), try rowFor(123, 150 + i, 150, 17, 6));
    try std.testing.expectError(error.LogprobPosition, rowFor(123, 149, 150, 17, 6));
    try std.testing.expectError(error.LogprobPosition, rowFor(123, 156, 150, 17, 6));
    try std.testing.expectError(error.LogprobPosition, rowFor(123, 155, 150, 44, 6));
}

test "reply logprobs merge failure fails only its request and the slot serves the next stream" {
    var st: State = .{};
    const slot: u32 = 5;
    // the first stream: prompt copy saved, its prompt token scored, then a decode row whose summary is non-finite
    try st.bookkeep(.{ .op = 0, .slot = slot });
    try st.bookkeep(.{ .op = 1, .slot = slot });
    var bad: [3]lp.Summary = undefined;
    for (&bad) |*x| x.* = std.mem.zeroes(lp.Summary);
    bad[0].maximum = std.math.nan(f32);
    try std.testing.expectError(error.LogprobRowRejected, mergeRow(bad, 7, 3));
    try std.testing.expectError(error.NoLogprobRow, mergeRow(null, 7, 3));
    // its release (op 3) runs as for any finished stream; the slot is free and unsaved on every rank
    try st.bookkeep(.{ .op = 3, .slot = slot });
    try std.testing.expectError(error.NoPromptLogits, st.bookkeep(.{ .op = 1, .slot = slot }));
    // the next stream on that slot saves and scores again
    try st.bookkeep(.{ .op = 0, .slot = slot });
    try st.bookkeep(.{ .op = 1, .slot = slot });
    var good: [3]lp.Summary = undefined;
    for (&good, 0..) |*x, r| {
        x.* = std.mem.zeroes(lp.Summary);
        x.maximum = 0;
        x.mass = 4;
        x.chosen = -std.math.inf(f32);
        x.owns = @intFromBool(r == 1);
        x.count = 0;
    }
    good[1].chosen = 0;
    const row = try mergeRow(good, 7, 0);
    try std.testing.expect(std.math.isFinite(row.chosen.logprob));
}
